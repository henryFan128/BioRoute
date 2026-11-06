import argparse
import json
import logging
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import yaml
from torch_geometric.data import HeteroData

try:
    from negative_sampler import NegativeSampler
except ImportError:
    from script.negative_sampler import NegativeSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.query_sampler import build_query_groups_from_edge_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

EdgeTriple = Tuple[str, str, str]
ENTITY_TYPE_MAP = {
    "chemicals": "chemical",
    "proteins": "protein",
    "diseases": "disease",
    "genes": "gene",
    "pathogens": "pathogen",
}


# =========================================================================
# Node ID → local index mapping
# =========================================================================


def build_node_index(
    entities: Dict[str, dict],
) -> Dict[str, Dict[int, int]]:
    """Create {entity_id: local_idx} mapping for each node type.

    Args:
        entities: {"chemicals": {id: {...}}, "proteins": ..., ...}

    Returns:
        {"chemical": {id: local_idx}, "protein": ..., ...}
    """
    node_index: Dict[str, Dict[int, int]] = {}
    for dict_key, type_name in ENTITY_TYPE_MAP.items():
        ids = sorted(entities[dict_key].keys())
        node_index[type_name] = {eid: idx for idx, eid in enumerate(ids)}
        logger.info(f"  {type_name}: {len(ids)} nodes")
    return node_index


def get_display_name(node_type: str, attrs: Dict[str, object]) -> str:
    """Return the human-readable label used in exported entity lookup files."""
    if node_type == "gene":
        return str(attrs.get("gene_symbols") or "").strip()
    return str(attrs.get("name") or "").strip()


def get_identifier(attrs: Dict[str, object]) -> str:
    """Return a normalized external identifier if present."""
    candidates = ("identifier", "Identifier", "id", "ID")
    for key in candidates:
        value = attrs.get(key)
        if value is None:
            continue
        value = str(value).strip()
        if value:
            return value
    return ""


def build_entity_lookup(
    entities: Dict[str, dict],
    node_index: Dict[str, Dict[int, int]],
) -> Dict[str, List[Dict[str, object]]]:
    """Create a unified entity lookup table for downstream inference.

    Each record links the original entity ID from preprocessing to the local
    graph index used inside the PyG graph, along with a display name.
    """
    entity_lookup: Dict[str, List[Dict[str, object]]] = {}

    for dict_key, node_type in ENTITY_TYPE_MAP.items():
        records: List[Dict[str, object]] = []
        for entity_id in sorted(entities[dict_key].keys()):
            attrs = entities[dict_key][entity_id]
            display_name = get_display_name(node_type, attrs)
            record: Dict[str, object] = {
                "entity_id": int(entity_id),
                "graph_index": int(node_index[node_type][entity_id]),
                "display_name": display_name,
            }
            if display_name:
                record["name"] = display_name
            identifier = get_identifier(attrs)
            if identifier:
                record["identifier"] = identifier
            records.append(record)
        entity_lookup[node_type] = records

    return entity_lookup


# =========================================================================
# Load feature matrices and attach to HeteroData
# =========================================================================


def load_feature_matrices(
    features_dir: str,
    node_index: Dict[str, Dict[int, int]],
) -> Dict[str, torch.Tensor]:
    """Load .npz feature files and return {node_type: Tensor[N, feat_dim]}.

    The .npz files store an 'ids' array that defines the row order of the
    feature matrix.  We re-index to match ``node_index`` ordering.
    """
    loaders = {
        "chemical": ("chemical_features.npz", ["chembert", "morgan"]),
        "protein": ("protein_features.npz", ["esm2"]),
        "disease": ("disease_features.npz", ["biobert"]),
        "gene": ("gene_features.npz", ["multihot"]),
        "pathogen": ("pathogen_features.npz", ["multihot"]),
    }

    x_dict: Dict[str, torch.Tensor] = {}

    for node_type, (fname, keys) in loaders.items():
        fpath = os.path.join(features_dir, fname)
        data = np.load(fpath, allow_pickle=True)
        ids = data["ids"].tolist()  # row order in the npz file

        # Build id→row lookup
        id2row = {eid: i for i, eid in enumerate(ids)}

        # Re-order rows to match node_index ordering
        local_order = sorted(
            node_index[node_type].keys(), key=lambda eid: node_index[node_type][eid]
        )

        matrices = []
        for key in keys:
            mat = data[key]  # [N_npz, feat_dim]
            reordered = np.stack([mat[id2row[eid]] for eid in local_order], axis=0)
            matrices.append(reordered)

        if len(matrices) == 1:
            feat = matrices[0]
        else:
            # Concatenate along feature dim (e.g. chemical: chembert ⊕ morgan)
            feat = np.concatenate(matrices, axis=1)  # [N, 768+2048]

        x_dict[node_type] = torch.tensor(feat, dtype=torch.float32)
        logger.info(f"  {node_type}: x.shape={x_dict[node_type].shape}")

    return x_dict


# =========================================================================
# Build edge_index tensors
# =========================================================================


def build_edge_indices(
    all_edges: Dict[EdgeTriple, List[Tuple[int, int]]],
    node_index: Dict[str, Dict[int, int]],
) -> Dict[EdgeTriple, torch.Tensor]:
    """Convert global entity ID pairs to local index pairs.

    Args:
        all_edges:   {(src_type, rel, dst_type): [(src_id, dst_id), ...]}
        node_index:  {node_type: {entity_id: local_idx}}

    Returns:
        {(src_type, rel, dst_type): LongTensor[2, num_edges]}
    """
    edge_index_dict: Dict[EdgeTriple, torch.Tensor] = {}

    for triple, id_pairs in all_edges.items():
        src_type, rel, dst_type = triple
        src_idx_map = node_index[src_type]
        dst_idx_map = node_index[dst_type]

        valid_pairs = [
            (src_idx_map[s], dst_idx_map[d])
            for s, d in id_pairs
            if s in src_idx_map and d in dst_idx_map
        ]
        skipped = len(id_pairs) - len(valid_pairs)
        if skipped:
            logger.warning(
                f"  Skipped {skipped} edges for {triple} (nodes not in index)"
            )

        if not valid_pairs:
            continue

        srcs, dsts = zip(*valid_pairs)
        edge_index = torch.tensor([list(srcs), list(dsts)], dtype=torch.long)
        edge_index_dict[triple] = edge_index
        logger.info(f"  {triple}: {edge_index.shape[1]} edges")

    return edge_index_dict


def add_logical_reverse_edges(
    edge_index_dict: Dict[EdgeTriple, torch.Tensor],
    reverse_specs: List[dict],
) -> Dict[EdgeTriple, torch.Tensor]:
    """Explicitly derive logical reverse relations from config specs."""
    out = dict(edge_index_dict)
    for spec in reverse_specs:
        forward = tuple(spec["forward"])
        reverse = tuple(spec["reverse"])
        if forward not in edge_index_dict:
            continue
        out[reverse] = edge_index_dict[forward].flip(0).contiguous()
        logger.info(f"  derived reverse relation: {forward} -> {reverse}")
    return out


def derive_reverse_split_edges(
    split_edges: Dict[EdgeTriple, torch.Tensor],
    reverse_relation_map: Dict[EdgeTriple, EdgeTriple],
) -> Dict[EdgeTriple, torch.Tensor]:
    """Derive split-aligned reverse supervision edges after forward split."""
    out: Dict[EdgeTriple, torch.Tensor] = {}
    for forward_triple, edge_index in split_edges.items():
        reverse_triple = reverse_relation_map.get(forward_triple)
        if reverse_triple is None:
            continue
        out[reverse_triple] = edge_index.flip(0).contiguous()
    return out


def save_query_groups(
    split_dir: str,
    split_name: str,
    query_edge_index: torch.Tensor,
    query_relation: EdgeTriple,
    num_queries: int,
    num_candidates: int,
    metadata: Optional[Dict[str, object]] = None,
) -> str:
    """Persist query-centric grouped supervision for disease-query retrieval."""
    query_groups = build_query_groups_from_edge_index(
        edge_index=query_edge_index,
        num_queries=num_queries,
        num_candidates=num_candidates,
        relation=query_relation,
        split=split_name,
        metadata=dict(metadata or {}),
    )
    out_path = os.path.join(split_dir, f"{split_name}_queries.pt")
    torch.save(query_groups.to_dict(), out_path)
    logger.info(
        f"{split_name}_queries.pt saved  active_queries={query_groups.num_active_queries} "
        f"positive_edges={query_groups.num_positive_edges} relation={query_relation}"
    )
    return out_path


# =========================================================================
# Supervision edge separation
# =========================================================================


def separate_supervision_edges(
    edge_index_dict: Dict[EdgeTriple, torch.Tensor],
    supervision_triples: List[EdgeTriple],
    background_exclude_triples: Optional[Set[EdgeTriple]] = None,
) -> Tuple[Dict[EdgeTriple, torch.Tensor], Dict[EdgeTriple, torch.Tensor]]:
    """Split edge_index_dict into background edges and supervision edges.

    Args:
        edge_index_dict:      Full edge dict.
        supervision_triples:  Edge types used as supervision targets.
        background_exclude_triples:
            Edge types that must not be used for message-passing background,
            such as logical reverse edges of held-out supervision edges.

    Returns:
        (background_edge_dict, supervision_edge_dict)
    """
    background: Dict[EdgeTriple, torch.Tensor] = {}
    supervision: Dict[EdgeTriple, torch.Tensor] = {}
    excluded_background_triples = set(background_exclude_triples or set())

    for triple, edge_index in edge_index_dict.items():
        if triple in supervision_triples:
            supervision[triple] = edge_index
        elif triple in excluded_background_triples:
            continue
        else:
            background[triple] = edge_index

    logger.info(f"Background edge types: {len(background)}")
    logger.info(f"Supervision edge types: {len(supervision)}")
    return background, supervision


# =========================================================================
# Split supervision edges
# =========================================================================


def split_supervision_edges(
    supervision_edges: Dict[EdgeTriple, torch.Tensor],
    split_ratio: List[float],
    primary_triple: EdgeTriple,
    query_axis: int,
    split_mode: str,
    seed: int = 42,
    *,
    test_seed: Optional[int] = None,
    split_seed: Optional[int] = None,
    freeze_test_split: bool = False,
) -> Tuple[
    Dict[EdgeTriple, torch.Tensor],
    Dict[EdgeTriple, torch.Tensor],
    Dict[EdgeTriple, torch.Tensor],
    Dict[str, Any],
]:
    """Split supervision edges for either transductive, query-level, or disease cold-start retrieval.

    Modes:
        - ``edge_level``: classic transductive split; every supervision edge is
          shuffled independently.
        - ``query_level``: grouped split by query node on the primary retrieval
          relation, keeping one query entirely inside one split.
        - ``disease_cold_start``: split by disease node IDs and assign supervision
          edges by disease membership to ensure node-disjoint train/val/test diseases.

    Returns:
        (train_sup, val_sup, test_sup, split_meta)
    """
    if query_axis not in (0, 1):
        raise ValueError(f"query_axis must be 0 or 1, got {query_axis}")
    if primary_triple not in supervision_edges:
        raise KeyError(f"Primary supervision triple not found: {primary_triple}")

    split_mode = str(split_mode).lower()
    if split_mode not in {"edge_level", "query_level", "disease_cold_start"}:
        raise ValueError(
            "Unsupported split_mode="
            f"{split_mode!r}. Expected 'edge_level', 'query_level', or 'disease_cold_start'."
        )

    if len(split_ratio) != 3:
        raise ValueError(f"split_ratio must have 3 values, got {split_ratio}")

    legacy_rng = random.Random(seed)
    effective_test_seed = int(seed if test_seed is None else test_seed)
    effective_split_seed = int(seed if split_seed is None else split_seed)

    def _split_indices_with_optional_fixed_test(
        indices: List[int],
    ) -> Tuple[List[int], List[int], List[int]]:
        total = len(indices)
        if total == 0:
            return [], [], []

        train_r, val_r, test_r = split_ratio
        if not freeze_test_split:
            shuffled = list(indices)
            legacy_rng.shuffle(shuffled)
            n_train = int(total * train_r)
            n_val = int(total * val_r)
            train_ids = shuffled[:n_train]
            val_ids = shuffled[n_train : n_train + n_val]
            test_ids = shuffled[n_train + n_val :]
            return train_ids, val_ids, test_ids

        test_rng = random.Random(effective_test_seed)
        split_rng = random.Random(effective_split_seed)

        shuffled = list(indices)
        test_rng.shuffle(shuffled)
        n_test = int(total * test_r)
        test_ids = shuffled[:n_test]
        remain_ids = shuffled[n_test:]

        if not remain_ids:
            return [], [], test_ids

        train_val_ratio = float(train_r + val_r)
        if train_val_ratio <= 0.0:
            raise ValueError(
                "freeze_test_split requires train_ratio + val_ratio > 0 "
                f"(got train={train_r}, val={val_r})"
            )

        split_rng.shuffle(remain_ids)
        train_fraction = float(train_r) / train_val_ratio
        n_train = int(len(remain_ids) * train_fraction)
        train_ids = remain_ids[:n_train]
        val_ids = remain_ids[n_train:]
        return train_ids, val_ids, test_ids

    train_sup: Dict[EdgeTriple, torch.Tensor] = {}
    val_sup: Dict[EdgeTriple, torch.Tensor] = {}
    test_sup: Dict[EdgeTriple, torch.Tensor] = {}

    split_meta: Dict[str, Any] = {
        "split_mode": split_mode,
        "seed": int(seed),
        "freeze_test_split": bool(freeze_test_split),
        "effective_test_seed": int(effective_test_seed),
        "effective_split_seed": int(effective_split_seed),
    }

    if split_mode == "edge_level":
        logger.info("  Split mode: transductive edge-level")
        for triple, edge_index in supervision_edges.items():
            indices = list(range(edge_index.shape[1]))
            t_idx, v_idx, s_idx = _split_indices_with_optional_fixed_test(indices)
            train_sup[triple] = edge_index[:, t_idx]
            val_sup[triple] = edge_index[:, v_idx]
            test_sup[triple] = edge_index[:, s_idx]
            logger.info(
                f"  {triple}: train={len(t_idx)}, val={len(v_idx)}, test={len(s_idx)}"
            )
        return train_sup, val_sup, test_sup, split_meta

    if split_mode == "disease_cold_start":
        cold_node_type = primary_triple[2]

        cold_node_ids = sorted(
            {
                int(node_id)
                for triple, edge_index in supervision_edges.items()
                if edge_index.numel() > 0
                for edge_axis, triple_pos in [(0, 0), (1, 2)]
                if triple[triple_pos] == cold_node_type
                for node_id in edge_index[edge_axis].tolist()
            }
        )
        if not cold_node_ids:
            raise ValueError(
                "No node ids found for disease cold-start split. "
                f"cold_node_type={cold_node_type}"
            )

        train_node_idx, val_node_idx, test_node_idx = _split_indices_with_optional_fixed_test(
            list(range(len(cold_node_ids)))
        )
        train_nodes = [cold_node_ids[i] for i in train_node_idx]
        val_nodes = [cold_node_ids[i] for i in val_node_idx]
        test_nodes = [cold_node_ids[i] for i in test_node_idx]

        node_to_split = {int(nid): "train" for nid in train_nodes}
        node_to_split.update({int(nid): "val" for nid in val_nodes})
        node_to_split.update({int(nid): "test" for nid in test_nodes})

        logger.info(
            "  Split mode: disease cold-start | "
            f"node_type={cold_node_type} train_nodes={len(train_nodes)}, "
            f"val_nodes={len(val_nodes)}, test_nodes={len(test_nodes)}"
        )

        for triple, edge_index in supervision_edges.items():
            if edge_index.numel() == 0:
                empty = edge_index[:, :0]
                train_sup[triple] = empty
                val_sup[triple] = empty
                test_sup[triple] = empty
                logger.info(f"  {triple}: train=0, val=0, test=0")
                continue

            if triple[0] == cold_node_type and triple[2] == cold_node_type:
                indices = list(range(edge_index.shape[1]))
                t_idx, v_idx, s_idx = _split_indices_with_optional_fixed_test(indices)
                train_sup[triple] = edge_index[:, t_idx]
                val_sup[triple] = edge_index[:, v_idx]
                test_sup[triple] = edge_index[:, s_idx]
                logger.info(
                    f"  {triple}: train={len(t_idx)}, val={len(v_idx)}, test={len(s_idx)} "
                    "(edge-level fallback for symmetric cold node type)"
                )
                continue

            cold_axis: Optional[int] = None
            if triple[0] == cold_node_type:
                cold_axis = 0
            elif triple[2] == cold_node_type:
                cold_axis = 1

            if cold_axis is None:
                indices = list(range(edge_index.shape[1]))
                t_idx, v_idx, s_idx = _split_indices_with_optional_fixed_test(indices)
                train_sup[triple] = edge_index[:, t_idx]
                val_sup[triple] = edge_index[:, v_idx]
                test_sup[triple] = edge_index[:, s_idx]
                logger.info(
                    f"  {triple}: train={len(t_idx)}, val={len(v_idx)}, test={len(s_idx)} "
                    "(edge-level fallback: no cold node axis)"
                )
                continue

            split_indices = {"train": [], "val": [], "test": []}
            for edge_id, node_id in enumerate(edge_index[cold_axis].tolist()):
                split_name = node_to_split.get(int(node_id))
                if split_name is None:
                    raise KeyError(
                        f"Missing cold-start split assignment for node_id={int(node_id)} "
                        f"in triple={triple}"
                    )
                split_indices[split_name].append(edge_id)

            train_sup[triple] = edge_index[:, split_indices["train"]]
            val_sup[triple] = edge_index[:, split_indices["val"]]
            test_sup[triple] = edge_index[:, split_indices["test"]]
            logger.info(
                f"  {triple}: train={len(split_indices['train'])}, "
                f"val={len(split_indices['val'])}, test={len(split_indices['test'])}"
            )

        primary_axis = 0 if primary_triple[0] == cold_node_type else 1
        train_primary_nodes = set(train_sup[primary_triple][primary_axis].tolist())
        val_primary_nodes = set(val_sup[primary_triple][primary_axis].tolist())
        test_primary_nodes = set(test_sup[primary_triple][primary_axis].tolist())
        if (
            (train_primary_nodes & val_primary_nodes)
            or (train_primary_nodes & test_primary_nodes)
            or (val_primary_nodes & test_primary_nodes)
        ):
            raise AssertionError(
                "Disease cold-start split has overlapping primary disease nodes across splits"
            )

        split_meta.update(
            {
                "cold_node_type": cold_node_type,
                "train_nodes": set(int(nid) for nid in train_nodes),
                "val_nodes": set(int(nid) for nid in val_nodes),
                "test_nodes": set(int(nid) for nid in test_nodes),
            }
        )
        return train_sup, val_sup, test_sup, split_meta

    triple_query_pos = 0 if query_axis == 0 else 2
    primary_query_type = primary_triple[triple_query_pos]
    grouped_query_ids = sorted(
        {
            int(query_id)
            for triple, edge_index in supervision_edges.items()
            if (
                triple[triple_query_pos] == primary_query_type
                and edge_index.numel() > 0
            )
            for query_id in edge_index[query_axis].tolist()
        }
    )
    if not grouped_query_ids:
        raise ValueError(
            "No query ids found for grouped supervision splitting. "
            f"primary_triple={primary_triple} query_axis={query_axis}"
        )

    train_query_idx, val_query_idx, test_query_idx = _split_indices_with_optional_fixed_test(
        list(range(len(grouped_query_ids)))
    )
    train_queries = [grouped_query_ids[i] for i in train_query_idx]
    val_queries = [grouped_query_ids[i] for i in val_query_idx]
    test_queries = [grouped_query_ids[i] for i in test_query_idx]

    query_to_split = {int(q): "train" for q in train_queries}
    query_to_split.update({int(q): "val" for q in val_queries})
    query_to_split.update({int(q): "test" for q in test_queries})
    logger.info(
        "  Split mode: query-level inductive | "
        f"query_type={primary_query_type} axis={query_axis} "
        f"train_queries={len(train_queries)}, val_queries={len(val_queries)}, "
        f"test_queries={len(test_queries)}"
    )

    for triple, edge_index in supervision_edges.items():
        if edge_index.numel() == 0:
            empty = edge_index[:, :0]
            train_sup[triple] = empty
            val_sup[triple] = empty
            test_sup[triple] = empty
            logger.info(f"  {triple}: train=0, val=0, test=0")
            continue

        if triple[triple_query_pos] == primary_query_type:
            split_indices = {"train": [], "val": [], "test": []}
            for edge_id, query_id in enumerate(edge_index[query_axis].tolist()):
                split_name = query_to_split.get(int(query_id))
                if split_name is None:
                    raise KeyError(
                        f"Missing split assignment for query_id={int(query_id)} "
                        f"while splitting {triple}"
                    )
                split_indices[split_name].append(edge_id)

            train_sup[triple] = edge_index[:, split_indices["train"]]
            val_sup[triple] = edge_index[:, split_indices["val"]]
            test_sup[triple] = edge_index[:, split_indices["test"]]

            train_q = len(
                {int(edge_index[query_axis, idx]) for idx in split_indices["train"]}
            )
            val_q = len(
                {int(edge_index[query_axis, idx]) for idx in split_indices["val"]}
            )
            test_q = len(
                {int(edge_index[query_axis, idx]) for idx in split_indices["test"]}
            )
            logger.info(
                f"  {triple}: train={len(split_indices['train'])} ({train_q} queries), "
                f"val={len(split_indices['val'])} ({val_q} queries), "
                f"test={len(split_indices['test'])} ({test_q} queries)"
            )
            continue

        indices = list(range(edge_index.shape[1]))
        t_idx, v_idx, s_idx = _split_indices_with_optional_fixed_test(indices)

        train_sup[triple] = edge_index[:, t_idx]
        val_sup[triple] = edge_index[:, v_idx]
        test_sup[triple] = edge_index[:, s_idx]
        logger.info(
            f"  {triple}: train={len(t_idx)}, val={len(v_idx)}, test={len(s_idx)} "
            "(edge-level fallback)"
        )

    return train_sup, val_sup, test_sup, split_meta


def filter_background_edges_by_node_split(
    background_edges: Dict[EdgeTriple, torch.Tensor],
    cold_node_type: str,
    allowed_nodes: Set[int],
) -> Dict[EdgeTriple, torch.Tensor]:
    """Filter background edges so only split-specific cold nodes are kept."""
    filtered: Dict[EdgeTriple, torch.Tensor] = {}
    for triple, edge_index in background_edges.items():
        if edge_index.numel() == 0:
            filtered[triple] = edge_index
            continue

        src_is_cold = triple[0] == cold_node_type
        dst_is_cold = triple[2] == cold_node_type
        if not src_is_cold and not dst_is_cold:
            filtered[triple] = edge_index
            continue

        mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
        if src_is_cold:
            src_allowed = torch.tensor(
                [int(node_id) in allowed_nodes for node_id in edge_index[0].tolist()],
                dtype=torch.bool,
                device=edge_index.device,
            )
            mask = mask & src_allowed
        if dst_is_cold:
            dst_allowed = torch.tensor(
                [int(node_id) in allowed_nodes for node_id in edge_index[1].tolist()],
                dtype=torch.bool,
                device=edge_index.device,
            )
            mask = mask & dst_allowed

        filtered[triple] = edge_index[:, mask]
    return filtered


# =========================================================================
# Assemble HeteroData objects
# =========================================================================


def assemble_heterodata(
    x_dict: Dict[str, torch.Tensor],
    edge_index_dict: Dict[EdgeTriple, torch.Tensor],
    sup_edge_index: Optional[torch.Tensor] = None,
    sup_edge_label: Optional[torch.Tensor] = None,
    sup_triple: Optional[EdgeTriple] = None,
) -> HeteroData:
    """Build a HeteroData object.

    Args:
        x_dict:          {node_type: Tensor[N, feat_dim]}
        edge_index_dict: {(src, rel, dst): Tensor[2, E]}  — background edges
        sup_edge_index:  [2, E_sup] supervision edge indices (label edges)
        sup_edge_label:  [E_sup] 0/1 labels
        sup_triple:      The (src, rel, dst) triple for supervision edges

    Returns:
        Populated HeteroData.
    """
    data = HeteroData()

    for node_type, x in x_dict.items():
        data[node_type].x = x

    for (src_t, rel, dst_t), edge_index in edge_index_dict.items():
        data[src_t, rel, dst_t].edge_index = edge_index

    if sup_edge_index is not None and sup_triple is not None:
        src_t, rel, dst_t = sup_triple
        data[src_t, rel, dst_t].edge_label_index = sup_edge_index
        if sup_edge_label is not None:
            data[src_t, rel, dst_t].edge_label = sup_edge_label

    return data


def sample_query_aligned_negatives(
    pos_edge_index: torch.Tensor,
    all_positive_edge_index: torch.Tensor,
    *,
    num_src: int,
    num_dst: int,
    query_axis: int,
    n_neg_per_pos: int,
    seed: int,
    degree_matched: bool = False,
    degree_match_quantiles: int = 5,
    strict: bool = False,
    with_replacement: bool = False,
    allowed_candidates: Optional[torch.Tensor] = None,
    insufficient_negatives_policy: str = "error",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample negatives by fixing each query node and corrupting the candidate node.

    For disease-to-chemical retrieval with the canonical supervision edge
    ``chemical -> disease``, this samples ``negative_chemical -> same_disease``.
    """
    if query_axis not in (0, 1):
        raise ValueError(f"query_axis must be 0 or 1, got {query_axis}")
    if pos_edge_index.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0)

    n_neg_per_pos = int(n_neg_per_pos)
    if n_neg_per_pos <= 0:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0)

    policy = str(insufficient_negatives_policy).lower()
    if policy not in {"error", "skip", "replace"}:
        raise ValueError(
            "insufficient_negatives_policy must be one of: error, skip, replace"
        )
    if strict:
        policy = "error"

    candidate_axis = 1 - int(query_axis)
    num_candidates = int(num_src if candidate_axis == 0 else num_dst)
    if num_candidates <= 0:
        raise ValueError("No candidate nodes available for query-aligned negatives")

    all_pos = all_positive_edge_index.detach().cpu()
    known_by_query: Dict[int, Set[int]] = {}
    for src_idx, dst_idx in zip(all_pos[0].tolist(), all_pos[1].tolist()):
        query_id = int(dst_idx if query_axis == 1 else src_idx)
        candidate_id = int(src_idx if candidate_axis == 0 else dst_idx)
        known_by_query.setdefault(query_id, set()).add(candidate_id)

    candidate_degree = torch.bincount(
        all_pos[candidate_axis].long(),
        minlength=num_candidates,
    ).cpu().numpy()
    if allowed_candidates is None:
        candidate_ids = np.arange(num_candidates, dtype=np.int64)
    else:
        candidate_ids = allowed_candidates.detach().cpu().numpy()
        candidate_ids = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
        if candidate_ids.size == 0:
            if strict:
                raise ValueError("No allowed query-aligned candidate nodes")
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0)
        if np.any(candidate_ids < 0) or np.any(candidate_ids >= num_candidates):
            raise ValueError("allowed_candidates contains an invalid node id")
        candidate_ids = np.unique(candidate_ids)

    candidate_bucket = np.zeros(num_candidates, dtype=np.int64)
    bucket_to_candidates: Dict[int, np.ndarray] = {0: candidate_ids}
    if degree_matched:
        q = max(1, int(degree_match_quantiles))
        boundaries = np.quantile(candidate_degree, np.linspace(0.0, 1.0, q + 1))
        candidate_bucket = np.searchsorted(
            boundaries[1:-1],
            candidate_degree,
            side="right",
        ).astype(np.int64)
        bucket_to_candidates = {
            int(bucket): candidate_ids[candidate_bucket == int(bucket)]
            for bucket in np.unique(candidate_bucket)
        }

    rng = np.random.default_rng(int(seed))
    used_by_query: Dict[int, Set[int]] = {}
    available_cache: Dict[Tuple[int, int, bool], np.ndarray] = {}

    def _available(query_id: int, bucket: int, include_used: bool) -> np.ndarray:
        cache_key = (int(query_id), int(bucket), bool(include_used))
        cached = available_cache.get(cache_key)
        if cached is not None:
            return cached
        base = bucket_to_candidates.get(int(bucket), candidate_ids)
        banned = set(known_by_query.get(int(query_id), set()))
        if not include_used:
            banned.update(used_by_query.get(int(query_id), set()))
        if not banned:
            out = np.asarray(base, dtype=np.int64)
        else:
            out = np.asarray(
                [int(candidate) for candidate in base.tolist() if int(candidate) not in banned],
                dtype=np.int64,
            )
        available_cache[cache_key] = out
        return out

    neg_src_parts: List[torch.Tensor] = []
    neg_dst_parts: List[torch.Tensor] = []
    pos_cpu = pos_edge_index.detach().cpu()

    for src_idx, dst_idx in zip(pos_cpu[0].tolist(), pos_cpu[1].tolist()):
        query_id = int(dst_idx if query_axis == 1 else src_idx)
        pos_candidate = int(src_idx if candidate_axis == 0 else dst_idx)
        bucket = int(candidate_bucket[pos_candidate]) if degree_matched else 0

        available = _available(query_id, bucket, include_used=False)
        if available.size < n_neg_per_pos and degree_matched:
            available = _available(query_id, 0, include_used=False)
        replace = bool(with_replacement)
        if available.size < n_neg_per_pos:
            available = _available(query_id, bucket if degree_matched else 0, include_used=True)
            if available.size < n_neg_per_pos and degree_matched:
                available = _available(query_id, 0, include_used=True)
            if not replace and policy == "replace":
                replace = True
        if available.size == 0:
            if policy == "error":
                raise ValueError(
                    f"No query-aligned negative candidates for query_id={query_id}"
                )
            continue
        if policy == "error" and not replace and available.size < n_neg_per_pos:
            raise ValueError(
                "insufficient query-aligned negatives: "
                f"query_id={query_id}, needed={n_neg_per_pos}, available={available.size}"
            )

        sampled = rng.choice(
            available,
            size=n_neg_per_pos,
            replace=replace or available.size < n_neg_per_pos,
        )
        used_by_query.setdefault(query_id, set()).update(int(x) for x in sampled.tolist())
        available_cache.clear()

        sampled_tensor = torch.as_tensor(sampled, dtype=torch.long)
        query_tensor = torch.full_like(sampled_tensor, int(query_id))
        if query_axis == 1:
            neg_src_parts.append(sampled_tensor)
            neg_dst_parts.append(query_tensor)
        else:
            neg_src_parts.append(query_tensor)
            neg_dst_parts.append(sampled_tensor)

    if not neg_src_parts:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0)

    neg_edge_index = torch.stack(
        [torch.cat(neg_src_parts), torch.cat(neg_dst_parts)],
        dim=0,
    )
    neg_label = torch.zeros(neg_edge_index.shape[1])
    return neg_edge_index, neg_label


# =========================================================================
# Leakage check
# =========================================================================


def assert_no_leakage(
    background_edges_by_split: Dict[str, Dict[EdgeTriple, torch.Tensor]],
    sup_edges_by_split: Dict[str, Dict[EdgeTriple, torch.Tensor]],
    *,
    strict_background_disease_isolation: bool = False,
    cold_node_type: Optional[str] = None,
    supervision_primary_triple: Optional[EdgeTriple] = None,
) -> None:
    """Assert that supervision edges do not appear in background graphs.

    Rules:
    - In all modes, val/test supervision edges must not leak into train background.
    - In disease_cold_start mode, each split supervision edges must not leak into
      its own split background.
    - In strict disease isolation mode, supervision cold nodes for one split must
      not appear in background cold-node sets of other splits.
    """

    def _edge_set(edge_index: torch.Tensor) -> Set[Tuple[int, int]]:
        return set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    checks: List[Tuple[str, str]] = [("val", "train"), ("test", "train")]
    if set(background_edges_by_split.keys()) == {"train", "val", "test"}:
        checks.extend([("val", "val"), ("test", "test")])

    violations: List[str] = []
    for sup_split, bg_split in checks:
        bg_edges = background_edges_by_split.get(bg_split, {})
        sup_edges = sup_edges_by_split.get(sup_split, {})
        for triple, sup_ei in sup_edges.items():
            if triple not in bg_edges:
                continue
            overlap = _edge_set(sup_ei) & _edge_set(bg_edges[triple])
            if overlap:
                sample = sorted(overlap)[:5]
                violations.append(
                    f"{sup_split}.sup ∩ {bg_split}.bg on {triple}: "
                    f"{len(overlap)} overlaps (first 5: {sample})"
                )

    if strict_background_disease_isolation:
        if cold_node_type is None or supervision_primary_triple is None:
            raise ValueError(
                "strict_background_disease_isolation requires cold_node_type and supervision_primary_triple"
            )

        def _cold_nodes_from_background(
            edge_dict: Dict[EdgeTriple, torch.Tensor],
            node_type: str,
        ) -> Set[int]:
            nodes: Set[int] = set()
            for triple, edge_index in edge_dict.items():
                if edge_index.numel() == 0:
                    continue
                if triple[0] == node_type:
                    nodes.update(int(v) for v in edge_index[0].tolist())
                if triple[2] == node_type:
                    nodes.update(int(v) for v in edge_index[1].tolist())
            return nodes

        def _cold_nodes_from_supervision(edge_index: torch.Tensor, triple: EdgeTriple, node_type: str) -> Set[int]:
            if edge_index.numel() == 0:
                return set()
            nodes: Set[int] = set()
            if triple[0] == node_type:
                nodes.update(int(v) for v in edge_index[0].tolist())
            if triple[2] == node_type:
                nodes.update(int(v) for v in edge_index[1].tolist())
            return nodes

        bg_cold_nodes = {
            split_name: _cold_nodes_from_background(edge_dict, cold_node_type)
            for split_name, edge_dict in background_edges_by_split.items()
        }

        sup_cold_nodes: Dict[str, Set[int]] = {}
        for split_name, sup_edge_dict in sup_edges_by_split.items():
            sup_ei = sup_edge_dict.get(supervision_primary_triple)
            if sup_ei is None:
                sup_cold_nodes[split_name] = set()
                continue
            sup_cold_nodes[split_name] = _cold_nodes_from_supervision(
                sup_ei,
                supervision_primary_triple,
                cold_node_type,
            )

        for sup_split, sup_nodes in sup_cold_nodes.items():
            for bg_split, bg_nodes in bg_cold_nodes.items():
                if sup_split == bg_split:
                    continue
                overlap_nodes = sup_nodes & bg_nodes
                if overlap_nodes:
                    sample = sorted(overlap_nodes)[:10]
                    violations.append(
                        f"{sup_split}.supervision_nodes ∩ {bg_split}.background_nodes: "
                        f"{len(overlap_nodes)} overlaps (sample={sample})"
                    )

    assert not violations, "LEAKAGE DETECTED:\n" + "\n".join(violations)
    logger.info("Leakage check passed for configured split backgrounds.")



# =========================================================================
# Main
# =========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PyG HeteroData and split datasets for IDKG subgraph"
    )
    parser.add_argument("--entities_path", default="data/idkg/entities.pkl")
    parser.add_argument("--edges_path", default="data/idkg/edges.pkl")
    parser.add_argument("--features_dir", default="data/features")
    parser.add_argument("--edge_mapping", default="config/edge_mapping.yaml")
    parser.add_argument("--output_graph", default="data/raw/full_graph_data.pt")
    parser.add_argument("--split_dir", default="data/split")
    parser.add_argument("--mapping_dir", default="data/mapping")
    parser.add_argument(
        "--train_neg_per_pos",
        type=int,
        default=None,
        help="override negative_sampling.bpr_neg_per_pos from config",
    )
    parser.add_argument(
        "--val_test_neg_per_pos",
        type=int,
        default=None,
        help="override negative_sampling.val_test_neg_per_pos from config",
    )
    parser.add_argument(
        "--hard_neg_ratio",
        type=float,
        default=None,
        help="override negative_sampling.hard_neg_ratio from config",
    )
    parser.add_argument(
        "--degree_matched_negatives",
        action="store_true",
        help="sample negatives from disease-degree buckets matched to each positive disease",
    )
    parser.add_argument(
        "--degree_match_quantiles",
        type=int,
        default=None,
        help="override negative_sampling.degree_match_quantiles from config",
    )
    parser.add_argument(
        "--query_aligned_negatives",
        action="store_true",
        help="for query-level retrieval, corrupt the candidate node while keeping the query node fixed",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test_seed",
        type=int,
        default=None,
        help="optional fixed seed for selecting test split partition",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=None,
        help="optional seed for train/val split within non-test remainder",
    )
    parser.add_argument(
        "--freeze_test_split",
        action="store_true",
        help="if enabled, keep test partition fixed and reseed train/val",
    )
    args = parser.parse_args()

    os.makedirs(args.split_dir, exist_ok=True)
    os.makedirs(args.mapping_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_graph), exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    with open(args.entities_path, "rb") as f:
        entities = pickle.load(f)
    with open(args.edges_path, "rb") as f:
        all_edges: Dict[EdgeTriple, List[Tuple[int, int]]] = pickle.load(f)
    with open(args.edge_mapping, "r") as f:
        edge_cfg = yaml.safe_load(f)

    logger.info(f"Using edge mapping config: {args.edge_mapping}")

    sup_cfg = edge_cfg["supervision"]
    supervision_triples: List[EdgeTriple] = [tuple(t) for t in sup_cfg["edge_types"]]
    split_ratio: List[float] = sup_cfg["split_ratio"]
    split_mode = str(sup_cfg.get("split_mode", "query_level")).lower()
    reverse_specs = edge_cfg.get("logical_reverse_relations", [])
    reverse_relation_map: Dict[EdgeTriple, EdgeTriple] = {
        tuple(spec["forward"]): tuple(spec["reverse"]) for spec in reverse_specs
    }
    query_cfg = edge_cfg.get("query_task", {})
    query_group_from = str(query_cfg.get("positive_group_from", "tail")).lower()
    if query_group_from not in {"head", "tail"}:
        raise ValueError(
            "query_task.positive_group_from must be either 'head' or 'tail'"
        )
    query_axis = 1 if query_group_from == "tail" else 0
    primary_forward_triple: EdgeTriple = tuple(
        sup_cfg.get("primary_forward_edge", supervision_triples[0])
    )
    default_query_triple = (
        primary_forward_triple[2],
        "treated_by",
        primary_forward_triple[0],
    )
    primary_query_triple: EdgeTriple = tuple(
        query_cfg.get(
            "primary_query_edge",
            reverse_relation_map.get(primary_forward_triple, default_query_triple),
        )
    )

    neg_cfg = edge_cfg.get("negative_sampling", {})
    n_neg_per_pos = neg_cfg.get("bpr_neg_per_pos", 30)
    val_test_neg_per_pos = neg_cfg.get("val_test_neg_per_pos", 1)
    hard_neg_ratio = neg_cfg.get("hard_neg_ratio", 0.33)
    hard_neg_via = neg_cfg.get("hard_neg_via_rels", ["bind_p", "activate_p"])
    hard_neg_reuse_cap = neg_cfg.get("hard_neg_reuse_cap", None)
    hard_neg_degree_quantiles = neg_cfg.get("hard_neg_degree_quantiles", 1)
    hard_neg_quantile_quota = neg_cfg.get("hard_neg_quantile_quota", "uniform")
    hard_neg_min_per_quantile = neg_cfg.get("hard_neg_min_per_quantile", 0)
    hard_neg_quota_strict = neg_cfg.get("hard_neg_quota_strict", False)
    degree_matched_negatives = bool(neg_cfg.get("degree_matched_negatives", False))
    query_aligned_negatives = bool(neg_cfg.get("query_aligned_negatives", False))
    degree_match_quantiles = int(neg_cfg.get("degree_match_quantiles", 5))
    effective_neg_per_pos_all_splits = neg_cfg.get(
        "effective_neg_per_pos_all_splits", None
    )
    strict_negative_ratio_contract = bool(
        neg_cfg.get("strict_negative_ratio_contract", False)
    )
    with_replacement = bool(neg_cfg.get("with_replacement", False))
    strict_background_disease_isolation = bool(
        sup_cfg.get("strict_background_disease_isolation", False)
    )
    freeze_test_split_cfg = bool(sup_cfg.get("freeze_test_split", False))
    config_test_seed = sup_cfg.get("test_seed", None)
    config_split_seed = sup_cfg.get("split_seed", None)

    freeze_test_split = bool(args.freeze_test_split or freeze_test_split_cfg)
    effective_test_seed = (
        int(args.test_seed)
        if args.test_seed is not None
        else int(config_test_seed)
        if config_test_seed is not None
        else int(args.seed)
    )
    effective_split_seed = (
        int(args.split_seed)
        if args.split_seed is not None
        else int(config_split_seed)
        if config_split_seed is not None
        else int(args.seed)
    )

    if args.train_neg_per_pos is not None:
        n_neg_per_pos = int(args.train_neg_per_pos)
    if args.val_test_neg_per_pos is not None:
        val_test_neg_per_pos = int(args.val_test_neg_per_pos)
    if args.hard_neg_ratio is not None:
        hard_neg_ratio = float(args.hard_neg_ratio)
    if args.degree_matched_negatives:
        degree_matched_negatives = True
    if args.query_aligned_negatives:
        query_aligned_negatives = True
    if args.degree_match_quantiles is not None:
        degree_match_quantiles = int(args.degree_match_quantiles)
    if (
        strict_negative_ratio_contract
        and args.train_neg_per_pos is not None
        and args.val_test_neg_per_pos is not None
        and int(n_neg_per_pos) == int(val_test_neg_per_pos)
    ):
        effective_neg_per_pos_all_splits = int(n_neg_per_pos)

    if strict_negative_ratio_contract:
        if effective_neg_per_pos_all_splits is None:
            raise ValueError(
                "strict_negative_ratio_contract=true requires "
                "negative_sampling.effective_neg_per_pos_all_splits"
            )
        unified_neg_per_pos = int(effective_neg_per_pos_all_splits)
        if (
            int(n_neg_per_pos) != unified_neg_per_pos
            or int(val_test_neg_per_pos) != unified_neg_per_pos
        ):
            raise ValueError(
                "strict_negative_ratio_contract violated: "
                f"bpr_neg_per_pos={n_neg_per_pos}, "
                f"val_test_neg_per_pos={val_test_neg_per_pos}, "
                f"effective_neg_per_pos_all_splits={unified_neg_per_pos}"
            )
        n_neg_per_pos = unified_neg_per_pos
        val_test_neg_per_pos = unified_neg_per_pos
        if split_mode == "disease_cold_start":
            strict_background_disease_isolation = True

    def assert_exact_k(
        split_name: str,
        pos_ei: Optional[torch.Tensor],
        neg_ei: torch.Tensor,
        neg_per_pos: int,
    ) -> None:
        if not strict_negative_ratio_contract or pos_ei is None:
            return
        expected = int(pos_ei.shape[1]) * int(neg_per_pos)
        actual = int(neg_ei.shape[1])
        if actual != expected:
            raise ValueError(
                f"{split_name}: insufficient negatives for strict contract, "
                f"expected={expected}, actual={actual}"
            )

    logger.info(
        "Supervision config: "
        f"edge_types={supervision_triples}, "
        f"primary_forward={primary_forward_triple}, "
        f"split_mode={split_mode}, "
        f"split_ratio={split_ratio}, "
        f"freeze_test_split={freeze_test_split}, "
        f"test_seed={effective_test_seed}, "
        f"split_seed={effective_split_seed}"
    )
    logger.info(
        "Negative sampling config: "
        f"train_neg_per_pos={n_neg_per_pos}, "
        f"val_test_neg_per_pos={val_test_neg_per_pos}, "
        f"hard_neg_ratio={hard_neg_ratio}, "
        f"query_aligned_negatives={query_aligned_negatives}, "
        f"degree_matched_negatives={degree_matched_negatives}, "
        f"degree_match_quantiles={degree_match_quantiles}"
    )

    # ------------------------------------------------------------------
    # Node index
    # ------------------------------------------------------------------
    logger.info("=== Stage A: Building node index ===")
    node_index = build_node_index(entities)

    # Serialise node_index (convert int keys → str for JSON)
    serialisable = {
        nt: {str(k): v for k, v in idx.items()} for nt, idx in node_index.items()
    }
    mapping_path = os.path.join(args.mapping_dir, "node_index.json")
    with open(mapping_path, "w") as f:
        json.dump(serialisable, f)
    logger.info(f"Node index saved → {mapping_path}")

    entity_lookup = build_entity_lookup(entities, node_index)
    entity_lookup_path = os.path.join(args.mapping_dir, "entity_lookup.json")
    with open(entity_lookup_path, "w") as f:
        json.dump(entity_lookup, f, ensure_ascii=False, indent=2)
    logger.info(f"Entity lookup saved → {entity_lookup_path}")

    # ------------------------------------------------------------------
    # Feature matrices
    # ------------------------------------------------------------------
    logger.info("=== Stage B: Loading feature matrices ===")
    x_dict = load_feature_matrices(args.features_dir, node_index)
    # Shapes:
    #   chemical: [N_chem, 2816]   protein: [N_prot, 1280]
    #   disease:  [N_dis,   768]   gene:    [N_gene,  768]
    #   pathogen: [N_path,  768]

    # ------------------------------------------------------------------
    # Edge indices
    # ------------------------------------------------------------------
    logger.info("=== Stage C: Building edge index tensors ===")
    edge_index_dict = build_edge_indices(all_edges, node_index)
    full_edge_index_dict = add_logical_reverse_edges(edge_index_dict, reverse_specs)

    # ------------------------------------------------------------------
    # Save full graph
    # ------------------------------------------------------------------
    full_graph = assemble_heterodata(x_dict, full_edge_index_dict)
    torch.save(full_graph, args.output_graph)
    logger.info(f"Full graph saved → {args.output_graph}")
    logger.info(f"  Node types:  {full_graph.node_types}")
    logger.info(f"  Edge types:  {full_graph.edge_types}")

    # ------------------------------------------------------------------
    # Separate supervision edges from background
    # ------------------------------------------------------------------
    logger.info("=== Stage D: Separating supervision edges ===")
    reverse_supervision_triples: Set[EdgeTriple] = {
        reverse_relation_map[triple]
        for triple in supervision_triples
        if triple in reverse_relation_map
    }
    background_edges, supervision_edges = separate_supervision_edges(
        full_edge_index_dict,
        supervision_triples,
        background_exclude_triples=reverse_supervision_triples,
    )

    # ------------------------------------------------------------------
    # Split supervision edges
    # ------------------------------------------------------------------
    logger.info(
        "=== Stage E: Splitting supervision edges "
        f"(mode={split_mode}, ratio={split_ratio}) ==="
    )
    train_sup, val_sup, test_sup, split_meta = split_supervision_edges(
        supervision_edges,
        split_ratio,
        primary_triple=primary_forward_triple,
        query_axis=query_axis,
        split_mode=split_mode,
        seed=args.seed,
        test_seed=effective_test_seed,
        split_seed=effective_split_seed,
        freeze_test_split=freeze_test_split,
    )
    train_query_sup = derive_reverse_split_edges(train_sup, reverse_relation_map)
    val_query_sup = derive_reverse_split_edges(val_sup, reverse_relation_map)
    test_query_sup = derive_reverse_split_edges(test_sup, reverse_relation_map)

    # ------------------------------------------------------------------
    # Build NegativeSampler (vectorized, per-epoch online)
    # ------------------------------------------------------------------
    logger.info("=== Stage F: Building NegativeSampler ===")
    logger.info(
        "Negative sampling config: "
        f"hard_neg_ratio={hard_neg_ratio}, "
        f"hard_neg_reuse_cap={hard_neg_reuse_cap}, "
        f"hard_neg_degree_quantiles={hard_neg_degree_quantiles}, "
        f"hard_neg_quantile_quota={hard_neg_quantile_quota}, "
        f"hard_neg_min_per_quantile={hard_neg_min_per_quantile}, "
        f"hard_neg_quota_strict={hard_neg_quota_strict}, "
        f"degree_matched_negatives={degree_matched_negatives}, "
        f"degree_match_quantiles={degree_match_quantiles}"
    )

    # Collect ALL supervision positive edges (train+val+test) for known-pos mask
    all_sup_ei_parts = [ei for ei in supervision_edges.values()]
    all_sup_ei = (
        torch.cat(all_sup_ei_parts, dim=1)
        if all_sup_ei_parts
        else torch.zeros(2, 0, dtype=torch.long)
    )

    num_chem = x_dict["chemical"].shape[0]
    num_dis = x_dict["disease"].shape[0]
    num_prot = x_dict["protein"].shape[0]
    num_gene = x_dict["gene"].shape[0]

    # Collect relay edges for hard-negative mining.
    # We adaptively choose the first schema-valid meta-path that ends at Disease.
    # Preferred order:
    #   1) Chemical → Protein → Gene → Disease
    #   2) Chemical → Protein → Chemical → Disease   (shared-target relay)
    #   3) Chemical → Chemical → Disease             (chemical-similarity relay)
    #   4) Chemical → Protein → Disease              (legacy fallback)
    hard_neg_edge_index_dict = full_edge_index_dict
    chem_prot_triples = [
        t
        for t in hard_neg_edge_index_dict
        if t[0] == "chemical" and t[2] == "protein" and t[1] in hard_neg_via
    ]
    gene_prot_triples = [
        t for t in hard_neg_edge_index_dict if t[0] == "gene" and t[2] == "protein"
    ]
    gene_dis_triples = [
        t for t in hard_neg_edge_index_dict if t[0] == "gene" and t[2] == "disease"
    ]
    chem_chem_triples = [
        t
        for t in hard_neg_edge_index_dict
        if t[0] == "chemical" and t[2] == "chemical"
    ]
    prot_dis_triples = [
        t for t in hard_neg_edge_index_dict if t[0] == "protein" and t[2] == "disease"
    ]

    cp_ei = (
        torch.cat([hard_neg_edge_index_dict[t] for t in chem_prot_triples], dim=1)
        if chem_prot_triples
        else None
    )
    gp_ei = (
        torch.cat([hard_neg_edge_index_dict[t] for t in gene_prot_triples], dim=1)
        if gene_prot_triples
        else None
    )
    gd_ei = (
        torch.cat([hard_neg_edge_index_dict[t] for t in gene_dis_triples], dim=1)
        if gene_dis_triples
        else None
    )
    cc_ei = (
        torch.cat([hard_neg_edge_index_dict[t] for t in chem_chem_triples], dim=1)
        if chem_chem_triples
        else None
    )
    pd_ei = (
        torch.cat([hard_neg_edge_index_dict[t] for t in prot_dis_triples], dim=1)
        if prot_dis_triples
        else None
    )
    primary_cd_ei = train_sup.get(primary_forward_triple)
    train_primary_sup_ei = train_sup.get(
        primary_forward_triple, torch.zeros((2, 0), dtype=torch.long)
    )

    relay_path_eis = None
    relay_path_shapes = None
    if cp_ei is not None and gp_ei is not None and gd_ei is not None:
        logger.info(
            "Using 3-hop relay path for hard negatives: Chemical→Protein→Gene→Disease"
        )
        relay_path_eis = [
            cp_ei,
            gp_ei.flip(0).contiguous(),
            gd_ei,
        ]
        relay_path_shapes = [
            (num_chem, num_prot),
            (num_prot, num_gene),
            (num_gene, num_dis),
        ]
    elif cp_ei is not None and primary_cd_ei is not None:
        logger.info(
            "Using 3-hop relay path for hard negatives: Chemical→Protein→Chemical→Disease"
        )
        relay_path_eis = [
            cp_ei,
            cp_ei.flip(0).contiguous(),
            primary_cd_ei,
        ]
        relay_path_shapes = [
            (num_chem, num_prot),
            (num_prot, num_chem),
            (num_chem, num_dis),
        ]
    elif cc_ei is not None and primary_cd_ei is not None:
        logger.info(
            "Using 2-hop relay path for hard negatives: Chemical→Chemical→Disease"
        )
        relay_path_eis = [
            cc_ei,
            primary_cd_ei,
        ]
        relay_path_shapes = [
            (num_chem, num_chem),
            (num_chem, num_dis),
        ]
    elif cp_ei is not None and pd_ei is not None:
        logger.info(
            "Falling back to 2-hop relay path for hard negatives: Chemical→Protein→Disease"
        )
    else:
        logger.warning(
            "No valid relay path found for hard-negative mining; sampler will use degree-aware random negatives only."
        )

    sampler = NegativeSampler.build_from_edges(
        sup_ei=train_primary_sup_ei,
        num_src=num_chem,
        num_dst=num_dis,
        cp_ei=cp_ei,
        pd_ei=pd_ei,
        num_mid=num_prot,
        relay_path_eis=relay_path_eis,
        relay_path_shapes=relay_path_shapes,
        hard_neg_ratio=hard_neg_ratio,
        hard_neg_reuse_cap=hard_neg_reuse_cap,
        hard_neg_degree_quantiles=hard_neg_degree_quantiles,
        hard_neg_quantile_quota=hard_neg_quantile_quota,
        hard_neg_min_per_quantile=hard_neg_min_per_quantile,
        hard_neg_quota_strict=hard_neg_quota_strict,
        degree_matched_negatives=degree_matched_negatives,
        degree_match_quantiles=degree_match_quantiles,
        known_pos_ei=train_primary_sup_ei,
        degree_pos_ei=train_primary_sup_ei,
        base_seed=args.seed,
    )
    sampler_path = os.path.join(args.split_dir, "neg_sampler.pkl")
    sampler.save(sampler_path)
    logger.info(f"NegativeSampler saved → {sampler_path}")
    for k, v in sampler.stats().items():
        logger.info(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Assemble and save split HeteroData objects
    # - transductive modes: val/test message-passing background == train background
    # - disease_cold_start: split-specific message-passing backgrounds
    # ------------------------------------------------------------------
    logger.info("=== Stage G: Assembling and saving split datasets ===")
    leakage_cold_node_type: Optional[str] = None
    primary_triple = primary_forward_triple
    all_primary_sup_ei = supervision_edges.get(
        primary_triple, torch.zeros((2, 0), dtype=torch.long)
    )
    train_primary_sup_ei = train_sup.get(
        primary_triple, torch.zeros((2, 0), dtype=torch.long)
    )
    val_primary_sup_ei = val_sup.get(
        primary_triple, torch.zeros((2, 0), dtype=torch.long)
    )
    test_primary_sup_ei = test_sup.get(
        primary_triple, torch.zeros((2, 0), dtype=torch.long)
    )
    split_exclusion_ei = {
        "train": train_primary_sup_ei,
        "val": torch.cat([train_primary_sup_ei, val_primary_sup_ei], dim=1),
        "test": torch.cat([train_primary_sup_ei, test_primary_sup_ei], dim=1),
    }
    negative_allowed_dst_by_split: Dict[str, Optional[torch.Tensor]] = {
        "train": None,
        "val": None,
        "test": None,
    }

    if split_mode == "disease_cold_start":
        cold_node_type = str(
            split_meta.get("cold_node_type", primary_forward_triple[2])
        )
        leakage_cold_node_type = cold_node_type
        train_nodes = set(split_meta.get("train_nodes", set()))
        val_nodes = set(split_meta.get("val_nodes", set()))
        test_nodes = set(split_meta.get("test_nodes", set()))
        if cold_node_type == primary_triple[2]:
            negative_allowed_dst_by_split = {
                "train": torch.tensor(sorted(train_nodes), dtype=torch.long),
                "val": torch.tensor(sorted(val_nodes), dtype=torch.long),
                "test": torch.tensor(sorted(test_nodes), dtype=torch.long),
            }
            logger.info(
                "Using split-aware destination candidates for disease_cold_start "
                f"negative sampling: train={len(train_nodes)}, "
                f"val={len(val_nodes)}, test={len(test_nodes)}."
            )
        else:
            logger.warning(
                "disease_cold_start cold node type is not the primary destination "
                f"({cold_node_type} vs {primary_triple[2]}); negative sampler "
                "destination candidates are not split-restricted."
            )
        train_background = filter_background_edges_by_node_split(
            background_edges,
            cold_node_type=cold_node_type,
            allowed_nodes=train_nodes,
        )
        val_background = filter_background_edges_by_node_split(
            background_edges,
            cold_node_type=cold_node_type,
            allowed_nodes=val_nodes,
        )
        test_background = filter_background_edges_by_node_split(
            background_edges,
            cold_node_type=cold_node_type,
            allowed_nodes=test_nodes,
        )

        for triple, ei in train_sup.items():
            train_background[triple] = ei
        for triple, ei in train_query_sup.items():
            train_background[triple] = ei

        leakage_backgrounds = {
            "train": train_background,
            "val": val_background,
            "test": test_background,
        }
        logger.info(
            "Using split-specific message-passing backgrounds for disease_cold_start."
        )
    else:
        train_background = dict(background_edges)
        for triple, ei in train_sup.items():
            train_background[triple] = ei
        for triple, ei in train_query_sup.items():
            train_background[triple] = ei

        val_background = train_background
        test_background = train_background
        leakage_backgrounds = {"train": train_background}
        logger.info(
            "Using train background graph for val/test message passing (transductive)."
        )

    def attach_negative_sampling_metadata(data, split_name: str) -> None:
        allowed_dst = negative_allowed_dst_by_split.get(split_name)
        if allowed_dst is not None:
            data.negative_allowed_dst = allowed_dst.clone()
            data.negative_allowed_dst_node_type = primary_triple[2]
            data.ranking_candidate_ids = allowed_dst.clone()
            data.ranking_candidate_node_type = primary_triple[2]
        data.negative_sampling_strict = bool(strict_negative_ratio_contract)
        data.negative_sampling_with_replacement = bool(with_replacement)
        if primary_triple in data.edge_types:
            data[primary_triple].known_positive_edge_index = (
                all_primary_sup_ei.clone()
            )

    # ---------- Train ----------
    train_pos_ei = train_sup.get(primary_triple)
    if train_pos_ei is not None:
        if query_aligned_negatives:
            neg_ei, neg_lbl = sample_query_aligned_negatives(
                train_pos_ei,
                split_exclusion_ei["train"],
                num_src=num_chem,
                num_dst=num_dis,
                query_axis=query_axis,
                n_neg_per_pos=n_neg_per_pos,
                seed=args.seed,
                degree_matched=degree_matched_negatives,
                degree_match_quantiles=degree_match_quantiles,
                strict=strict_negative_ratio_contract,
                with_replacement=with_replacement,
                insufficient_negatives_policy="error",
                allowed_candidates=(
                    negative_allowed_dst_by_split["train"]
                    if (1 - query_axis) == 1
                    else None
                ),
            )
        else:
            neg_ei, neg_lbl = sampler.sample_epoch(
                train_pos_ei,
                n_neg_per_pos,
                epoch=0,
                strict=strict_negative_ratio_contract,
                insufficient_negatives_policy="error",
                with_replacement=with_replacement,
                allowed_dst=negative_allowed_dst_by_split["train"],
                exclude_pos_ei=split_exclusion_ei["train"],
            )
        assert_exact_k("train", train_pos_ei, neg_ei, n_neg_per_pos)
        # Combine pos + neg
        pos_lbl = torch.ones(train_pos_ei.shape[1])
        sup_ei = torch.cat([train_pos_ei, neg_ei], dim=1)
        sup_lbl = torch.cat([pos_lbl, neg_lbl])
    else:
        sup_ei, sup_lbl = None, None

    total_train_sup_edges = sum(ei.shape[1] for ei in train_sup.values())
    total_val_sup_edges = sum(ei.shape[1] for ei in val_sup.values())
    total_test_sup_edges = sum(ei.shape[1] for ei in test_sup.values())

    train_data = assemble_heterodata(
        x_dict, train_background, sup_ei, sup_lbl, primary_triple
    )
    train_data.split_mode = split_mode
    train_data.query_aligned_negatives = bool(query_aligned_negatives)
    train_data.freeze_test_split = bool(split_meta.get("freeze_test_split", False))
    train_data.test_seed = int(split_meta.get("effective_test_seed", args.seed))
    train_data.split_seed = int(split_meta.get("effective_split_seed", args.seed))
    attach_negative_sampling_metadata(train_data, "train")
    torch.save(train_data, os.path.join(args.split_dir, "train_data.pt"))
    logger.info(
        "train_data.pt saved  "
        f"primary_pos_edges={train_pos_ei.shape[1] if train_pos_ei is not None else 0} "
        f"all_sup_edges={total_train_sup_edges}"
    )

    # ---------- Val ----------
    val_pos_ei = val_sup.get(primary_triple)
    val_neg_ei = torch.zeros((2, 0), dtype=torch.long)
    if val_pos_ei is not None:
        if query_aligned_negatives:
            val_neg_ei, val_neg_lbl = sample_query_aligned_negatives(
                val_pos_ei,
                split_exclusion_ei["val"],
                num_src=num_chem,
                num_dst=num_dis,
                query_axis=query_axis,
                n_neg_per_pos=val_test_neg_per_pos,
                seed=args.seed + 10_000,
                degree_matched=degree_matched_negatives,
                degree_match_quantiles=degree_match_quantiles,
                strict=strict_negative_ratio_contract,
                with_replacement=with_replacement,
                insufficient_negatives_policy="error",
                allowed_candidates=(
                    negative_allowed_dst_by_split["val"]
                    if (1 - query_axis) == 1
                    else None
                ),
            )
        else:
            val_neg_ei, val_neg_lbl = sampler.sample_epoch(
                val_pos_ei,
                val_test_neg_per_pos,
                epoch=10_000,
                strict=strict_negative_ratio_contract,
                insufficient_negatives_policy="error",
                with_replacement=with_replacement,
                allowed_dst=negative_allowed_dst_by_split["val"],
                exclude_pos_ei=split_exclusion_ei["val"],
            )
        assert_exact_k("val", val_pos_ei, val_neg_ei, val_test_neg_per_pos)
        val_pos_lbl = torch.ones(val_pos_ei.shape[1])
        val_sup_ei = torch.cat([val_pos_ei, val_neg_ei], dim=1)
        val_lbl = torch.cat([val_pos_lbl, val_neg_lbl])
    else:
        val_sup_ei = None
        val_lbl = None

    val_data = assemble_heterodata(
        x_dict, val_background, val_sup_ei, val_lbl, primary_triple
    )
    val_data.split_mode = split_mode
    val_data.query_aligned_negatives = bool(query_aligned_negatives)
    val_data.freeze_test_split = bool(split_meta.get("freeze_test_split", False))
    val_data.test_seed = int(split_meta.get("effective_test_seed", args.seed))
    val_data.split_seed = int(split_meta.get("effective_split_seed", args.seed))
    attach_negative_sampling_metadata(val_data, "val")
    torch.save(val_data, os.path.join(args.split_dir, "val_data.pt"))
    logger.info(
        "val_data.pt   saved  primary_pos_edges="
        f"{val_pos_ei.shape[1] if val_pos_ei is not None else 0} "
        f"neg_edges={val_neg_ei.shape[1] if val_pos_ei is not None else 0} "
        f"all_sup_edges={total_val_sup_edges}"
    )

    # ---------- Test ----------
    test_pos_ei = test_sup.get(primary_triple)
    test_neg_ei = torch.zeros((2, 0), dtype=torch.long)
    if test_pos_ei is not None:
        if query_aligned_negatives:
            test_neg_ei, test_neg_lbl = sample_query_aligned_negatives(
                test_pos_ei,
                split_exclusion_ei["test"],
                num_src=num_chem,
                num_dst=num_dis,
                query_axis=query_axis,
                n_neg_per_pos=val_test_neg_per_pos,
                seed=args.seed + 20_000,
                degree_matched=degree_matched_negatives,
                degree_match_quantiles=degree_match_quantiles,
                strict=strict_negative_ratio_contract,
                with_replacement=with_replacement,
                insufficient_negatives_policy="error",
                allowed_candidates=(
                    negative_allowed_dst_by_split["test"]
                    if (1 - query_axis) == 1
                    else None
                ),
            )
        else:
            test_neg_ei, test_neg_lbl = sampler.sample_epoch(
                test_pos_ei,
                val_test_neg_per_pos,
                epoch=20_000,
                strict=strict_negative_ratio_contract,
                insufficient_negatives_policy="error",
                with_replacement=with_replacement,
                allowed_dst=negative_allowed_dst_by_split["test"],
                exclude_pos_ei=split_exclusion_ei["test"],
            )
        assert_exact_k("test", test_pos_ei, test_neg_ei, val_test_neg_per_pos)
        test_pos_lbl = torch.ones(test_pos_ei.shape[1])
        test_sup_ei = torch.cat([test_pos_ei, test_neg_ei], dim=1)
        test_lbl = torch.cat([test_pos_lbl, test_neg_lbl])
    else:
        test_sup_ei = None
        test_lbl = None

    test_data = assemble_heterodata(
        x_dict, test_background, test_sup_ei, test_lbl, primary_triple
    )
    test_data.split_mode = split_mode
    test_data.query_aligned_negatives = bool(query_aligned_negatives)
    test_data.freeze_test_split = bool(split_meta.get("freeze_test_split", False))
    test_data.test_seed = int(split_meta.get("effective_test_seed", args.seed))
    test_data.split_seed = int(split_meta.get("effective_split_seed", args.seed))
    attach_negative_sampling_metadata(test_data, "test")
    torch.save(test_data, os.path.join(args.split_dir, "test_data.pt"))
    logger.info(
        "test_data.pt  saved  primary_pos_edges="
        f"{test_pos_ei.shape[1] if test_pos_ei is not None else 0} "
        f"neg_edges={test_neg_ei.shape[1] if test_pos_ei is not None else 0} "
        f"all_sup_edges={total_test_sup_edges}"
    )

    # ------------------------------------------------------------------
    # Export query-centric supervision metadata
    # ------------------------------------------------------------------
    logger.info("=== Stage H: Exporting query-centric supervision ===")
    num_query_nodes = x_dict[primary_query_triple[0]].shape[0]
    num_candidate_nodes = x_dict[primary_query_triple[2]].shape[0]
    query_metadata = {
        "query_mode": query_cfg.get("query_mode", "disease_to_chemical"),
        "query_node_type": primary_query_triple[0],
        "candidate_node_type": primary_query_triple[2],
        "forward_relation": list(primary_forward_triple),
        "query_relation": list(primary_query_triple),
        "positive_group_from": query_cfg.get("positive_group_from", "tail"),
    }
    save_query_groups(
        split_dir=args.split_dir,
        split_name="train",
        query_edge_index=train_query_sup[primary_query_triple],
        query_relation=primary_query_triple,
        num_queries=num_query_nodes,
        num_candidates=num_candidate_nodes,
        metadata=query_metadata,
    )
    save_query_groups(
        split_dir=args.split_dir,
        split_name="val",
        query_edge_index=val_query_sup[primary_query_triple],
        query_relation=primary_query_triple,
        num_queries=num_query_nodes,
        num_candidates=num_candidate_nodes,
        metadata=query_metadata,
    )
    save_query_groups(
        split_dir=args.split_dir,
        split_name="test",
        query_edge_index=test_query_sup[primary_query_triple],
        query_relation=primary_query_triple,
        num_queries=num_query_nodes,
        num_candidates=num_candidate_nodes,
        metadata=query_metadata,
    )

    # ------------------------------------------------------------------
    # Leakage check
    # ------------------------------------------------------------------
    sup_edges_for_check = {
        "train": train_sup,
        "val": val_sup,
        "test": test_sup,
    }
    for split_name, query_sup in (
        ("train", train_query_sup),
        ("val", val_query_sup),
        ("test", test_query_sup),
    ):
        sup_edges_for_check[split_name].update(query_sup)

    assert_no_leakage(
        leakage_backgrounds,
        sup_edges_for_check,
        strict_background_disease_isolation=strict_background_disease_isolation,
        cold_node_type=leakage_cold_node_type,
        supervision_primary_triple=primary_forward_triple,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("\n=== Build complete ===")
    for nt in full_graph.node_types:
        logger.info(f"  {nt}: {full_graph[nt].x.shape}")
    logger.info(f"  Edge types ({len(full_graph.edge_types)}): {full_graph.edge_types}")


if __name__ == "__main__":
    main()