import argparse
import json
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch
import yaml
from torch_geometric.data import HeteroData

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.model.model import KGCModel


BIOLOGIC_EXCLUDE_RE = re.compile(
    r"("
    r"\b[a-z0-9-]*(?:mab|ximab|zumab|umab|momab)\b|"
    r"antibody|antibodies|immunoglobulin|immune globulin|antiserum|"
    r"vaccine|vaccinia|toxoid|antitoxin|toxin|botulinum|"
    r"collagenase|prothrombin|coagulation factor|blood coagulation factor|"
    r"interferon|interleukin"
    r")",
    re.IGNORECASE,
)


def resolve_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    if str(path_value).strip() == "":
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def get_args():
    parser = argparse.ArgumentParser(
        description="inference script for the simplified BRIGHT model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/inference.yaml",
        help="Path to the config file (YAML format)",
    )
    parser.add_argument("--graph_data", type=str, help="path to the graph data file")
    parser.add_argument(
        "--entity_mapping",
        type=str,
        help="path to unified entity lookup json with entity_id, graph_index and display_name",
    )
    parser.add_argument(
        "--disease_mapping",
        type=str,
        help="legacy path to disease mapping file",
    )
    parser.add_argument(
        "--drug_mapping",
        type=str,
        help="legacy path to drug mapping file",
    )
    parser.add_argument("--model_path", type=str, help="path to the trained model file")
    parser.add_argument(
        "--entities_path",
        type=str,
        help="path to entities.pkl for resolving disease/drug names",
    )
    parser.add_argument("--disease", type=str, help="disease name for inference")
    parser.add_argument(
        "--disease_id",
        type=int,
        help="original entity id of the disease for exact lookup",
    )
    parser.add_argument("--hidden_channels", type=int, help="hidden channels")
    parser.add_argument("--out_channels", type=int, help="output channels")
    parser.add_argument("--num_srr_layers", type=int, help="number of SRR blocks")
    parser.add_argument(
        "--num_d2m_layers",
        type=int,
        help="legacy alias for --num_srr_layers",
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        help="number of attention heads shared by SRR and global router",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument(
        "--use_global_router",
        action="store_true",
        default=True,
        help="enable the global router branch",
    )
    parser.add_argument(
        "--disable_global_router",
        action="store_true",
        help="disable the global router branch",
    )
    parser.add_argument(
        "--mechanism_dropout",
        type=float,
        default=0.1,
        help="dropout rate for the router branch",
    )
    parser.add_argument(
        "--decoder_type",
        type=str,
        choices=["bilinear"],
        help="decoder family used by the checkpoint",
    )
    parser.add_argument(
        "--scoring_mode",
        type=str,
        choices=["decoder", "cosine"],
        default="decoder",
        help="candidate scoring mode; decoder matches the backend inference default",
    )
    parser.add_argument("--top_k", type=int, help="top k predictions to return")
    parser.add_argument(
        "--candidate_filter",
        type=str,
        choices=["all", "exclude_biologics", "small_molecule_like"],
        help="candidate filtering before top-k ranking",
    )
    parser.add_argument(
        "--exclude_candidate_regex",
        type=str,
        help="extra case-insensitive regex for excluding candidate chemical names",
    )
    parser.add_argument(
        "--require_smiles",
        action="store_true",
        help="only keep candidates with a non-empty SMILES string",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="path to save recommendations (csv/tsv). If omitted, only print.",
    )
    parser.add_argument("--verbose", action="store_true", help="whether to print detailed logs")
    parser.add_argument(
        "--use_raw_scores",
        action="store_true",
        help="whether to use raw scores instead of probabilities",
    )
    parser.add_argument(
        "--known_edge_filter_enabled",
        action="store_true",
        default=None,
        help="mask externally known treatment edges before Top-K ranking",
    )
    parser.add_argument(
        "--disable_known_edge_filter",
        action="store_true",
        help="disable known treatment edge masking even if enabled in config",
    )
    parser.add_argument(
        "--known_edge_filter_source",
        type=str,
        default="data/known_treat_edges.json",
        help="path to known treatment edge JSON, relative to BioRoute root unless absolute",
    )
    parser.add_argument(
        "--known_edge_filter_on_missing",
        type=str,
        choices=["warn_keep", "error"],
        default="warn_keep",
        help="behavior when the known-edge filter JSON is missing",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for inference")

    temp_args, _ = parser.parse_known_args()
    config = None
    config_path = resolve_path(temp_args.config)
    if config_path is not None and config_path.exists():
        print(f"Loading the config from '{config_path}' ...")
        with config_path.open("r") as f:
            config = yaml.safe_load(f)

        defaults = {}
        if config:
            for key, value in config.items():
                if isinstance(value, dict):
                    if key == "known_edge_filter":
                        if "enabled" in value:
                            defaults["known_edge_filter_enabled"] = value["enabled"]
                        if "source" in value:
                            defaults["known_edge_filter_source"] = value["source"]
                        if "on_missing" in value:
                            defaults["known_edge_filter_on_missing"] = value["on_missing"]
                    else:
                        defaults.update(value)
                else:
                    defaults[key] = value
        parser.set_defaults(**defaults)

    args = parser.parse_args()
    if args.disable_global_router:
        args.use_global_router = False
    if args.disable_known_edge_filter:
        args.known_edge_filter_enabled = False
    if args.known_edge_filter_enabled is None:
        args.known_edge_filter_enabled = False
    if args.num_srr_layers is None:
        args.num_srr_layers = args.num_d2m_layers

    for attr in (
        "config",
        "graph_data",
        "entity_mapping",
        "disease_mapping",
        "drug_mapping",
        "model_path",
        "entities_path",
        "output",
    ):
        value = getattr(args, attr, None)
        if value is not None:
            resolved_value = resolve_path(value)
            setattr(args, attr, str(resolved_value) if resolved_value is not None else None)

    if args.entity_mapping is None:
        args.entity_mapping = str(resolve_path("data/mapping/entity_lookup.json"))
    if args.entities_path is None:
        args.entities_path = str(resolve_path("data/idkg/entities.pkl"))
    if args.top_k is None:
        args.top_k = 10
    if args.candidate_filter is None:
        args.candidate_filter = "all"
    return args


def load_mappings(
    file_path,
    node_type=None,
    entities_path=None,
    name_col="Name",
    index_col="Graph_Index",
):
    try:
        suffix = Path(file_path).suffix.lower()
        if suffix == ".json":
            if node_type is None:
                raise ValueError("node_type is required when loading JSON mapping")
            if entities_path is None:
                raise ValueError(
                    "entities_path is required when loading JSON mapping by node type",
                )

            with open(file_path, "r") as f:
                payload = json.load(f)
            node_map = payload[node_type]

            with open(entities_path, "rb") as f:
                entities = pickle.load(f)

            entity_key = {
                "disease": "diseases",
                "chemical": "chemicals",
            }.get(node_type)
            typed_entities = entities[entity_key]

            name_to_index = {}
            for global_id_str, local_idx in node_map.items():
                entity = typed_entities.get(int(global_id_str))
                if entity is None:
                    continue
                entity_name = entity.get("name")
                if entity_name:
                    name_to_index[str(entity_name)] = int(local_idx)
            return name_to_index

        df = pd.read_csv(file_path)
        return df.set_index(name_col)[index_col].to_dict()
    except FileNotFoundError:
        raise FileNotFoundError(f"Not found the mapping file: '{file_path}'")
    except KeyError as e:
        raise KeyError(
            f"Mapping file '{file_path}' is missing required key/column: {e}",
        )
    except Exception as e:
        raise Exception(f"Error occurred while loading mapping file '{file_path}': {e}")


def normalize_name(name: str | None) -> str:
    return str(name or "").strip().lower()


def get_identifier(attrs: dict | None) -> str:
    if not attrs:
        return ""
    candidates = ("identifier", "Identifier", "id", "ID")
    for key in candidates:
        value = attrs.get(key)
        if value is None:
            continue
        value = str(value).strip()
        if value:
            return value
    return ""


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                return nested
        return checkpoint
    raise ValueError("Unsupported checkpoint format; expected a state_dict-like dict.")


def sigmoid_score(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return float(torch.sigmoid(torch.tensor(float(value))).item())


def score_candidates(
    model,
    z_dict: dict[str, torch.Tensor],
    disease_idx: int,
    num_chemicals: int,
    device: torch.device,
    scoring_mode: str,
) -> torch.Tensor:
    candidate_indices = torch.arange(
        num_chemicals,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0)
    disease_indices = torch.tensor([disease_idx], device=device, dtype=torch.long)
    candidate_mask = torch.ones_like(candidate_indices, dtype=torch.bool)

    if scoring_mode == "cosine":
        return KGCModel.score_recommendation_candidates(
            z_dict=z_dict,
            disease_index=disease_indices,
            candidate_chemical_index=candidate_indices,
            candidate_mask=candidate_mask,
        ).squeeze(0)

    if scoring_mode == "decoder":
        return model.score_grouped_candidates(
            z_dict=z_dict,
            source_index=disease_indices,
            candidate_index=candidate_indices,
            candidate_mask=candidate_mask,
            relation="treats",
            group_axis="dst",
        ).squeeze(0)

    raise ValueError("scoring_mode must be 'decoder' or 'cosine'")


def load_known_edge_filter(args) -> dict[str, Any]:
    if not bool(args.known_edge_filter_enabled):
        return {
            "enabled": False,
            "status": "disabled",
            "edges_by_disease": {},
            "removed_edge_count": 0,
            "cache_key": "known-edge-filter:disabled",
        }

    source = str(args.known_edge_filter_source or "data/known_treat_edges.json").strip()
    on_missing = str(args.known_edge_filter_on_missing or "warn_keep").strip().lower()
    if on_missing not in {"warn_keep", "error"}:
        raise ValueError("known_edge_filter.on_missing must be 'warn_keep' or 'error'")

    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    if not source_path.exists():
        message = f"known-edge filter file not found: {source_path}"
        if on_missing == "error":
            raise FileNotFoundError(message)
        print(f"Warning: {message}; keeping unfiltered model rankings")
        return {
            "enabled": False,
            "status": "missing_warn_keep",
            "edges_by_disease": {},
            "removed_edge_count": 0,
            "source": str(source_path),
            "cache_key": f"known-edge-filter:missing:{source_path}",
        }

    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    raw_edges = payload.get("edges_by_disease", payload)
    if not isinstance(raw_edges, dict):
        raise ValueError("known-edge filter payload must contain edges_by_disease")

    edges_by_disease: dict[int, set[int]] = {}
    edge_count = 0
    for disease_index, chemical_indices in raw_edges.items():
        if not isinstance(chemical_indices, list):
            continue
        disease_key = int(disease_index)
        chemical_set = {int(index) for index in chemical_indices}
        edges_by_disease[disease_key] = chemical_set
        edge_count += len(chemical_set)

    return {
        "enabled": True,
        "status": "loaded",
        "edges_by_disease": edges_by_disease,
        "removed_edge_count": edge_count,
        "source": str(source_path),
        "cache_key": f"known-edge-filter:{source_path}:{edge_count}",
    }


def apply_known_edge_filter(
    scores: torch.Tensor,
    disease_idx: int,
    known_edge_filter: Mapping[str, Any],
) -> tuple[torch.Tensor, str, int]:
    status = str(known_edge_filter.get("status", "disabled"))
    if not bool(known_edge_filter.get("enabled", False)):
        return scores, status, 0

    edges_by_disease = known_edge_filter.get("edges_by_disease", {})
    known_indices = edges_by_disease.get(int(disease_idx), set())
    if not known_indices:
        return scores, "no_known_edges_for_disease", 0

    valid_indices = sorted(
        index for index in known_indices if 0 <= int(index) < int(scores.numel())
    )
    if not valid_indices:
        return scores, "no_valid_known_edges_for_disease", 0

    filtered_scores = scores.clone()
    index_tensor = torch.tensor(valid_indices, device=scores.device, dtype=torch.long)
    filtered_scores[index_tensor] = float("-inf")
    return filtered_scores, "applied", len(valid_indices)


def load_entity_lookup(file_path: str) -> dict[str, list[dict]]:
    with open(file_path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Entity lookup '{file_path}' must be a dict keyed by node type.")
    return payload


def build_entity_views(records: list[dict]) -> dict[str, dict]:
    by_entity_id: dict[int, dict] = {}
    by_graph_index: dict[int, dict] = {}
    by_name: dict[str, list[dict]] = {}

    for raw_record in records:
        record = dict(raw_record)
        entity_id = int(record["entity_id"])
        graph_index = int(record["graph_index"])
        display_name = str(record.get("display_name") or record.get("name") or "").strip()
        record["entity_id"] = entity_id
        record["graph_index"] = graph_index
        record["display_name"] = display_name

        by_entity_id[entity_id] = record
        by_graph_index[graph_index] = record
        key = normalize_name(display_name)
        if key:
            by_name.setdefault(key, []).append(record)

    return {
        "by_entity_id": by_entity_id,
        "by_graph_index": by_graph_index,
        "by_name": by_name,
    }


def enrich_entity_views_from_entities(
    entity_views: dict[str, dict[str, dict]],
    entities_path: str | None,
) -> dict[str, dict[str, dict]]:
    if not entities_path:
        return entity_views

    entities_file = Path(entities_path)
    if not entities_file.exists():
        return entity_views

    with entities_file.open("rb") as f:
        entities = pickle.load(f)

    chemical_entities = entities.get("chemicals", {})
    chemical_views = entity_views.get("chemical")
    if chemical_views is None:
        return entity_views

    for entity_id, record in chemical_views["by_entity_id"].items():
        attrs = chemical_entities.get(int(entity_id), {})
        if "identifier" not in record or not str(record.get("identifier") or "").strip():
            identifier = get_identifier(attrs)
            if identifier:
                record["identifier"] = identifier
        if "smiles" not in record or not str(record.get("smiles") or "").strip():
            smiles = attrs.get("smiles") or attrs.get("SMILES") or attrs.get("canonicalSmiles")
            if smiles is not None and str(smiles).strip() and str(smiles).strip().lower() != "none":
                record["smiles"] = str(smiles).strip()

    return entity_views


def _has_smiles(record: dict | None) -> bool:
    if not record:
        return False
    smiles = str(record.get("smiles") or "").strip()
    return bool(smiles) and smiles.lower() != "none"


def _is_biologic_like_candidate(
    record: dict | None,
    extra_exclude_re: re.Pattern | None = None,
) -> bool:
    if not record:
        return False
    name = str(record.get("display_name") or record.get("name") or "")
    identifier = str(record.get("identifier") or "")
    text = f"{name} {identifier}"
    if BIOLOGIC_EXCLUDE_RE.search(text):
        return True
    if extra_exclude_re is not None and extra_exclude_re.search(text):
        return True
    return False


def build_candidate_keep_mask(
    chemical_views: dict[str, dict],
    num_candidates: int,
    device: torch.device,
    candidate_filter: str = "all",
    exclude_candidate_regex: str | None = None,
    require_smiles: bool = False,
    verbose: bool = False,
) -> torch.Tensor:
    candidate_filter = str(candidate_filter or "all").lower()
    extra_exclude_re = (
        re.compile(exclude_candidate_regex, re.IGNORECASE)
        if exclude_candidate_regex
        else None
    )

    keep = []
    for graph_idx in range(num_candidates):
        record = chemical_views["by_graph_index"].get(int(graph_idx))
        keep_candidate = True

        if require_smiles or candidate_filter == "small_molecule_like":
            keep_candidate = keep_candidate and _has_smiles(record)

        if candidate_filter in {"exclude_biologics", "small_molecule_like"}:
            keep_candidate = keep_candidate and not _is_biologic_like_candidate(
                record,
                extra_exclude_re=extra_exclude_re,
            )
        elif extra_exclude_re is not None:
            keep_candidate = keep_candidate and not _is_biologic_like_candidate(
                record,
                extra_exclude_re=extra_exclude_re,
            )

        keep.append(bool(keep_candidate))

    keep_mask = torch.tensor(keep, dtype=torch.bool, device=device)
    if verbose:
        kept = int(keep_mask.sum().item())
        print(
            f"  - Candidate filter: {candidate_filter}; "
            f"kept {kept}/{num_candidates} chemicals"
        )
    return keep_mask


def build_views_from_legacy_mapping(
    mapping_path: str,
    node_type: str,
    entities_path: str,
) -> dict[str, dict]:
    name_to_index = load_mappings(
        mapping_path,
        node_type=node_type,
        entities_path=entities_path,
    )
    with open(entities_path, "rb") as f:
        entities = pickle.load(f)

    entity_key = {
        "chemical": "chemicals",
        "disease": "diseases",
    }[node_type]
    with open(mapping_path, "r") as f:
        mapping_payload = json.load(f)
    entity_to_graph_index = {
        int(entity_id): int(local_idx)
        for entity_id, local_idx in mapping_payload[node_type].items()
    }

    records: list[dict] = []
    for entity_id, attrs in entities[entity_key].items():
        graph_index = entity_to_graph_index.get(int(entity_id))
        if graph_index is None:
            continue
        display_name = attrs.get("name") or ""
        records.append(
            {
                "entity_id": int(entity_id),
                "graph_index": int(graph_index),
                "display_name": str(display_name),
                "name": str(display_name),
            }
        )

    views = build_entity_views(records)
    for name, graph_index in name_to_index.items():
        key = normalize_name(name)
        if key and key not in views["by_name"]:
            record = views["by_graph_index"].get(int(graph_index))
            if record is not None:
                views["by_name"][key] = [record]
    return views


def load_entity_lookup_views(args) -> dict[str, dict[str, dict]]:
    entity_mapping_path = Path(args.entity_mapping) if args.entity_mapping else None
    if entity_mapping_path is not None and entity_mapping_path.exists():
        payload = load_entity_lookup(str(entity_mapping_path))
        views = {
            node_type: build_entity_views(payload.get(node_type, []))
            for node_type in ("chemical", "disease")
        }
        return enrich_entity_views_from_entities(views, args.entities_path)

    if args.disease_mapping and args.drug_mapping and args.entities_path:
        views = {
            "disease": build_views_from_legacy_mapping(
                args.disease_mapping,
                node_type="disease",
                entities_path=args.entities_path,
            ),
            "chemical": build_views_from_legacy_mapping(
                args.drug_mapping,
                node_type="chemical",
                entities_path=args.entities_path,
            ),
        }
        return enrich_entity_views_from_entities(views, args.entities_path)

    raise FileNotFoundError(
        "No usable entity mapping found. Provide --entity_mapping or legacy mapping files."
    )


def resolve_disease_record(
    disease_views: dict[str, dict],
    disease_name: str | None = None,
    disease_id: int | None = None,
) -> dict:
    if disease_id is not None:
        record = disease_views["by_entity_id"].get(int(disease_id))
        if record is None:
            available = list(disease_views["by_entity_id"].keys())[:5]
            raise ValueError(
                f"Not found disease_id={disease_id}. Available disease ids examples: {available}"
            )
        return record

    key = normalize_name(disease_name)
    matches = disease_views["by_name"].get(key, [])
    if not matches:
        available = [
            record["display_name"]
            for record in list(disease_views["by_graph_index"].values())[:5]
        ]
        raise ValueError(
            f"Not found the disease '{disease_name}' in the entity lookup. Available examples: {available}"
        )
    if len(matches) > 1:
        candidate_ids = [record["entity_id"] for record in matches]
        raise ValueError(
            f"Disease name '{disease_name}' is ambiguous. Please use --disease_id. Candidate entity ids: {candidate_ids}"
        )
    return matches[0]


def recommend_drugs(
    model,
    full_graph,
    disease_record,
    chemical_views,
    top_k=10,
    use_probability=True,
    scoring_mode="decoder",
    known_edge_filter=None,
    candidate_filter="all",
    exclude_candidate_regex=None,
    require_smiles=False,
    verbose=False,
):
    model.eval()
    disease_idx = int(disease_record["graph_index"])
    disease_name = str(disease_record["display_name"])

    with torch.no_grad():
        if verbose:
            print(
                f"Calculating recommendations for disease '{disease_name}' "
                f"(entity_id={disease_record['entity_id']}, graph_index={disease_idx})...",
            )

        z_dict = model(full_graph.x_dict, full_graph.edge_index_dict)
        num_drugs = int(z_dict["chemical"].shape[0])
        raw_scores = score_candidates(
            model=model,
            z_dict=z_dict,
            disease_idx=disease_idx,
            num_chemicals=num_drugs,
            device=z_dict["chemical"].device,
            scoring_mode=scoring_mode,
        )

        ranking_scores, known_filter_status, known_removed_count = apply_known_edge_filter(
            scores=raw_scores,
            disease_idx=disease_idx,
            known_edge_filter=known_edge_filter or {},
        )

        candidate_keep_mask = build_candidate_keep_mask(
            chemical_views=chemical_views,
            num_candidates=num_drugs,
            device=ranking_scores.device,
            candidate_filter=candidate_filter,
            exclude_candidate_regex=exclude_candidate_regex,
            require_smiles=require_smiles,
            verbose=verbose,
        )
        ranking_scores = ranking_scores.masked_fill(~candidate_keep_mask, float("-inf"))

        score_type = "Probability" if use_probability else "Raw Score"
        valid_count = int(torch.isfinite(ranking_scores).sum().item())
        if valid_count == 0:
            raise ValueError(
                "No candidate chemicals remain after filtering and known-edge removal."
            )
        top_scores, top_indices = torch.topk(
            ranking_scores,
            k=min(top_k, valid_count),
            largest=True,
        )

        if verbose:
            finite_scores = raw_scores[torch.isfinite(raw_scores)]
            print(
                f"  - Raw Score Range: [{finite_scores.min().item():.6f}, {finite_scores.max().item():.6f}]",
            )

    recommendations = []
    for i in range(len(top_scores)):
        drug_idx = top_indices[i].item()
        raw_score = raw_scores[drug_idx].item()
        drug_record = chemical_views["by_graph_index"].get(
            int(drug_idx),
            {
                "entity_id": None,
                "display_name": f"Unknown_Drug_{drug_idx}",
                "graph_index": int(drug_idx),
            },
        )
        if use_probability:
            recommendations.append(
                {
                    "drug_name": drug_record["display_name"],
                    "drug_entity_id": drug_record["entity_id"],
                    "drug_graph_index": int(drug_idx),
                    "identifier": str(drug_record.get("identifier") or ""),
                    "score": sigmoid_score(raw_score),
                    "raw_score": raw_score,
                }
            )
        else:
            recommendations.append(
                {
                    "drug_name": drug_record["display_name"],
                    "drug_entity_id": drug_record["entity_id"],
                    "drug_graph_index": int(drug_idx),
                    "identifier": str(drug_record.get("identifier") or ""),
                    "score": raw_score,
                    "probability": sigmoid_score(raw_score),
                }
            )
    return recommendations, score_type, known_filter_status, known_removed_count


def main():
    args = get_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if args.verbose:
        print(f"Using device: {device}")

    entity_views = load_entity_lookup_views(args)
    disease_record = resolve_disease_record(
        entity_views["disease"],
        disease_name=args.disease,
        disease_id=args.disease_id,
    )

    graph_path = Path(args.graph_data)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph data file '{graph_path}' does not exist.")

    try:
        graph_data = torch.load(graph_path, map_location="cpu", weights_only=False)
        if not isinstance(graph_data, HeteroData):
            raise ValueError("Loaded graph data is not a HeteroData object.")
    except Exception as e:
        raise Exception(f"Error loading graph data from '{args.graph_data}': {e}")

    graph_data = graph_data.to(device)

    model = KGCModel(
        data=graph_data,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        num_srr_layers=args.num_srr_layers,
        num_attention_heads=args.num_attention_heads,
        dropout=args.dropout,
        use_global_router=args.use_global_router,
        mechanism_dropout=args.mechanism_dropout,
        decoder_type=args.decoder_type,
    )

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file '{model_path}' does not exist.")

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(extract_state_dict(checkpoint))
        if args.verbose:
            print(f"Loaded model weights from '{model_path}'.")
    except Exception as e:
        raise Exception(f"Error loading model weights from '{args.model_path}': {e}")

    model = model.to(device)

    known_edge_filter = load_known_edge_filter(args)
    if args.verbose:
        print(
            "Known-edge filter:"
            f" status={known_edge_filter.get('status')}, "
            f"removed_edges={known_edge_filter.get('removed_edge_count', 0)}"
        )

    recommendations, score_type, known_filter_status, known_removed_count = recommend_drugs(
        model=model,
        full_graph=graph_data,
        disease_record=disease_record,
        chemical_views=entity_views["chemical"],
        top_k=args.top_k,
        use_probability=not args.use_raw_scores,
        scoring_mode=args.scoring_mode,
        known_edge_filter=known_edge_filter,
        candidate_filter=args.candidate_filter,
        exclude_candidate_regex=args.exclude_candidate_regex,
        require_smiles=bool(args.require_smiles),
        verbose=args.verbose,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for rank, rec in enumerate(recommendations, start=1):
            if score_type == "Probability":
                rows.append(
                    {
                        "Rank": rank,
                        "Disease": disease_record["display_name"],
                        "DiseaseEntityID": int(disease_record["entity_id"]),
                        "Drug": rec["drug_name"],
                        "DrugEntityID": rec["drug_entity_id"],
                        "Identifier": rec["identifier"],
                        "DrugGraphIndex": rec["drug_graph_index"],
                        "Probability": float(rec["score"]),
                        "RawScore": float(rec["raw_score"]),
                        "KnownEdgeFilterStatus": known_filter_status,
                        "KnownEdgesRemovedCount": int(known_removed_count),
                    }
                )
            else:
                rows.append(
                    {
                        "Rank": rank,
                        "Disease": disease_record["display_name"],
                        "DiseaseEntityID": int(disease_record["entity_id"]),
                        "Drug": rec["drug_name"],
                        "DrugEntityID": rec["drug_entity_id"],
                        "Identifier": rec["identifier"],
                        "DrugGraphIndex": rec["drug_graph_index"],
                        "RawScore": float(rec["score"]),
                        "Probability": float(rec["probability"]),
                        "KnownEdgeFilterStatus": known_filter_status,
                        "KnownEdgesRemovedCount": int(known_removed_count),
                    }
                )
        df_out = pd.DataFrame(rows)
        sep = "\t" if out_path.suffix.lower() == ".tsv" else ","
        df_out.to_csv(out_path, index=False, sep=sep)
        print(f"Saved recommendations to: {out_path}")

    print(
        f"\nTop-{args.top_k} drug recommendations for disease "
        f"'{disease_record['display_name']}' (entity_id={disease_record['entity_id']}):"
    )
    if score_type == "Probability":
        print(
            f"{'Rank':<5} {'Drug Name':<30} {'Entity ID':<12} {'Identifier':<15} "
            f"{'Probability':<15} {'Raw Score':<15}"
        )
    else:
        print(
            f"{'Rank':<5} {'Drug Name':<30} {'Entity ID':<12} {'Identifier':<15} "
            f"{'Raw Score':<15} {'Probability':<15}"
        )
    print("-" * 70)
    print(
        f"Known-edge filter status: {known_filter_status}; "
        f"known edges removed for disease: {known_removed_count}"
    )
    for rank, rec in enumerate(recommendations, start=1):
        other_metric = rec.get("raw_score", rec.get("probability"))
        print(
            f"{rank:<5} {rec['drug_name']:<30} {str(rec['drug_entity_id']):<12} "
            f"{str(rec['identifier']):<15} "
            f"{rec['score']:<15.6f} {other_metric:<15.6f}"
        )
    print("-" * 70)


if __name__ == "__main__":
    main()
