from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import torch


def ranking_group_axis_to_int(group_axis: str) -> int:
    value = str(group_axis).lower()
    if value in {"src", "source", "chemical", "0"}:
        return 0
    if value in {"dst", "target", "disease", "1"}:
        return 1
    raise ValueError(
        "group_axis must be one of src/source/chemical/0 or dst/target/disease/1"
    )


def _empty_metrics() -> Dict[str, float]:
    return {
        "MRR": 0.0,
        "MAP": 0.0,
        "Hits@1": 0.0,
        "Hits@3": 0.0,
        "Hits@10": 0.0,
    }


def _metrics_from_sorted_labels(sorted_labels_by_group: Iterable[np.ndarray]) -> Dict[str, float]:
    reciprocal_ranks = []
    average_precisions = []
    hits_at_1 = []
    hits_at_3 = []
    hits_at_10 = []

    for labels in sorted_labels_by_group:
        positive_positions = np.flatnonzero(np.asarray(labels, dtype=np.float64) > 0.5) + 1
        if positive_positions.size == 0:
            continue

        reciprocal_ranks.append(1.0 / float(positive_positions[0]))
        precision_at_positive = (
            np.arange(1, positive_positions.size + 1, dtype=np.float64)
            / positive_positions.astype(np.float64)
        )
        average_precisions.append(float(np.mean(precision_at_positive)))
        hits_at_1.append(float(np.any(positive_positions <= 1)))
        hits_at_3.append(float(np.any(positive_positions <= 3)))
        hits_at_10.append(float(np.any(positive_positions <= 10)))

    if not reciprocal_ranks:
        return _empty_metrics()

    return {
        "MRR": float(np.mean(reciprocal_ranks)),
        "MAP": float(np.mean(average_precisions)),
        "Hits@1": float(np.mean(hits_at_1)),
        "Hits@3": float(np.mean(hits_at_3)),
        "Hits@10": float(np.mean(hits_at_10)),
    }


def sampled_grouped_ranking_metrics(
    edge_label_index: torch.Tensor,
    edge_label: torch.Tensor,
    y_score: np.ndarray,
    group_axis: str = "dst",
) -> Dict[str, float]:
    axis = ranking_group_axis_to_int(group_axis)
    edge_index_cpu = edge_label_index.detach().cpu()
    labels = edge_label.detach().float().cpu().numpy()
    scores = np.asarray(y_score, dtype=np.float64)

    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for edge_idx, label in enumerate(labels.tolist()):
        group_id = int(edge_index_cpu[axis, edge_idx].item())
        grouped[group_id].append((float(scores[edge_idx]), float(label)))

    sorted_labels = []
    for values in grouped.values():
        if not any(label > 0.5 for _, label in values):
            continue
        ordered = sorted(values, key=lambda item: item[0], reverse=True)
        sorted_labels.append(np.asarray([label for _, label in ordered], dtype=np.float64))

    return _metrics_from_sorted_labels(sorted_labels)


def collect_positive_pairs(datasets, edge_type: tuple[str, str, str]) -> set[tuple[int, int]]:
    positives: set[tuple[int, int]] = set()
    for data in datasets:
        if data is None or edge_type not in getattr(data, "edge_types", []):
            continue
        store = data[edge_type]
        if hasattr(store, "edge_label_index") and hasattr(store, "edge_label"):
            edge_index = store.edge_label_index.detach().cpu()
            positive_mask = store.edge_label.detach().cpu() > 0.5
            edge_index = edge_index[:, positive_mask]
        elif hasattr(store, "edge_index"):
            edge_index = store.edge_index.detach().cpu()
        else:
            continue

        for src_idx, dst_idx in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            positives.add((int(src_idx), int(dst_idx)))
    return positives


def _score_candidate_edges(
    score_edges_fn: Callable[[torch.Tensor], np.ndarray],
    group_id: int,
    candidates: np.ndarray,
    axis: int,
    candidate_batch_size: int,
) -> np.ndarray:
    scores = []
    batch_size = max(1, int(candidate_batch_size))
    for start in range(0, int(candidates.size), batch_size):
        chunk = candidates[start : start + batch_size]
        if axis == 0:
            edge_index = torch.stack(
                [
                    torch.full((chunk.size,), int(group_id), dtype=torch.long),
                    torch.as_tensor(chunk, dtype=torch.long),
                ],
                dim=0,
            )
        else:
            edge_index = torch.stack(
                [
                    torch.as_tensor(chunk, dtype=torch.long),
                    torch.full((chunk.size,), int(group_id), dtype=torch.long),
                ],
                dim=0,
            )
        scores.append(np.asarray(score_edges_fn(edge_index), dtype=np.float64))
    return np.concatenate(scores, axis=0) if scores else np.zeros(0, dtype=np.float64)


def full_candidate_filtered_ranking_metrics(
    edge_label_index: torch.Tensor,
    edge_label: torch.Tensor,
    num_src_nodes: int,
    num_dst_nodes: int,
    score_edges_fn: Callable[[torch.Tensor], np.ndarray],
    known_positive_pairs: Optional[set[tuple[int, int]]] = None,
    group_axis: str = "dst",
    candidate_batch_size: int = 65536,
) -> Dict[str, float]:
    axis = ranking_group_axis_to_int(group_axis)
    edge_index_cpu = edge_label_index.detach().cpu()
    labels = edge_label.detach().cpu() > 0.5
    known_positive_pairs = known_positive_pairs or set()

    positives_by_group: dict[int, set[int]] = defaultdict(set)
    for edge_idx, is_positive in enumerate(labels.tolist()):
        if not bool(is_positive):
            continue
        src_idx = int(edge_index_cpu[0, edge_idx].item())
        dst_idx = int(edge_index_cpu[1, edge_idx].item())
        if axis == 0:
            positives_by_group[src_idx].add(dst_idx)
        else:
            positives_by_group[dst_idx].add(src_idx)

    if not positives_by_group:
        return _empty_metrics()

    candidate_count = int(num_dst_nodes if axis == 0 else num_src_nodes)
    all_candidates = np.arange(candidate_count, dtype=np.int64)
    sorted_labels = []

    for group_id, positives in positives_by_group.items():
        if not positives:
            continue
        kept_candidates = []
        for candidate in all_candidates.tolist():
            pair = (group_id, candidate) if axis == 0 else (candidate, group_id)
            if pair in known_positive_pairs and candidate not in positives:
                continue
            kept_candidates.append(candidate)

        candidates = np.asarray(kept_candidates, dtype=np.int64)
        if candidates.size == 0:
            continue
        labels_for_candidates = np.asarray(
            [1.0 if int(candidate) in positives else 0.0 for candidate in candidates],
            dtype=np.float64,
        )
        if labels_for_candidates.sum() <= 0:
            continue

        scores = _score_candidate_edges(
            score_edges_fn=score_edges_fn,
            group_id=group_id,
            candidates=candidates,
            axis=axis,
            candidate_batch_size=candidate_batch_size,
        )
        order = np.argsort(-scores)
        sorted_labels.append(labels_for_candidates[order])

    return _metrics_from_sorted_labels(sorted_labels)
