from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .utils.distributed import (
    DistributedState,
    barrier,
    broadcast_scalars,
    is_main_process,
    log0,
)
from .utils.evaluation import monitor_gradients
from .utils.ranking import (
    collect_positive_pairs,
    full_candidate_filtered_ranking_metrics,
    ranking_group_axis_to_int,
    sampled_grouped_ranking_metrics,
)

try:
    from negative_sampler import NegativeSampler
except ImportError:
    NegativeSampler = None


@dataclass
class TrainerConfig:
    hidden_channels: int
    out_channels: int
    num_srr_layers: int
    num_attention_heads: int
    dropout: float
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    epochs: int
    eval_interval: int
    early_stopping_patience: int
    grad_clip_norm: float
    monitor_gradients: bool
    infonce_temperature: float
    infonce_loss_mode: str
    edge_type_to_predict: tuple[str, str, str]
    aux_bce_weight: float = 0.1
    source_batch_size: int = 256
    decoder_type: str = "complex"
    primary_metric: str = "aupr"
    ranking_group_axis: str = "src"
    training_group_axis: str = "src"
    full_ranking_candidate_batch_size: int = 65536
    find_unused_parameters: bool = False
    use_global_router: bool = True
    mechanism_dropout: float = 0.1
    train_neg_per_pos: int = 5
    online_negative_sampling: bool = True
    use_contraindication_negatives: bool = True
    contraindication_relation: tuple[str, str, str] = (
        "chemical",
        "contraindicates_cd",
        "disease",
    )
    negative_curriculum_epochs: int = 0
    negative_start_factor: float = 1.0
    hard_negative_ratio_start: float = 0.0
    min_learning_rate_ratio: float = 0.1
    aux_focal_enabled: bool = False
    aux_focal_alpha: float = 0.25
    aux_focal_gamma: float = 2.0
    aux_focal_weight: float = 0.0
    use_ema: bool = False
    ema_decay: float = 0.999
    ema_update_after_step: int = 0
    ema_update_every: int = 1
    edge_dropout_p: float = 0.0
    log_interval: int = 1


class Trainer:
    def __init__(
        self,
        cfg: TrainerConfig,
        dist_state: DistributedState,
        neg_sampler=None,
        trial_reporter=None,
    ) -> None:
        self.cfg = cfg
        self.dist_state = dist_state
        self.neg_sampler = neg_sampler
        self.trial_reporter = trial_reporter
        self.last_run_summary: Optional[Dict[str, Dict[str, float] | float]] = None
        self._ema_state: Optional[Dict[str, torch.Tensor]] = None
        self._ema_backup_state: Optional[Dict[str, torch.Tensor]] = None
        self._global_step: int = 0
        self._known_positive_pairs: Optional[set[tuple[int, int]]] = None

    @staticmethod
    def _base_model(model: torch.nn.Module) -> torch.nn.Module:
        return model.module if isinstance(model, DDP) else model

    @staticmethod
    def _to_scalar(x) -> float:
        if torch.is_tensor(x):
            return float(x.detach().item())
        return float(x)

    def _maybe_init_ema(self, model: torch.nn.Module) -> None:
        self._ema_state: Optional[Dict[str, torch.Tensor]]
        self._ema_backup_state: Optional[Dict[str, torch.Tensor]]
        self._ema_state = None
        self._ema_backup_state = None
        if not bool(self.cfg.use_ema):
            return

        base_model = self._base_model(model)
        self._ema_state = {
            name: param.detach().clone()
            for name, param in base_model.named_parameters()
            if param.requires_grad
        }

    def _update_ema(self, model: torch.nn.Module, global_step: int) -> None:
        if self._ema_state is None or not bool(self.cfg.use_ema):
            return
        if global_step < int(self.cfg.ema_update_after_step):
            return
        update_every = max(1, int(self.cfg.ema_update_every))
        if global_step % update_every != 0:
            return

        decay = float(self.cfg.ema_decay)
        decay = min(max(decay, 0.0), 1.0)
        base_model = self._base_model(model)
        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if not param.requires_grad or name not in self._ema_state:
                    continue
                shadow = self._ema_state[name]
                shadow.mul_(decay).add_(param.detach(), alpha=1.0 - decay)

    def _swap_in_ema_weights(self, model: torch.nn.Module) -> None:
        if self._ema_state is None:
            return
        base_model = self._base_model(model)
        self._ema_backup_state = {}
        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if not param.requires_grad or name not in self._ema_state:
                    continue
                self._ema_backup_state[name] = param.detach().clone()
                param.copy_(self._ema_state[name])

    def _restore_original_weights(self, model: torch.nn.Module) -> None:
        if self._ema_state is None:
            return
        if self._ema_backup_state is None:
            raise RuntimeError("EMA backup weights are missing")

        base_model = self._base_model(model)
        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if not param.requires_grad or name not in self._ema_backup_state:
                    continue
                param.copy_(self._ema_backup_state[name])
        self._ema_backup_state = None

    def _apply_edge_dropout(
        self,
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[Tuple[str, str, str], torch.Tensor]:
        drop_prob = float(self.cfg.edge_dropout_p)
        if drop_prob <= 0.0:
            return edge_index_dict

        drop_prob = min(max(drop_prob, 0.0), 1.0)
        keep_prob = 1.0 - drop_prob
        dropped: Dict[Tuple[str, str, str], torch.Tensor] = {}

        for edge_type, edge_index in edge_index_dict.items():
            if edge_index.numel() == 0 or edge_index.dim() != 2 or edge_index.size(1) == 0:
                dropped[edge_type] = edge_index
                continue

            num_edges = int(edge_index.size(1))
            keep_mask = torch.rand(num_edges, device=edge_index.device) < keep_prob
            if not bool(keep_mask.any()):
                keep_mask[torch.randint(0, num_edges, (1,), device=edge_index.device)] = True
            dropped[edge_type] = edge_index[:, keep_mask]

        return dropped

    def _selection_score(self, metrics: Dict[str, float]) -> float:
        primary_metric = str(self.cfg.primary_metric).lower()
        key_by_metric = {
            "auc": "AUC",
            "aupr": "AUPR",
            "mrr": "MRR",
            "map": "MAP",
            "hits@1": "Hits@1",
            "hits@3": "Hits@3",
            "hits@10": "Hits@10",
            "f1": "F1",
            "maxf1": "MaxF1",
            "mcc": "MCC",
        }
        return float(metrics.get(key_by_metric.get(primary_metric, "AUPR"), 0.0))

    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator > 0.0 else 0.0

    @classmethod
    def _binary_metrics_at_threshold(
        cls,
        y_true: np.ndarray,
        y_score: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        labels = np.asarray(y_true, dtype=np.float64) > 0.5
        preds = np.asarray(y_score, dtype=np.float64) >= float(threshold)

        tp = float(np.logical_and(preds, labels).sum())
        tn = float(np.logical_and(~preds, ~labels).sum())
        fp = float(np.logical_and(preds, ~labels).sum())
        fn = float(np.logical_and(~preds, labels).sum())

        precision = cls._safe_div(tp, tp + fp)
        recall = cls._safe_div(tp, tp + fn)
        specificity = cls._safe_div(tn, tn + fp)
        f1 = cls._safe_div(2.0 * precision * recall, precision + recall)
        denom = float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc = cls._safe_div(tp * tn - fp * fn, denom)

        return {
            "Accuracy": cls._safe_div(tp + tn, tp + tn + fp + fn),
            "BalancedAccuracy": 0.5 * (recall + specificity),
            "Precision": precision,
            "Recall": recall,
            "Specificity": specificity,
            "F1": f1,
            "MCC": mcc,
        }

    @classmethod
    def _max_f1_metric(cls, y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
        labels = np.asarray(y_true, dtype=np.float64) > 0.5
        scores = np.asarray(y_score, dtype=np.float64)
        total_pos = float(labels.sum())
        if total_pos <= 0.0 or labels.size == 0:
            return {"MaxF1": 0.0}

        order = np.argsort(-scores)
        ranked_labels = labels[order]
        tp = np.cumsum(ranked_labels).astype(np.float64)
        fp = np.cumsum(~ranked_labels).astype(np.float64)
        precision = tp / np.maximum(tp + fp, 1.0)
        recall = tp / total_pos
        f1 = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision, dtype=np.float64),
            where=(precision + recall) > 0.0,
        )
        return {"MaxF1": float(f1.max()) if f1.size else 0.0}

    @staticmethod
    def _ranking_group_axis_to_int(group_axis: str) -> int:
        return ranking_group_axis_to_int(group_axis)

    @classmethod
    def _grouped_ranking_metrics(
        cls,
        edge_label_index: torch.Tensor,
        edge_label: torch.Tensor,
        y_score: np.ndarray,
        group_axis: str = "dst",
    ) -> Dict[str, float]:
        return sampled_grouped_ranking_metrics(
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            y_score=y_score,
            group_axis=group_axis,
        )

    def _full_candidate_ranking_metrics(
        self,
        model,
        z_dict: Dict[str, torch.Tensor],
        data,
        edge_label_index: torch.Tensor,
        edge_label: torch.Tensor,
        filtered_positive_pairs: Optional[set[tuple[int, int]]] = None,
    ) -> Dict[str, float]:
        src_type, _, dst_type = self.cfg.edge_type_to_predict
        num_src_nodes = int(data[src_type].num_nodes)
        num_dst_nodes = int(data[dst_type].num_nodes)
        device = self.dist_state.device

        def score_edges(edge_index_cpu: torch.Tensor) -> np.ndarray:
            edge_index = edge_index_cpu.to(device=device)
            logits = model.decode_edges(
                z_dict=z_dict,
                edge_label_index=edge_index,
                relation=self.cfg.edge_type_to_predict,
            )
            return torch.sigmoid(logits).detach().float().cpu().numpy()

        return full_candidate_filtered_ranking_metrics(
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            num_src_nodes=num_src_nodes,
            num_dst_nodes=num_dst_nodes,
            score_edges_fn=score_edges,
            known_positive_pairs=filtered_positive_pairs,
            group_axis=self.cfg.ranking_group_axis,
            candidate_batch_size=int(self.cfg.full_ranking_candidate_batch_size),
        )

    def _init_model_and_optimizer(self, train_data):
        from .model.model import KGCModel

        model = KGCModel(
            data=train_data,
            hidden_channels=self.cfg.hidden_channels,
            out_channels=self.cfg.out_channels,
            num_srr_layers=self.cfg.num_srr_layers,
            num_attention_heads=self.cfg.num_attention_heads,
            dropout=self.cfg.dropout,
            use_global_router=self.cfg.use_global_router,
            mechanism_dropout=self.cfg.mechanism_dropout,
            decoder_type=self.cfg.decoder_type,
        ).to(self.dist_state.device)

        if self.dist_state.distributed:
            ddp_kwargs = {"find_unused_parameters": self.cfg.find_unused_parameters}
            if self.dist_state.device.type == "cuda":
                model = DDP(
                    model,
                    device_ids=[self.dist_state.local_rank],
                    output_device=self.dist_state.local_rank,
                    **ddp_kwargs,
                )
            else:
                model = DDP(model, **ddp_kwargs)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        warmup_epochs = max(0, int(self.cfg.warmup_epochs))
        total_epochs = max(1, int(self.cfg.epochs))
        min_lr_ratio = min(max(float(self.cfg.min_learning_rate_ratio), 0.0), 1.0)
        if warmup_epochs > 0 and total_epochs > warmup_epochs:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=0.1,
                        total_iters=warmup_epochs,
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=max(1, total_epochs - warmup_epochs),
                        eta_min=float(self.cfg.learning_rate) * min_lr_ratio,
                    ),
                ],
                milestones=[warmup_epochs],
            )
        elif warmup_epochs > 0:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=warmup_epochs,
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs,
                eta_min=float(self.cfg.learning_rate) * min_lr_ratio,
            )
        return model, optimizer, scheduler

    def _edge_store(self, data):
        edge_store = data[self.cfg.edge_type_to_predict]
        if not hasattr(edge_store, "edge_label_index") or not hasattr(
            edge_store,
            "edge_label",
        ):
            raise ValueError(
                "Missing edge_label_index/edge_label for edge type "
                f"{self.cfg.edge_type_to_predict!r}"
            )
        return edge_store

    @staticmethod
    def _build_source_groups(
        edge_label_index,
        edge_label,
        group_axis: str = "src",
    ) -> List[Tuple[int, torch.Tensor, torch.Tensor]]:
        axis = ranking_group_axis_to_int(group_axis)
        src = edge_label_index[0].detach().cpu()
        dst = edge_label_index[1].detach().cpu()
        lbl = edge_label.detach().cpu()

        groups: dict[int, dict[str, list[int]]] = {}
        for src_idx, dst_idx, label in zip(src.tolist(), dst.tolist(), lbl.tolist()):
            group_idx = int(dst_idx if axis == 1 else src_idx)
            candidate_idx = int(src_idx if axis == 1 else dst_idx)
            bucket = groups.setdefault(group_idx, {"pos": [], "neg": []})
            if float(label) > 0.5:
                bucket["pos"].append(candidate_idx)
            else:
                bucket["neg"].append(candidate_idx)

        out: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
        for group_idx, payload in groups.items():
            if not payload["pos"] or not payload["neg"]:
                continue
            pos = torch.tensor(sorted(set(payload["pos"])), dtype=torch.long)
            neg = torch.tensor(sorted(set(payload["neg"])), dtype=torch.long)
            if pos.numel() == 0 or neg.numel() == 0:
                continue
            out.append((int(group_idx), pos, neg))
        return out

    @staticmethod
    def _positive_edge_index(edge_label_index, edge_label) -> torch.Tensor:
        pos_mask = edge_label.detach().cpu() > 0.5
        return edge_label_index.detach().cpu()[:, pos_mask]

    def _collect_known_positive_pairs(self, datasets) -> set[tuple[int, int]]:
        return collect_positive_pairs(datasets, self.cfg.edge_type_to_predict)

    @staticmethod
    def _filter_known_positive_edges(
        edge_index: Optional[torch.Tensor],
        known_positive_pairs: Optional[set[tuple[int, int]]],
    ) -> Optional[torch.Tensor]:
        if edge_index is None or edge_index.numel() == 0 or not known_positive_pairs:
            return edge_index

        ei = edge_index.detach().cpu()
        keep_mask = [
            (int(src), int(dst)) not in known_positive_pairs
            for src, dst in zip(ei[0].tolist(), ei[1].tolist())
        ]
        if all(keep_mask):
            return ei
        if not any(keep_mask):
            return torch.zeros((2, 0), dtype=torch.long)
        return ei[:, torch.tensor(keep_mask, dtype=torch.bool)]

    @staticmethod
    def _init_group_buckets(
        pos_edge_index: torch.Tensor,
        group_axis: str = "src",
    ) -> dict[int, dict[str, set[int]]]:
        axis = ranking_group_axis_to_int(group_axis)
        groups: dict[int, dict[str, set[int]]] = {}
        for src_idx, dst_idx in zip(
            pos_edge_index[0].tolist(),
            pos_edge_index[1].tolist(),
        ):
            group_idx = int(dst_idx if axis == 1 else src_idx)
            candidate_idx = int(src_idx if axis == 1 else dst_idx)
            bucket = groups.setdefault(group_idx, {"pos": set(), "neg": set()})
            bucket["pos"].add(candidate_idx)
        return groups

    @staticmethod
    def _add_negative_edges(
        groups: dict[int, dict[str, set[int]]],
        negative_edge_index: Optional[torch.Tensor],
        group_axis: str = "src",
    ) -> None:
        if negative_edge_index is None or negative_edge_index.numel() == 0:
            return
        axis = ranking_group_axis_to_int(group_axis)
        neg_ei = negative_edge_index.detach().cpu()
        for src_idx, dst_idx in zip(neg_ei[0].tolist(), neg_ei[1].tolist()):
            group_idx = int(dst_idx if axis == 1 else src_idx)
            candidate_idx = int(src_idx if axis == 1 else dst_idx)
            bucket = groups.setdefault(group_idx, {"pos": set(), "neg": set()})
            bucket["neg"].add(candidate_idx)

    @staticmethod
    def _finalize_group_buckets(
        groups: dict[int, dict[str, set[int]]],
    ) -> List[Tuple[int, torch.Tensor, torch.Tensor]]:
        out: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
        for src_idx, payload in groups.items():
            pos_set = set(payload["pos"])
            neg_set = set(payload["neg"]) - pos_set
            if not pos_set or not neg_set:
                continue
            pos = torch.tensor(sorted(pos_set), dtype=torch.long)
            neg = torch.tensor(sorted(neg_set), dtype=torch.long)
            if pos.numel() == 0 or neg.numel() == 0:
                continue
            out.append((int(src_idx), pos, neg))
        return out

    def _curriculum_progress(self, epoch: int) -> float:
        curriculum_epochs = max(0, int(self.cfg.negative_curriculum_epochs))
        if curriculum_epochs <= 1:
            return 1.0
        progress = float(epoch - 1) / float(max(1, curriculum_epochs - 1))
        return min(max(progress, 0.0), 1.0)

    def _effective_train_neg_per_pos(self, epoch: int) -> int:
        target = max(1, int(self.cfg.train_neg_per_pos))
        if target <= 1:
            return target
        start_factor = min(max(float(self.cfg.negative_start_factor), 0.0), 1.0)
        start_value = max(1, int(round(target * start_factor)))
        if start_value >= target:
            return target
        progress = self._curriculum_progress(epoch)
        current = start_value + int(round((target - start_value) * progress))
        return min(target, max(1, current))

    def _effective_hard_negative_ratio(self, epoch: int) -> float:
        target = 0.0
        if self.neg_sampler is not None and hasattr(self.neg_sampler, "hard_neg_ratio"):
            target = float(self.neg_sampler.hard_neg_ratio)
        start = min(max(float(self.cfg.hard_negative_ratio_start), 0.0), 1.0)
        target = min(max(float(target), 0.0), 1.0)
        if target <= start:
            return target
        progress = self._curriculum_progress(epoch)
        return start + (target - start) * progress

    def _sample_online_negatives(
        self,
        pos_edge_index: torch.Tensor,
        epoch: int,
    ) -> Optional[torch.Tensor]:
        if (
            not self.cfg.online_negative_sampling
            or self.neg_sampler is None
            or pos_edge_index.numel() == 0
        ):
            return None

        effective_neg_per_pos = self._effective_train_neg_per_pos(epoch)
        effective_hard_ratio = self._effective_hard_negative_ratio(epoch)
        original_ratio = float(getattr(self.neg_sampler, "hard_neg_ratio", 0.0))
        try:
            self.neg_sampler.hard_neg_ratio = effective_hard_ratio
            neg_ei, _ = self.neg_sampler.sample_epoch(
                pos_edge_index,
                n_neg_per_pos=effective_neg_per_pos,
                epoch=epoch,
            )
        finally:
            self.neg_sampler.hard_neg_ratio = original_ratio
        return neg_ei

    def _build_training_source_groups(
        self,
        train_data,
        epoch: int,
    ) -> tuple[List[Tuple[int, torch.Tensor, torch.Tensor]], Dict[str, float]]:
        edge_store = self._edge_store(train_data)
        pos_edge_index = self._positive_edge_index(
            edge_store.edge_label_index,
            edge_store.edge_label,
        )
        training_group_axis = str(getattr(self.cfg, "training_group_axis", "src"))
        groups = self._init_group_buckets(
            pos_edge_index=pos_edge_index,
            group_axis=training_group_axis,
        )

        online_neg_ei = self._sample_online_negatives(
            pos_edge_index=pos_edge_index,
            epoch=epoch,
        )

        if online_neg_ei is not None:
            self._add_negative_edges(
                groups,
                online_neg_ei,
                group_axis=training_group_axis,
            )
        else:
            offline_neg_mask = edge_store.edge_label.detach().cpu() <= 0.5
            offline_neg_ei = edge_store.edge_label_index.detach().cpu()[
                :, offline_neg_mask
            ]
            self._add_negative_edges(
                groups,
                offline_neg_ei,
                group_axis=training_group_axis,
            )

        contraindication_edges = None
        if self.cfg.use_contraindication_negatives:
            contra_rel = self.cfg.contraindication_relation
            if contra_rel in train_data.edge_types:
                contraindication_edges = train_data[contra_rel].edge_index
                contraindication_edges = self._filter_known_positive_edges(
                    contraindication_edges,
                    self._known_positive_pairs,
                )
                self._add_negative_edges(
                    groups,
                    contraindication_edges,
                    group_axis=training_group_axis,
                )

        source_groups = self._finalize_group_buckets(groups)
        avg_candidates = 0.0
        if source_groups:
            avg_candidates = float(
                np.mean([pos.numel() + neg.numel() for _, pos, neg in source_groups])
            )

        diagnostics = {
            "effective_train_neg_per_pos": float(
                self._effective_train_neg_per_pos(epoch)
            ),
            "effective_hard_neg_ratio": float(
                self._effective_hard_negative_ratio(epoch)
            ),
            "num_source_groups": float(len(source_groups)),
            "avg_candidates": avg_candidates,
            "online_neg_edges": float(online_neg_ei.shape[1])
            if online_neg_ei is not None
            else 0.0,
            "contra_neg_edges": float(contraindication_edges.shape[1])
            if contraindication_edges is not None
            else 0.0,
            "training_group_axis": float(ranking_group_axis_to_int(training_group_axis)),
        }
        return source_groups, diagnostics

    def _iterate_source_batches(
        self,
        source_groups: List[Tuple[int, torch.Tensor, torch.Tensor]],
        shuffle: bool,
        device: torch.device,
    ):
        if not source_groups:
            return

        order = torch.arange(len(source_groups))
        if shuffle:
            order = order[torch.randperm(order.shape[0])]

        batch_size = max(1, int(self.cfg.source_batch_size))
        for start in range(0, order.shape[0], batch_size):
            batch_ids = order[start : start + batch_size].tolist()
            batch_groups = [source_groups[int(idx)] for idx in batch_ids]
            width = max(pos.numel() + neg.numel() for _, pos, neg in batch_groups)

            source_index = torch.zeros(len(batch_groups), dtype=torch.long, device=device)
            candidate_index = torch.zeros(
                (len(batch_groups), width),
                dtype=torch.long,
                device=device,
            )
            positive_mask = torch.zeros(
                (len(batch_groups), width),
                dtype=torch.bool,
                device=device,
            )
            candidate_mask = torch.zeros(
                (len(batch_groups), width),
                dtype=torch.bool,
                device=device,
            )

            for row_idx, (src_idx, pos, neg) in enumerate(batch_groups):
                pos = pos.to(device=device, dtype=torch.long)
                neg = neg.to(device=device, dtype=torch.long)
                candidates = torch.cat([pos, neg], dim=0)
                count = int(candidates.numel())
                source_index[row_idx] = int(src_idx)
                candidate_index[row_idx, :count] = candidates
                positive_mask[row_idx, : pos.numel()] = True
                candidate_mask[row_idx, :count] = True

            yield source_index, candidate_index, positive_mask, candidate_mask

    def _compute_grouped_batch_loss(
        self,
        base_model,
        z_dict,
        source_index,
        candidate_index,
        positive_mask,
        candidate_mask,
    ):
        return base_model.loss(
            z_dict=z_dict,
            source_index=source_index,
            candidate_index=candidate_index,
            positive_mask=positive_mask,
            candidate_mask=candidate_mask,
            relation=self.cfg.edge_type_to_predict,
            infonce_temperature=self.cfg.infonce_temperature,
            infonce_loss_mode=self.cfg.infonce_loss_mode,
            aux_bce_weight=self.cfg.aux_bce_weight,
            aux_focal_enabled=self.cfg.aux_focal_enabled,
            aux_focal_alpha=self.cfg.aux_focal_alpha,
            aux_focal_gamma=self.cfg.aux_focal_gamma,
            aux_focal_weight=self.cfg.aux_focal_weight,
            group_axis=str(getattr(self.cfg, "training_group_axis", "src")),
        )

    @torch.no_grad()
    def evaluate(
        self,
        model,
        data,
        edge_index_dict_for_message_passing,
        filtered_positive_pairs: Optional[set[tuple[int, int]]] = None,
    ) -> Dict[str, float]:
        using_ema = self._ema_state is not None and bool(self.cfg.use_ema)
        if using_ema:
            self._swap_in_ema_weights(model)

        device = self.dist_state.device
        data_device = next(iter(data.x_dict.values())).device
        moved_eval_data = data_device != device
        if moved_eval_data:
            data = data.to(device)

        edge_index_dict_for_message_passing = {
            edge_type: edge_index.to(device)
            if getattr(edge_index, "device", device) != device
            else edge_index
            for edge_type, edge_index in edge_index_dict_for_message_passing.items()
        }

        try:
            model.eval()
            z_dict = model(data.x_dict, edge_index_dict_for_message_passing)
            edge_store = self._edge_store(data)
            source_groups = self._build_source_groups(
                edge_store.edge_label_index,
                edge_store.edge_label,
                group_axis=self.cfg.ranking_group_axis,
            )

            loss_total = 0.0
            loss_steps = 0
            for source_index, candidate_index, positive_mask, candidate_mask in self._iterate_source_batches(
                source_groups=source_groups,
                shuffle=False,
                device=self.dist_state.device,
            ):
                loss, _, _ = model.loss(
                    z_dict=z_dict,
                    source_index=source_index,
                    candidate_index=candidate_index,
                    positive_mask=positive_mask,
                    candidate_mask=candidate_mask,
                    relation=self.cfg.edge_type_to_predict,
                    infonce_temperature=self.cfg.infonce_temperature,
                    infonce_loss_mode=self.cfg.infonce_loss_mode,
                    aux_bce_weight=self.cfg.aux_bce_weight,
                    aux_focal_enabled=self.cfg.aux_focal_enabled,
                    aux_focal_alpha=self.cfg.aux_focal_alpha,
                    aux_focal_gamma=self.cfg.aux_focal_gamma,
                    aux_focal_weight=self.cfg.aux_focal_weight,
                    group_axis=self.cfg.ranking_group_axis,
                )
                loss_total += float(loss.detach().item())
                loss_steps += 1

            logits = model.decode_edges(
                z_dict=z_dict,
                edge_label_index=edge_store.edge_label_index,
                relation=self.cfg.edge_type_to_predict,
            )

            y_true = edge_store.edge_label.detach().float().cpu().numpy()
            y_score = torch.sigmoid(logits).detach().float().cpu().numpy()

            if np.unique(y_true).size < 2:
                auc_value = 0.0
                aupr_value = 0.0
            else:
                try:
                    auc_value = float(roc_auc_score(y_true, y_score))
                except Exception:
                    auc_value = 0.0
                try:
                    aupr_value = float(average_precision_score(y_true, y_score))
                except Exception:
                    aupr_value = 0.0

            metrics = {
                "Loss": loss_total / max(loss_steps, 1),
                "AUC": auc_value,
                "AUPR": aupr_value,
            }
            metrics.update(self._binary_metrics_at_threshold(y_true, y_score))
            metrics.update(self._max_f1_metric(y_true, y_score))
            sampled_metrics = self._grouped_ranking_metrics(
                edge_label_index=edge_store.edge_label_index,
                edge_label=edge_store.edge_label,
                y_score=y_score,
                group_axis=self.cfg.ranking_group_axis,
            )
            metrics.update(
                {
                    f"Sampled{metric_name}": metric_value
                    for metric_name, metric_value in sampled_metrics.items()
                }
            )
            metrics.update(
                self._full_candidate_ranking_metrics(
                    model=model,
                    z_dict=z_dict,
                    data=data,
                    edge_label_index=edge_store.edge_label_index,
                    edge_label=edge_store.edge_label,
                    filtered_positive_pairs=filtered_positive_pairs,
                )
            )
            return metrics
        finally:
            if using_ema:
                self._restore_original_weights(model)
            if moved_eval_data:
                data.to("cpu")
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    def _train_epoch(self, model, optimizer, train_data, epoch: int) -> Dict[str, float]:
        model.train()
        base_model = self._base_model(model)
        source_groups, diagnostics = self._build_training_source_groups(
            train_data=train_data,
            epoch=epoch,
        )
        if not source_groups:
            raise ValueError("No valid source groups with both positive and negative labels were found")

        loss_sum = 0.0
        info_nce_sum = 0.0
        aux_bce_sum = 0.0
        grad_norm_pre_values = []
        grad_norm_post_values = []
        clip_ratio_values = []
        num_steps = 0

        for source_index, candidate_index, positive_mask, candidate_mask in self._iterate_source_batches(
            source_groups=source_groups,
            shuffle=True,
            device=self.dist_state.device,
        ):
            optimizer.zero_grad()
            edge_index_dict_for_train = self._apply_edge_dropout(train_data.edge_index_dict)
            z_dict = model(train_data.x_dict, edge_index_dict_for_train)
            loss, components, _ = self._compute_grouped_batch_loss(
                base_model=base_model,
                z_dict=z_dict,
                source_index=source_index,
                candidate_index=candidate_index,
                positive_mask=positive_mask,
                candidate_mask=candidate_mask,
            )
            loss.backward()

            if self.cfg.monitor_gradients and is_main_process(self.dist_state):
                grad_norm_pre_values.append(float(monitor_gradients(base_model)))

            clip_total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=self.cfg.grad_clip_norm,
            )
            clip_total_norm = self._to_scalar(clip_total_norm)
            clip_ratio = 1.0
            if clip_total_norm > 0.0 and float(self.cfg.grad_clip_norm) > 0.0:
                clip_ratio = min(1.0, float(self.cfg.grad_clip_norm) / clip_total_norm)

            if self.cfg.monitor_gradients and is_main_process(self.dist_state):
                grad_norm_post_values.append(float(monitor_gradients(base_model)))

            optimizer.step()
            self._global_step += 1
            self._update_ema(model, global_step=self._global_step)

            loss_sum += float(loss.detach().item())
            info_nce_sum += self._to_scalar(components.get("info_nce", 0.0))
            aux_bce_sum += self._to_scalar(components.get("aux_bce", 0.0))
            clip_ratio_values.append(float(clip_ratio))
            num_steps += 1

        return {
            "loss": loss_sum / max(num_steps, 1),
            "info_nce": info_nce_sum / max(num_steps, 1),
            "aux_bce": aux_bce_sum / max(num_steps, 1),
            "clip_ratio": float(np.mean(clip_ratio_values)) if clip_ratio_values else 1.0,
            "grad_norm_pre": float(np.mean(grad_norm_pre_values)) if grad_norm_pre_values else None,
            "grad_norm_post": float(np.mean(grad_norm_post_values)) if grad_norm_post_values else None,
            "effective_train_neg_per_pos": diagnostics["effective_train_neg_per_pos"],
            "effective_hard_neg_ratio": diagnostics["effective_hard_neg_ratio"],
            "num_source_groups": diagnostics["num_source_groups"],
            "avg_candidates": diagnostics["avg_candidates"],
            "online_neg_edges": diagnostics["online_neg_edges"],
            "contra_neg_edges": diagnostics["contra_neg_edges"],
        }

    def _format_train_status(
        self,
        epoch: int,
        train_log: Dict[str, float],
        learning_rate: float,
    ) -> str:
        message = (
            f"[Epoch {epoch:04d}] "
            f"loss={train_log['loss']:.4f} "
            f"info_nce={train_log['info_nce']:.4f} "
            f"aux_bce={train_log['aux_bce']:.4f} "
            f"lr={learning_rate:.2e} "
            f"neg_per_pos={int(train_log['effective_train_neg_per_pos'])} "
            f"hard_ratio={train_log['effective_hard_neg_ratio']:.2f} "
            f"groups={int(train_log['num_source_groups'])} "
            f"cand={train_log['avg_candidates']:.1f}"
        )
        if train_log["grad_norm_pre"] is not None:
            message += f" gnorm_pre={train_log['grad_norm_pre']:.3f}"
        if train_log["grad_norm_post"] is not None:
            message += f" gnorm_post={train_log['grad_norm_post']:.3f}"
        message += f" clip_ratio={train_log['clip_ratio']:.4f}"
        return message

    def train_and_evaluate_model(
        self,
        train_data,
        valid_data,
        test_data,
        model_save_path,
    ) -> Dict[str, float]:
        model, optimizer, scheduler = self._init_model_and_optimizer(train_data)
        base_model = self._base_model(model)
        self._maybe_init_ema(model)
        self._global_step = 0

        log0(self.dist_state, "\n--- Start training ---")

        split_mode = str(getattr(train_data, "split_mode", "edge_level")).lower()
        self._known_positive_pairs = self._collect_known_positive_pairs(
            (train_data, valid_data, test_data)
        )
        val_filtered_positive_pairs = self._collect_known_positive_pairs((train_data,))
        test_filtered_positive_pairs = self._collect_known_positive_pairs(
            (train_data, valid_data)
        )
        use_train_background_for_eval = split_mode != "disease_cold_start"
        val_message_passing_edges = (
            train_data.edge_index_dict
            if use_train_background_for_eval
            else valid_data.edge_index_dict
        )
        test_message_passing_edges = (
            train_data.edge_index_dict
            if use_train_background_for_eval
            else test_data.edge_index_dict
        )

        best_val_score = float("-inf")
        best_val_metrics: Dict[str, float] | None = None
        epochs_no_improve = 0

        iterator = tqdm(
            range(1, self.cfg.epochs + 1),
            desc="Epoch",
            unit="ep",
            dynamic_ncols=True,
            disable=not is_main_process(self.dist_state),
        )

        for epoch in iterator:
            train_log = self._train_epoch(
                model=model,
                optimizer=optimizer,
                train_data=train_data,
                epoch=epoch,
            )
            scheduler.step()

            current_lr = float(optimizer.param_groups[0].get("lr", 0.0))
            eval_interval = max(1, int(self.cfg.eval_interval))
            log_interval = max(1, int(getattr(self.cfg, "log_interval", 1)))
            should_eval = epoch % eval_interval == 0 or epoch == 1

            if not should_eval:
                if is_main_process(self.dist_state) and epoch % log_interval == 0:
                    log0(
                        self.dist_state,
                        self._format_train_status(epoch, train_log, current_lr),
                    )
                continue

            val_score = 0.0
            improved = False
            should_stop = False

            if is_main_process(self.dist_state):
                val_metrics = self.evaluate(
                    base_model,
                    valid_data,
                    val_message_passing_edges,
                    filtered_positive_pairs=val_filtered_positive_pairs,
                )
                val_score = self._selection_score(val_metrics)

                log0(
                    self.dist_state,
                    self._format_train_status(epoch, train_log, current_lr)
                    + " "
                    f"val_loss={val_metrics.get('Loss', 0.0):.4f} "
                    f"AUC={val_metrics.get('AUC', 0.0):.4f} "
                    f"AUPR={val_metrics.get('AUPR', 0.0):.4f} "
                    f"MRR={val_metrics.get('MRR', 0.0):.4f} "
                    f"MAP={val_metrics.get('MAP', 0.0):.4f} "
                    f"Hits@1={val_metrics.get('Hits@1', 0.0):.4f} "
                    f"Sel={val_score:.4f}",
                )

                if self.trial_reporter is not None:
                    self.trial_reporter.report_intermediate(
                        metrics=val_metrics,
                        epoch=epoch,
                        score=val_score,
                    )

                if val_score > best_val_score:
                    best_val_score = val_score
                    best_val_metrics = dict(val_metrics)
                    epochs_no_improve = 0
                    improved = True

                    save_dir = Path(model_save_path).parent
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(base_model.state_dict(), str(model_save_path))
                    log0(
                        self.dist_state,
                        f"  -> Improved; checkpoint saved to '{model_save_path}'",
                    )
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.cfg.early_stopping_patience:
                    should_stop = True
                    log0(
                        self.dist_state,
                        "Early stopping triggered: "
                        f"no improvement for {self.cfg.early_stopping_patience} eval steps.",
                    )

            val_score, improved, should_stop = broadcast_scalars(
                self.dist_state,
                val_score,
                improved,
                should_stop,
            )
            barrier(self.dist_state)
            if should_stop:
                break

        if Path(model_save_path).exists():
            log0(self.dist_state, f"Loading best checkpoint from '{model_save_path}' ...")
            base_model.load_state_dict(
                torch.load(str(model_save_path), map_location=self.dist_state.device)
            )
        else:
            log0(
                self.dist_state,
                "Warning: checkpoint not found; using current model weights for test evaluation.",
            )

        test_metrics = self.evaluate(
            base_model,
            test_data,
            test_message_passing_edges,
            filtered_positive_pairs=test_filtered_positive_pairs,
        )
        self.last_run_summary = {
            "best_val_score": float(best_val_score),
            "best_val_metrics": dict(best_val_metrics or {}),
            "test_metrics": dict(test_metrics),
        }
        if self.trial_reporter is not None:
            self.trial_reporter.report_final(
                summary=self.last_run_summary,
            )
        return test_metrics
