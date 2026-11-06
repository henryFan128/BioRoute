from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix

from .hard_sampling import HardSamplingMixin
from .random_sampling import RandomSamplingMixin

logger = logging.getLogger(__name__)


class _NegativeSamplerCore(RandomSamplingMixin, HardSamplingMixin):
    """Vectorized negative sampler backed by scipy CSR matrices.

    Supports source-fixed sampling over the supervision matrix:
      (chemical, treats, ?disease)

    Attributes:
        num_src (int): Number of source (Chemical) nodes.
        num_dst (int): Number of destination (Disease) nodes.
        known_pos_csr: [N_src, N_dst] bool CSR — all known positive pairs.
        relay_csr: [N_src, N_dst] bool CSR — Chemical→Disease relay adjacency
            mined from one configured multi-hop path.
        deg_probs (np.ndarray): [N_dst] degree-proportional Disease sampling prior.
        hard_neg_ratio (float): Fraction of negatives that should be structural
            hard negatives when relay candidates are available.
        base_seed (int): Base RNG seed. Per-epoch seed = ``base_seed + epoch``.
    """

    def __init__(
        self,
        num_src: int,
        num_dst: int,
        known_pos_csr: csr_matrix,
        relay_csr: Optional[csr_matrix],
        deg_probs: np.ndarray,
        hard_neg_ratio: float = 0.33,
        base_seed: int = 42,
        hard_neg_reuse_cap: Optional[int] = None,
        hard_neg_degree_quantiles: int = 1,
        hard_neg_quantile_quota: str = "uniform",
        hard_neg_min_per_quantile: int = 0,
        hard_neg_quota_strict: bool = False,
        random_uniform_mix: float = 0.5,
    ) -> None:
        self.num_src = num_src
        self.num_dst = num_dst
        self.known_pos_csr = known_pos_csr.astype(bool).tocsr()
        self.relay_csr = (
            relay_csr.astype(bool).tocsr() if relay_csr is not None else None
        )
        self.hard_neg_ratio = hard_neg_ratio
        self.base_seed = base_seed

        self.hard_neg_reuse_cap = (
            int(hard_neg_reuse_cap)
            if hard_neg_reuse_cap is not None and int(hard_neg_reuse_cap) > 0
            else None
        )
        self.hard_neg_degree_quantiles = max(1, int(hard_neg_degree_quantiles))
        self.hard_neg_quantile_quota = str(hard_neg_quantile_quota)
        if self.hard_neg_quantile_quota not in {
            "uniform",
            "inverse_degree_mass",
            "proportional_degree_mass",
        }:
            raise ValueError(
                "hard_neg_quantile_quota must be one of: "
                "uniform, inverse_degree_mass, proportional_degree_mass"
            )
        self.hard_neg_min_per_quantile = max(0, int(hard_neg_min_per_quantile))
        self.hard_neg_quota_strict = bool(hard_neg_quota_strict)
        self.random_uniform_mix = min(max(float(random_uniform_mix), 0.0), 1.0)
        self.deg_probs = self._normalise_probs(deg_probs, self.num_dst)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build_from_edges(
        cls,
        sup_ei: torch.Tensor,
        num_src: int,
        num_dst: int,
        cp_ei: Optional[torch.Tensor] = None,
        pd_ei: Optional[torch.Tensor] = None,
        num_mid: int = 0,
        relay_path_eis: Optional[list[torch.Tensor]] = None,
        relay_path_shapes: Optional[list[tuple[int, int]]] = None,
        hard_neg_ratio: float = 0.33,
        hard_neg_reuse_cap: Optional[int] = None,
        hard_neg_degree_quantiles: int = 1,
        hard_neg_quantile_quota: str = "uniform",
        hard_neg_min_per_quantile: int = 0,
        hard_neg_quota_strict: bool = False,
        random_uniform_mix: float = 0.5,
        base_seed: int = 42,
        degree_matched_negatives: bool = False,
        degree_match_quantiles: int = 5,
        known_pos_ei: Optional[torch.Tensor] = None,
        degree_pos_ei: Optional[torch.Tensor] = None,
    ) -> "NegativeSampler":
        """Build a NegativeSampler from supervision and relay edge tensors.

        Args:
            sup_ei:       [2, E] positive edges used as the default train mask
                and degree prior.
            num_src:      Total Chemical nodes.
            num_dst:      Total Disease nodes.
            cp_ei:        [2, E_cp] Chemical→Protein edges for legacy 2-hop relay.
            pd_ei:        [2, E_pd] Protein→Disease edges for legacy 2-hop relay.
            num_mid:      Total Protein nodes for the legacy 2-hop path.
            relay_path_eis:
                Optional sequence of edge_index tensors defining one directed
                relay path from source Chemical nodes to destination Disease nodes.
                Supports 3-hop or longer paths, e.g. Chemical→Protein→Gene→Disease.
            relay_path_shapes:
                Shapes for each relay step matrix. Must align with relay_path_eis.
            hard_neg_ratio: Fraction of negatives that are hard (relay) ones.
            hard_neg_reuse_cap:
                Optional cap on how many times one (query, candidate) pair may
                be reused within one sampling call. ``None``/<=0 disables cap.
            hard_neg_degree_quantiles:
                Number of quantile buckets for hard candidates by candidate
                degree. ``1`` disables quantile balancing.
            hard_neg_quantile_quota:
                Quota strategy across quantiles: ``uniform``,
                ``inverse_degree_mass``, or ``proportional_degree_mass``.
            hard_neg_min_per_quantile:
                Minimum quota per quantile when quantile balancing is enabled.
            hard_neg_quota_strict:
                If True, do not backfill quota deficits from other quantiles;
                leave ``-1`` padding when a bucket is exhausted.
            degree_matched_negatives:
                Enable degree-quantile balancing for hard-negative candidates.
            degree_match_quantiles:
                Number of degree buckets used when degree matching is enabled.
            known_pos_ei:
                Optional explicit positive edge set used to build the exclusion
                mask. It defaults to ``sup_ei``.
            degree_pos_ei:
                Optional explicit positive edge set used for the destination
                degree prior. It defaults to ``known_pos_ei``.
            base_seed:    Base RNG seed.
        """
        mask_ei = known_pos_ei if known_pos_ei is not None else sup_ei
        prior_ei = degree_pos_ei if degree_pos_ei is not None else mask_ei
        mask_ei = cls._validate_edge_index(mask_ei, "known_pos_ei")
        prior_ei = cls._validate_edge_index(prior_ei, "degree_pos_ei")

        sup_rows = mask_ei[0].cpu().numpy().astype(np.int32)
        sup_cols = mask_ei[1].cpu().numpy().astype(np.int32)
        data_ones = np.ones(len(sup_rows), dtype=bool)
        known_pos_csr = csr_matrix(
            (data_ones, (sup_rows, sup_cols)), shape=(num_src, num_dst)
        )
        logger.info(
            f"known_pos_csr: {known_pos_csr.nnz} positive pairs "
            f"({num_src} src × {num_dst} dst)"
        )

        prior_cols = prior_ei[1].cpu().numpy().astype(np.int32)
        deg = np.bincount(prior_cols, minlength=num_dst).astype(np.float64)
        deg += 1.0
        deg_probs = deg / deg.sum()

        relay_csr: Optional[csr_matrix] = None
        if relay_path_eis is not None and len(relay_path_eis) > 0:
            if relay_path_shapes is None or len(relay_path_shapes) != len(
                relay_path_eis
            ):
                raise ValueError(
                    "relay_path_shapes must be provided and match relay_path_eis length"
                )
            if (
                relay_path_shapes[0][0] != num_src
                or relay_path_shapes[-1][1] != num_dst
            ):
                raise ValueError(
                    "Configured relay path must start from num_src and end at num_dst"
                )

            relay_product: Optional[csr_matrix] = None
            for edge_index, shape in zip(relay_path_eis, relay_path_shapes):
                step_rows = edge_index[0].cpu().numpy().astype(np.int32)
                step_cols = edge_index[1].cpu().numpy().astype(np.int32)
                step_csr = csr_matrix(
                    (np.ones(len(step_rows), dtype=np.float32), (step_rows, step_cols)),
                    shape=shape,
                )
                relay_product = (
                    step_csr if relay_product is None else (relay_product @ step_csr)
                )

            relay_raw = (
                relay_product.astype(bool) if relay_product is not None else None
            )
            if relay_raw is not None:
                relay_from_path = relay_raw - relay_raw.multiply(known_pos_csr)
                relay_from_path.eliminate_zeros()
                relay_csr = relay_from_path
                logger.info(
                    f"relay_csr ({len(relay_path_eis)}-hop Chemical→Disease): {relay_from_path.nnz} candidate pairs "
                    f"across {(relay_from_path.sum(axis=1) > 0).sum()} source chemicals"
                )
        elif cp_ei is not None and pd_ei is not None and num_mid > 0:
            cp_rows = cp_ei[0].cpu().numpy().astype(np.int32)
            cp_cols = cp_ei[1].cpu().numpy().astype(np.int32)
            csr_cp = csr_matrix(
                (np.ones(len(cp_rows), dtype=np.float32), (cp_rows, cp_cols)),
                shape=(num_src, num_mid),
            )

            pd_rows = pd_ei[0].cpu().numpy().astype(np.int32)
            pd_cols = pd_ei[1].cpu().numpy().astype(np.int32)
            csr_pd = csr_matrix(
                (np.ones(len(pd_rows), dtype=np.float32), (pd_rows, pd_cols)),
                shape=(num_mid, num_dst),
            )

            relay_raw = (csr_cp @ csr_pd).astype(bool)
            relay_2hop = relay_raw - relay_raw.multiply(known_pos_csr)
            relay_2hop.eliminate_zeros()
            relay_csr = relay_2hop
            logger.info(
                f"relay_csr (2-hop Chemical→Disease): {relay_2hop.nnz} candidate pairs "
                f"across {(relay_2hop.sum(axis=1) > 0).sum()} source chemicals"
            )
        else:
            logger.warning(
                "Relay edges not provided; hard negatives will be skipped "
                "(all negatives will be degree-proportional random)."
            )

        return cls(
            num_src=num_src,
            num_dst=num_dst,
            known_pos_csr=known_pos_csr,
            relay_csr=relay_csr,
            deg_probs=deg_probs,
            hard_neg_ratio=hard_neg_ratio,
            hard_neg_reuse_cap=hard_neg_reuse_cap,
            hard_neg_degree_quantiles=(
                max(1, int(degree_match_quantiles))
                if degree_matched_negatives
                else hard_neg_degree_quantiles
            ),
            hard_neg_quantile_quota=hard_neg_quantile_quota,
            hard_neg_min_per_quantile=hard_neg_min_per_quantile,
            hard_neg_quota_strict=hard_neg_quota_strict,
            random_uniform_mix=random_uniform_mix,
            base_seed=base_seed,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def restrict_to_positive_edges(self, pos_ei: torch.Tensor) -> None:
        """Rebuild the exclusion mask and degree prior from one split only."""
        pos_ei = self._validate_edge_index(pos_ei, "pos_ei")
        rows = pos_ei[0].numpy().astype(np.int32)
        cols = pos_ei[1].numpy().astype(np.int32)
        if np.any(rows < 0) or np.any(rows >= self.num_src):
            raise ValueError("pos_ei contains an invalid source id")
        if np.any(cols < 0) or np.any(cols >= self.num_dst):
            raise ValueError("pos_ei contains an invalid destination id")

        self.known_pos_csr = csr_matrix(
            (np.ones(rows.size, dtype=bool), (rows, cols)),
            shape=(self.num_src, self.num_dst),
        )
        deg = np.bincount(cols, minlength=self.num_dst).astype(np.float64)
        deg += 1.0
        self.deg_probs = self._normalise_probs(deg, self.num_dst)
        if self.relay_csr is not None:
            self.relay_csr = (
                self.relay_csr - self.relay_csr.multiply(self.known_pos_csr)
            ).astype(bool).tocsr()
            self.relay_csr.eliminate_zeros()

    @staticmethod
    def _validate_edge_index(edge_index: torch.Tensor, name: str) -> torch.Tensor:
        if not torch.is_tensor(edge_index):
            raise TypeError(f"{name} must be a torch.Tensor")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"{name} must have shape [2, E]")
        return edge_index.detach().cpu().long()

    def _coerce_allowed_dst(self, allowed_dst) -> np.ndarray:
        if allowed_dst is None:
            return np.arange(self.num_dst, dtype=np.int32)
        if torch.is_tensor(allowed_dst):
            values = allowed_dst.detach().cpu().numpy()
        else:
            values = np.asarray(allowed_dst)
        values = np.asarray(values, dtype=np.int64).reshape(-1)
        if values.size == 0:
            return np.array([], dtype=np.int32)
        if np.any(values < 0) or np.any(values >= self.num_dst):
            raise ValueError(
                "allowed_dst contains an id outside the destination node range"
            )
        return np.unique(values).astype(np.int32, copy=False)

    @staticmethod
    def _with_extra_known_pairs(
        known_pos_csr: csr_matrix,
        extra_pos_ei: Optional[torch.Tensor],
        num_src: int,
        num_dst: int,
    ) -> csr_matrix:
        if extra_pos_ei is None or extra_pos_ei.numel() == 0:
            return known_pos_csr
        extra_pos_ei = _NegativeSamplerCore._validate_edge_index(
            extra_pos_ei, "exclude_pos_ei"
        )
        rows = extra_pos_ei[0].numpy().astype(np.int32)
        cols = extra_pos_ei[1].numpy().astype(np.int32)
        if np.any(rows < 0) or np.any(rows >= num_src):
            raise ValueError("exclude_pos_ei contains an invalid source id")
        if np.any(cols < 0) or np.any(cols >= num_dst):
            raise ValueError("exclude_pos_ei contains an invalid destination id")
        extra = csr_matrix(
            (np.ones(rows.size, dtype=bool), (rows, cols)),
            shape=(num_src, num_dst),
        )
        return known_pos_csr.maximum(extra).astype(bool).tocsr()

    @staticmethod
    def _normalise_probs(probs: np.ndarray, expected_size: int) -> np.ndarray:
        arr = np.asarray(probs, dtype=np.float64).reshape(-1)
        if arr.size != expected_size:
            raise ValueError(
                f"Probability vector size mismatch: got {arr.size}, expected {expected_size}"
            )
        total = float(arr.sum())
        if total <= 0.0:
            arr = np.ones(expected_size, dtype=np.float64) / max(expected_size, 1)
        else:
            arr = arr / total
        return arr

    @staticmethod
    def _blend_sampling_probs(probs: np.ndarray, uniform_mix: float) -> np.ndarray:
        base = np.asarray(probs, dtype=np.float64).reshape(-1)
        if base.size == 0:
            return base

        mix = min(max(float(uniform_mix), 0.0), 1.0)
        uniform = np.full(base.shape, 1.0 / float(base.size), dtype=np.float64)
        if mix <= 0.0:
            total = float(base.sum())
            return uniform if total <= 0.0 else (base / total)

        blended = (1.0 - mix) * base + mix * uniform
        total = float(blended.sum())
        return uniform if total <= 0.0 else (blended / total)

    def _ensure_runtime_defaults(self) -> None:
        """Backfill fields for samplers loaded from older pickle versions."""
        self.known_pos_csr = self.known_pos_csr.astype(bool).tocsr()

        if not hasattr(self, "relay_csr"):
            self.relay_csr = None
        elif self.relay_csr is not None:
            self.relay_csr = self.relay_csr.astype(bool).tocsr()

        if not hasattr(self, "deg_probs") or self.deg_probs is None:
            deg = np.asarray(self.known_pos_csr.sum(axis=0)).ravel().astype(np.float64)
            deg += 1.0
            self.deg_probs = self._normalise_probs(deg, self.num_dst)
        else:
            self.deg_probs = self._normalise_probs(self.deg_probs, self.num_dst)

        if not hasattr(self, "hard_neg_reuse_cap"):
            self.hard_neg_reuse_cap = None
        elif self.hard_neg_reuse_cap is not None:
            self.hard_neg_reuse_cap = int(self.hard_neg_reuse_cap)
            if self.hard_neg_reuse_cap <= 0:
                self.hard_neg_reuse_cap = None

        if not hasattr(self, "hard_neg_degree_quantiles"):
            self.hard_neg_degree_quantiles = 1
        else:
            self.hard_neg_degree_quantiles = max(1, int(self.hard_neg_degree_quantiles))

        if not hasattr(self, "hard_neg_quantile_quota"):
            self.hard_neg_quantile_quota = "uniform"
        else:
            self.hard_neg_quantile_quota = str(self.hard_neg_quantile_quota)
            if self.hard_neg_quantile_quota not in {
                "uniform",
                "inverse_degree_mass",
                "proportional_degree_mass",
            }:
                self.hard_neg_quantile_quota = "uniform"

        if not hasattr(self, "hard_neg_min_per_quantile"):
            self.hard_neg_min_per_quantile = 0
        else:
            self.hard_neg_min_per_quantile = max(0, int(self.hard_neg_min_per_quantile))

        if not hasattr(self, "hard_neg_quota_strict"):
            self.hard_neg_quota_strict = False
        else:
            self.hard_neg_quota_strict = bool(self.hard_neg_quota_strict)

        if not hasattr(self, "random_uniform_mix"):
            self.random_uniform_mix = 0.5
        else:
            self.random_uniform_mix = min(max(float(self.random_uniform_mix), 0.0), 1.0)

    @staticmethod
    def _coerce_pos_ei(pos_ei: torch.Tensor) -> np.ndarray:
        pos_np = pos_ei.detach().cpu().numpy().astype(np.int32)
        if pos_np.ndim != 2 or pos_np.shape[0] != 2:
            raise ValueError("pos_ei must have shape [2, P]")
        return pos_np

    @staticmethod
    def _flatten_src_candidate_matrix(
        src_ids: np.ndarray,
        neg_matrix: torch.Tensor,
        valid_mask: torch.Tensor,
        deduplicate_pairs: bool = True,
    ) -> torch.Tensor:
        src_ids = np.asarray(src_ids, dtype=np.int64)
        neg_np = neg_matrix.detach().cpu().numpy().astype(np.int64, copy=False)
        mask_np = valid_mask.detach().cpu().numpy().astype(bool, copy=False)

        head_parts = []
        tail_parts = []
        seen_pairs: set[tuple[int, int]] = set()
        for i, src in enumerate(src_ids):
            valid = RandomSamplingMixin._unique_preserve_order(
                neg_np[i, mask_np[i]]
            )
            if deduplicate_pairs:
                valid = np.asarray(
                    [
                        int(candidate)
                        for candidate in valid.tolist()
                        if (int(src), int(candidate)) not in seen_pairs
                    ],
                    dtype=np.int64,
                )
            if valid.size == 0:
                continue
            if deduplicate_pairs:
                seen_pairs.update((int(src), int(candidate)) for candidate in valid.tolist())
            head_parts.append(np.full(valid.size, src, dtype=np.int64))
            tail_parts.append(valid.astype(np.int64, copy=False))

        if head_parts:
            heads = np.concatenate(head_parts)
            tails = np.concatenate(tail_parts)
        else:
            heads = np.array([], dtype=np.int64)
            tails = np.array([], dtype=np.int64)

        return torch.tensor(np.stack([heads, tails]), dtype=torch.long)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_epoch(
        self,
        pos_ei: torch.Tensor,
        n_neg_per_pos: int = 30,
        epoch: int = 0,
        *,
        allowed_dst=None,
        strict: bool = False,
        insufficient_negatives_policy: str = "skip",
        with_replacement: bool = False,
        exclude_pos_ei: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Source-fixed sampler returning flattened negative edges.

        ``allowed_dst`` restricts the destination candidate domain. When
        ``with_replacement`` is false, each ``(source, destination)`` pair is
        emitted at most once per call, including repeated source rows in
        ``pos_ei``. ``strict`` or ``insufficient_negatives_policy='error'``
        requires exactly ``n_neg_per_pos`` negatives for every positive row.
        """
        self._ensure_runtime_defaults()
        pos_np = self._coerce_pos_ei(pos_ei)
        n_neg_per_pos = int(n_neg_per_pos)
        if n_neg_per_pos <= 0 or pos_np.shape[1] == 0:
            empty = torch.zeros((2, 0), dtype=torch.long)
            return empty, torch.zeros(0, dtype=torch.float)

        policy = str(insufficient_negatives_policy).lower()
        if policy not in {"skip", "error"}:
            raise ValueError(
                "insufficient_negatives_policy must be either 'skip' or 'error'"
            )
        if strict:
            policy = "error"

        if np.any(pos_np[0] < 0) or np.any(pos_np[0] >= self.num_src):
            raise ValueError("pos_ei contains an invalid source id")
        if np.any(pos_np[1] < 0) or np.any(pos_np[1] >= self.num_dst):
            raise ValueError("pos_ei contains an invalid destination id")

        query_ids = pos_np[0]
        allowed_candidates = self._coerce_allowed_dst(allowed_dst)
        if allowed_candidates.size == 0:
            if policy == "error":
                raise ValueError("No allowed destination candidates are available")
            empty = torch.zeros((2, 0), dtype=torch.long)
            return empty, torch.zeros(0, dtype=torch.float)

        extra_parts = [pos_ei.detach().cpu().long()]
        if exclude_pos_ei is not None and exclude_pos_ei.numel() > 0:
            extra_parts.append(exclude_pos_ei.detach().cpu().long())
        known_pos_csr = self._with_extra_known_pairs(
            self.known_pos_csr,
            torch.cat(extra_parts, dim=1),
            self.num_src,
            self.num_dst,
        )

        n_hard = (
            min(
                n_neg_per_pos,
                int(n_neg_per_pos * min(max(float(self.hard_neg_ratio), 0.0), 1.0)),
            )
            if self.relay_csr is not None
            else 0
        )
        n_rand = n_neg_per_pos - n_hard
        rng = np.random.default_rng(self.base_seed + epoch)

        hard_matrix = self._sample_hard_matrix(
            query_ids=query_ids,
            known_pos_csr=known_pos_csr,
            relay_csr=self.relay_csr,
            n_hard=n_hard,
            epoch=epoch,
            allowed_candidates=allowed_candidates,
        )
        rand_matrix = self._sample_random_matrix(
            query_ids=query_ids,
            known_pos_csr=known_pos_csr,
            num_candidates=self.num_dst,
            probs=self.deg_probs,
            n_rand=n_rand,
            rng=rng,
            allowed_candidates=allowed_candidates,
        )

        neg_np = (
            np.concatenate([hard_matrix, rand_matrix], axis=1)
            if n_hard > 0
            else rand_matrix
        )
        effective_probs = self._blend_sampling_probs(
            self.deg_probs,
            uniform_mix=self.random_uniform_mix,
        )
        known_by_query = {
            int(query_id): set(known_pos_csr.getrow(int(query_id)).indices.tolist())
            for query_id in np.unique(query_ids)
        }
        used_by_query: dict[int, set[int]] = {}
        allowed_set = set(int(candidate) for candidate in allowed_candidates.tolist())
        neg_src_parts: list[np.ndarray] = []
        neg_dst_parts: list[np.ndarray] = []

        for row_idx, query_id_raw in enumerate(query_ids.tolist()):
            query_id = int(query_id_raw)
            known = known_by_query[query_id]
            used = used_by_query.setdefault(query_id, set())
            selected: list[int] = []
            selected_set: set[int] = set()

            for candidate in neg_np[row_idx].tolist():
                candidate = int(candidate)
                if candidate < 0 or candidate not in allowed_set:
                    continue
                if candidate in known or candidate in selected_set:
                    continue
                if not with_replacement and candidate in used:
                    continue
                selected.append(candidate)
                selected_set.add(candidate)
                if len(selected) >= n_neg_per_pos:
                    break

            base_available = np.asarray(
                [
                    int(candidate)
                    for candidate in allowed_candidates.tolist()
                    if int(candidate) not in known
                    and int(candidate) not in selected_set
                    and (with_replacement or int(candidate) not in used)
                ],
                dtype=np.int32,
            )
            remaining = n_neg_per_pos - len(selected)
            if remaining > 0 and base_available.size > 0:
                available_probs = effective_probs[base_available]
                prob_sum = float(available_probs.sum())
                if prob_sum <= 0.0:
                    available_probs = np.full(
                        base_available.shape,
                        1.0 / float(base_available.size),
                        dtype=np.float64,
                    )
                else:
                    available_probs = available_probs / prob_sum
                take = min(remaining, int(base_available.size))
                sampled = rng.choice(
                    base_available,
                    size=take,
                    replace=False,
                    p=available_probs,
                )
                selected.extend(int(candidate) for candidate in sampled.tolist())
                selected_set.update(int(candidate) for candidate in sampled.tolist())

            remaining = n_neg_per_pos - len(selected)
            if remaining > 0 and with_replacement:
                replacement_pool = np.asarray(
                    [
                        int(candidate)
                        for candidate in allowed_candidates.tolist()
                        if int(candidate) not in known
                    ],
                    dtype=np.int32,
                )
                if replacement_pool.size > 0:
                    replacement_probs = effective_probs[replacement_pool]
                    prob_sum = float(replacement_probs.sum())
                    if prob_sum <= 0.0:
                        replacement_probs = np.full(
                            replacement_pool.shape,
                            1.0 / float(replacement_pool.size),
                            dtype=np.float64,
                        )
                    else:
                        replacement_probs = replacement_probs / prob_sum
                    sampled = rng.choice(
                        replacement_pool,
                        size=remaining,
                        replace=True,
                        p=replacement_probs,
                    )
                    selected.extend(int(candidate) for candidate in sampled.tolist())

            if len(selected) < n_neg_per_pos:
                if policy == "error":
                    available_count = sum(
                        int(candidate) not in known and int(candidate) not in used
                        for candidate in allowed_candidates.tolist()
                    )
                    raise ValueError(
                        "insufficient negatives: "
                        f"row={row_idx}, query_id={query_id}, "
                        f"needed={n_neg_per_pos}, selected={len(selected)}, "
                        f"available={available_count}, "
                        f"allowed_dst={allowed_candidates.size}"
                    )

            if selected:
                neg_src_parts.append(np.full(len(selected), query_id, dtype=np.int64))
                neg_dst_parts.append(np.asarray(selected, dtype=np.int64))
                if not with_replacement:
                    used.update(selected)

        if neg_src_parts:
            neg_ei = torch.tensor(
                np.stack(
                    [np.concatenate(neg_src_parts), np.concatenate(neg_dst_parts)]
                ),
                dtype=torch.long,
            )
        else:
            neg_ei = torch.zeros((2, 0), dtype=torch.long)
        neg_labels = torch.zeros(neg_ei.shape[1], dtype=torch.float)
        return neg_ei, neg_labels

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Pickle the sampler to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"NegativeSampler saved → {path}")

    @classmethod
    def load(cls, path: str) -> "NegativeSampler":
        """Load a pickled sampler from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"NegativeSampler loaded ← {path}")
        return obj

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return a dict of key statistics for logging."""
        self._ensure_runtime_defaults()
        relay_nnz = self.relay_csr.nnz if self.relay_csr is not None else 0
        relay_srcs = (
            int((self.relay_csr.sum(axis=1) > 0).sum())
            if self.relay_csr is not None
            else 0
        )
        return {
            "num_src": self.num_src,
            "num_dst": self.num_dst,
            "known_pos_pairs": self.known_pos_csr.nnz,
            "relay_candidate_pairs": relay_nnz,
            "relay_active_sources": relay_srcs,
            "hard_neg_ratio": self.hard_neg_ratio,
            "hard_neg_reuse_cap": self.hard_neg_reuse_cap,
            "hard_neg_degree_quantiles": self.hard_neg_degree_quantiles,
            "hard_neg_quantile_quota": self.hard_neg_quantile_quota,
            "hard_neg_min_per_quantile": self.hard_neg_min_per_quantile,
            "hard_neg_quota_strict": self.hard_neg_quota_strict,
            "random_uniform_mix": self.random_uniform_mix,
        }
