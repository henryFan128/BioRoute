"""
scripts/negative_sampler.py

Vectorized negative sampler for supervision edges in the BioRoute drug-repurposing pipeline.

Usage (offline in build_graph.py):
    sampler = NegativeSampler.build_from_edges(
        sup_ei=all_sup_ei,
        num_src=num_chem,
        num_dst=num_dis,
        relay_path_eis=[chem_prot_ei, prot_gene_ei, gene_dis_ei],
        relay_path_shapes=[
            (num_chem, num_prot),
            (num_prot, num_gene),
            (num_gene, num_dis),
        ],
        hard_neg_ratio=0.33,
        base_seed=42,
    )
    sampler.save("data/split/neg_sampler.pkl")

Usage (online):
    neg_ei, neg_lbl = sampler.sample_epoch(train_pos_ei, n_neg_per_pos=30, epoch=0)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


class NegativeSampler:
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
    ) -> "NegativeSampler":
        """Build a NegativeSampler from supervision and relay edge tensors.

        Args:
            sup_ei:       [2, E] ALL supervision positive edges (train+val+test).
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
            base_seed:    Base RNG seed.
        """
        sup_rows = sup_ei[0].cpu().numpy().astype(np.int32)
        sup_cols = sup_ei[1].cpu().numpy().astype(np.int32)
        data_ones = np.ones(len(sup_rows), dtype=bool)
        known_pos_csr = csr_matrix(
            (data_ones, (sup_rows, sup_cols)), shape=(num_src, num_dst)
        )
        logger.info(
            f"known_pos_csr: {known_pos_csr.nnz} positive pairs "
            f"({num_src} src × {num_dst} dst)"
        )

        deg = np.bincount(sup_cols, minlength=num_dst).astype(np.float64)
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
            hard_neg_degree_quantiles=hard_neg_degree_quantiles,
            hard_neg_quantile_quota=hard_neg_quantile_quota,
            hard_neg_min_per_quantile=hard_neg_min_per_quantile,
            hard_neg_quota_strict=hard_neg_quota_strict,
            random_uniform_mix=random_uniform_mix,
            base_seed=base_seed,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
    def _refill_unique_candidates(
        query_id: int,
        existing: np.ndarray,
        known_pos_csr: csr_matrix,
        num_candidates: int,
        rng: np.random.Generator,
        target_size: int,
    ) -> np.ndarray:
        if existing.size >= target_size:
            return existing[:target_size]
        if num_candidates <= 0:
            return existing

        fallback = rng.permutation(num_candidates).astype(np.int32, copy=False)
        row = np.full(num_candidates, int(query_id), dtype=np.int32)
        is_known = np.asarray(known_pos_csr[row, fallback]).ravel().astype(bool)
        fallback = fallback[~is_known]
        if existing.size > 0:
            fallback = fallback[~np.isin(fallback, existing, assume_unique=False)]
        if fallback.size == 0:
            return existing
        need = max(0, target_size - existing.size)
        return np.concatenate([existing, fallback[:need]]).astype(np.int32, copy=False)

    def _sample_random_matrix(
        self,
        query_ids: np.ndarray,
        known_pos_csr: csr_matrix,
        num_candidates: int,
        probs: np.ndarray,
        n_rand: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        P = len(query_ids)
        if n_rand <= 0:
            return np.full((P, 0), -1, dtype=np.int32)

        oversample = 3
        n_cands = max(n_rand * oversample, n_rand)
        effective_probs = self._blend_sampling_probs(
            probs,
            uniform_mix=self.random_uniform_mix,
        )
        flat_cands = rng.choice(
            num_candidates,
            size=P * n_cands,
            replace=True,
            p=effective_probs,
        )
        cands_2d = flat_cands.reshape(P, n_cands).astype(np.int32, copy=False)

        row_idx = np.repeat(query_ids.astype(np.int32, copy=False), n_cands)
        col_idx = cands_2d.ravel()
        is_known = np.asarray(known_pos_csr[row_idx, col_idx]).ravel().astype(bool)
        valid_2d = ~is_known.reshape(P, n_cands)

        rand_negs = np.full((P, n_rand), -1, dtype=np.int32)
        for i in range(P):
            valid = np.unique(cands_2d[i, valid_2d[i]]).astype(np.int32, copy=False)
            valid = self._refill_unique_candidates(
                query_id=int(query_ids[i]),
                existing=valid,
                known_pos_csr=known_pos_csr,
                num_candidates=num_candidates,
                rng=rng,
                target_size=n_rand,
            )
            n = min(n_rand, len(valid))
            rand_negs[i, :n] = valid[:n]
        return rand_negs

    @staticmethod
    def _unique_preserve_order(values: np.ndarray) -> np.ndarray:
        if values.size <= 1:
            return values.astype(np.int32, copy=False)
        _, first_idx = np.unique(values, return_index=True)
        return values[np.sort(first_idx)].astype(np.int32, copy=False)

    def _candidate_degree_vector(self) -> np.ndarray:
        self._ensure_runtime_defaults()
        deg = np.asarray(self.known_pos_csr.sum(axis=0)).ravel().astype(np.float64)
        deg += 1.0
        return deg

    def _split_candidates_by_quantile(
        self,
        candidates: np.ndarray,
        candidate_degree: np.ndarray,
        n_quantiles: int,
    ) -> list[np.ndarray]:
        if candidates.size == 0 or n_quantiles <= 1:
            return [candidates.astype(np.int32, copy=False)]

        n_quantiles = max(1, int(n_quantiles))
        deg_vals = candidate_degree[candidates]
        if np.allclose(deg_vals, deg_vals[0]):
            buckets = [np.array([], dtype=np.int32) for _ in range(n_quantiles)]
            buckets[0] = candidates.astype(np.int32, copy=False)
            return buckets

        quantile_edges = np.quantile(deg_vals, np.linspace(0.0, 1.0, n_quantiles + 1))
        buckets: list[np.ndarray] = []
        assigned = np.zeros(candidates.size, dtype=bool)
        for q in range(n_quantiles):
            lo = quantile_edges[q]
            hi = quantile_edges[q + 1]
            if q == n_quantiles - 1:
                mask = (deg_vals >= lo) & (deg_vals <= hi)
            else:
                mask = (deg_vals >= lo) & (deg_vals < hi)
            buckets.append(candidates[mask].astype(np.int32, copy=False))
            assigned |= mask

        if not np.all(assigned):
            remainder = candidates[~assigned].astype(np.int32, copy=False)
            if buckets:
                buckets[-1] = np.concatenate([buckets[-1], remainder]).astype(
                    np.int32, copy=False
                )
            else:
                buckets = [remainder]

        if len(buckets) < n_quantiles:
            buckets.extend(
                [
                    np.array([], dtype=np.int32)
                    for _ in range(n_quantiles - len(buckets))
                ]
            )
        return buckets

    def _allocate_quantile_quota(
        self,
        buckets: list[np.ndarray],
        candidate_degree: np.ndarray,
        n_hard: int,
    ) -> np.ndarray:
        n_quantiles = len(buckets)
        quotas = np.zeros(n_quantiles, dtype=np.int32)
        if n_hard <= 0 or n_quantiles == 0:
            return quotas

        active = np.array(
            [i for i, b in enumerate(buckets) if b.size > 0], dtype=np.int32
        )
        if active.size == 0:
            return quotas

        reserve = 0
        min_per_q = int(self.hard_neg_min_per_quantile)
        if min_per_q > 0:
            reserve_target = min_per_q * int(active.size)
            reserve = min(n_hard, reserve_target)
            for j in range(reserve):
                quotas[int(active[j % active.size])] += 1

        remaining = n_hard - int(quotas.sum())
        if remaining <= 0:
            return quotas

        masses = np.zeros(active.size, dtype=np.float64)
        for idx, q in enumerate(active):
            masses[idx] = float(candidate_degree[buckets[int(q)]].sum())

        if self.hard_neg_quantile_quota == "uniform":
            weights = np.ones(active.size, dtype=np.float64)
        elif self.hard_neg_quantile_quota == "inverse_degree_mass":
            safe = np.where(masses > 0.0, masses, 1.0)
            weights = 1.0 / safe
        else:  # proportional_degree_mass
            weights = np.where(masses > 0.0, masses, 1.0)

        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            weights = np.ones(active.size, dtype=np.float64) / float(active.size)
        else:
            weights = weights / weight_sum

        expected = weights * float(remaining)
        add = np.floor(expected).astype(np.int32)
        quotas[active] += add
        rem = remaining - int(add.sum())
        if rem > 0:
            frac = expected - add.astype(np.float64)
            order = np.argsort(-frac)
            for k in range(rem):
                quotas[int(active[int(order[k % order.size])])] += 1

        return quotas

    def _sample_hard_matrix(
        self,
        query_ids: np.ndarray,
        known_pos_csr: csr_matrix,
        relay_csr: Optional[csr_matrix],
        n_hard: int,
        epoch: int,
    ) -> np.ndarray:
        P = len(query_ids)
        hard_negs = np.full((P, n_hard), -1, dtype=np.int32)
        if n_hard <= 0 or relay_csr is None:
            return hard_negs

        # Backward-compatible fast path: preserve legacy sampling behaviour.
        if self.hard_neg_reuse_cap is None and self.hard_neg_degree_quantiles <= 1:
            unique_queries = np.unique(query_ids.astype(np.int32, copy=False))
            query_to_candidates: dict[int, np.ndarray] = {}
            for query_id in unique_queries:
                row = relay_csr.getrow(int(query_id))
                cols = row.indices.copy().astype(np.int32, copy=False)
                if len(cols) == 0:
                    query_to_candidates[int(query_id)] = np.array([], dtype=np.int32)
                    continue
                rng_q = np.random.default_rng(
                    self.base_seed + epoch * 999983 + int(query_id)
                )
                rng_q.shuffle(cols)
                query_to_candidates[int(query_id)] = cols

            for i in range(P):
                query_id = int(query_ids[i])
                cols = query_to_candidates[query_id]
                if len(cols) == 0:
                    continue
                row = np.full(len(cols), query_id, dtype=np.int32)
                is_known = np.asarray(known_pos_csr[row, cols]).ravel().astype(bool)
                valid = np.unique(cols[~is_known]).astype(np.int32, copy=False)
                n = min(n_hard, len(valid))
                hard_negs[i, :n] = valid[:n]
            return hard_negs

        candidate_degree = self._candidate_degree_vector()
        unique_queries = np.unique(query_ids.astype(np.int32, copy=False))
        query_to_candidates: dict[int, np.ndarray] = {}
        for query_id in unique_queries:
            row = relay_csr.getrow(int(query_id))
            cols = row.indices.copy().astype(np.int32, copy=False)
            if len(cols) == 0:
                query_to_candidates[int(query_id)] = np.array([], dtype=np.int32)
                continue
            rng_q = np.random.default_rng(
                self.base_seed + epoch * 999983 + int(query_id)
            )
            rng_q.shuffle(cols)
            row_idx = np.full(len(cols), int(query_id), dtype=np.int32)
            is_known = np.asarray(known_pos_csr[row_idx, cols]).ravel().astype(bool)
            cols = self._unique_preserve_order(cols[~is_known])
            query_to_candidates[int(query_id)] = cols

        reuse_counter: dict[tuple[int, int], int] = {}

        for i in range(P):
            query_id = int(query_ids[i])
            candidates = query_to_candidates[query_id]
            if candidates.size == 0:
                continue

            if self.hard_neg_reuse_cap is not None:
                available = np.array(
                    [
                        c
                        for c in candidates
                        if reuse_counter.get((query_id, int(c)), 0)
                        < self.hard_neg_reuse_cap
                    ],
                    dtype=np.int32,
                )
            else:
                available = candidates

            if available.size == 0:
                continue

            n_quantiles = max(1, int(self.hard_neg_degree_quantiles))
            if n_quantiles <= 1:
                take = available[:n_hard]
                if take.size > 0:
                    hard_negs[i, : take.size] = take
                    if self.hard_neg_reuse_cap is not None:
                        for cand in take:
                            key = (query_id, int(cand))
                            reuse_counter[key] = reuse_counter.get(key, 0) + 1
                continue

            buckets = self._split_candidates_by_quantile(
                candidates=available,
                candidate_degree=candidate_degree,
                n_quantiles=n_quantiles,
            )
            quotas = self._allocate_quantile_quota(
                buckets=buckets,
                candidate_degree=candidate_degree,
                n_hard=n_hard,
            )

            selected_parts: list[np.ndarray] = []
            for q, bucket in enumerate(buckets):
                if bucket.size == 0 or quotas[q] <= 0:
                    continue
                take_n = min(int(quotas[q]), int(bucket.size))
                chosen = bucket[:take_n]
                selected_parts.append(chosen)
                if self.hard_neg_reuse_cap is not None:
                    for cand in chosen:
                        key = (query_id, int(cand))
                        reuse_counter[key] = reuse_counter.get(key, 0) + 1

            selected = (
                np.concatenate(selected_parts).astype(np.int32, copy=False)
                if selected_parts
                else np.array([], dtype=np.int32)
            )

            remain = n_hard - int(selected.size)
            if remain > 0 and not self.hard_neg_quota_strict:
                overflow = np.concatenate(buckets).astype(np.int32, copy=False)
                if selected.size > 0:
                    overflow = overflow[
                        ~np.isin(overflow, selected, assume_unique=False)
                    ].astype(np.int32, copy=False)
                if overflow.size > 0:
                    extra = overflow[:remain]
                    selected = np.concatenate([selected, extra]).astype(
                        np.int32, copy=False
                    )
                    if self.hard_neg_reuse_cap is not None:
                        for cand in extra:
                            key = (query_id, int(cand))
                            reuse_counter[key] = reuse_counter.get(key, 0) + 1

            fill_n = min(n_hard, int(selected.size))
            if fill_n > 0:
                hard_negs[i, :fill_n] = selected[:fill_n]

        return hard_negs

    @staticmethod
    def _flatten_src_candidate_matrix(
        src_ids: np.ndarray,
        neg_matrix: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        src_ids = np.asarray(src_ids, dtype=np.int64)
        neg_np = neg_matrix.detach().cpu().numpy().astype(np.int64, copy=False)
        mask_np = valid_mask.detach().cpu().numpy().astype(bool, copy=False)

        head_parts = []
        tail_parts = []
        for i, src in enumerate(src_ids):
            valid = np.unique(neg_np[i, mask_np[i]])
            if valid.size == 0:
                continue
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Source-fixed sampler returning flattened negative edges."""
        self._ensure_runtime_defaults()
        pos_np = self._coerce_pos_ei(pos_ei)
        query_ids = pos_np[0]

        n_hard = (
            int(n_neg_per_pos * self.hard_neg_ratio) if self.relay_csr is not None else 0
        )
        n_rand = n_neg_per_pos - n_hard
        rng = np.random.default_rng(self.base_seed + epoch)

        hard_matrix = self._sample_hard_matrix(
            query_ids=query_ids,
            known_pos_csr=self.known_pos_csr,
            relay_csr=self.relay_csr,
            n_hard=n_hard,
            epoch=epoch,
        )
        rand_matrix = self._sample_random_matrix(
            query_ids=query_ids,
            known_pos_csr=self.known_pos_csr,
            num_candidates=self.num_dst,
            probs=self.deg_probs,
            n_rand=n_rand,
            rng=rng,
        )

        neg_np = (
            np.concatenate([hard_matrix, rand_matrix], axis=1)
            if n_hard > 0
            else rand_matrix
        )
        valid_mask = neg_np >= 0
        neg_matrix = torch.tensor(neg_np, dtype=torch.long)
        valid_mask_tensor = torch.tensor(valid_mask, dtype=torch.bool)

        neg_ei = self._flatten_src_candidate_matrix(
            src_ids=query_ids,
            neg_matrix=neg_matrix,
            valid_mask=valid_mask_tensor,
        )
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
