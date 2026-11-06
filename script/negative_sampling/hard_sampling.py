from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix


class HardSamplingMixin:
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
        allowed_candidates: Optional[np.ndarray] = None,
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
                if allowed_candidates is not None:
                    cols = cols[
                        np.isin(cols, allowed_candidates, assume_unique=False)
                    ]
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
                valid = self._unique_preserve_order(cols[~is_known])
                n = min(n_hard, len(valid))
                hard_negs[i, :n] = valid[:n]
            return hard_negs

        candidate_degree = self._candidate_degree_vector()
        unique_queries = np.unique(query_ids.astype(np.int32, copy=False))
        query_to_candidates: dict[int, np.ndarray] = {}
        for query_id in unique_queries:
            row = relay_csr.getrow(int(query_id))
            cols = row.indices.copy().astype(np.int32, copy=False)
            if allowed_candidates is not None:
                cols = cols[
                    np.isin(cols, allowed_candidates, assume_unique=False)
                ]
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
