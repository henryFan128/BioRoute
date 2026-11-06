from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy.sparse import csr_matrix


class RandomSamplingMixin:
    @staticmethod
    def _refill_unique_candidates(
        query_id: int,
        existing: np.ndarray,
        known_pos_csr: csr_matrix,
        num_candidates: int,
        rng: np.random.Generator,
        target_size: int,
        allowed_candidates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if existing.size >= target_size:
            return existing[:target_size]
        if allowed_candidates is None:
            candidate_pool = np.arange(num_candidates, dtype=np.int32)
        else:
            candidate_pool = np.asarray(allowed_candidates, dtype=np.int32).reshape(-1)
        if candidate_pool.size <= 0:
            return existing

        fallback = rng.permutation(candidate_pool)
        row = np.full(fallback.size, int(query_id), dtype=np.int32)
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
        allowed_candidates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        P = len(query_ids)
        if n_rand <= 0:
            return np.full((P, 0), -1, dtype=np.int32)

        if allowed_candidates is None:
            candidate_pool = np.arange(num_candidates, dtype=np.int32)
        else:
            candidate_pool = np.asarray(allowed_candidates, dtype=np.int32).reshape(-1)
        if candidate_pool.size == 0:
            return np.full((P, n_rand), -1, dtype=np.int32)

        oversample = 3
        n_cands = max(n_rand * oversample, n_rand)
        effective_probs = self._blend_sampling_probs(
            probs,
            uniform_mix=self.random_uniform_mix,
        )
        pool_probs = effective_probs[candidate_pool]
        pool_prob_sum = float(pool_probs.sum())
        if pool_prob_sum <= 0.0:
            pool_probs = np.full(
                candidate_pool.shape,
                1.0 / float(candidate_pool.size),
                dtype=np.float64,
            )
        else:
            pool_probs = pool_probs / pool_prob_sum
        flat_cands = rng.choice(
            candidate_pool,
            size=P * n_cands,
            replace=True,
            p=pool_probs,
        )
        cands_2d = flat_cands.reshape(P, n_cands).astype(np.int32, copy=False)

        row_idx = np.repeat(query_ids.astype(np.int32, copy=False), n_cands)
        col_idx = cands_2d.ravel()
        is_known = np.asarray(known_pos_csr[row_idx, col_idx]).ravel().astype(bool)
        valid_2d = ~is_known.reshape(P, n_cands)

        rand_negs = np.full((P, n_rand), -1, dtype=np.int32)
        for i in range(P):
            valid = self._unique_preserve_order(cands_2d[i, valid_2d[i]])
            valid = self._refill_unique_candidates(
                query_id=int(query_ids[i]),
                existing=valid,
                known_pos_csr=known_pos_csr,
                num_candidates=num_candidates,
                rng=rng,
                target_size=n_rand,
                allowed_candidates=candidate_pool,
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
