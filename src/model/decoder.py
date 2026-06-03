from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplExDecoder(nn.Module):
    """Relation-aware ComplEx edge decoder."""

    def __init__(
        self,
        hidden_channels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_channels % 2 != 0:
            raise ValueError("hidden_channels must be even for ComplEx decoding")

        self.hidden_channels = int(hidden_channels)
        self.real_dim = self.hidden_channels // 2

        self.src_norm = nn.LayerNorm(hidden_channels)
        self.dst_norm = nn.LayerNorm(hidden_channels)
        self.rel_norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def _coerce_relation_embed(
        self,
        relation_embed: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if relation_embed.ndim == 1:
            return relation_embed.unsqueeze(0).expand(batch_size, -1)
        if relation_embed.ndim == 2:
            if relation_embed.shape[0] == 1:
                return relation_embed.expand(batch_size, -1)
            if relation_embed.shape[0] == batch_size:
                return relation_embed
        raise ValueError(
            "relation_embed must have shape [H], [1, H], or [B, H] for ComplEx decoding"
        )

    def _split_complex(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x[..., : self.real_dim], x[..., self.real_dim :]

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1, eps=1e-8)

    def score_edges(
        self,
        src_embed: torch.Tensor,
        dst_embed: torch.Tensor,
        relation_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if relation_embed is None:
            raise ValueError("ComplEx decoding requires relation_embed")
        if src_embed.shape != dst_embed.shape:
            raise ValueError("src_embed and dst_embed must share the same shape")

        batch_size = int(src_embed.shape[0])
        rel = self._coerce_relation_embed(relation_embed, batch_size=batch_size)

        src = self._normalize(self.src_norm(self.dropout(src_embed)))
        dst = self._normalize(self.dst_norm(self.dropout(dst_embed)))
        rel = self._normalize(self.rel_norm(self.dropout(rel)))

        src_re, src_im = self._split_complex(src)
        rel_re, rel_im = self._split_complex(rel)
        dst_re, dst_im = self._split_complex(dst)

        return (
            src_re * rel_re * dst_re
            + src_im * rel_re * dst_im
            + src_re * rel_im * dst_im
            - src_im * rel_im * dst_re
        ).sum(dim=-1)


class BilinearDecoder(nn.Module):
    """Simple bilinear edge decoder."""

    def __init__(
        self,
        hidden_channels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.src_norm = nn.LayerNorm(hidden_channels)
        self.dst_norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.scorer = nn.Bilinear(hidden_channels, hidden_channels, 1)

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1, eps=1e-8)

    def score_edges(
        self,
        src_embed: torch.Tensor,
        dst_embed: torch.Tensor,
        relation_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _ = relation_embed
        if src_embed.shape != dst_embed.shape:
            raise ValueError("src_embed and dst_embed must share the same shape")
        src = self._normalize(self.src_norm(self.dropout(src_embed)))
        dst = self._normalize(self.dst_norm(self.dropout(dst_embed)))
        return self.scorer(src, dst).squeeze(-1)



class PairMLPDecoder(nn.Module):
    """Relation-aware pairwise MLP decoder for binary edge classification."""

    def __init__(
        self,
        hidden_channels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)

        self.src_norm = nn.LayerNorm(hidden_channels)
        self.dst_norm = nn.LayerNorm(hidden_channels)
        self.rel_norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)

        pair_input_dim = hidden_channels * 5
        hidden_dim = max(hidden_channels, hidden_channels // 2)
        self.mlp = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _coerce_relation_embed(
        self,
        relation_embed: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if relation_embed.ndim == 1:
            return relation_embed.unsqueeze(0).expand(batch_size, -1)
        if relation_embed.ndim == 2:
            if relation_embed.shape[0] == 1:
                return relation_embed.expand(batch_size, -1)
            if relation_embed.shape[0] == batch_size:
                return relation_embed
        raise ValueError(
            "relation_embed must have shape [H], [1, H], or [B, H] for PairMLP decoding"
        )

    def score_edges(
        self,
        src_embed: torch.Tensor,
        dst_embed: torch.Tensor,
        relation_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if relation_embed is None:
            raise ValueError("PairMLP decoding requires relation_embed")
        if src_embed.shape != dst_embed.shape:
            raise ValueError("src_embed and dst_embed must share the same shape")

        batch_size = int(src_embed.shape[0])
        rel = self._coerce_relation_embed(relation_embed, batch_size=batch_size)

        src = self.src_norm(self.dropout(src_embed))
        dst = self.dst_norm(self.dropout(dst_embed))
        rel = self.rel_norm(self.dropout(rel))

        pair_features = torch.cat(
            [
                src,
                dst,
                src * dst,
                torch.abs(src - dst),
                rel,
            ],
            dim=-1,
        )
        return self.mlp(pair_features).squeeze(-1)
