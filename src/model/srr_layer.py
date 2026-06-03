from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
from torch_geometric.utils import softmax

from .relation_encoder import RelationEncoderState, RelationTriple


@dataclass
class SRRBlockOutput:
    """Outputs of one dual-stream masked message-passing layer."""

    x_dict: Dict[str, torch.Tensor]
    intra_messages: Dict[str, torch.Tensor]
    inter_messages: Dict[str, torch.Tensor]
    relation_state: RelationEncoderState


class SRRLayer(nn.Module):
    """Dual-Stream Masked Message Passing block.

    Stage 1: intra-domain purity
        Applies masked self-attention only on same-type relations.

    Stage 2: inter-domain interaction
        Applies relation-aware cross-type attention on heterotypic edges.

    Edge evolution
        Updates relation representations inside the block and returns the evolved
        relation state for the next SRR layer / downstream decoder.

    This implementation replaces the previous degree-normalised mean aggregation
    with a lightweight multi-head attention variant while keeping the migration
    path incremental and compatible with the current model wrapper.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by num_heads ({num_heads})"
            )

        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads

        # Stage 1: intra-domain masked self-attention.
        self.intra_q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.intra_k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.intra_v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.intra_out_proj = nn.Linear(hidden_channels, hidden_channels)

        # Stage 2: inter-domain relation-aware attention.
        self.inter_q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.inter_k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.inter_v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.inter_rel_proj = nn.Linear(hidden_channels, hidden_channels)
        self.inter_out_proj = nn.Linear(hidden_channels, hidden_channels)

        self.intra_update = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.inter_update = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Relation / edge evolution: E_r^(l) = E_r^(l-1) + MLP(mean(h_src || h_dst)).
        self.relation_summary_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.relation_update_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.intra_norm = nn.LayerNorm(hidden_channels)
        self.inter_norm = nn.LayerNorm(hidden_channels)
        self.relation_norm = nn.LayerNorm(hidden_channels)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], self.num_heads, self.head_dim)

    def _init_message_buffers(self, x_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
        node_type: x.new_zeros(x.shape[0], self.hidden_channels)
        for node_type, x in x_dict.items()
        }
    
    def _prepare_relation_state(
        self,
        x_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[RelationTriple, torch.Tensor],
        relation_state: Optional[RelationEncoderState],
    ) -> RelationEncoderState:
        ref_tensor = next(iter(x_dict.values()))
        relation_keys: set[RelationTriple] = set(edge_index_dict.keys())
        if relation_state is not None:
            relation_keys.update(relation_state.base.keys())
            relation_keys.update(relation_state.evolved.keys())
            relation_keys.update(relation_state.pair_summary.keys())

        base: Dict[RelationTriple, torch.Tensor] = {}
        evolved: Dict[RelationTriple, torch.Tensor] = {}
        pair_summary: Dict[RelationTriple, torch.Tensor] = {}
        for triple in relation_keys:
            zero = ref_tensor.new_zeros(self.hidden_channels)
            if relation_state is None:
                base[triple] = zero
                evolved[triple] = zero
                pair_summary[triple] = zero
                continue

            base_vec = relation_state.base.get(triple, zero)
            evolved_vec = relation_state.evolved.get(triple, base_vec)
            summary_vec = relation_state.pair_summary.get(triple, zero)
            base[triple] = base_vec
            evolved[triple] = evolved_vec
            pair_summary[triple] = summary_vec

        return RelationEncoderState(
            base=base,
            evolved=evolved,
            pair_summary=pair_summary,
        )

    def _relation_vector(
        self,
        triple: RelationTriple,
        relation_state: RelationEncoderState,
        ref_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if triple in relation_state.evolved:
            return relation_state.evolved[triple]
        if triple in relation_state.base:
            return relation_state.base[triple]
        return ref_tensor.new_zeros(self.hidden_channels)

    def _aggregate_messages(
        self,
        x_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[RelationTriple, torch.Tensor],
        relation_state: RelationEncoderState,
        intra_only: bool,
    ) -> Dict[str, torch.Tensor]:
        messages = self._init_message_buffers(x_dict)

        for triple, edge_index in edge_index_dict.items():
            src_type, _, dst_type = triple
            if src_type not in x_dict or dst_type not in x_dict:
                continue
            if edge_index.numel() == 0:
                continue

            is_intra = src_type == dst_type
            if intra_only and not is_intra:
                continue
            if (not intra_only) and is_intra:
                continue

            src_idx = edge_index[0]
            dst_idx = edge_index[1]
            src_feat = x_dict[src_type][src_idx]
            dst_feat = x_dict[dst_type][dst_idx]

            if intra_only:
                q = self._split_heads(self.intra_q_proj(dst_feat))
                k = self._split_heads(self.intra_k_proj(src_feat))
                v = self._split_heads(self.intra_v_proj(src_feat))
            else:
                rel_vec = self._relation_vector(triple, relation_state, src_feat)
                rel_feat = self.inter_rel_proj(rel_vec).unsqueeze(0).expand(src_feat.shape[0], -1)
                q = self._split_heads(self.inter_q_proj(dst_feat))
                k = self._split_heads(self.inter_k_proj(src_feat + rel_feat))
                v = self._split_heads(self.inter_v_proj(src_feat))

            scores = (q * k).sum(dim=-1) / math.sqrt(self.head_dim)
            alpha = softmax(scores, dst_idx, num_nodes=x_dict[dst_type].shape[0])
            alpha = self.attn_dropout(alpha)
            msg = (alpha.unsqueeze(-1) * v).reshape(src_feat.shape[0], self.hidden_channels)
            messages[dst_type] = messages[dst_type].index_add(0, dst_idx, msg)

        out_proj = self.intra_out_proj if intra_only else self.inter_out_proj
        return {
            node_type: out_proj(messages[node_type])
            for node_type in messages
        }

    def _residual_update(
        self,
        x_dict: Mapping[str, torch.Tensor],
        messages: Mapping[str, torch.Tensor],
        updater: nn.Module,
        norm: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for node_type, x in x_dict.items():
            delta = updater(torch.cat([x, messages[node_type]], dim=-1))
            out[node_type] = norm(x + self.dropout(delta))
        return out

    def _evolve_relations(
        self,
        x_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[RelationTriple, torch.Tensor],
        relation_state: RelationEncoderState,
    ) -> RelationEncoderState:
        ref_tensor = next(iter(x_dict.values()))
        relation_keys: set[RelationTriple] = set(edge_index_dict.keys())
        relation_keys.update(relation_state.base.keys())
        relation_keys.update(relation_state.evolved.keys())

        base = dict(relation_state.base)
        evolved: Dict[RelationTriple, torch.Tensor] = {}
        pair_summary: Dict[RelationTriple, torch.Tensor] = {}

        for triple in relation_keys:
            prev_rel = relation_state.evolved.get(
                triple,
                relation_state.base.get(triple, ref_tensor.new_zeros(self.hidden_channels)),
            )
            base.setdefault(triple, relation_state.base.get(triple, prev_rel))

            edge_index = edge_index_dict.get(triple)
            src_type, _, dst_type = triple
            if (
                edge_index is None
                or edge_index.numel() == 0
                or src_type not in x_dict
                or dst_type not in x_dict
            ):
                evolved[triple] = prev_rel
                pair_summary[triple] = relation_state.pair_summary.get(
                    triple, ref_tensor.new_zeros(self.hidden_channels)
                )
                continue

            src_feat = x_dict[src_type][edge_index[0]]
            dst_feat = x_dict[dst_type][edge_index[1]]
            pooled_pair = torch.cat([src_feat, dst_feat], dim=-1).mean(dim=0)
            summary = self.relation_summary_proj(pooled_pair.unsqueeze(0)).squeeze(0)
            delta = self.relation_update_mlp(torch.cat([prev_rel, pooled_pair], dim=-1))
            evolved[triple] = self.relation_norm(prev_rel + self.dropout(delta))
            pair_summary[triple] = summary

        return RelationEncoderState(
            base=base,
            evolved=evolved,
            pair_summary=pair_summary,
        )

    def forward(
        self,
        x_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[RelationTriple, torch.Tensor],
        relation_state: Optional[RelationEncoderState] = None,
    ) -> SRRBlockOutput:
        seed_relation_state = self._prepare_relation_state(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            relation_state=relation_state,
        )

        intra_messages = self._aggregate_messages(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            relation_state=seed_relation_state,
            intra_only=True,
        )
        intra_out = self._residual_update(
            x_dict=x_dict,
            messages=intra_messages,
            updater=self.intra_update,
            norm=self.intra_norm,
        )

        inter_messages = self._aggregate_messages(
            x_dict=intra_out,
            edge_index_dict=edge_index_dict,
            relation_state=seed_relation_state,
            intra_only=False,
        )
        inter_out = self._residual_update(
            x_dict=intra_out,
            messages=inter_messages,
            updater=self.inter_update,
            norm=self.inter_norm,
        )

        # Edge evolution stays inside the SRR block. Following the architecture
        # note, the update is built from intra-stage node states and returned for
        # the next layer / downstream decoder.
        evolved_relation_state = self._evolve_relations(
            x_dict=intra_out,
            edge_index_dict=edge_index_dict,
            relation_state=seed_relation_state,
        )

        return SRRBlockOutput(
            x_dict=inter_out,
            intra_messages=intra_messages,
            inter_messages=inter_messages,
            relation_state=evolved_relation_state,
        )
