from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

RelationTriple = Tuple[str, str, str]


@dataclass
class RelationEncoderState:
    """Relation table and its evolved per-layer state."""

    base: Dict[RelationTriple, torch.Tensor]
    evolved: Dict[RelationTriple, torch.Tensor]
    pair_summary: Dict[RelationTriple, torch.Tensor]


class RelationEncoder(nn.Module):
    """Learnable relation embeddings with lightweight edge-type evolution."""

    def __init__(
        self,
        relations: Sequence[RelationTriple],
        hidden_channels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.relations: List[RelationTriple] = [tuple(rel) for rel in relations]
        self._rel_to_key = {rel: self._triple_key(rel) for rel in self.relations}

        self.relation_embed = nn.ParameterDict(
            {
                self._triple_key(rel): nn.Parameter(torch.empty(hidden_channels))
                for rel in self.relations
            }
        )
        for param in self.relation_embed.values():
            nn.init.xavier_uniform_(param.view(1, -1))

        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.norm = nn.LayerNorm(hidden_channels)

    @staticmethod
    def _triple_key(triple: RelationTriple) -> str:
        return "__".join(triple)

    def relation_triples(self) -> List[RelationTriple]:
        return list(self.relations)

    def init_state(self, device: Optional[torch.device] = None) -> RelationEncoderState:
        base = {}
        for triple in self.relations:
            tensor = self.relation_embed[self._rel_to_key[triple]]
            if device is not None:
                tensor = tensor.to(device)
            base[triple] = tensor
        evolved = {triple: emb for triple, emb in base.items()}
        pair_summary = {
            triple: emb.new_zeros(self.hidden_channels) for triple, emb in base.items()
        }
        return RelationEncoderState(
            base=base,
            evolved=evolved,
            pair_summary=pair_summary,
        )

    def forward(
        self,
        x_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[RelationTriple, torch.Tensor],
        prev_state: Optional[RelationEncoderState] = None,
    ) -> RelationEncoderState:
        if len(x_dict) == 0:
            raise ValueError("x_dict must not be empty")
        device = next(iter(x_dict.values())).device
        state = prev_state or self.init_state(device=device)

        evolved: Dict[RelationTriple, torch.Tensor] = {}
        pair_summary: Dict[RelationTriple, torch.Tensor] = {}
        base = {triple: state.base[triple] for triple in self.relations}

        for triple in self.relations:
            rel_vec = state.evolved.get(triple, state.base[triple])
            src_type, _, dst_type = triple
            edge_index = edge_index_dict.get(triple)
            if edge_index is None or edge_index.numel() == 0:
                summary = rel_vec.new_zeros(self.hidden_channels)
                evolved[triple] = rel_vec
                pair_summary[triple] = summary
                continue

            src_x = x_dict[src_type][edge_index[0]]
            dst_x = x_dict[dst_type][edge_index[1]]
            src_mean = src_x.mean(dim=0)
            dst_mean = dst_x.mean(dim=0)
            summary = 0.5 * (src_mean + dst_mean)
            pair_summary[triple] = summary

            update_input = torch.cat([rel_vec, src_mean, dst_mean], dim=-1)
            delta = self.update_mlp(update_input)
            evolved[triple] = self.norm(rel_vec + delta)

        return RelationEncoderState(
            base=base,
            evolved=evolved,
            pair_summary=pair_summary,
        )

    def get_relation_embedding(
        self,
        relation: RelationTriple,
        state: Optional[RelationEncoderState] = None,
        evolved: bool = True,
    ) -> torch.Tensor:
        rel = tuple(relation)
        if rel not in self._rel_to_key:
            raise KeyError(f"Unknown relation triple: {rel}")
        if state is None:
            return self.relation_embed[self._rel_to_key[rel]]
        if evolved:
            return state.evolved[rel]
        return state.base[rel]

    def export_relation_table(
        self,
        state: Optional[RelationEncoderState] = None,
    ) -> Dict[RelationTriple, torch.Tensor]:
        if state is None:
            return {
                triple: self.relation_embed[self._rel_to_key[triple]]
                for triple in self.relations
            }
        return {triple: emb for triple, emb in state.evolved.items()}
