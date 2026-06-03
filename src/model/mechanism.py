from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .relation_encoder import RelationEncoderState, RelationTriple


@dataclass(frozen=True)
class MechanismSpec:
    """Definition of one mechanism-induced local subgraph."""

    name: str
    node_types: Tuple[str, ...]
    allowed_type_pairs: Tuple[Tuple[str, str], ...]
    blocked_relations: Tuple[str, ...] = ()


class MechanismRouter(nn.Module):
    """Local mechanism router with virtual mechanism nodes.

    The block is intentionally edge-driven instead of global-pooling-based:

    1. Induce a mechanism-specific subgraph from the full heterogeneous graph.
    2. Aggregate edge-local evidence onto the nodes inside that subgraph.
    3. Read the active subgraph into a virtual mechanism node / token.
    4. Write the mechanism token back only to nodes that participate in the
       induced local subgraph.

    This keeps the router compatible with the current model interface while
    avoiding the previous "broadcast one global token to everything" behavior.
    """

    def __init__(
        self,
        hidden_channels: int,
        dropout: float = 0.1,
        therapy_types: Sequence[str] = ("chemical", "protein", "gene", "disease"),
        pathogenesis_types: Sequence[str] = ("pathogen", "gene", "disease"),
    ) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.dropout = nn.Dropout(dropout)

        self.mechanism_specs = (
            MechanismSpec(
                name="therapy",
                node_types=tuple(dict.fromkeys(therapy_types)),
                allowed_type_pairs=(
                    ("chemical", "chemical"),
                    ("chemical", "protein"),
                    ("protein", "chemical"),
                    ("protein", "protein"),
                    ("gene", "protein"),
                    ("protein", "gene"),
                    ("gene", "gene"),
                    ("gene", "disease"),
                    ("disease", "gene"),
                ),
                blocked_relations=(
                    "treats",
                    "treated_by",
                    "contraindicates_cd",
                    "contraindicated_by",
                ),
            ),
            MechanismSpec(
                name="pathogenesis",
                node_types=tuple(dict.fromkeys(pathogenesis_types)),
                allowed_type_pairs=(
                    ("pathogen", "gene"),
                    ("gene", "pathogen"),
                    ("gene", "gene"),
                    ("gene", "disease"),
                    ("disease", "gene"),
                ),
                blocked_relations=(
                    "treats",
                    "treated_by",
                    "contraindicates_cd",
                    "contraindicated_by",
                ),
            ),
        )
        self._node_sets = {
            spec.name: set(spec.node_types) for spec in self.mechanism_specs
        }
        self._pair_sets = {
            spec.name: set(spec.allowed_type_pairs) for spec in self.mechanism_specs
        }

        edge_input_dim = hidden_channels * 5
        node_to_token_dim = hidden_channels * 2
        token_to_node_dim = hidden_channels * 3

        self.edge_mlps = nn.ModuleDict()
        self.edge_gate_mlps = nn.ModuleDict()
        self.edge_to_src = nn.ModuleDict()
        self.edge_to_dst = nn.ModuleDict()
        self.node_to_token = nn.ModuleDict()
        self.readout_gate = nn.ModuleDict()
        self.token_update = nn.ModuleDict()
        self.token_to_node = nn.ModuleDict()
        self.node_gate = nn.ModuleDict()
        self.local_norms = nn.ModuleDict()
        self.token_norms = nn.ModuleDict()

        self.token_embeddings = nn.ParameterDict()

        for spec in self.mechanism_specs:
            self.edge_mlps[spec.name] = nn.Sequential(
                nn.Linear(edge_input_dim, hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.edge_gate_mlps[spec.name] = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, 1),
            )
            self.edge_to_src[spec.name] = nn.Linear(hidden_channels, hidden_channels)
            self.edge_to_dst[spec.name] = nn.Linear(hidden_channels, hidden_channels)
            self.node_to_token[spec.name] = nn.Sequential(
                nn.Linear(node_to_token_dim, hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.readout_gate[spec.name] = nn.Linear(node_to_token_dim, 1)
            self.token_update[spec.name] = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.token_to_node[spec.name] = nn.Sequential(
                nn.Linear(token_to_node_dim, hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.node_gate[spec.name] = nn.Linear(token_to_node_dim, 1)
            self.local_norms[spec.name] = nn.LayerNorm(hidden_channels)
            self.token_norms[spec.name] = nn.LayerNorm(hidden_channels)

            token = nn.Parameter(torch.empty(hidden_channels))
            nn.init.xavier_uniform_(token.view(1, -1))
            self.token_embeddings[spec.name] = token

        self.output_norm = nn.LayerNorm(hidden_channels)

    def _empty_feature_buffers(
        self,
        x_dict: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {
            node_type: x.new_zeros(x.shape[0], self.hidden_channels)
            for node_type, x in x_dict.items()
        }

    def _empty_mask_buffers(
        self,
        x_dict: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {
            node_type: torch.zeros(
                x.shape[0],
                device=x.device,
                dtype=torch.bool,
            )
            for node_type, x in x_dict.items()
        }

    def _empty_count_buffers(
        self,
        x_dict: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {
            node_type: x.new_zeros(x.shape[0], 1)
            for node_type, x in x_dict.items()
        }

    def _induce_subgraph_edges(
        self,
        mechanism: MechanismSpec,
        edge_index_dict: Mapping[RelationTriple, torch.Tensor],
    ) -> Dict[RelationTriple, torch.Tensor]:
        allowed_types = self._node_sets[mechanism.name]
        allowed_pairs = self._pair_sets[mechanism.name]
        induced: Dict[RelationTriple, torch.Tensor] = {}
        for triple, edge_index in edge_index_dict.items():
            src_type, relation_name, dst_type = triple
            if src_type not in allowed_types or dst_type not in allowed_types:
                continue
            if (src_type, dst_type) not in allowed_pairs:
                continue
            if relation_name in mechanism.blocked_relations:
                continue
            if edge_index is None or edge_index.numel() == 0:
                continue
            induced[triple] = edge_index
        return induced

    def _relation_vector(
        self,
        triple: RelationTriple,
        relation_state: Optional[RelationEncoderState],
        ref_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if relation_state is None:
            return ref_tensor.new_zeros(self.hidden_channels)
        if triple in relation_state.evolved:
            return relation_state.evolved[triple]
        if triple in relation_state.base:
            return relation_state.base[triple]
        return ref_tensor.new_zeros(self.hidden_channels)

    def _edge_localize(
        self,
        mechanism: MechanismSpec,
        x_dict: Mapping[str, torch.Tensor],
        subgraph_edges: Mapping[RelationTriple, torch.Tensor],
        relation_state: Optional[RelationEncoderState],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int]:
        local_messages = self._empty_feature_buffers(x_dict)
        node_counts = self._empty_count_buffers(x_dict)
        active_masks = self._empty_mask_buffers(x_dict)
        total_edges = 0

        for triple, edge_index in subgraph_edges.items():
            src_type, _, dst_type = triple
            src_idx = edge_index[0]
            dst_idx = edge_index[1]
            if src_idx.numel() == 0:
                continue

            src_feat = x_dict[src_type][src_idx]
            dst_feat = x_dict[dst_type][dst_idx]
            rel_feat = self._relation_vector(
                triple,
                relation_state=relation_state,
                ref_tensor=src_feat,
            ).unsqueeze(0).expand(src_feat.shape[0], -1)

            edge_input = torch.cat(
                [
                    src_feat,
                    dst_feat,
                    src_feat * dst_feat,
                    torch.abs(src_feat - dst_feat),
                    rel_feat,
                ],
                dim=-1,
            )
            edge_hidden = self.edge_mlps[mechanism.name](edge_input)
            edge_gate = torch.sigmoid(self.edge_gate_mlps[mechanism.name](edge_hidden))

            src_msg = self.edge_to_src[mechanism.name](edge_hidden) * edge_gate
            dst_msg = self.edge_to_dst[mechanism.name](edge_hidden) * edge_gate

            local_messages[src_type].index_add_(0, src_idx, src_msg)
            local_messages[dst_type].index_add_(0, dst_idx, dst_msg)

            ones = src_feat.new_ones((src_idx.shape[0], 1))
            node_counts[src_type].index_add_(0, src_idx, ones)
            node_counts[dst_type].index_add_(0, dst_idx, ones)

            active_masks[src_type][src_idx] = True
            active_masks[dst_type][dst_idx] = True
            total_edges += int(src_idx.shape[0])

        local_summary: Dict[str, torch.Tensor] = {}
        for node_type, x in x_dict.items():
            normed = local_messages[node_type] / node_counts[node_type].clamp_min(1.0)
            local_summary[node_type] = self.local_norms[mechanism.name](normed)

        return local_summary, active_masks, total_edges

    def _pool_virtual_node(
        self,
        mechanism: MechanismSpec,
        x_dict: Mapping[str, torch.Tensor],
        local_summary: Mapping[str, torch.Tensor],
        active_masks: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        token_seed = self.token_embeddings[mechanism.name]

        weighted_sum: Optional[torch.Tensor] = None
        weight_sum: Optional[torch.Tensor] = None
        active_nodes = 0

        for node_type in mechanism.node_types:
            if node_type not in x_dict:
                continue
            mask = active_masks[node_type]
            if int(mask.sum().item()) == 0:
                continue

            active_nodes += int(mask.sum().item())
            node_input = torch.cat(
                [
                    x_dict[node_type][mask],
                    local_summary[node_type][mask],
                ],
                dim=-1,
            )
            token_contrib = self.node_to_token[mechanism.name](node_input)
            contrib_gate = torch.sigmoid(self.readout_gate[mechanism.name](node_input))

            contrib_sum = (token_contrib * contrib_gate).sum(dim=0)
            gate_sum = contrib_gate.sum(dim=0)

            if weighted_sum is None:
                weighted_sum = contrib_sum
                weight_sum = gate_sum
            else:
                weighted_sum = weighted_sum + contrib_sum
                weight_sum = weight_sum + gate_sum

        if weighted_sum is None or weight_sum is None:
            return self.token_norms[mechanism.name](token_seed), active_nodes

        pooled = weighted_sum / weight_sum.clamp_min(1e-6)
        token_delta = self.token_update[mechanism.name](
            torch.cat([token_seed, pooled], dim=-1)
        )
        token = self.token_norms[mechanism.name](
            token_seed + self.dropout(token_delta)
        )
        return token, active_nodes

    def _write_back(
        self,
        mechanism: MechanismSpec,
        x_dict: Mapping[str, torch.Tensor],
        local_summary: Mapping[str, torch.Tensor],
        active_masks: Mapping[str, torch.Tensor],
        mechanism_token: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        node_delta = self._empty_feature_buffers(x_dict)

        for node_type in mechanism.node_types:
            if node_type not in x_dict:
                continue
            mask = active_masks[node_type]
            if int(mask.sum().item()) == 0:
                continue

            local_feat = local_summary[node_type][mask]
            node_feat = x_dict[node_type][mask]
            token_feat = mechanism_token.unsqueeze(0).expand(node_feat.shape[0], -1)
            write_input = torch.cat(
                [node_feat, local_feat, token_feat],
                dim=-1,
            )
            delta = self.token_to_node[mechanism.name](write_input)
            gate = torch.sigmoid(self.node_gate[mechanism.name](write_input))
            node_delta[node_type][mask] = delta * gate

        return node_delta

    def forward(
        self,
        x_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[RelationTriple, torch.Tensor],
        relation_state: Optional[RelationEncoderState] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        combined_delta = self._empty_feature_buffers(x_dict)
        mechanism_hits = self._empty_count_buffers(x_dict)
        router_state: Dict[str, torch.Tensor] = {}

        for mechanism in self.mechanism_specs:
            subgraph_edges = self._induce_subgraph_edges(
                mechanism=mechanism,
                edge_index_dict=edge_index_dict,
            )
            local_summary, active_masks, active_edges = self._edge_localize(
                mechanism=mechanism,
                x_dict=x_dict,
                subgraph_edges=subgraph_edges,
                relation_state=relation_state,
            )
            mechanism_token, active_nodes = self._pool_virtual_node(
                mechanism=mechanism,
                x_dict=x_dict,
                local_summary=local_summary,
                active_masks=active_masks,
            )
            mechanism_delta = self._write_back(
                mechanism=mechanism,
                x_dict=x_dict,
                local_summary=local_summary,
                active_masks=active_masks,
                mechanism_token=mechanism_token,
            )

            for node_type in combined_delta:
                combined_delta[node_type] = combined_delta[node_type] + mechanism_delta[node_type]
                mechanism_hits[node_type] = mechanism_hits[node_type] + active_masks[
                    node_type
                ].float().unsqueeze(-1)

            router_state[f"{mechanism.name}_token"] = mechanism_token
            router_state[f"{mechanism.name}_active_nodes"] = mechanism_token.new_tensor(
                float(active_nodes)
            )
            router_state[f"{mechanism.name}_active_edges"] = mechanism_token.new_tensor(
                float(active_edges)
            )
            router_state[
                f"{mechanism.name}_active_relations"
            ] = mechanism_token.new_tensor(float(len(subgraph_edges)))

        node_context: Dict[str, torch.Tensor] = {}
        for node_type, x in x_dict.items():
            hit_mask = mechanism_hits[node_type].squeeze(-1) > 0
            if not bool(hit_mask.any().item()):
                node_context[node_type] = x
                continue

            updated = x.clone()
            delta = combined_delta[node_type] / mechanism_hits[node_type].clamp_min(1.0)
            updated_active = self.output_norm(
                x[hit_mask] + self.dropout(delta[hit_mask])
            )
            updated[hit_mask] = updated_active
            node_context[node_type] = updated

        return node_context, router_state
