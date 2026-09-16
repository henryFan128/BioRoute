from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn

from .mechanism import MechanismRouter
from .relation_encoder import RelationEncoderState

RelationTriple = Tuple[str, str, str]


@dataclass
class GlobalRouterOutput:
    """Outputs of the global routing block."""

    node_context: Dict[str, torch.Tensor]
    router_state: Dict[str, torch.Tensor]
    explain: Dict[str, Dict[str, Any]] | None = None


class GlobalRouter(nn.Module):
    """Wrapper around the mechanism-induced local router block."""

    def __init__(
        self,
        hidden_channels: int,
        dropout: float = 0.1,
        therapy_types: Sequence[str] = ("chemical", "protein", "gene", "disease"),
        pathogenesis_types: Sequence[str] = ("pathogen", "gene", "disease"),
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.router = MechanismRouter(
            hidden_channels=hidden_channels,
            dropout=dropout,
            therapy_types=therapy_types,
            pathogenesis_types=pathogenesis_types,
        )

    def forward(
        self,
        x_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[RelationTriple, torch.Tensor],
        relation_state: RelationEncoderState | None = None,
        return_explain: bool = False,
    ) -> GlobalRouterOutput:
        router_result = self.router(
            x_dict=dict(x_dict),
            edge_index_dict=dict(edge_index_dict),
            relation_state=relation_state,
            return_explain=return_explain,
        )
        if return_explain:
            node_context, router_state, explain = router_result
        else:
            node_context, router_state = router_result
            explain = None
        if relation_state is not None:
            router_state = dict(router_state)
            router_state["num_relations"] = torch.tensor(
                float(len(relation_state.evolved)),
                device=next(iter(node_context.values())).device,
            )
        return GlobalRouterOutput(
            node_context=node_context,
            router_state=router_state,
            explain=explain,
        )
