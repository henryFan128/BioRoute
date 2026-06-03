from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import Linear

from ..feature_ge_extra import GOResMLP
from .srr_layer import SRRLayer
from .global_router import GlobalRouter, GlobalRouterOutput
from .relation_encoder import RelationEncoder, RelationEncoderState, RelationTriple
from .decoder import BilinearDecoder, ComplExDecoder, PairMLPDecoder


class KGCModel(nn.Module):
    """Backbone + decoder runtime for edge scoring on the knowledge graph."""

    def __init__(
        self,
        data: HeteroData,
        hidden_channels: int,
        out_channels: int,
        num_srr_layers: int,
        num_attention_heads: int,
        dropout: float = 0.1,
        use_global_router: bool = True,
        mechanism_dropout: float = 0.1,
        decoder_type: str = "complex",
    ) -> None:
        super().__init__()

        self.hidden_channels = int(hidden_channels)
        self.out_channels = int(out_channels)
        self.num_srr_layers = int(num_srr_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.dropout = float(dropout)
        self.use_global_router = bool(use_global_router)
        self.decoder_type = str(decoder_type).lower()

        self.default_relation_triple: RelationTriple = (
            "chemical",
            "treats",
            "disease",
        )

        self._latest_relation_state: Optional[RelationEncoderState] = None
        self._latest_router_output: Optional[GlobalRouterOutput] = None
        self._latest_gate_weights: Optional[Dict[str, torch.Tensor]] = None

        self.dropout_layer = nn.Dropout(dropout)

        self.proj_linears = nn.ModuleDict()
        for node_type in data.node_types:
            in_channels = data[node_type].x.shape[1]
            if node_type in ("gene", "pathogen"):
                self.proj_linears[node_type] = GOResMLP(
                    vocab_size=in_channels,
                    hidden_dim=512,
                    out_dim=hidden_channels,
                )
            else:
                self.proj_linears[node_type] = Linear(in_channels, hidden_channels)

        relation_triples = [tuple(t) for t in data.edge_types]
        self.relation_encoder = RelationEncoder(
            relations=relation_triples,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )

        self.srr_layers = nn.ModuleList(
            [
                SRRLayer(
                    hidden_channels=hidden_channels,
                    num_heads=num_attention_heads,
                    dropout=dropout,
                )
                for _ in range(num_srr_layers)
            ]
        )

        if self.use_global_router:
            self.global_router = GlobalRouter(
                hidden_channels=hidden_channels,
                dropout=mechanism_dropout,
            )
        else:
            self.global_router = None

        self.semantic_out_linears = nn.ModuleDict()
        self.srr_out_linears = nn.ModuleDict()
        for node_type in data.node_types:
            self.semantic_out_linears[node_type] = Linear(hidden_channels, out_channels)
            self.srr_out_linears[node_type] = Linear(hidden_channels, out_channels)
        if self.use_global_router:
            self.router_out_linears = nn.ModuleDict()
            for node_type in data.node_types:
                self.router_out_linears[node_type] = Linear(
                    hidden_channels,
                    out_channels,
                )
        else:
            self.router_out_linears = None

        gate_input_dim = out_channels * (3 if self.use_global_router else 2)
        gate_output_dim = 3 if self.use_global_router else 2
        self.fusion_gate_mlps = nn.ModuleDict()
        for node_type in data.node_types:
            self.fusion_gate_mlps[node_type] = nn.Sequential(
                Linear(gate_input_dim, out_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                Linear(out_channels, gate_output_dim),
            )
        self.relation_out = Linear(hidden_channels, out_channels)

        if self.decoder_type == "complex":
            self.decoder = ComplExDecoder(
                hidden_channels=out_channels,
                dropout=dropout,
            )
        elif self.decoder_type == "bilinear":
            self.decoder = BilinearDecoder(
                hidden_channels=out_channels,
                dropout=dropout,
            )
        elif self.decoder_type == "pair_mlp":
            self.decoder = PairMLPDecoder(
                hidden_channels=out_channels,
                dropout=dropout,
            )
        else:
            raise ValueError(
                f"Unsupported decoder_type={decoder_type!r}. Expected 'complex', 'bilinear', or 'pair_mlp'."
            )

    def _semantic_init(
        self,
        x_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {
            node_type: self.dropout_layer(self.proj_linears[node_type](x).relu())
            for node_type, x in x_dict.items()
        }

    def _run_srr_backbone(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[RelationTriple, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], RelationEncoderState]:
        device = next(iter(x_dict.values())).device
        relation_state = self.relation_encoder.init_state(device=device)
        srr_input = dict(x_dict)
        for layer in self.srr_layers:
            block_out = layer(
                srr_input,
                edge_index_dict,
                relation_state=relation_state,
            )
            srr_input = {
                node_type: self.dropout_layer(feat)
                for node_type, feat in block_out.x_dict.items()
            }
            relation_state = block_out.relation_state
        return srr_input, relation_state

    def _run_router(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[RelationTriple, torch.Tensor],
        relation_state: RelationEncoderState,
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[GlobalRouterOutput]]:
        if not self.use_global_router or self.global_router is None:
            return None, None
        router_output = self.global_router(
            x_dict,
            edge_index_dict,
            relation_state=relation_state,
        )
        return router_output.node_context, router_output

    def _project_branch(
        self,
        x_dict: Dict[str, torch.Tensor],
        proj_linears: nn.ModuleDict,
    ) -> Dict[str, torch.Tensor]:
        return {
            node_type: proj_linears[node_type](x)
            for node_type, x in x_dict.items()
        }

    def _fuse_projected_branches(
        self,
        semantic_proj: Dict[str, torch.Tensor],
        srr_proj: Dict[str, torch.Tensor],
        router_proj: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        fused: Dict[str, torch.Tensor] = {}
        gate_weights: Dict[str, torch.Tensor] = {}

        for node_type in semantic_proj:
            if router_proj is None:
                branch_stack = torch.cat(
                    [semantic_proj[node_type], srr_proj[node_type]],
                    dim=-1,
                )
                weights = torch.softmax(
                    self.fusion_gate_mlps[node_type](branch_stack),
                    dim=-1,
                )
                fused[node_type] = (
                    weights[:, 0:1] * semantic_proj[node_type]
                    + weights[:, 1:2] * srr_proj[node_type]
                )
            else:
                branch_stack = torch.cat(
                    [
                        semantic_proj[node_type],
                        srr_proj[node_type],
                        router_proj[node_type],
                    ],
                    dim=-1,
                )
                weights = torch.softmax(
                    self.fusion_gate_mlps[node_type](branch_stack),
                    dim=-1,
                )
                fused[node_type] = (
                    weights[:, 0:1] * semantic_proj[node_type]
                    + weights[:, 1:2] * srr_proj[node_type]
                    + weights[:, 2:3] * router_proj[node_type]
                )
            gate_weights[node_type] = weights

        return fused, gate_weights

    def forward(
        self,
        x_dict,
        edge_index_dict,
        return_attention_weights: bool = False,
    ):
        semantic_features = self._semantic_init(x_dict)
        srr_features, relation_state = self._run_srr_backbone(
            semantic_features,
            edge_index_dict,
        )
        routed_features, router_output = self._run_router(
            semantic_features,
            edge_index_dict,
            relation_state,
        )

        semantic_proj = self._project_branch(
            semantic_features,
            self.semantic_out_linears,
        )
        srr_proj = self._project_branch(
            srr_features,
            self.srr_out_linears,
        )
        router_proj = None
        if routed_features is not None and self.router_out_linears is not None:
            router_proj = self._project_branch(
                routed_features,
                self.router_out_linears,
            )
        final_x_dict, gate_weights = self._fuse_projected_branches(
            semantic_proj=semantic_proj,
            srr_proj=srr_proj,
            router_proj=router_proj,
        )

        self._latest_relation_state = relation_state
        self._latest_router_output = router_output
        self._latest_gate_weights = gate_weights

        if return_attention_weights:
            aux = {
                "relation_state": relation_state,
                "semantic": semantic_features,
                "SRR": srr_features,
                "routed": routed_features,
                "semantic_proj": semantic_proj,
                "SRR_proj": srr_proj,
                "gate_weights": gate_weights,
            }
            if router_output is not None:
                aux["router_output"] = router_output
            if router_proj is not None:
                aux["router_proj"] = router_proj
            return final_x_dict, aux
        return final_x_dict

    def extract_components(
        self,
        x_dict,
        edge_index_dict,
        return_attention_weights: bool = True,
    ):
        semantic_features = self._semantic_init(x_dict)
        srr_features, relation_state = self._run_srr_backbone(
            semantic_features,
            edge_index_dict,
        )
        routed_features, router_output = self._run_router(
            semantic_features,
            edge_index_dict,
            relation_state,
        )
        semantic_proj = self._project_branch(
            semantic_features,
            self.semantic_out_linears,
        )
        srr_proj = self._project_branch(
            srr_features,
            self.srr_out_linears,
        )
        router_proj = None
        if routed_features is not None and self.router_out_linears is not None:
            router_proj = self._project_branch(
                routed_features,
                self.router_out_linears,
            )
        final_x_dict, gate_weights = self._fuse_projected_branches(
            semantic_proj=semantic_proj,
            srr_proj=srr_proj,
            router_proj=router_proj,
        )

        result = {
            "semantic": semantic_features,
            "SRR": srr_features,
            "routed": routed_features,
            "semantic_proj": semantic_proj,
            "SRR_proj": srr_proj,
            "router_proj": router_proj,
            "gate_weights": gate_weights,
            "final": final_x_dict,
            "relation_state": relation_state,
        }
        if router_output is not None:
            result["router_output"] = router_output
        if return_attention_weights and router_output is not None:
            result["attention"] = {
                "router_state": router_output.router_state,
            }
        return result

    def _resolve_relation_triple(
        self,
        relation: str | RelationTriple,
    ) -> RelationTriple:
        if isinstance(relation, tuple):
            return relation
        if relation == "treats":
            return self.default_relation_triple
        raise KeyError(f"Unsupported relation name: {relation}")

    def _relation_embed(
        self,
        relation: str | RelationTriple = "treats",
    ) -> torch.Tensor:
        rel_triple = self._resolve_relation_triple(relation)
        if self._latest_relation_state is None:
            raise RuntimeError("forward() must be called before decoding edges")
        relation_hidden = self.relation_encoder.get_relation_embedding(
            rel_triple,
            state=self._latest_relation_state,
            evolved=True,
        )
        return self.relation_out(relation_hidden)

    def decode_edges(
        self,
        z_dict: Dict[str, torch.Tensor],
        edge_label_index: torch.Tensor,
        relation: str | RelationTriple = "treats",
    ) -> torch.Tensor:
        rel_triple = self._resolve_relation_triple(relation)
        src_type, _, dst_type = rel_triple
        src_embed = z_dict[src_type][edge_label_index[0]]
        dst_embed = z_dict[dst_type][edge_label_index[1]]
        relation_embed = self._relation_embed(relation=rel_triple)
        return self.decoder.score_edges(
            src_embed=src_embed,
            dst_embed=dst_embed,
            relation_embed=relation_embed,
        )

    @staticmethod
    def score_recommendation_candidates(
        z_dict: Dict[str, torch.Tensor],
        disease_index: torch.Tensor,
        candidate_chemical_index: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        query_node_type: str = "disease",
        candidate_node_type: str = "chemical",
    ) -> torch.Tensor:
        """Score disease-query/full-chemical-candidate batches by cosine dot product.

        This is the recommendation path used by the full-candidate drug
        repurposing trainer. It deliberately bypasses the KGC relation decoder:
        the KG encoder produces node representations, and the recommender ranks
        chemicals for each disease by normalized query-item similarity.
        """
        if query_node_type not in z_dict:
            raise KeyError(f"query node type not found in z_dict: {query_node_type}")
        if candidate_node_type not in z_dict:
            raise KeyError(
                f"candidate node type not found in z_dict: {candidate_node_type}"
            )

        device = z_dict[query_node_type].device
        disease_index = torch.as_tensor(
            disease_index,
            device=device,
            dtype=torch.long,
        )
        candidate_chemical_index = torch.as_tensor(
            candidate_chemical_index,
            device=device,
            dtype=torch.long,
        )
        if disease_index.ndim != 1:
            raise ValueError("disease_index must have shape [B]")
        if candidate_chemical_index.ndim != 2:
            raise ValueError("candidate_chemical_index must have shape [B, C]")
        if disease_index.shape[0] != candidate_chemical_index.shape[0]:
            raise ValueError(
                "disease_index must match candidate_chemical_index batch size"
            )

        if candidate_mask is None:
            candidate_mask = torch.ones_like(candidate_chemical_index, dtype=torch.bool)
        else:
            candidate_mask = torch.as_tensor(
                candidate_mask,
                device=device,
                dtype=torch.bool,
            )
        if candidate_mask.shape != candidate_chemical_index.shape:
            raise ValueError("candidate_mask must match candidate_chemical_index shape")

        disease_embed = F.normalize(
            z_dict[query_node_type][disease_index],
            p=2,
            dim=-1,
            eps=1e-8,
        )
        chemical_embed = F.normalize(
            z_dict[candidate_node_type][candidate_chemical_index],
            p=2,
            dim=-1,
            eps=1e-8,
        )
        logits = (disease_embed.unsqueeze(1) * chemical_embed).sum(dim=-1)
        return logits.masked_fill(~candidate_mask, float("-inf"))

    @staticmethod
    def recommendation_info_nce_loss(
        logits: torch.Tensor,
        positive_mask: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """Interaction-level InfoNCE over full candidate chemicals.

        Each known disease-chemical treatment pair contributes one numerator,
        while the denominator is the candidate chemical universe for that
        disease. Multi-positive diseases therefore contribute one term per
        observed positive pair, matching the common recommendation objective.
        """
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        positive_mask = torch.as_tensor(
            positive_mask,
            device=logits.device,
            dtype=torch.bool,
        )
        if candidate_mask is None:
            candidate_mask = torch.ones_like(positive_mask, dtype=torch.bool)
        else:
            candidate_mask = torch.as_tensor(
                candidate_mask,
                device=logits.device,
                dtype=torch.bool,
            )
        if positive_mask.shape != logits.shape:
            raise ValueError("positive_mask must match logits shape")
        if candidate_mask.shape != logits.shape:
            raise ValueError("candidate_mask must match logits shape")

        valid_positive_mask = positive_mask & candidate_mask
        valid_negative_mask = candidate_mask & ~positive_mask
        if not torch.all(valid_positive_mask.any(dim=1)):
            raise ValueError("Each disease batch row must have a positive candidate")
        if not torch.all(valid_negative_mask.any(dim=1)):
            raise ValueError(
                "Each disease batch row must have at least one non-positive candidate"
            )

        scaled_logits = logits / float(temperature)
        masked_logits = scaled_logits.masked_fill(~candidate_mask, float("-inf"))
        log_denom = torch.logsumexp(masked_logits, dim=1, keepdim=True)
        per_positive_loss = -(scaled_logits - log_denom)
        return per_positive_loss[valid_positive_mask].mean()

    @staticmethod
    def recommendation_loss(
        z_dict: Dict[str, torch.Tensor],
        disease_index: torch.Tensor,
        candidate_chemical_index: torch.Tensor,
        positive_mask: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        temperature: float = 0.1,
        query_node_type: str = "disease",
        candidate_node_type: str = "chemical",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        logits = KGCModel.score_recommendation_candidates(
            z_dict=z_dict,
            disease_index=disease_index,
            candidate_chemical_index=candidate_chemical_index,
            candidate_mask=candidate_mask,
            query_node_type=query_node_type,
            candidate_node_type=candidate_node_type,
        )
        info_nce_loss = KGCModel.recommendation_info_nce_loss(
            logits=logits,
            positive_mask=positive_mask,
            candidate_mask=candidate_mask,
            temperature=temperature,
        )
        return info_nce_loss, {"info_nce": info_nce_loss.detach()}, logits

    @staticmethod
    def _grouped_info_nce_loss(
        logits: torch.Tensor,
        positive_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        temperature: float = 0.1,
        loss_mode: str = "pooled",
    ) -> torch.Tensor:
        if temperature <= 0.0:
            raise ValueError("infonce_temperature must be positive")

        positive_mask = torch.as_tensor(
            positive_mask,
            device=logits.device,
            dtype=torch.bool,
        )
        candidate_mask = torch.as_tensor(
            candidate_mask,
            device=logits.device,
            dtype=torch.bool,
        )
        if positive_mask.shape != logits.shape:
            raise ValueError("positive_mask must match logits shape")
        if candidate_mask.shape != logits.shape:
            raise ValueError("candidate_mask must match logits shape")

        negative_mask = candidate_mask & ~positive_mask
        if not torch.all(positive_mask.any(dim=1)):
            raise ValueError("Each grouped source batch row must have a positive candidate")
        if not torch.all(negative_mask.any(dim=1)):
            raise ValueError("Each grouped source batch row must have a negative candidate")

        scaled_logits = logits / temperature
        masked_logits = scaled_logits.masked_fill(~candidate_mask, float("-inf"))
        log_denom = torch.logsumexp(masked_logits, dim=1, keepdim=True)

        loss_mode = str(loss_mode).lower()
        if loss_mode == "per_positive":
            per_positive_loss = -(scaled_logits - log_denom)
            valid_positive_loss = per_positive_loss[positive_mask]
            return valid_positive_loss.mean()

        if loss_mode == "pooled":
            masked_positive_logits = scaled_logits.masked_fill(
                ~positive_mask,
                float("-inf"),
            )
            log_pos = torch.logsumexp(masked_positive_logits, dim=1, keepdim=True)
            row_loss = -(log_pos - log_denom)
            return row_loss.mean()

        raise ValueError(
            f"Unsupported infonce loss mode: {loss_mode!r}. "
            "Expected 'pooled' or 'per_positive'."
        )

    def score_grouped_candidates(
        self,
        z_dict: Dict[str, torch.Tensor],
        source_index: torch.Tensor,
        candidate_index: torch.Tensor,
        candidate_mask: torch.Tensor,
        relation: str | RelationTriple = "treats",
        group_axis: str = "src",
    ) -> torch.Tensor:
        device = next(iter(z_dict.values())).device
        source_index = torch.as_tensor(source_index, device=device, dtype=torch.long)
        candidate_index = torch.as_tensor(
            candidate_index,
            device=device,
            dtype=torch.long,
        )
        candidate_mask = torch.as_tensor(
            candidate_mask,
            device=device,
            dtype=torch.bool,
        )
        if candidate_index.ndim != 2:
            raise ValueError("candidate_index must have shape [B, C]")
        if candidate_mask.shape != candidate_index.shape:
            raise ValueError("candidate_mask must match candidate_index shape")
        if source_index.ndim != 1 or source_index.shape[0] != candidate_index.shape[0]:
            raise ValueError("source_index must have shape [B] and match candidate_index batch size")

        batch_size, width = candidate_index.shape
        expanded_group = source_index.unsqueeze(1).expand(batch_size, width)
        group_axis_value = str(group_axis).lower()
        if group_axis_value in {"src", "source", "chemical", "0"}:
            flat_src = expanded_group[candidate_mask]
            flat_dst = candidate_index[candidate_mask]
        elif group_axis_value in {"dst", "target", "disease", "1"}:
            flat_src = candidate_index[candidate_mask]
            flat_dst = expanded_group[candidate_mask]
        else:
            raise ValueError(
                "group_axis must be one of src/source/chemical/0 or "
                "dst/target/disease/1"
            )
        edge_label_index = torch.stack([flat_src, flat_dst], dim=0)
        flat_scores = self.decode_edges(
            z_dict=z_dict,
            edge_label_index=edge_label_index,
            relation=relation,
        )

        logits = torch.full(
            (batch_size, width),
            fill_value=float("-inf"),
            device=device,
            dtype=flat_scores.dtype,
        )
        logits[candidate_mask] = flat_scores
        return logits

    def loss(
        self,
        z_dict: Dict[str, torch.Tensor],
        source_index: torch.Tensor,
        candidate_index: torch.Tensor,
        positive_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        relation: str | RelationTriple = "treats",
        infonce_temperature: float = 0.1,
        infonce_loss_mode: str = "pooled",
        aux_bce_weight: float = 0.0,
        aux_focal_enabled: bool = False,
        aux_focal_alpha: float = 0.25,
        aux_focal_gamma: float = 2.0,
        aux_focal_weight: float = 0.0,
        group_axis: str = "src",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        logits = self.score_grouped_candidates(
            z_dict=z_dict,
            source_index=source_index,
            candidate_index=candidate_index,
            candidate_mask=candidate_mask,
            relation=relation,
            group_axis=group_axis,
        )
        info_nce_loss = self._grouped_info_nce_loss(
            logits=logits,
            positive_mask=positive_mask,
            candidate_mask=candidate_mask,
            temperature=infonce_temperature,
            loss_mode=infonce_loss_mode,
        )
        aux_bce_loss = logits.new_zeros(())
        aux_focal_loss = logits.new_zeros(())
        valid_logits = logits[candidate_mask]
        valid_targets = positive_mask[candidate_mask].to(
            device=logits.device,
            dtype=logits.dtype,
        )

        if float(aux_bce_weight) > 0.0:
            pos_mask = valid_targets > 0.5
            neg_mask = ~pos_mask
            if torch.any(pos_mask) and torch.any(neg_mask):
                pos_loss = F.binary_cross_entropy_with_logits(
                    valid_logits[pos_mask],
                    valid_targets[pos_mask],
                    reduction="mean",
                )
                neg_loss = F.binary_cross_entropy_with_logits(
                    valid_logits[neg_mask],
                    valid_targets[neg_mask],
                    reduction="mean",
                )
                aux_bce_loss = 0.5 * (pos_loss + neg_loss)
            else:
                aux_bce_loss = F.binary_cross_entropy_with_logits(
                    valid_logits,
                    valid_targets,
                    reduction="mean",
                )

        if bool(aux_focal_enabled) and float(aux_focal_weight) > 0.0:
            bce_per_elem = F.binary_cross_entropy_with_logits(
                valid_logits,
                valid_targets,
                reduction="none",
            )
            probs = torch.sigmoid(valid_logits)
            pt = torch.where(valid_targets > 0.5, probs, 1.0 - probs)
            alpha_t = torch.where(
                valid_targets > 0.5,
                torch.full_like(valid_targets, float(aux_focal_alpha)),
                torch.full_like(valid_targets, 1.0 - float(aux_focal_alpha)),
            )
            focal_factor = torch.pow(1.0 - pt.clamp(min=1e-8, max=1.0), float(aux_focal_gamma))
            aux_focal_loss = (alpha_t * focal_factor * bce_per_elem).mean()

        total_loss = (
            info_nce_loss
            + float(aux_bce_weight) * aux_bce_loss
            + float(aux_focal_weight) * aux_focal_loss
        )
        components: Dict[str, torch.Tensor] = {
            "info_nce": info_nce_loss.detach(),
            "aux_bce": aux_bce_loss.detach(),
        }
        if bool(aux_focal_enabled) and float(aux_focal_weight) > 0.0:
            components["aux_focal"] = aux_focal_loss.detach()
        return total_loss, components, logits
