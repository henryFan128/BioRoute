import json
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MLP block: Dropout(BN(ReLU(Wx + b)))
# ---------------------------------------------------------------------------


class MLPBlock(nn.Module):
    """Single MLP block: Linear → BN → ReLU → Dropout.

    Implements:
        MLP(x) = Dropout(BN(ReLU(W·x + b)))
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor. Shape: [N, in_dim]

        Returns:
            Tensor of shape [N, out_dim]
        """
        return self.dropout(self.relu(self.bn(self.linear(x))))  # [N, out_dim]


# ---------------------------------------------------------------------------
# ResMLP encoder
# ---------------------------------------------------------------------------


class GOResMLP(nn.Module):
    """GO multi-hot → MLP → ResMLP encoder.

    Architecture:
        y_mlp     = MLP(x)                   # first MLP block
        y_resmlp  = y_mlp + MLP(y_mlp)       # residual connection
        out       = proj(y_resmlp)            # optional projection to out_dim
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        out_dim: int = 768,
        dropout: float = 0.2,
    ) -> None:
        """
        Args:
            vocab_size:  Size of GO vocabulary (multi-hot input dimension).
            hidden_dim:  Intermediate hidden dimension.
            out_dim:     Final output feature dimension (should match other node types).
            dropout:     Dropout probability.
        """
        super().__init__()

        # First MLP block: vocab_size → hidden_dim
        self.mlp1 = MLPBlock(vocab_size, hidden_dim, dropout)

        # Second MLP block inside ResMLP: hidden_dim → hidden_dim
        self.mlp2 = MLPBlock(hidden_dim, hidden_dim, dropout)

        # Final linear projection: hidden_dim → out_dim
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: GO multi-hot encodings. Shape: [N, vocab_size]

        Returns:
            Gene feature vectors. Shape: [N, out_dim]
        """
        y_mlp = self.mlp1(x)  # [N, hidden_dim]
        y_resmlp = y_mlp + self.mlp2(y_mlp)  # [N, hidden_dim]  residual
        out = self.proj(y_resmlp)  # [N, out_dim]
        return out


# ---------------------------------------------------------------------------
# GO vocab helper
# ---------------------------------------------------------------------------


def load_go_vocab(vocab_path: str) -> Dict[str, int]:
    """Load GO vocabulary and return go_id → index mapping.

    Args:
        vocab_path: Path to ``go_vocab.json``.

    Returns:
        Dictionary mapping GO term IDs to integer indices.
    """
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    return vocab["go2idx"]  # {GO:xxxxxxx: int_index}


def build_multihot(
    go_terms: List[str],
    go2idx: Dict[str, int],
    vocab_size: int,
) -> np.ndarray:
    """Build a single GO multi-hot vector.

    Args:
        go_terms:   List of GO term IDs associated with one gene.
        go2idx:     GO term → index mapping.
        vocab_size: Total vocabulary size.

    Returns:
        Multi-hot numpy array. Shape: [vocab_size]
    """
    vec = np.zeros(vocab_size, dtype=np.float32)  # [vocab_size]
    for term in go_terms:
        idx = go2idx.get(term)
        if idx is not None:
            vec[idx] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Main extractor class  (mirrors the interface of the other *Extractor classes)
# ---------------------------------------------------------------------------


class GeneGOExtractor:
    """Extract gene features from GO annotation multi-hot encodings via ResMLP.

    Pipeline:
        1. Build GO multi-hot vectors (vocab_size = 14945).
        2. Forward-pass through ``GOResMLP`` (no gradient, eval mode).
        3. Return numpy feature vectors aligned to ``out_dim`` (default 768).
    """

    def __init__(
        self,
        vocab_path: str,
        hidden_dim: int = 512,
        out_dim: int = 768,
        dropout: float = 0.2,
        model_weights_path: Optional[str] = None,
    ) -> None:
        """
        Args:
            vocab_path:          Path to ``go_vocab.json``.
            hidden_dim:          Hidden dimension inside ``GOResMLP``.
            out_dim:             Output feature dimension (should match other nodes, default 768).
            dropout:             Dropout probability used during forward pass.
            model_weights_path:  Optional path to a pre-saved ``GOResMLP`` ``.pt`` file.
                                 If ``None``, the model is randomly initialised (useful when
                                 the encoder is trained end-to-end inside ``KGCModel``).
        """
        self.go2idx = load_go_vocab(vocab_path)
        self.vocab_size = len(self.go2idx)
        self.out_dim = out_dim
        logger.info(f"Loaded GO vocabulary: {self.vocab_size} terms")

        self.model = GOResMLP(
            vocab_size=self.vocab_size,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
        )

        if model_weights_path and os.path.exists(model_weights_path):
            state = torch.load(model_weights_path, map_location="cpu")
            self.model.load_state_dict(state)
            logger.info(f"Loaded GOResMLP weights from: {model_weights_path}")
        else:
            if model_weights_path:
                logger.warning(
                    f"Model weights not found at {model_weights_path}. "
                    "Using random initialisation."
                )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"GOResMLP loaded on device: {self.device}")

    # ------------------------------------------------------------------
    # Core encoding methods
    # ------------------------------------------------------------------

    def _build_multihot_matrix(
        self,
        gene_go_map: Dict[str, List[str]],
    ) -> tuple:
        """Build multi-hot matrix for all genes.

        Args:
            gene_go_map: {gene_id: [GO:xxxxxxx, ...]}

        Returns:
            Tuple of (gene_ids, multihot_matrix).
            ``multihot_matrix`` has shape [N, vocab_size].
        """
        gene_ids = list(gene_go_map.keys())
        matrix = np.stack(
            [
                build_multihot(gene_go_map[gid], self.go2idx, self.vocab_size)
                for gid in gene_ids
            ],
            axis=0,
        )  # [N, vocab_size]
        return gene_ids, matrix

    def batch_extract_features(
        self,
        gene_go_map: Dict[str, List[str]],
        batch_size: int = 256,
    ) -> Dict[str, np.ndarray]:
        """Encode genes in batches.

        Args:
            gene_go_map: {gene_id: [GO term list]}
            batch_size:  Number of genes processed per batch.

        Returns:
            {gene_id: feature_vector (numpy, shape [out_dim])}
        """
        gene_ids, matrix = self._build_multihot_matrix(gene_go_map)
        n = len(gene_ids)
        all_features: List[np.ndarray] = []

        for start in tqdm(range(0, n, batch_size), desc="Encoding GO features"):
            end = min(start + batch_size, n)
            batch = torch.tensor(matrix[start:end], dtype=torch.float32).to(
                self.device
            )  # [B, vocab_size]

            with torch.no_grad():
                feats = self.model(batch)  # [B, out_dim]

            all_features.append(feats.cpu().numpy())

        all_features_np = np.concatenate(all_features, axis=0)  # [N, out_dim]
        return {gid: all_features_np[i] for i, gid in enumerate(gene_ids)}

    def extract_features(
        self,
        genes_dict: Dict[str, dict],
        batch_size: int = 256,
    ) -> Dict[str, np.ndarray]:
        """Extract gene features from a genes dictionary.

        This method mirrors the ``extract_features`` interface of the other
        ``*Extractor`` classes so it can be plugged into ``feature_extractor.py``.

        Args:
            genes_dict: {gene_id: {"gene_symbols": ..., "go_ids": ["GO:0000001", ...]}}
                Note: also accepts the legacy key ``go_terms`` for backward compatibility.
            batch_size: Encoding batch size.

        Returns:
            {gene_id: numpy feature vector, shape [out_dim]}
        """
        # --- Separate genes with / without GO annotations ---
        gene_go_map: Dict[str, List[str]] = {}
        missing_go_genes: List[str] = []

        for gene_id, info in genes_dict.items():
            # Accept both the new ``go_ids`` key and the legacy ``go_terms`` key
            go_terms = info.get("go_ids") or info.get("go_terms", [])
            if go_terms:
                gene_go_map[gene_id] = go_terms
            else:
                missing_go_genes.append(gene_id)
                logger.warning(
                    f"Gene {gene_id} has no GO annotations, will use default features."
                )

        # --- Encode genes that have GO terms ---
        if gene_go_map:
            features = self.batch_extract_features(gene_go_map, batch_size)
        else:
            features = {}

        # --- Handle missing genes ---
        if missing_go_genes:
            logger.info(
                f"Generating default features for {len(missing_go_genes)} genes "
                "with missing GO annotations."
            )
            if features:
                default_feat = np.mean(
                    np.stack(list(features.values())), axis=0
                )  # [out_dim]
                logger.info("Using mean of existing gene features as default.")
            else:
                default_feat = np.zeros(self.out_dim, dtype=np.float32)  # [out_dim]

            for gene_id in missing_go_genes:
                # Add tiny noise so each default vector is unique
                features[gene_id] = default_feat + np.random.normal(
                    0, 0.01, self.out_dim
                ).astype(np.float32)

        return features
