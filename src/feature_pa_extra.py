import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from .feature_ge_extra import GOResMLP, build_multihot  # noqa: F401 re-export

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_taxo_vocab(vocab_path: str) -> Dict[str, int]:
    """Load taxonomy vocabulary and return term → index mapping.

    Args:
        vocab_path: Path to ``taxo_vocab.json``.

    Returns:
        Dictionary mapping taxonomy term IDs to integer indices.
    """
    import json

    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    # Support nested formats produced by different build scripts:
    #   "taxo2idx" — new canonical key
    #   "tax2idx"  — key used by build_taxonomy_vocab() in notebooks
    #   "go2idx"   — shared-format fallback
    #   flat dict  — {term: idx} directly
    if "taxo2idx" in vocab:
        return vocab["taxo2idx"]
    if "tax2idx" in vocab:
        return vocab["tax2idx"]
    if "go2idx" in vocab:
        return vocab["go2idx"]
    # Flat dict: must be {term: idx} with no wrapper keys
    if all(isinstance(v, int) for v in vocab.values()):
        return vocab
    raise ValueError(
        f"Cannot parse taxonomy vocab at {vocab_path!r}: "
        "expected keys 'taxo2idx', 'tax2idx', or 'go2idx', "
        f"got top-level keys {list(vocab.keys())}"
    )


class PathogenTaxoExtractor:
    """Extract pathogen features from NCBI taxonomy multi-hot encodings via TaxoResMLP.

    Architecture (identical to GeneGOExtractor):
        taxonomy multi-hot (vocab_size) → GOResMLP → out_dim (768)

    Interface mirrors GeneGOExtractor / BioClinicalBERTExtractor so it can be
    plugged directly into ``feature_extractor.py`` without changes to the caller.
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
            vocab_path:          Path to ``taxo_vocab.json``.
            hidden_dim:          Hidden dimension inside ``GOResMLP``.
            out_dim:             Output feature dimension (default 768, aligned with Disease/Gene).
            dropout:             Dropout probability.
            model_weights_path:  Optional path to a pre-saved ``.pt`` file for the encoder.
        """
        import torch

        self.taxo2idx = load_taxo_vocab(vocab_path)
        self.vocab_size = len(self.taxo2idx)
        self.out_dim = out_dim
        logger.info(f"Loaded taxonomy vocabulary: {self.vocab_size} terms")

        # Reuse GOResMLP architecture (Dropout(BN(ReLU(Wx+b))) + residual)
        self.model = GOResMLP(
            vocab_size=self.vocab_size,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
        )

        if model_weights_path and os.path.exists(model_weights_path):
            state = torch.load(model_weights_path, map_location="cpu")
            self.model.load_state_dict(state)
            logger.info(f"Loaded TaxoResMLP weights from: {model_weights_path}")
        elif model_weights_path:
            logger.warning(
                f"Weights not found at {model_weights_path}. Using random initialisation."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"TaxoResMLP loaded on device: {self.device}")

    # ------------------------------------------------------------------
    def _build_multihot_matrix(
        self,
        pathogen_taxo_map: Dict[str, List[str]],
    ) -> tuple:
        """Build taxonomy multi-hot matrix for all pathogens.

        Args:
            pathogen_taxo_map: {pathogen_id: [taxonomy_term, ...]}

        Returns:
            (pathogen_ids, matrix) where matrix has shape [N, vocab_size].
        """
        pathogen_ids = list(pathogen_taxo_map.keys())
        matrix = np.stack(
            [
                build_multihot(pathogen_taxo_map[pid], self.taxo2idx, self.vocab_size)
                for pid in pathogen_ids
            ],
            axis=0,
        )  # [N, vocab_size]
        return pathogen_ids, matrix

    def batch_extract_features(
        self,
        pathogen_taxo_map: Dict[str, List[str]],
        batch_size: int = 256,
    ) -> Dict[str, np.ndarray]:
        """Encode pathogens in batches.

        Args:
            pathogen_taxo_map: {pathogen_id: [taxonomy term list]}
            batch_size:        Number of pathogens processed per batch.

        Returns:
            {pathogen_id: np.ndarray shape [out_dim]}
        """
        import torch

        pathogen_ids, matrix = self._build_multihot_matrix(pathogen_taxo_map)
        n = len(pathogen_ids)
        all_features = []

        for start in tqdm(range(0, n, batch_size), desc="Encoding taxonomy features"):
            end = min(start + batch_size, n)
            batch = torch.tensor(matrix[start:end], dtype=torch.float32).to(
                self.device
            )  # [B, vocab_size]

            with torch.no_grad():
                feats = self.model(batch)  # [B, out_dim]

            all_features.append(feats.cpu().numpy())

        all_features_np = np.concatenate(all_features, axis=0)  # [N, out_dim]
        return {pid: all_features_np[i] for i, pid in enumerate(pathogen_ids)}

    def extract_features(
        self,
        pathogens_dict: Dict[str, dict],
        batch_size: int = 256,
    ) -> Dict[str, np.ndarray]:
        """Extract pathogen features from a pathogens dictionary.

        Args:
            pathogens_dict: {pathogen_id: {"name": ..., "taxonomy_terms": ["term1", ...]}}
            batch_size:     Encoding batch size.

        Returns:
            {pathogen_id: np.ndarray shape [out_dim]}
        """
        pathogen_taxo_map: Dict[str, List[str]] = {}
        missing_pathogens: List[str] = []

        for pathogen_id, info in pathogens_dict.items():
            taxo_terms = info.get("taxonomy_terms", [])
            if taxo_terms:
                pathogen_taxo_map[pathogen_id] = taxo_terms
            else:
                missing_pathogens.append(pathogen_id)
                logger.warning(
                    f"Pathogen {pathogen_id} has no taxonomy terms, will use default features."
                )

        if pathogen_taxo_map:
            features = self.batch_extract_features(pathogen_taxo_map, batch_size)
        else:
            features = {}

        if missing_pathogens:
            logger.info(
                f"Generating default features for {len(missing_pathogens)} "
                "pathogens with missing taxonomy annotations."
            )
            if features:
                default_feat = np.mean(
                    np.stack(list(features.values())), axis=0
                )  # [out_dim]
            else:
                default_feat = np.zeros(self.out_dim, dtype=np.float32)  # [out_dim]

            for pathogen_id in missing_pathogens:
                features[pathogen_id] = default_feat + np.random.normal(
                    0, 0.01, self.out_dim
                ).astype(np.float32)

        return features
