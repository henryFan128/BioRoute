# BioRoute

BioRoute is a biomedical heterogeneous-graph learning framework for drug-disease treatment prediction. The current release focuses on the supervised relation:


## Pretrained Feature Models

Download pretrained models manually from Hugging Face. Suggested local placement:

```text
data/pretrained/
```

| Feature type | Model | URL |
| --- | --- | --- |
| Protein sequence | ESM-2 `facebook/esm2_t33_650M_UR50D` | https://huggingface.co/facebook/esm2_t33_650M_UR50D |
| Disease text | BioClinicalBERT `emilyalsentzer/Bio_ClinicalBERT` | https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT |
| Chemical SMILES | ChemBERTa `seyonec/ChemBERTa-zinc-base-v1` | https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1 |

## Installation

The provided `requirements.txt` is an environment export from the development environment. CUDA, PyTorch, and PyTorch Geometric packages may need to be adapted to the target machine.

```bash
conda create --name bioroute --file requirements.txt
conda activate bioroute
```

## Training

Default command:

```bash
python script/train.py --config config/train.yaml
```

## Repository Layout

```text
BioRoute/
├── src/
│   ├── model/                   # KGC model, SRR layers, relation encoders, decoders, router modules
│   ├── utils/                   # Ranking, evaluation, and distributed-training utilities
│   ├── trainer.py               # Core training and evaluation loop
│   ├── feature_dr_extra.py      # Chemical features: ChemBERTa and Morgan fingerprints
│   ├── feature_di_extra.py      # Disease text features: BioClinicalBERT
│   ├── feature_pr_extra.py      # Protein sequence features: ESM-2
│   ├── feature_ge_extra.py      # Gene GO multi-hot features and GOResMLP encoder
│   └── feature_pa_extra.py      # Pathogen taxonomy multi-hot features
├── script/
│   ├── train.py                 # Main training/evaluation entry point
│   └── negative_sampler.py      # negative sampler for chemical-disease supervision
├── config/
│   └── train.yaml               # Default experiment configuration
├── data/                        # Local graph/features directory
├── requirements.txt             # Development environment export
└── README.md                    # Project overview
```
