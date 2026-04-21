# SignAlign

SignAlign is a modular framework for multimodal alignment between textual identities and handwritten signatures using vision-language models such as CLIP and SigLIP.

<p align="center">
  <img src="docs/images/signature_retrieval_pipeline.png" alt="Identity-Conditioned Signature Retrieval Pipeline" width="800"/>
</p>
<p align="center">
  <em>Figure: Identity-Conditioned Signature Retrieval pipeline integrated into a document processing workflow.</em>
</p>

## Objective

The goal of this project is to train and evaluate multimodal learning models for the task of **retrieving handwritten signatures conditioned on textual identity**.

Given a textual query (e.g., a person’s name) and a set of candidate signature images, the model aims to identify the signature that corresponds to the given identity.

## Repository Structure

```
src/
├── config/           # Configuration management (dataclasses + YAML)
├── data/             # Dataset handling, augmentations, batch construction
├── models/           # Wrappers for CLIP, TinyCLIP, and SigLIP
├── losses/           # Loss functions (InfoNCE, Sigmoid, Triplet, Combined)
├── training/         # Training loops and schedulers
├── evaluation/       # Metrics and retrieval evaluation
├── explainability/   # Attention rollout and interpretability tools
├── inference/        # Prediction and visualization utilities
└── utils/            # Reproducibility, logging, and utilities

configs/
├── base.yaml         # Base configuration
└── grid/             # Configurations for grid search experiments

scripts/
├── run_experiment.py       # Execute a single experiment
├── run_all_experiments.sh  # Run full experimental grid
└── generate_experiment_grid.py

notebooks/
├── SignAlign.ipynb   # Original experiments
└── analysis.ipynb    # Result analysis and comparisons
```

## Installation and Usage

### Installation

```bash
pip install -r requirements.txt
```

### Running Experiments

```bash
python scripts/run_experiment.py --config configs/base.yaml
```

### Debug Mode (Reduced Dataset)

```bash
python scripts/run_experiment.py --config configs/base.yaml --max-samples 100 --epochs 2
```

## Evaluation Metrics

The framework supports a range of evaluation metrics commonly used in retrieval and verification tasks:

- Retrieval Accuracy (with 1, 2, and 3 negative samples)
- Equal Error Rate (EER)
- Area Under the Curve (AUC)
- Recall@k (k = 1, 5, 10)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG)

## Experimental Grid

The framework includes a predefined grid of experiments exploring different models, training strategies, and loss functions:

| Model           | Variants                                      |
|-----------------|-----------------------------------------------|
| TinyCLIP        | Frozen, InfoNCE, Combined (λ = 0.1–0.5)       |
| CLIP ViT-B/32   | Frozen, InfoNCE, Combined (λ = 0.1–0.5)       |
| SigLIP          | Frozen, Sigmoid, Combined (λ = 0.1–0.5)       |

Each configuration is evaluated with and without data augmentation.

To execute the full experimental grid:

```bash
bash scripts/run_all_experiments.sh
```

## Inference

Example usage for computing similarity between a textual identity and a signature image:

```python
from src.inference import SignaturePredictor

predictor = SignaturePredictor.from_experiment("experiments/your_experiment")
similarity = predictor.compute_similarity("João Silva", "signature.png")
```

## Dataset

The dataset used in this work is designed to reflect real-world document processing scenarios, including multiple candidate signatures per document and variability in layout and content.

Access to the dataset is available upon request or through the following link:

**[Dataset Link Placeholder]**

(Replace this placeholder with the official dataset URL or access instructions.)

## References

- CLIP: https://arxiv.org/abs/2103.00020  
- SigLIP: https://arxiv.org/abs/2303.15343  
- TinyCLIP: https://arxiv.org/abs/2309.12314  

## License

This project is developed as part of a Master's research at the Federal University of Paraná (UFPR). Licensing terms will be defined upon public release.
