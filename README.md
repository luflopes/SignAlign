# 📝 SignAlign

Framework modular para alinhamento multimodal texto-assinatura usando modelos CLIP/SigLIP.

## 🎯 Objetivo

Treinar e avaliar modelos de aprendizado multimodal para **recuperação de assinaturas manuscritas a partir de nomes textuais**.

Dado um nome (texto) e um conjunto de assinaturas candidatas, o modelo seleciona a assinatura correta.

## 🏗️ Estrutura

```
src/
├── config/           # Configurações via dataclasses + YAML
├── data/             # Dataset, augmentations, batch builder
├── models/           # Wrappers para CLIP, TinyCLIP, SigLIP
├── losses/           # InfoNCE, Sigmoid, Triplet, Combined
├── training/         # Trainer e schedulers
├── evaluation/       # Métricas e retrieval
├── explainability/   # Attention rollout maps
├── inference/        # Predictor e visualizações
└── utils/            # Seed, logging

configs/
├── base.yaml         # Configuração base
└── grid/             # 40 configs para grid search

scripts/
├── run_experiment.py       # Executa um experimento
├── run_all_experiments.sh  # Executa grid completo
└── generate_experiment_grid.py

notebooks/
├── SignAlign.ipynb   # Experimentos originais
└── analysis.ipynb    # Análise e comparação
```

## 🚀 Quick Start

```bash
# Instalar dependências
pip install -r requirements.txt

# Executar experimento
python scripts/run_experiment.py --config configs/base.yaml

# Modo teste (poucas amostras)
python scripts/run_experiment.py --config configs/base.yaml --max-samples 100 --epochs 2
```

## 📊 Métricas

- **Retrieval Accuracy**: 1, 2 e 3 exemplos negativos
- **EER / AUC**: Equal Error Rate e Area Under Curve
- **Recall@k**: k = 1, 5, 10
- **MRR / NDCG**: Mean Reciprocal Rank e Normalized DCG

## 🔬 Grid de Experimentos

| Modelo | Variações |
|--------|-----------|
| TinyCLIP | Frozen, InfoNCE, Combined (λ = 0.1-0.5) |
| CLIP ViT-B/32 | Frozen, InfoNCE, Combined (λ = 0.1-0.5) |
| SigLIP | Frozen, Sigmoid, Combined (λ = 0.1-0.5) |

Cada variação disponível com e sem data augmentation.

```bash
# Executar grid completo
bash scripts/run_all_experiments.sh
```

## 🧪 Inferência

```python
from src.inference import SignaturePredictor

predictor = SignaturePredictor.from_experiment("experiments/meu_experimento")
sim = predictor.compute_similarity("João Silva", "assinatura.png")
```

## 📚 Referências

- [CLIP](https://arxiv.org/abs/2103.00020)
- [SigLIP](https://arxiv.org/abs/2303.15343)
- [TinyCLIP](https://arxiv.org/abs/2309.12314)

## 📝 Licença

Projeto de pesquisa - Mestrado UFPR.
