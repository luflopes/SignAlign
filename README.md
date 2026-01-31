# 📝 SignAlign

Framework modular para alinhamento multimodal texto-assinatura usando modelos CLIP/SigLIP.

## 🎯 Objetivo

Treinar e avaliar modelos de aprendizado multimodal para **recuperação de assinaturas manuscritas a partir de nomes textuais**.

Dado um nome (texto) e um conjunto de assinaturas candidatas, o modelo deve selecionar a assinatura correta.

## 🏗️ Arquitetura

```
signalign/
├── config/           # Configurações via dataclasses + YAML
├── data/             # Dataset, augmentations, batch builder
├── models/           # Wrappers para CLIP, TinyCLIP, SigLIP
├── losses/           # InfoNCE, Sigmoid, Triplet, Combined
├── training/         # Trainer e schedulers
├── evaluation/       # Métricas e retrieval
├── explainability/   # Attention rollout
├── inference/        # Predictor e visualizações
└── utils/            # Seed, logging
```

## 🚀 Quick Start

### 1. Instalação

```bash
pip install -r requirements.txt
```

### 2. Executar Experimento

```bash
python scripts/run_experiment.py --config configs/base.yaml
```

### 3. Parâmetros via CLI

```bash
python scripts/run_experiment.py \
    --config configs/base.yaml \
    --name meu_experimento \
    --epochs 30 \
    --lr 1e-5 \
    --freeze-vision
```

### 4. Grid Search

```bash
python scripts/run_grid_search.py --base-config configs/base.yaml
```

## 📊 Configurações Disponíveis

| Config | Modelo | Loss | Descrição |
|--------|--------|------|-----------|
| `base.yaml` | TinyCLIP | CE + Triplet | Baseline completo |
| `tinyclip_infonce.yaml` | TinyCLIP | InfoNCE | Sem triplet loss |
| `tinyclip_combined.yaml` | TinyCLIP | CE + Triplet | Loss combinada |
| `tinyclip_frozen.yaml` | TinyCLIP | CE + Triplet | Vision encoder congelado |
| `siglip_sigmoid.yaml` | SigLIP | Sigmoid | Loss nativa do SigLIP |
| `clip_base.yaml` | CLIP ViT-B/32 | CE + Triplet | CLIP completo |

## 📈 Métricas

- **Accuracy@k**: Proporção de queries onde o item correto está no top-k
- **EER**: Equal Error Rate
- **AUC**: Area Under ROC Curve
- **Recall@k**: Recall nos top k=1,5,10
- **MRR**: Mean Reciprocal Rank
- **NDCG**: Normalized Discounted Cumulative Gain

## 🔬 Análise

O notebook `notebooks/analysis.ipynb` carrega automaticamente todos os experimentos e permite:

1. Comparar métricas entre experimentos
2. Visualizar curvas de treinamento
3. Gerar attention maps
4. Criar matriz de similaridade gráfica

## 📁 Estrutura de Saída

```
experiments/
└── {nome_experimento}/
    ├── config.json          # Configuração do experimento
    ├── metadata.json        # Melhor época, métricas
    ├── checkpoints/
    │   ├── best/            # Melhor modelo
    │   └── epoch_XXX/       # Checkpoints periódicos
    ├── metrics/
    │   ├── train_history.json
    │   └── val_history.json
    ├── visualizations/      # Attention maps
    └── tensorboard/         # Logs TensorBoard
```

## 🧪 Uso Programático

### Inferência

```python
from signalign.inference import SignaturePredictor

predictor = SignaturePredictor.from_experiment(
    "experiments/meu_experimento",
    checkpoint="best"
)

# Similaridade texto-imagem
sim = predictor.compute_similarity("João Silva", "assinatura.png")

# Verificação
result = predictor.verify_signature(
    claimed_name="João Silva",
    signature_image="assinatura.png",
    threshold=0.15
)
```

### Matriz de Similaridade

```python
from signalign.inference import SimilarityMatrixVisualizer

visualizer = SimilarityMatrixVisualizer(predictor)
sim_matrix = visualizer.plot_similarity_matrix(
    texts=["Nome 1", "Nome 2", "Nome 3"],
    images=["img1.png", "img2.png", "img3.png"]
)
```

### Attention Rollout

```python
from signalign.explainability import visualize_attention

visualize_attention(
    model=model,
    processor=processor,
    image_path="assinatura.png",
    text="João Silva",
    device=device
)
```

## 📚 Referências

- [CLIP](https://arxiv.org/abs/2103.00020) - Learning Transferable Visual Models From Natural Language Supervision
- [SigLIP](https://arxiv.org/abs/2303.15343) - Sigmoid Loss for Language Image Pre-Training
- [TinyCLIP](https://arxiv.org/abs/2309.12314) - CLIP Distillation via Affinity Mimicking and Weight Inheritance

## 📝 Licença

Este projeto é parte de pesquisa de mestrado na UFPR.

