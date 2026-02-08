#!/bin/bash
# Script para rodar os 3 experimentos no dataset interno
# 
# Experimento 1: Avaliar modelo pré-treinado (clip_vit_b32_combined_tw05) no dataset interno
# Experimento 2: Fine-tuning no dataset interno (mesma configuração)
# Experimento 3: CLIP Frozen (baseline)
#
# Uso:
#   bash scripts/run_internal_experiments.sh
#   bash scripts/run_internal_experiments.sh --test-mode  # Para teste rápido
#   bash scripts/run_internal_experiments.sh --epochs 5   # Customizar épocas

set -e

echo "============================================================"
echo "🏢 EXPERIMENTOS NO DATASET INTERNO"
echo "============================================================"

# Argumentos extras
EXTRA_ARGS="$@"

# Verificar se o dataset interno existe
if [ ! -d "datasets/internal-dataset" ]; then
    echo "❌ ERRO: Dataset interno não encontrado em datasets/internal-dataset"
    exit 1
fi

echo "📂 Dataset interno encontrado!"
echo "   - train.csv: $(wc -l < datasets/internal-dataset/train.csv) linhas"
echo "   - val.csv: $(wc -l < datasets/internal-dataset/val.csv) linhas"
echo ""

# Criar diretório de experimentos internos
mkdir -p experiments/internal

# ============================================================
# Experimento 1: Avaliar modelo pré-treinado
# ============================================================
echo ""
echo "============================================================"
echo "🔬 EXPERIMENTO 1: Avaliar modelo pré-treinado no dataset interno"
echo "============================================================"
echo "   Modelo: clip_vit_b32_combined_tw05 (treinado no dataset original)"
echo "   Modo: Apenas avaliação (sem treino)"
echo ""

# Verificar se o checkpoint existe
if [ -d "experiments/clip_vit_b32_combined_tw05/checkpoints/best" ]; then
    python scripts/run_experiment.py \
        --config configs/internal/1_evaluate_pretrained_model.yaml \
        --pretrained-checkpoint experiments/clip_vit_b32_combined_tw05/checkpoints/best \
        --val-csv datasets/internal-dataset/val.csv \
        --eval-only \
        $EXTRA_ARGS
    echo "✅ Experimento 1 concluído!"
else
    echo "⚠️ AVISO: Checkpoint do modelo pré-treinado não encontrado!"
    echo "   Esperado: experiments/clip_vit_b32_combined_tw05/checkpoints/best"
    echo "   Pulando experimento 1..."
fi

# ============================================================
# Experimento 2: Fine-tuning no dataset interno
# ============================================================
echo ""
echo "============================================================"
echo "🏋️ EXPERIMENTO 2: Fine-tuning no dataset interno"
echo "============================================================"
echo "   Modelo: CLIP ViT-B/32 (iniciando do zero)"
echo "   Configuração: Mesma do clip_vit_b32_combined_tw05"
echo "   Loss: InfoNCE + Triplet (weight=0.5)"
echo "   Split: 85:15 por indivíduos (do train.csv)"
echo ""

# NOTA: NÃO passamos --val-csv aqui!
# O train.csv será dividido internamente 85:15 por indivíduos
python scripts/run_experiment.py \
    --config configs/internal/2_finetune_internal.yaml \
    $EXTRA_ARGS

echo "✅ Experimento 2 concluído!"

# ============================================================
# Experimento 3: CLIP Frozen (baseline)
# ============================================================
echo ""
echo "============================================================"
echo "❄️ EXPERIMENTO 3: CLIP Frozen (baseline)"
echo "============================================================"
echo "   Modelo: CLIP ViT-B/32 original (sem fine-tuning)"
echo "   Modo: Apenas avaliação"
echo ""

python scripts/run_experiment.py \
    --config configs/internal/3_clip_frozen_eval.yaml \
    --val-csv datasets/internal-dataset/val.csv \
    --eval-only \
    $EXTRA_ARGS

echo "✅ Experimento 3 concluído!"

# ============================================================
# Resumo
# ============================================================
echo ""
echo "============================================================"
echo "📊 RESUMO DOS EXPERIMENTOS"
echo "============================================================"
echo ""
echo "Experimentos salvos em: experiments/internal/"
echo ""
echo "📁 Estrutura:"
echo "   experiments/internal/"
echo "   ├── internal_eval_pretrained/   (Exp 1: modelo pré-treinado)"
echo "   ├── internal_finetune_combined/ (Exp 2: fine-tuning)"
echo "   └── internal_clip_frozen/       (Exp 3: baseline frozen)"
echo ""
echo "🎯 Para comparar resultados:"
echo "   cat experiments/internal/*/final_metrics.json | jq ."
echo ""
echo "✅ Todos os experimentos concluídos!"

