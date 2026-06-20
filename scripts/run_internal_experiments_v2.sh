#!/bin/bash
# Estudo do dataset interno com os 3 melhores modelos (CLIP, TinyCLIP, SigLIP),
# em 3 esquemas cada: frozen, pretrained (transfer) e finetuned -> 9 experimentos.
#
#   frozen     : modelo base off-the-shelf, encoders congelados, eval-only no val interno
#   pretrained : melhor checkpoint público do modelo (transfer), eval-only no val interno
#   finetuned  : finetuning no train.csv interno com a melhor config do modelo
#
# O dataset interno usa train.csv/val.csv separados (sem split fixo). As métricas
# ficam em experiments/internal/<name>/metrics/val_history.json (não há test split).
#
# Uso:
#   bash scripts/run_internal_experiments_v2.sh
#   bash scripts/run_internal_experiments_v2.sh --test-mode      # smoke test rápido
#   bash scripts/run_internal_experiments_v2.sh --epochs 5       # repassado ao run_experiment.py
#   bash scripts/run_internal_experiments_v2.sh tinyclip         # roda só um modelo
#   bash scripts/run_internal_experiments_v2.sh siglip --test-mode

set -e

CONFIG_DIR="configs/internal"
VAL_CSV="datasets/internal-dataset/val.csv"

# Checkpoint público de cada modelo (esquema pretrained/transfer).
# 1ª escolha: melhor seed do grid de significância; fallback: rodada principal do grid.
declare -A PRETRAINED_CKPT=(
    [clip_vit_b32]="experiments/significance/seed_1/clip_vit_b32_combined_tw05/checkpoints/best"
    [tinyclip]="experiments/significance/seed_2/tinyclip_combined_tw03/checkpoints/best"
    [siglip]="experiments/significance/seed_1/siglip_combined_tw03/checkpoints/best"
)
declare -A PRETRAINED_CKPT_FALLBACK=(
    [clip_vit_b32]="experiments/clip_vit_b32_combined_tw05/checkpoints/best"
    [tinyclip]="experiments/tinyclip_combined_tw03/checkpoints/best"
    [siglip]="experiments/siglip_combined_tw03/checkpoints/best"
)

# Retorna (via echo) o primeiro checkpoint existente para o modelo, ou vazio.
resolve_ckpt() {
    local model="$1"
    if [ -d "${PRETRAINED_CKPT[$model]}" ]; then
        echo "${PRETRAINED_CKPT[$model]}"
    elif [ -d "${PRETRAINED_CKPT_FALLBACK[$model]}" ]; then
        echo "${PRETRAINED_CKPT_FALLBACK[$model]}"
    else
        echo ""
    fi
}

# Seleção opcional de modelo(s) via 1º argumento posicional
MODELS=("clip_vit_b32" "tinyclip" "siglip")
if [[ -n "$1" && "$1" != --* ]]; then
    MODELS=("$1")
    shift
fi
EXTRA_ARGS="$@"

echo "============================================================"
echo "🏢 DATASET INTERNO — 3 modelos × {frozen, pretrained, finetuned}"
echo "============================================================"

if [ ! -d "datasets/internal-dataset" ]; then
    echo "❌ ERRO: dataset interno não encontrado em datasets/internal-dataset"
    exit 1
fi
echo "📂 train.csv: $(wc -l < datasets/internal-dataset/train.csv) linhas | val.csv: $(wc -l < $VAL_CSV) linhas"
mkdir -p experiments/internal

run_scheme() {
    local model="$1"; local scheme="$2"
    local config="${CONFIG_DIR}/${model}_${scheme}.yaml"
    echo ""
    echo "------------------------------------------------------------"
    echo "▶️  ${model} | ${scheme}"
    echo "------------------------------------------------------------"
    if [ ! -f "$config" ]; then
        echo "⚠️  config não encontrado: $config — pulando."
        return
    fi

    case "$scheme" in
        frozen)
            python scripts/run_experiment.py \
                --config "$config" \
                --val-csv "$VAL_CSV" \
                --eval-only \
                $EXTRA_ARGS
            ;;
        pretrained)
            local ckpt
            ckpt="$(resolve_ckpt "$model")"
            if [ -z "$ckpt" ]; then
                echo "⚠️  checkpoint público não encontrado para ${model}:"
                echo "      ${PRETRAINED_CKPT[$model]}"
                echo "      ${PRETRAINED_CKPT_FALLBACK[$model]}"
                echo "    (esquema 'pretrained' de ${model} pulado — verifique o caminho/seed)"
                return
            fi
            echo "   checkpoint: $ckpt"
            python scripts/run_experiment.py \
                --config "$config" \
                --pretrained-checkpoint "$ckpt" \
                --val-csv "$VAL_CSV" \
                --eval-only \
                $EXTRA_ARGS
            ;;
        finetuned)
            python scripts/run_experiment.py \
                --config "$config" \
                --val-csv "$VAL_CSV" \
                $EXTRA_ARGS
            ;;
    esac
    echo "✅ ${model} | ${scheme} concluído."
}

for model in "${MODELS[@]}"; do
    echo ""
    echo "############################################################"
    echo "# MODELO: ${model}"
    echo "############################################################"
    run_scheme "$model" "frozen"
    run_scheme "$model" "pretrained"
    run_scheme "$model" "finetuned"
done

echo ""
echo "============================================================"
echo "📊 RESUMO"
echo "============================================================"
echo "Resultados em: experiments/internal/internal_<modelo>_<esquema>/"
echo "Métricas (sem test split): experiments/internal/*/metrics/val_history.json"
echo "Analise no notebook: seção '🏢 Análise do Dataset Interno' (analysis_test.ipynb)"
echo "✅ Concluído."
