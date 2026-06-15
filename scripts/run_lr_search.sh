#!/bin/bash
# Busca de Learning Rate (finetuning simples, sem augmentação/combined).
#
# Uso:
#   ./scripts/run_lr_search.sh                 # Execução completa
#   ./scripts/run_lr_search.sh --test-mode     # Modo teste rápido

set -e

EXTRA_ARGS="$@"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

OUTPUT_DIR="experiments/lr_search"

echo "====================================="
echo "Busca de LR: 9 experimentos"
echo "====================================="

CONFIGS=(
    "tinyclip_lr1e4"
    "tinyclip_lr1e5"
    "tinyclip_lr1e6"
    "clip_vit_b32_lr1e4"
    "clip_vit_b32_lr1e5"
    "clip_vit_b32_lr1e6"
    "siglip_lr1e4"
    "siglip_lr1e5"
    "siglip_lr1e6"
)

TOTAL=${#CONFIGS[@]}
COUNT=0
FAILED=0

for CONFIG in "${CONFIGS[@]}"; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "[$COUNT/$TOTAL] Executando: $CONFIG"
    echo "====================================="
    
    if python scripts/run_experiment.py --config "configs/lr_search/${CONFIG}.yaml" --output-dir "$OUTPUT_DIR" $EXTRA_ARGS; then
        echo "✅ $CONFIG concluído"
    else
        echo "❌ $CONFIG FALHOU"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "====================================="
if [ $FAILED -eq 0 ]; then
    echo "🎉 Busca de LR concluída! Rode: python scripts/analyze_lr_search.py"
else
    echo "⚠️ $FAILED de $TOTAL experimentos falharam"
fi
echo "====================================="
