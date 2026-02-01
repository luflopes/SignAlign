#!/bin/bash
# Script para executar todos os experimentos do grid
# 
# Uso:
#   ./scripts/run_all_experiments.sh                    # Execução completa
#   ./scripts/run_all_experiments.sh --test-mode        # Modo teste (50 amostras, 1 época)
#   ./scripts/run_all_experiments.sh --max-samples 100  # Limitar amostras
#   ./scripts/run_all_experiments.sh --max-samples 200 --epochs 3  # Customizado

set -e  # Parar em caso de erro

# Capturar todos os argumentos para passar aos experimentos
EXTRA_ARGS="$@"

# Ativar ambiente virtual se existir
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "====================================="
echo "Executando 40 experimentos"
if [ -n "$EXTRA_ARGS" ]; then
    echo "Argumentos extras: $EXTRA_ARGS"
fi
echo "====================================="

# Lista de configurações
CONFIGS=(
    "tinyclip_frozen_eval"
    "clip_vit_b32_frozen_eval"
    "clip_vit_b16_frozen_eval"
    "siglip_frozen_eval"
    "tinyclip_infonce"
    "tinyclip_combined_tw01"
    "tinyclip_combined_tw02"
    "tinyclip_combined_tw03"
    "tinyclip_combined_tw04"
    "tinyclip_combined_tw05"
    "tinyclip_infonce_noaug"
    "tinyclip_combined_tw01_noaug"
    "tinyclip_combined_tw02_noaug"
    "tinyclip_combined_tw03_noaug"
    "tinyclip_combined_tw04_noaug"
    "tinyclip_combined_tw05_noaug"
    "clip_vit_b32_infonce"
    "clip_vit_b32_combined_tw01"
    "clip_vit_b32_combined_tw02"
    "clip_vit_b32_combined_tw03"
    "clip_vit_b32_combined_tw04"
    "clip_vit_b32_combined_tw05"
    "clip_vit_b32_infonce_noaug"
    "clip_vit_b32_combined_tw01_noaug"
    "clip_vit_b32_combined_tw02_noaug"
    "clip_vit_b32_combined_tw03_noaug"
    "clip_vit_b32_combined_tw04_noaug"
    "clip_vit_b32_combined_tw05_noaug"
    "siglip_sigmoid"
    "siglip_combined_tw01"
    "siglip_combined_tw02"
    "siglip_combined_tw03"
    "siglip_combined_tw04"
    "siglip_combined_tw05"
    "siglip_sigmoid_noaug"
    "siglip_combined_tw01_noaug"
    "siglip_combined_tw02_noaug"
    "siglip_combined_tw03_noaug"
    "siglip_combined_tw04_noaug"
    "siglip_combined_tw05_noaug"
)

TOTAL=${#CONFIGS[@]}
COUNT=0
FAILED=0

for CONFIG in "${CONFIGS[@]}"; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "[$COUNT/$TOTAL] Executando: $CONFIG"
    echo "====================================="
    
    if python scripts/run_experiment.py --config "configs/grid/${CONFIG}.yaml" $EXTRA_ARGS; then
        echo "✅ $CONFIG concluído"
    else
        echo "❌ $CONFIG FALHOU"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "====================================="
if [ $FAILED -eq 0 ]; then
    echo "🎉 Todos os $TOTAL experimentos concluídos com sucesso!"
else
    echo "⚠️ $FAILED de $TOTAL experimentos falharam"
fi
echo "====================================="
