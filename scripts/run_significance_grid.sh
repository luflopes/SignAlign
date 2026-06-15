#!/bin/bash
# Grid de significância estatística: executa todas as configs de configs/grid/
# para múltiplas seeds, usando a melhor LR por modelo (já embutida nas configs,
# desde que tenham sido geradas com:
#   python scripts/generate_experiment_grid.py --target grid --best-lr
#
# Cada seed escreve em experiments/significance/seed_<s>/ para posterior
# agregação por scripts/analyze_significance.py.
#
# Uso:
#   ./scripts/run_significance_grid.sh                 # 40 configs x 5 seeds
#   ./scripts/run_significance_grid.sh --test-mode     # Modo teste rápido
#   SEEDS="5932 1 2" ./scripts/run_significance_grid.sh   # Seeds customizadas

set -e

EXTRA_ARGS="$@"

# Ativar ambiente virtual se existir
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Seeds para repetição (override via variável de ambiente SEEDS)
SEEDS="${SEEDS:-5932 1 2 3 4}"

BASE_OUTPUT_DIR="experiments/significance"

# Descobrir todas as configs do grid
CONFIGS=()
for cfg in configs/grid/*.yaml; do
    name=$(basename "$cfg" .yaml)
    CONFIGS+=("$name")
done

TOTAL_CONFIGS=${#CONFIGS[@]}
NUM_SEEDS=$(echo $SEEDS | wc -w)
TOTAL_RUNS=$((TOTAL_CONFIGS * NUM_SEEDS))

echo "====================================="
echo "Grid de significância"
echo "  Configs: $TOTAL_CONFIGS"
echo "  Seeds:   $SEEDS ($NUM_SEEDS)"
echo "  Total:   $TOTAL_RUNS execuções"
if [ -n "$EXTRA_ARGS" ]; then
    echo "  Args extras: $EXTRA_ARGS"
fi
echo "====================================="

COUNT=0
FAILED=0

for SEED in $SEEDS; do
    SEED_OUTPUT_DIR="${BASE_OUTPUT_DIR}/seed_${SEED}"
    echo ""
    echo "############ SEED ${SEED} -> ${SEED_OUTPUT_DIR} ############"

    for CONFIG in "${CONFIGS[@]}"; do
        COUNT=$((COUNT + 1))
        echo ""
        echo "[$COUNT/$TOTAL_RUNS] seed=$SEED config=$CONFIG"
        echo "-------------------------------------"

        if [[ "$CONFIG" == *"frozen_eval"* ]]; then
            # Experimentos frozen: apenas avaliação (remover --epochs se vier)
            FILTERED_ARGS=$(echo "$EXTRA_ARGS" | sed 's/--epochs[[:space:]]*[0-9]*//g')
            if python scripts/run_experiment.py \
                --config "configs/grid/${CONFIG}.yaml" \
                --seed "$SEED" \
                --output-dir "$SEED_OUTPUT_DIR" \
                --eval-only $FILTERED_ARGS; then
                echo "✅ $CONFIG (seed $SEED) concluído"
            else
                echo "❌ $CONFIG (seed $SEED) FALHOU"
                FAILED=$((FAILED + 1))
            fi
        else
            if python scripts/run_experiment.py \
                --config "configs/grid/${CONFIG}.yaml" \
                --seed "$SEED" \
                --output-dir "$SEED_OUTPUT_DIR" $EXTRA_ARGS; then
                echo "✅ $CONFIG (seed $SEED) concluído"
            else
                echo "❌ $CONFIG (seed $SEED) FALHOU"
                FAILED=$((FAILED + 1))
            fi
        fi
    done
done

echo ""
echo "====================================="
if [ $FAILED -eq 0 ]; then
    echo "🎉 Todas as $TOTAL_RUNS execuções concluídas!"
    echo "Rode: python scripts/analyze_significance.py"
else
    echo "⚠️ $FAILED de $TOTAL_RUNS execuções falharam"
fi
echo "====================================="
