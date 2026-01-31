#!/bin/bash
# Script gerado automaticamente para executar todos os experimentos
# Uso: ./scripts/run_all_experiments.sh

set -e  # Parar em caso de erro

# Ativar ambiente virtual se existir
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "====================================="
echo "Executando 40 experimentos"
echo "====================================="

echo ""
echo "[1/40] Executando: tinyclip_frozen_eval"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/tinyclip_frozen_eval.yaml
echo "✅ tinyclip_frozen_eval concluído"

echo ""
echo "[2/40] Executando: clip_vit_b32_frozen_eval"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/clip_vit_b32_frozen_eval.yaml
echo "✅ clip_vit_b32_frozen_eval concluído"

echo ""
echo "[3/40] Executando: clip_vit_b16_frozen_eval"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/clip_vit_b16_frozen_eval.yaml
echo "✅ clip_vit_b16_frozen_eval concluído"

echo ""
echo "[4/40] Executando: siglip_frozen_eval"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/siglip_frozen_eval.yaml
echo "✅ siglip_frozen_eval concluído"

echo ""
echo "[5/40] Executando: tinyclip_infonce"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/tinyclip_infonce.yaml
echo "✅ tinyclip_infonce concluído"

echo ""
echo "[6/40] Executando: tinyclip_combined_tw01"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/tinyclip_combined_tw01.yaml
echo "✅ tinyclip_combined_tw01 concluído"

echo ""
echo "[7/40] Executando: tinyclip_combined_tw02"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/tinyclip_combined_tw02.yaml
echo "✅ tinyclip_combined_tw02 concluído"

echo ""
echo "[8/40] Executando: tinyclip_combined_tw03"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/tinyclip_combined_tw03.yaml
echo "✅ tinyclip_combined_tw03 concluído"

echo ""
echo "[9/40] Executando: tinyclip_combined_tw04"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/tinyclip_combined_tw04.yaml
echo "✅ tinyclip_combined_tw04 concluído"

echo ""
echo "[10/40] Executando: tinyclip_combined_tw05"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/tinyclip_combined_tw05.yaml
echo "✅ tinyclip_combined_tw05 concluído"

echo ""
echo "[11/40] Executando: tinyclip_infonce_noaug"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/tinyclip_infonce_noaug.yaml
echo "✅ tinyclip_infonce_noaug concluído"

echo ""
echo "[12/40] Executando: tinyclip_combined_tw01_noaug"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/tinyclip_combined_tw01_noaug.yaml
echo "✅ tinyclip_combined_tw01_noaug concluído"

echo ""
echo "[13/40] Executando: tinyclip_combined_tw02_noaug"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/tinyclip_combined_tw02_noaug.yaml
echo "✅ tinyclip_combined_tw02_noaug concluído"

echo ""
echo "[14/40] Executando: tinyclip_combined_tw03_noaug"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/tinyclip_combined_tw03_noaug.yaml
echo "✅ tinyclip_combined_tw03_noaug concluído"

echo ""
echo "[15/40] Executando: tinyclip_combined_tw04_noaug"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/tinyclip_combined_tw04_noaug.yaml
echo "✅ tinyclip_combined_tw04_noaug concluído"

echo ""
echo "[16/40] Executando: tinyclip_combined_tw05_noaug"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/tinyclip_combined_tw05_noaug.yaml
echo "✅ tinyclip_combined_tw05_noaug concluído"

echo ""
echo "[17/40] Executando: clip_vit_b32_infonce"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/clip_vit_b32_infonce.yaml
echo "✅ clip_vit_b32_infonce concluído"

echo ""
echo "[18/40] Executando: clip_vit_b32_combined_tw01"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/clip_vit_b32_combined_tw01.yaml
echo "✅ clip_vit_b32_combined_tw01 concluído"

echo ""
echo "[19/40] Executando: clip_vit_b32_combined_tw02"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/clip_vit_b32_combined_tw02.yaml
echo "✅ clip_vit_b32_combined_tw02 concluído"

echo ""
echo "[20/40] Executando: clip_vit_b32_combined_tw03"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/clip_vit_b32_combined_tw03.yaml
echo "✅ clip_vit_b32_combined_tw03 concluído"

echo ""
echo "[21/40] Executando: clip_vit_b32_combined_tw04"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/clip_vit_b32_combined_tw04.yaml
echo "✅ clip_vit_b32_combined_tw04 concluído"

echo ""
echo "[22/40] Executando: clip_vit_b32_combined_tw05"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/clip_vit_b32_combined_tw05.yaml
echo "✅ clip_vit_b32_combined_tw05 concluído"

echo ""
echo "[23/40] Executando: clip_vit_b32_infonce_noaug"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/clip_vit_b32_infonce_noaug.yaml
echo "✅ clip_vit_b32_infonce_noaug concluído"

echo ""
echo "[24/40] Executando: clip_vit_b32_combined_tw01_noaug"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/clip_vit_b32_combined_tw01_noaug.yaml
echo "✅ clip_vit_b32_combined_tw01_noaug concluído"

echo ""
echo "[25/40] Executando: clip_vit_b32_combined_tw02_noaug"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/clip_vit_b32_combined_tw02_noaug.yaml
echo "✅ clip_vit_b32_combined_tw02_noaug concluído"

echo ""
echo "[26/40] Executando: clip_vit_b32_combined_tw03_noaug"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/clip_vit_b32_combined_tw03_noaug.yaml
echo "✅ clip_vit_b32_combined_tw03_noaug concluído"

echo ""
echo "[27/40] Executando: clip_vit_b32_combined_tw04_noaug"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/clip_vit_b32_combined_tw04_noaug.yaml
echo "✅ clip_vit_b32_combined_tw04_noaug concluído"

echo ""
echo "[28/40] Executando: clip_vit_b32_combined_tw05_noaug"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/clip_vit_b32_combined_tw05_noaug.yaml
echo "✅ clip_vit_b32_combined_tw05_noaug concluído"

echo ""
echo "[29/40] Executando: siglip_sigmoid"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/siglip_sigmoid.yaml
echo "✅ siglip_sigmoid concluído"

echo ""
echo "[30/40] Executando: siglip_combined_tw01"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/siglip_combined_tw01.yaml
echo "✅ siglip_combined_tw01 concluído"

echo ""
echo "[31/40] Executando: siglip_combined_tw02"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/siglip_combined_tw02.yaml
echo "✅ siglip_combined_tw02 concluído"

echo ""
echo "[32/40] Executando: siglip_combined_tw03"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/siglip_combined_tw03.yaml
echo "✅ siglip_combined_tw03 concluído"

echo ""
echo "[33/40] Executando: siglip_combined_tw04"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/siglip_combined_tw04.yaml
echo "✅ siglip_combined_tw04 concluído"

echo ""
echo "[34/40] Executando: siglip_combined_tw05"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/siglip_combined_tw05.yaml
echo "✅ siglip_combined_tw05 concluído"

echo ""
echo "[35/40] Executando: siglip_sigmoid_noaug"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/siglip_sigmoid_noaug.yaml
echo "✅ siglip_sigmoid_noaug concluído"

echo ""
echo "[36/40] Executando: siglip_combined_tw01_noaug"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/siglip_combined_tw01_noaug.yaml
echo "✅ siglip_combined_tw01_noaug concluído"

echo ""
echo "[37/40] Executando: siglip_combined_tw02_noaug"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/siglip_combined_tw02_noaug.yaml
echo "✅ siglip_combined_tw02_noaug concluído"

echo ""
echo "[38/40] Executando: siglip_combined_tw03_noaug"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/siglip_combined_tw03_noaug.yaml
echo "✅ siglip_combined_tw03_noaug concluído"

echo ""
echo "[39/40] Executando: siglip_combined_tw04_noaug"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/siglip_combined_tw04_noaug.yaml
echo "✅ siglip_combined_tw04_noaug concluído"

echo ""
echo "[40/40] Executando: siglip_combined_tw05_noaug"
echo "====================================="
python scripts/run_experiment.py --config configs/grid/siglip_combined_tw05_noaug.yaml
echo "✅ siglip_combined_tw05_noaug concluído"

echo ""
echo "====================================="
echo "🎉 Todos os experimentos concluídos!"
echo "====================================="
