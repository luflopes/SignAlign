#!/usr/bin/env python3
"""
Gera configurações YAML para o grid de experimentos.

Uso:
    python scripts/generate_experiment_grid.py
    
    Gera os arquivos em configs/grid/
"""

import os
from pathlib import Path
import yaml

# Diretório de saída
OUTPUT_DIR = Path("configs/grid")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Modelos disponíveis
MODELS = {
    "tinyclip": {
        "name": "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M",
        "type": "tinyclip",
        "image_size": 224,
    },
    "clip_vit_b32": {
        "name": "openai/clip-vit-base-patch32",
        "type": "clip",
        "image_size": 224,
    },
    "clip_vit_b16": {
        "name": "openai/clip-vit-base-patch16",
        "type": "clip",
        "image_size": 224,
    },
    "siglip": {
        "name": "google/siglip-base-patch16-224",
        "type": "siglip",
        "image_size": 224,
    },
}

# Triplet weights para testar
TRIPLET_WEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5]

# Configuração base
BASE_CONFIG = {
    "seed": 5932,
    "output_dir": "experiments",
    "data": {
        "dataset_csv": "datasets/dataset-sign-align/dataset.csv",
        "images_base_path": "datasets/dataset-sign-align",
        "train_ratio": 0.85,
        "val_ratio": 0.15,
        "batch_size": 32,
        "num_workers": 4,
        "exclude_unknown": True,
        "max_samples": None,
        "augmentation": {
            "enabled": True,
            "shift_limit": [-0.0625, 0.0625],
            "scale_limit": [-0.1, 0.1],
            "rotate_limit": [-10, 15],
            "downscale_range": [0.4, 1.0],
            "motion_blur_limit": [3, 5],
            "noise_std_range": [0.1, 0.2],
            "brightness_limit": [-0.2, 0.2],
            "contrast_limit": [-0.2, 0.2],
            "compression_quality_range": [40, 100],
            "geometric_p": 0.2,
            "blur_noise_p": 0.2,
            "color_p": 0.2,
            "compression_p": 0.2,
        }
    },
    "training": {
        "epochs": 20,
        "learning_rate": 1e-6,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "use_amp": True,
        "save_every": 5,
        "save_top_k": 3,
        "save_only_best": True,
        "log_interval": 50,
        "early_stopping_patience": 10,
        "scheduler": {
            "name": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 3,
            "min_lr": 1e-8,
            "mode": "max",
        }
    },
    "evaluation": {
        "negative_samples": [1, 2, 3],
        "compute_eer": True,
        "compute_auc": True,
        "compute_recall_at_k": True,
        "recall_k_values": [1, 5, 10],
        "compute_mrr": True,
        "compute_ndcg": True,
        "num_visualization_samples": 10,
        "save_attention_maps": True,
    },
    "loss": {
        "primary_loss": "cross_entropy",
        "temperature": 0.07,
        "use_triplet_loss": False,
        "triplet_weight": 0.2,
        "triplet_margin": 0.2,
        "sigmoid_bias": -10.0,
    }
}


def create_config(
    model_key: str,
    experiment_name: str,
    freeze_vision: bool = False,
    augmentation: bool = True,
    loss_type: str = "cross_entropy",  # cross_entropy, infonce, sigmoid
    triplet_weight: float = None,
    epochs: int = 20,
) -> dict:
    """Cria uma configuração de experimento."""
    
    import copy
    config = copy.deepcopy(BASE_CONFIG)
    
    # Nome do experimento
    config["name"] = experiment_name
    
    # Modelo
    model_info = MODELS[model_key]
    config["model"] = {
        "name": model_info["name"],
        "type": model_info["type"],
        "image_size": model_info["image_size"],
        "freeze_vision_encoder": freeze_vision,
        "freeze_text_encoder": False,
        "output_attentions": True,
    }
    
    # Augmentation
    config["data"]["augmentation"]["enabled"] = augmentation
    
    # Loss
    config["loss"]["primary_loss"] = loss_type
    if triplet_weight is not None:
        config["loss"]["use_triplet_loss"] = True
        config["loss"]["triplet_weight"] = triplet_weight
    else:
        config["loss"]["use_triplet_loss"] = False
    
    # Epochs
    config["training"]["epochs"] = epochs
    
    return config


def save_config(config: dict, filename: str):
    """Salva configuração em YAML."""
    path = OUTPUT_DIR / f"{filename}.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  ✅ {path}")


def generate_all_configs():
    """Gera todas as configurações do grid."""
    
    configs_generated = []
    
    print("🔧 Gerando configurações do grid de experimentos...\n")
    
    # ==============================================================
    # 1. EXPERIMENTOS FROZEN (apenas avaliação, sem treino)
    # ==============================================================
    print("📦 Experimentos Frozen (avaliação sem treino):")
    
    for model_key in ["tinyclip", "clip_vit_b32", "clip_vit_b16", "siglip"]:
        exp_name = f"{model_key}_frozen_eval"
        config = create_config(
            model_key=model_key,
            experiment_name=exp_name,
            freeze_vision=True,
            augmentation=False,
            loss_type="cross_entropy",
            triplet_weight=None,
            epochs=0,  # Sem treino
        )
        save_config(config, exp_name)
        configs_generated.append(exp_name)
    
    print()
    
    # ==============================================================
    # 2. TINYCLIP - Com fine-tuning
    # ==============================================================
    print("📦 TinyCLIP (fine-tuning):")
    
    # InfoNCE sem triplet
    for aug in [True, False]:
        suffix = "" if aug else "_noaug"
        exp_name = f"tinyclip_infonce{suffix}"
        config = create_config(
            model_key="tinyclip",
            experiment_name=exp_name,
            freeze_vision=False,
            augmentation=aug,
            loss_type="infonce",
            triplet_weight=None,
        )
        save_config(config, exp_name)
        configs_generated.append(exp_name)
    
    # InfoNCE + Triplet (variando peso)
    for tw in TRIPLET_WEIGHTS:
        tw_str = f"tw{int(tw*10):02d}"
        for aug in [True, False]:
            suffix = "" if aug else "_noaug"
            exp_name = f"tinyclip_combined_{tw_str}{suffix}"
            config = create_config(
                model_key="tinyclip",
                experiment_name=exp_name,
                freeze_vision=False,
                augmentation=aug,
                loss_type="infonce",
                triplet_weight=tw,
            )
            save_config(config, exp_name)
            configs_generated.append(exp_name)
    
    print()
    
    # ==============================================================
    # 3. CLIP ViT-B/32 - Com fine-tuning
    # ==============================================================
    print("📦 CLIP ViT-B/32 (fine-tuning):")
    
    # InfoNCE sem triplet
    for aug in [True, False]:
        suffix = "" if aug else "_noaug"
        exp_name = f"clip_vit_b32_infonce{suffix}"
        config = create_config(
            model_key="clip_vit_b32",
            experiment_name=exp_name,
            freeze_vision=False,
            augmentation=aug,
            loss_type="infonce",
            triplet_weight=None,
        )
        save_config(config, exp_name)
        configs_generated.append(exp_name)
    
    # InfoNCE + Triplet
    for tw in TRIPLET_WEIGHTS:
        tw_str = f"tw{int(tw*10):02d}"
        for aug in [True, False]:
            suffix = "" if aug else "_noaug"
            exp_name = f"clip_vit_b32_combined_{tw_str}{suffix}"
            config = create_config(
                model_key="clip_vit_b32",
                experiment_name=exp_name,
                freeze_vision=False,
                augmentation=aug,
                loss_type="infonce",
                triplet_weight=tw,
            )
            save_config(config, exp_name)
            configs_generated.append(exp_name)
    
    print()
    
    # ==============================================================
    # 4. SIGLIP - Com fine-tuning (Sigmoid loss)
    # ==============================================================
    print("📦 SigLIP (fine-tuning):")
    
    # Sigmoid sem triplet
    for aug in [True, False]:
        suffix = "" if aug else "_noaug"
        exp_name = f"siglip_sigmoid{suffix}"
        config = create_config(
            model_key="siglip",
            experiment_name=exp_name,
            freeze_vision=False,
            augmentation=aug,
            loss_type="sigmoid",
            triplet_weight=None,
        )
        save_config(config, exp_name)
        configs_generated.append(exp_name)
    
    # Sigmoid + Triplet
    for tw in TRIPLET_WEIGHTS:
        tw_str = f"tw{int(tw*10):02d}"
        for aug in [True, False]:
            suffix = "" if aug else "_noaug"
            exp_name = f"siglip_combined_{tw_str}{suffix}"
            config = create_config(
                model_key="siglip",
                experiment_name=exp_name,
                freeze_vision=False,
                augmentation=aug,
                loss_type="sigmoid",
                triplet_weight=tw,
            )
            save_config(config, exp_name)
            configs_generated.append(exp_name)
    
    print()
    print(f"✅ Total: {len(configs_generated)} configurações geradas em {OUTPUT_DIR}/")
    
    # Gerar script de execução
    generate_run_script(configs_generated)
    
    return configs_generated


def generate_run_script(configs: list):
    """Gera script bash para executar todos os experimentos."""
    
    script_path = Path("scripts/run_all_experiments.sh")
    
    lines = [
        "#!/bin/bash",
        "# Script para executar todos os experimentos do grid",
        "# ",
        "# Uso:",
        "#   ./scripts/run_all_experiments.sh                    # Execução completa",
        "#   ./scripts/run_all_experiments.sh --test-mode        # Modo teste (50 amostras, 1 época)",
        "#   ./scripts/run_all_experiments.sh --max-samples 100  # Limitar amostras",
        "#   ./scripts/run_all_experiments.sh --max-samples 200 --epochs 3  # Customizado",
        "",
        "set -e  # Parar em caso de erro",
        "",
        "# Capturar todos os argumentos para passar aos experimentos",
        'EXTRA_ARGS="$@"',
        "",
        "# Ativar ambiente virtual se existir",
        'if [ -d "venv" ]; then',
        "    source venv/bin/activate",
        "fi",
        "",
        'echo "====================================="',
        f'echo "Executando {len(configs)} experimentos"',
        'if [ -n "$EXTRA_ARGS" ]; then',
        '    echo "Argumentos extras: $EXTRA_ARGS"',
        "fi",
        'echo "====================================="',
        "",
        "# Lista de configurações",
        "CONFIGS=(",
    ]
    
    for config in configs:
        lines.append(f'    "{config}"')
    
    lines.extend([
        ")",
        "",
        "TOTAL=${#CONFIGS[@]}",
        "COUNT=0",
        "FAILED=0",
        "",
        'for CONFIG in "${CONFIGS[@]}"; do',
        "    COUNT=$((COUNT + 1))",
        '    echo ""',
        '    echo "[$COUNT/$TOTAL] Executando: $CONFIG"',
        '    echo "====================================="',
        "    ",
        '    if python scripts/run_experiment.py --config "configs/grid/${CONFIG}.yaml" $EXTRA_ARGS; then',
        '        echo "✅ $CONFIG concluído"',
        "    else",
        '        echo "❌ $CONFIG FALHOU"',
        "        FAILED=$((FAILED + 1))",
        "    fi",
        "done",
        "",
        'echo ""',
        'echo "====================================="',
        "if [ $FAILED -eq 0 ]; then",
        '    echo "🎉 Todos os $TOTAL experimentos concluídos com sucesso!"',
        "else",
        '    echo "⚠️ $FAILED de $TOTAL experimentos falharam"',
        "fi",
        'echo "====================================="',
    ])
    
    with open(script_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    
    os.chmod(script_path, 0o755)
    print(f"📜 Script de execução gerado: {script_path}")


if __name__ == "__main__":
    generate_all_configs()
