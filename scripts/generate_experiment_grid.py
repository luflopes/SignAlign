#!/usr/bin/env python3
"""
Gera configurações YAML para o grid de experimentos.

Uso:
    python scripts/generate_experiment_grid.py
    
    Gera os arquivos em configs/grid/
"""

import os
import json
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

# Loss nativa por modelo (usada no finetuning simples / LR search)
NATIVE_LOSS = {
    "tinyclip": "infonce",
    "clip_vit_b32": "infonce",
    "clip_vit_b16": "infonce",
    "siglip": "sigmoid",
}

# Modelos principais para a busca de LR
LR_SEARCH_MODELS = ["tinyclip", "clip_vit_b32", "siglip"]

# Learning rates a testar na busca
LR_SEARCH_VALUES = [1e-4, 1e-5, 1e-6]

# Diretório de saída das configs de busca de LR
LR_SEARCH_DIR = Path("configs/lr_search")

# Melhor LR por modelo (preenchido a partir de best_lr.json, se disponível).
# Quando vazio, create_config usa a LR default da BASE_CONFIG (1e-6).
BEST_LR: dict = {}


def load_best_lr(path: str = "experiments/lr_search/best_lr.json") -> dict:
    """Carrega a melhor LR por modelo gerada por analyze_lr_search.py."""
    p = Path(path)
    if not p.exists():
        print(f"⚠️ best_lr.json não encontrado em {p}. Usando LR default ({BASE_CONFIG['training']['learning_rate']:.0e}).")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    best = {k: float(v) for k, v in data.get("best_lr", {}).items()}
    print(f"📥 Melhor LR carregada de {p}: {best}")
    return best


def _lr_tag(lr: float) -> str:
    """Gera um sufixo curto e legível para a LR (ex.: 1e-4 -> lr1e4)."""
    return "lr" + f"{lr:.0e}".replace("-0", "").replace("-", "").replace("+", "")

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
    learning_rate: float = None,
    use_weighted_sampler: bool = True,
    weight_scheme: str = "inv_sqrt",
) -> dict:
    """Cria uma configuração de experimento.

    Por padrão o grid usa amostragem ponderada por indivíduo
    (use_weighted_sampler=True). A busca de LR passa False explicitamente.
    """
    
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
    
    # Amostragem ponderada por indivíduo
    config["data"]["use_weighted_sampler"] = use_weighted_sampler
    config["data"]["weight_scheme"] = weight_scheme
    
    # Loss
    config["loss"]["primary_loss"] = loss_type
    if triplet_weight is not None:
        config["loss"]["use_triplet_loss"] = True
        config["loss"]["triplet_weight"] = triplet_weight
    else:
        config["loss"]["use_triplet_loss"] = False
    
    # Epochs
    config["training"]["epochs"] = epochs
    
    # Learning rate: prioridade para argumento explícito, depois BEST_LR
    # (melhor LR encontrada na busca), por fim o default da BASE_CONFIG.
    if learning_rate is not None:
        config["training"]["learning_rate"] = learning_rate
    elif model_key in BEST_LR:
        config["training"]["learning_rate"] = BEST_LR[model_key]
    
    return config


def save_config(config: dict, filename: str, output_dir: Path = OUTPUT_DIR):
    """Salva configuração em YAML."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{filename}.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  ✅ {path}")


def generate_all_configs(best_lr_path: str = None):
    """
    Gera todas as configurações do grid.

    Args:
        best_lr_path: Se fornecido, carrega a melhor LR por modelo desse arquivo
            (gerado por analyze_lr_search.py) e aplica em todas as configs.
    """
    global BEST_LR
    if best_lr_path is not None:
        BEST_LR = load_best_lr(best_lr_path)

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


def generate_lr_search_configs():
    """
    Gera configs para a busca de Learning Rate (finetuning simples).

    Para cada modelo principal e cada LR em {1e-4, 1e-5, 1e-6}, cria uma
    config de finetuning simples: SEM augmentação, SEM triplet/combined,
    SEM weighted sampler, usando a loss nativa do modelo.
    """
    configs_generated = []

    print("🔧 Gerando configs de busca de LR (finetuning simples)...\n")

    for model_key in LR_SEARCH_MODELS:
        for lr in LR_SEARCH_VALUES:
            exp_name = f"{model_key}_{_lr_tag(lr)}"
            config = create_config(
                model_key=model_key,
                experiment_name=exp_name,
                freeze_vision=False,
                augmentation=False,
                loss_type=NATIVE_LOSS[model_key],
                triplet_weight=None,
                learning_rate=lr,
                use_weighted_sampler=False,
            )
            save_config(config, exp_name, output_dir=LR_SEARCH_DIR)
            configs_generated.append(exp_name)

    print(f"\n✅ Total: {len(configs_generated)} configs de LR em {LR_SEARCH_DIR}/")

    generate_lr_search_script(configs_generated)

    return configs_generated


def generate_lr_search_script(configs: list):
    """Gera script bash para executar a busca de LR."""

    script_path = Path("scripts/run_lr_search.sh")

    lines = [
        "#!/bin/bash",
        "# Busca de Learning Rate (finetuning simples, sem augmentação/combined).",
        "#",
        "# Uso:",
        "#   ./scripts/run_lr_search.sh                 # Execução completa",
        "#   ./scripts/run_lr_search.sh --test-mode     # Modo teste rápido",
        "",
        "set -e",
        "",
        'EXTRA_ARGS="$@"',
        "",
        'if [ -d "venv" ]; then',
        "    source venv/bin/activate",
        "fi",
        "",
        'OUTPUT_DIR="experiments/lr_search"',
        "",
        'echo "====================================="',
        f'echo "Busca de LR: {len(configs)} experimentos"',
        'echo "====================================="',
        "",
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
        '    if python scripts/run_experiment.py --config "configs/lr_search/${CONFIG}.yaml" --output-dir "$OUTPUT_DIR" $EXTRA_ARGS; then',
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
        '    echo "🎉 Busca de LR concluída! Rode: python scripts/analyze_lr_search.py"',
        "else",
        '    echo "⚠️ $FAILED de $TOTAL experimentos falharam"',
        "fi",
        'echo "====================================="',
    ])

    with open(script_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    os.chmod(script_path, 0o755)
    print(f"📜 Script de busca de LR gerado: {script_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gera configs de experimentos")
    parser.add_argument(
        "--target",
        choices=["grid", "lr_search", "all"],
        default="grid",
        help="Quais configs gerar: grid completo, busca de LR, ou ambos",
    )
    parser.add_argument(
        "--best-lr",
        type=str,
        nargs="?",
        const="experiments/lr_search/best_lr.json",
        default=None,
        help="Usar a melhor LR por modelo no grid (default: experiments/lr_search/best_lr.json)",
    )
    args = parser.parse_args()

    if args.target in ("grid", "all"):
        generate_all_configs(best_lr_path=args.best_lr)
    if args.target in ("lr_search", "all"):
        generate_lr_search_configs()
