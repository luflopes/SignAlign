#!/usr/bin/env python3
"""
Gerador automático de configurações para grid de experimentos.

Gera todas as combinações de:
- Modelo: TinyCLIP, CLIP, SigLIP
- Loss: InfoNCE, Sigmoid, Combined (com Triplet)
- Triplet Weight: 0.1, 0.2, 0.3, 0.4, 0.5
- Data Augmentation: True, False
- Fine-tuning: True, False (frozen)
"""

import os
import yaml
from pathlib import Path
from itertools import product
from typing import Dict, List, Any


# Configurações base compartilhadas
BASE_CONFIG = {
    "seed": 5932,
    "output_dir": "experiments",
    "data": {
        "dataset_csv": "datasets/dataset-sign-align/dataset.csv",
        "images_base_path": "datasets/dataset-sign-align",
        "val_ratio": 0.15,
        "exclude_unknown": True,
        "batch_size": 32,
        "max_samples": None,  # None = usar todos
    },
    "training": {
        "epochs": 30,
        "learning_rate": 1e-6,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "use_amp": True,
        "save_every": 5,
        "save_only_best": True,
        "early_stopping_patience": 10,
        "scheduler": {
            "type": "plateau",
            "factor": 0.5,
            "patience": 3,
            "min_lr": 1e-8,
        },
    },
    "evaluation": {
        "negative_samples": [1, 2, 3],
        "compute_eer": True,
        "compute_auc": True,
    },
}

# Definições de modelos
MODELS = {
    "tinyclip": {
        "type": "tinyclip",
        "pretrained_name": "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M",
        "image_size": 224,
    },
    "clip_vit_b32": {
        "type": "clip",
        "pretrained_name": "openai/clip-vit-base-patch32",
        "image_size": 224,
    },
    "clip_vit_b16": {
        "type": "clip",
        "pretrained_name": "openai/clip-vit-base-patch16",
        "image_size": 224,
    },
    "siglip": {
        "type": "siglip",
        "pretrained_name": "google/siglip-base-patch16-224",
        "image_size": 224,
    },
}

# Triplet weights para grid
TRIPLET_WEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5]

# Data augmentation options
AUGMENTATION_OPTIONS = [True, False]


def generate_experiment_name(
    model_name: str,
    loss_type: str,
    triplet_weight: float = None,
    augmentation: bool = True,
    frozen: bool = False
) -> str:
    """Gera nome único para o experimento."""
    parts = [model_name]
    
    if frozen:
        parts.append("frozen")
    
    parts.append(loss_type)
    
    if triplet_weight is not None:
        parts.append(f"tw{triplet_weight:.1f}".replace(".", ""))
    
    if not augmentation:
        parts.append("noaug")
    
    return "_".join(parts)


def create_config(
    name: str,
    model_config: Dict,
    loss_config: Dict,
    augmentation: bool = True,
    frozen: bool = False,
    eval_only: bool = False
) -> Dict:
    """Cria configuração completa para um experimento."""
    config = {
        "name": name,
        **{k: v.copy() if isinstance(v, dict) else v for k, v in BASE_CONFIG.items()},
    }
    
    # Deep copy nested dicts
    config["data"] = BASE_CONFIG["data"].copy()
    config["training"] = BASE_CONFIG["training"].copy()
    config["training"]["scheduler"] = BASE_CONFIG["training"]["scheduler"].copy()
    config["evaluation"] = BASE_CONFIG["evaluation"].copy()
    
    # Modelo
    config["model"] = {
        **model_config,
        "freeze_vision_encoder": frozen,
        "freeze_text_encoder": False,
    }
    
    # Loss
    config["loss"] = loss_config
    
    # Augmentation
    config["data"]["augmentation"] = {
        "enabled": augmentation,
        "random_rotation": 5,
        "random_scale": [0.95, 1.05],
        "random_brightness": 0.1,
        "random_contrast": 0.1,
    }
    
    # Se eval_only, configurar para 0 épocas
    if eval_only:
        config["training"]["epochs"] = 0
        config["training"]["eval_only"] = True
    
    return config


def generate_all_configs(output_dir: str = "configs/grid") -> List[str]:
    """Gera todas as configurações do grid de experimentos."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    # =====================================================
    # 1. Experimentos SEM fine-tuning (apenas avaliação)
    # =====================================================
    print("\n📦 Gerando configurações SEM fine-tuning (eval only)...")
    
    for model_name, model_config in MODELS.items():
        name = generate_experiment_name(model_name, "eval", frozen=True)
        
        config = create_config(
            name=name,
            model_config=model_config,
            loss_config={
                "primary_loss": "cross_entropy",
                "temperature": 0.07,
                "use_triplet_loss": False,
            },
            augmentation=False,
            frozen=True,
            eval_only=True
        )
        
        filepath = output_path / f"{name}.yaml"
        with open(filepath, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        generated_files.append(str(filepath))
        print(f"   ✓ {name}")
    
    # =====================================================
    # 2. TinyCLIP / CLIP com fine-tuning
    # =====================================================
    print("\n📦 Gerando configurações TinyCLIP/CLIP com fine-tuning...")
    
    for model_name in ["tinyclip", "clip_vit_b32"]:
        model_config = MODELS[model_name]
        
        for aug in AUGMENTATION_OPTIONS:
            # 2a. Apenas InfoNCE (sem Triplet)
            name = generate_experiment_name(model_name, "infonce", augmentation=aug)
            config = create_config(
                name=name,
                model_config=model_config,
                loss_config={
                    "primary_loss": "cross_entropy",
                    "temperature": 0.07,
                    "use_triplet_loss": False,
                },
                augmentation=aug
            )
            
            filepath = output_path / f"{name}.yaml"
            with open(filepath, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            generated_files.append(str(filepath))
            print(f"   ✓ {name}")
            
            # 2b. InfoNCE + Triplet (variar peso)
            for tw in TRIPLET_WEIGHTS:
                name = generate_experiment_name(
                    model_name, "combined", triplet_weight=tw, augmentation=aug
                )
                config = create_config(
                    name=name,
                    model_config=model_config,
                    loss_config={
                        "primary_loss": "cross_entropy",
                        "temperature": 0.07,
                        "use_triplet_loss": True,
                        "triplet_weight": tw,
                        "triplet_margin": 0.2,
                    },
                    augmentation=aug
                )
                
                filepath = output_path / f"{name}.yaml"
                with open(filepath, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                generated_files.append(str(filepath))
                print(f"   ✓ {name}")
    
    # =====================================================
    # 3. SigLIP com fine-tuning
    # =====================================================
    print("\n📦 Gerando configurações SigLIP com fine-tuning...")
    
    model_config = MODELS["siglip"]
    
    for aug in AUGMENTATION_OPTIONS:
        # 3a. Apenas Sigmoid Loss
        name = generate_experiment_name("siglip", "sigmoid", augmentation=aug)
        config = create_config(
            name=name,
            model_config=model_config,
            loss_config={
                "primary_loss": "sigmoid",
                "temperature": 1.0,
                "sigmoid_bias": -10.0,
                "use_triplet_loss": False,
            },
            augmentation=aug
        )
        
        filepath = output_path / f"{name}.yaml"
        with open(filepath, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        generated_files.append(str(filepath))
        print(f"   ✓ {name}")
        
        # 3b. Sigmoid + Triplet (variar peso)
        for tw in TRIPLET_WEIGHTS:
            name = generate_experiment_name(
                "siglip", "combined", triplet_weight=tw, augmentation=aug
            )
            config = create_config(
                name=name,
                model_config=model_config,
                loss_config={
                    "primary_loss": "sigmoid",
                    "temperature": 1.0,
                    "sigmoid_bias": -10.0,
                    "use_triplet_loss": True,
                    "triplet_weight": tw,
                    "triplet_margin": 0.2,
                },
                augmentation=aug
            )
            
            filepath = output_path / f"{name}.yaml"
            with open(filepath, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            generated_files.append(str(filepath))
            print(f"   ✓ {name}")
    
    return generated_files


def generate_run_script(config_files: List[str], output_file: str = "scripts/run_all_experiments.sh"):
    """Gera script bash para executar todos os experimentos."""
    with open(output_file, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Script gerado automaticamente para executar todos os experimentos\n")
        f.write("# Uso: ./scripts/run_all_experiments.sh\n\n")
        
        f.write("set -e  # Parar em caso de erro\n\n")
        
        f.write("# Ativar ambiente virtual se existir\n")
        f.write("if [ -d \"venv\" ]; then\n")
        f.write("    source venv/bin/activate\n")
        f.write("fi\n\n")
        
        f.write("echo \"=====================================\"\n")
        f.write(f"echo \"Executando {len(config_files)} experimentos\"\n")
        f.write("echo \"=====================================\"\n\n")
        
        for i, config in enumerate(config_files, 1):
            exp_name = Path(config).stem
            f.write(f"echo \"\"\n")
            f.write(f"echo \"[{i}/{len(config_files)}] Executando: {exp_name}\"\n")
            f.write(f"echo \"=====================================\"\n")
            f.write(f"python scripts/run_experiment.py --config {config}\n")
            f.write(f"echo \"✅ {exp_name} concluído\"\n\n")
        
        f.write("echo \"\"\n")
        f.write("echo \"=====================================\"\n")
        f.write("echo \"🎉 Todos os experimentos concluídos!\"\n")
        f.write("echo \"=====================================\"\n")
    
    os.chmod(output_file, 0o755)
    print(f"\n📜 Script de execução salvo em: {output_file}")


def main():
    """Função principal."""
    print("=" * 60)
    print("🔬 Gerador de Grid de Experimentos - SignAlign")
    print("=" * 60)
    
    # Gerar todas as configurações
    config_files = generate_all_configs()
    
    # Gerar script de execução
    generate_run_script(config_files)
    
    # Resumo
    print("\n" + "=" * 60)
    print(f"📊 Resumo:")
    print(f"   Total de configurações geradas: {len(config_files)}")
    print(f"   Diretório: configs/grid/")
    print(f"   Script de execução: scripts/run_all_experiments.sh")
    print("=" * 60)
    
    # Estatísticas detalhadas
    print("\n📈 Distribuição de experimentos:")
    
    eval_only = len([f for f in config_files if "frozen" in f or "eval" in f])
    tinyclip = len([f for f in config_files if "tinyclip" in f])
    clip = len([f for f in config_files if "clip_vit" in f])
    siglip = len([f for f in config_files if "siglip" in f])
    with_triplet = len([f for f in config_files if "combined" in f])
    no_aug = len([f for f in config_files if "noaug" in f])
    
    print(f"   - Avaliação apenas (frozen): {eval_only}")
    print(f"   - TinyCLIP: {tinyclip}")
    print(f"   - CLIP: {clip}")
    print(f"   - SigLIP: {siglip}")
    print(f"   - Com Triplet Loss: {with_triplet}")
    print(f"   - Sem augmentation: {no_aug}")


if __name__ == "__main__":
    main()

