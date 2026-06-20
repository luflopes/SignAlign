#!/usr/bin/env python3
"""
Gera os configs do estudo do dataset interno: 3 modelos × 3 esquemas = 9 configs.

Esquemas:
  - frozen     : modelo base off-the-shelf, encoders congelados, eval-only no val interno
  - pretrained : melhor checkpoint treinado no dataset público (transfer), eval-only no val interno
  - finetuned  : finetuning no train.csv interno usando a melhor config do modelo

Melhores versões por modelo (grid de significância, accuracy_3_neg médio entre seeds;
empates resolvidos pelo melhor pico por seed / consistência com o modelo já baixado):

  - clip_vit_b32: clip_vit_b32_combined_tw05  (melhor seed: 1)
  - tinyclip    : tinyclip_combined_tw03      (melhor seed: 2)
  - siglip      : siglip_combined_tw03        (melhor seed: 1)

O checkpoint usado no esquema "pretrained" é o da MELHOR seed dessa config:
  experiments/significance/seed_<best_seed>/<best_config>/checkpoints/best

Uso:
    python scripts/generate_internal_configs.py
"""

from pathlib import Path

import yaml

OUT_DIR = Path(__file__).parent.parent / "configs" / "internal"
SEED = 5932
IMAGES_BASE = "datasets/internal-dataset"
TRAIN_CSV = "datasets/internal-dataset/train.csv"
VAL_CSV = "datasets/internal-dataset/val.csv"
FINETUNE_EPOCHS = 20

# Melhor versão por modelo (config pública + seed do checkpoint)
MODELS = {
    "clip_vit_b32": {
        "model_name": "openai/clip-vit-base-patch32",
        "model_type": "clip",
        "primary_loss": "infonce",
        "triplet_weight": 0.5,
        "learning_rate": 1.0e-06,
        "best_config": "clip_vit_b32_combined_tw05",
        "best_seed": 1,
    },
    "tinyclip": {
        "model_name": "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M",
        "model_type": "tinyclip",
        "primary_loss": "infonce",
        "triplet_weight": 0.3,
        "learning_rate": 1.0e-06,
        "best_config": "tinyclip_combined_tw03",
        "best_seed": 2,
    },
    "siglip": {
        "model_name": "google/siglip-base-patch16-224",
        "model_type": "siglip",
        "primary_loss": "sigmoid",
        "triplet_weight": 0.3,
        "learning_rate": 1.0e-05,
        "best_config": "siglip_combined_tw03",
        "best_seed": 1,
    },
}

AUGMENTATION = {
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

EVALUATION = {
    "negative_samples": [1, 2, 3],
    "compute_eer": True,
    "compute_auc": True,
    "compute_recall_at_k": True,
    "recall_k_values": [1, 5, 10],
    "compute_mrr": True,
    "compute_ndcg": True,
    "num_visualization_samples": 10,
    "save_attention_maps": False,
}

SCHEDULER = {
    "name": "reduce_on_plateau",
    "factor": 0.5,
    "patience": 3,
    "min_lr": 1.0e-08,
    "mode": "max",
}


def training_block(epochs: int, lr: float) -> dict:
    return {
        "epochs": epochs,
        "learning_rate": lr,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "use_amp": True,
        "save_every": 5,
        "save_top_k": 3,
        "save_only_best": True,
        "log_interval": 50,
        "early_stopping_patience": 10,
        "scheduler": dict(SCHEDULER),
    }


def loss_block(spec: dict) -> dict:
    return {
        "primary_loss": spec["primary_loss"],
        "temperature": 0.07,
        "use_triplet_loss": True,
        "triplet_weight": spec["triplet_weight"],
        "triplet_margin": 0.2,
        "sigmoid_bias": -10.0,
    }


def model_block(spec: dict, freeze: bool) -> dict:
    return {
        "name": spec["model_name"],
        "type": spec["model_type"],
        "image_size": 224,
        "freeze_vision_encoder": freeze,
        "freeze_text_encoder": freeze,
        "output_attentions": False,
    }


def make_config(model_key: str, scheme: str) -> dict:
    spec = MODELS[model_key]
    name = f"internal_{model_key}_{scheme}"

    if scheme == "finetuned":
        dataset_csv = TRAIN_CSV
        train_ratio, val_ratio = 1.0, 0.0
        epochs = FINETUNE_EPOCHS
        freeze = False
        aug = dict(AUGMENTATION)
    else:  # frozen / pretrained -> eval-only no val interno
        dataset_csv = VAL_CSV
        train_ratio, val_ratio = 0.0, 1.0
        epochs = 0
        freeze = True
        aug = {"enabled": False}

    return {
        "seed": SEED,
        "output_dir": "experiments/internal",
        "name": name,
        "data": {
            "dataset_csv": dataset_csv,
            "images_base_path": IMAGES_BASE,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "batch_size": 32,
            "num_workers": 4,
            "exclude_unknown": False,
            "max_samples": None,
            "augmentation": aug,
            "split_path": None,
        },
        "training": training_block(epochs, spec["learning_rate"]),
        "evaluation": dict(EVALUATION),
        "loss": loss_block(spec),
        "model": model_block(spec, freeze),
    }


HEADER = {
    "frozen": "Esquema FROZEN: modelo base off-the-shelf, encoders congelados, eval-only no val interno.",
    "pretrained": ("Esquema PRETRANSFER: avalia o melhor checkpoint público do modelo no val interno (eval-only).\n"
                   "# Rode com: --pretrained-checkpoint experiments/significance/seed_<best_seed>/<best_config>/checkpoints/best"),
    "finetuned": "Esquema FINETUNED: finetuning no train.csv interno com a melhor config do modelo (use --val-csv val.csv).",
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for model_key, spec in MODELS.items():
        for scheme in ["frozen", "pretrained", "finetuned"]:
            cfg = make_config(model_key, scheme)
            path = OUT_DIR / f"{model_key}_{scheme}.yaml"
            header = (
                f"# {cfg['name']}\n"
                f"# {HEADER[scheme]}\n"
                f"# Modelo: {spec['model_name']} ({spec['model_type']})\n"
            )
            if scheme == "pretrained":
                header += (f"# Checkpoint: experiments/significance/seed_{spec['best_seed']}/"
                           f"{spec['best_config']}/checkpoints/best\n")
            with open(path, "w", encoding="utf-8") as f:
                f.write(header)
                yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False,
                               allow_unicode=True)
            written.append(path.name)

    print(f"✅ {len(written)} configs gerados em {OUT_DIR}:")
    for name in written:
        print(f"   - {name}")


if __name__ == "__main__":
    main()
