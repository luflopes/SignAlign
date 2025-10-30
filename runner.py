import os
import argparse
import csv
import logging
import itertools
from pathlib import Path
from types import SimpleNamespace
import time
import torch
import pandas as pd

# Import train.main (used quando finetune=True)
from src import train as train_module
from src.models import create_model, create_processor
from src.validate import validate_and_compute_metrics
from src.dataset import make_transform


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_MAP = {
    "tinyclip": "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M",
    "clip": "openai/clip-vit-base-patch32"
}

def build_config(base, overrides: dict):
    cfg = vars(base).copy()
    cfg.update(overrides)
    return SimpleNamespace(**cfg)

def main():
    parser = argparse.ArgumentParser(description="Experiment runner for signname-align")
    parser.add_argument("--train-csv", type=Path, default=Path("train.csv"), required=False)
    parser.add_argument("--val-csv", type=Path, default=Path("val.csv"), required=False)
    parser.add_argument("--train-triplets-csv", type=Path, default="train_triplets.csv", required=False, help="Optional triplets CSV for triplet training")
    parser.add_argument("--output-root", type=Path, default=Path("experiments"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=5932)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--model-variant", choices=["tinyclip","clip"], default="tinyclip")
    parser.add_argument("--model-name", type=str, default=None, help="If provided overrides the model map")
    parser.add_argument("--finetune", action="store_true", help="Enable finetuning of backbone")
    parser.add_argument("--loss-mode", choices=["ce","triplet","ce+triplet"], default="triplet")
    parser.add_argument("--use-augmentations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-scheduler", action="store_true")
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-8)
    parser.add_argument("--scheduler-threshold", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--contrastive-weight", type=float, default=0.2)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--neg-samples", type=int, default=3)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    results_csv = args.output_root / "results.csv"

    base = SimpleNamespace(
        seed=args.seed,
        neg_samples=args.neg_samples,
        image_size=224,
        model_name=None,
        model_variant=args.model_variant,
        finetune=False,
        output_dir=Path(args.output_root) / "run_default",
        train_csv_path=args.train_csv,
    train_triplets_csv_path=args.train_triplets_csv,
        val_csv_path=args.val_csv,
        best_model_path=None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        margin=args.margin,
        contrastive_weight=args.contrastive_weight,
        use_amp=args.use_amp,
        max_grad_norm=args.max_grad_norm,
        num_workers=args.num_workers,
        save_every=args.save_every,
        log_interval=args.log_interval,
        loss_mode=args.loss_mode,
        use_augmentations=args.use_augmentations,
        use_scheduler=args.use_scheduler,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        scheduler_min_lr=args.scheduler_min_lr,
        scheduler_threshold=args.scheduler_threshold,
        early_stopping_patience=args.early_stopping_patience
    )

    # Experiment grid (customize if needed)
    loss_modes = [args.loss_mode]
    augmentations = [args.use_augmentations]
    model_variants = [args.model_variant]
    finetune_opts = [True] #[args.finetune]

    header = [
        "experiment_id", "timestamp", "model_variant", "model_name", "finetune",
        "loss_mode", "use_augmentations", "seed", "output_dir", "best_val_accuracy", "validation_accuracies"
    ]
    if not results_csv.exists():
        with results_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    exp_id = 0
    for model_variant, finetune, loss_mode, use_augment in itertools.product(model_variants, finetune_opts, loss_modes, augmentations):
        exp_id += 1
        model_name = args.model_name or MODEL_MAP[model_variant]
        out_dir = args.output_root / f"exp_{exp_id:03d}_{model_variant}_{'ft' if finetune else 'noft'}_{loss_mode}_{'aug' if use_augment else 'noaug'}"
        override = {
            "model_variant": model_variant,
            "model_name": model_name,
            "finetune": finetune,
            "loss_mode": loss_mode,
            "use_augmentations": use_augment,
            "output_dir": out_dir,
            "best_model_path": out_dir / "best",
            "train_csv_path": args.train_csv,
            "val_csv_path": args.val_csv,
            "seed": args.seed
        }
        config = build_config(base, override)

        logging.info(f"Starting experiment {exp_id}: {override}")
        start = time.time()

        if not finetune:
            # Only run validation on pretrained model (no fine-tuning)
            try:
                model = create_model(model_name)
                processor = create_processor(model_name)
                device = torch.device(config.device)
                model.to(device)
                model.eval()

                # load val pairs
                val_df = pd.read_csv(config.val_csv_path)
                val_pairs = list(zip(val_df["img"].tolist(), val_df["name"].tolist()))
                
                val_transform = make_transform(config, is_train=False)
                # run validation to get metrics (use neg_samples from config)
                val_metrics = validate_and_compute_metrics(model, processor, val_pairs, val_transform, num_negative_samples=getattr(config, "neg_samples", 1))

                current_val_accuracy = val_metrics.get("custom_accuracy") if isinstance(val_metrics, dict) else None
                metrics = {
                    "best_val_accuracy": float(current_val_accuracy) if current_val_accuracy is not None else None,
                    "validation_accuracies": [float(current_val_accuracy)] if current_val_accuracy is not None else []
                }
                logging.info(f"Validation-only experiment done: val_acc={current_val_accuracy}")
            except Exception as e:
                logging.exception(f"Validation-only experiment failed: {e}")
                metrics = {"best_val_accuracy": None, "validation_accuracies": []}
        else:
            # Run full training loop
            try:
                metrics = train_module.main(config)
            except Exception as e:
                logging.exception(f"Training experiment failed: {e}")
                metrics = {"best_val_accuracy": None, "validation_accuracies": []}

        duration = time.time() - start

        row = [
            exp_id,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            model_variant,
            model_name,
            finetune,
            loss_mode,
            use_augment,
            args.seed,
            str(out_dir),
            metrics.get("best_val_accuracy"),
            ";".join([f"{v:.6f}" for v in (metrics.get("validation_accuracies") or [])])
        ]
        with results_csv.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        logging.info(f"Finished exp {exp_id} in {duration:.1f}s — best_val_accuracy={metrics.get('best_val_accuracy')}")
    logging.info(f"All experiments finished. Results in {results_csv}")

if __name__ == "__main__":
    main()
