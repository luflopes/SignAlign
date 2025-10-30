import argparse
import random
import torch
import torch.optim as optim
import pandas as pd
from pathlib import Path
from src.dataset import (
    SignatureNameDataset,
    collate_fn,
    build_unique_name_batches,
    make_transform,
    TripletDataset,
    collate_fn_triplets,
)
from torch.utils.data import DataLoader
import torch.nn.functional as F
from src.utils import set_seed
from src.validate import validate_and_compute_metrics, validate_and_visualize
from src.loss import CrossEntropyLoss, ContrastiveLoss, TripletLoss
from tqdm.auto import tqdm
from src.models import create_model, create_processor



def main(config):

    # Set seed for reproducibility
    set_seed(config.seed)

    # convert/normalize args
    output_dir: Path = Path(config.output_dir)
    device = torch.device(config.device)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameters
    print("Training configuration:")
    print(f"  seed={config.seed}")
    print(f"  image_size={config.image_size} neg_samples={config.neg_samples}")
    print(f"  model_name={config.model_name} model_variant={getattr(config,'model_variant',None)} finetune={getattr(config,'finetune',False)}")
    print(f"  output_dir={config.output_dir}")
    print(f"  best_model_path={config.best_model_path}")
    print(f"  epochs={config.epochs} batch_size={config.batch_size} lr={config.lr} weight_decay={config.weight_decay}")
    print(f"  device={device} use_amp={config.use_amp} num_workers={config.num_workers}")
    print(f"  margin={config.margin} contrastive_weight={config.contrastive_weight}")
    print(f"  max_grad_norm={config.max_grad_norm} save_every={config.save_every} log_interval={config.log_interval}")
    print(f"  loss_mode={getattr(config,'loss_mode', 'ce')} use_augmentations={getattr(config,'use_augmentations', True)}")
    temperature = getattr(config, "temperature", 1.0)
    print(f"  scheduler_use={getattr(config,'use_scheduler', False)} scheduler_patience={getattr(config,'scheduler_patience',None)} scheduler_factor={getattr(config,'scheduler_factor',None)}")
    print(f"  early_stopping_patience={getattr(config,'early_stopping_patience', None)}")

    best_val_accuracy = 0.0 # Initialize with 0.0 since we want to maximize accuracy
    best_model_path = None
    validation_accuracies = [] # To store accuracy for plotting later

    model = create_model(config.model_name)
    processor = create_processor(config.model_name)
    model.to(device)

    # if finetune == False -> freeze backbone parameters (user requested flexible finetune)
    if hasattr(config, "finetune") and not config.finetune:
        for p in model.parameters():
            p.requires_grad = False

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, weight_decay=config.weight_decay)

    # Scheduler based on validation accuracy (ReduceLROnPlateau)
    scheduler = None
    if getattr(config, "use_scheduler", False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",  # we monitor validation accuracy (higher is better)
            factor=getattr(config, "scheduler_factor", 0.5),
            patience=getattr(config, "scheduler_patience", 2),
            threshold=getattr(config, "scheduler_threshold", 1e-4),
            min_lr=getattr(config, "scheduler_min_lr", 1e-8),
            verbose=True
        )

    # Early stopping control
    early_stopping_patience = getattr(config, "early_stopping_patience", None)
    epochs_since_improvement = 0

    # instantiate losses (we will use / combine according to config.loss_mode)
    cross_entropy_loss = CrossEntropyLoss()
    contrastive_loss = ContrastiveLoss()
    triplet_loss = TripletLoss(margin=config.margin)

    # If training with explicit triplets (CSV), create TripletDataset + DataLoader
    triplet_loader = None
    using_triplet_dataset = getattr(config, "loss_mode", "ce") in ("triplet", "ce+triplet") and getattr(config, "train_triplets_csv_path", None) is not None
    if using_triplet_dataset:
        triplet_ds = TripletDataset(csv_path=Path(config.train_triplets_csv_path), transform=make_transform(config, is_train=True))
        triplet_loader = DataLoader(triplet_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=lambda b: collate_fn_triplets(b, processor))

    train_df = pd.read_csv(config.train_csv_path)
    val_df = pd.read_csv(config.val_csv_path)

    train_pairs = list(zip(train_df["img"].tolist(), train_df["name"].tolist()))
    val_pairs = list(zip(val_df["img"].tolist(), val_df["name"].tolist()))

    print(f"Size of training set: {len(train_pairs)}")
    print(f"Size of validation set: {len(val_pairs)}")

    # Use augmentations only if requested
    if getattr(config, "use_augmentations", True):
        train_transform = make_transform(config, is_train=True)
    else:
        train_transform = make_transform(config, is_train=False)

    val_transform = make_transform(config, is_train=False)

    # Prepara batches únicos
    unique_batches = build_unique_name_batches(train_pairs, batch_size=config.batch_size)
    print(f"🔹 Total unique batches: {len(unique_batches)}")


    # Training loop
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        processed_batches = 0
        sum_pos_sim = 0.0
        sum_neg_sim = 0.0 # Keep tracking negative sim for reporting

        if using_triplet_dataset and triplet_loader is not None:
            batch_iterator = tqdm(triplet_loader, desc=f"Epoch {epoch+1}/{config.epochs} TripletTraining", leave=False)
            for step, batch in enumerate(batch_iterator, start=1):
                # batch is a dict from collate_fn_triplets
                text_input_ids = batch["text_input_ids"].to(device)
                text_attention_mask = batch.get("text_attention_mask")
                if text_attention_mask is not None:
                    text_attention_mask = text_attention_mask.to(device)
                pos_pixel_values = batch["pos_pixel_values"].to(device)
                neg_pixel_values = batch["neg_pixel_values"].to(device)

                with torch.amp.autocast('cuda', enabled=(config.use_amp and device.type == "cuda")):
                    optimizer.zero_grad()

                    # compute embeddings
                    anchor_emb = model.get_text_features(input_ids=text_input_ids, attention_mask=text_attention_mask)
                    pos_emb = model.get_image_features(pixel_values=pos_pixel_values)
                    neg_emb = model.get_image_features(pixel_values=neg_pixel_values)

                    # Triplet loss (explicit)
                    trip_loss_val = triplet_loss(anchor_emb, pos_emb, neg_emb)

                    # If also want CE term, build per-anchor logits over [pos,neg]
                    loss_mode = getattr(config, "loss_mode", "ce")
                    if loss_mode == "triplet":
                        loss = trip_loss_val
                    elif loss_mode == "ce+triplet":
                        # compute similarity between anchor and [pos,neg]
                        anchor_n = F.normalize(anchor_emb, dim=-1)
                        pos_n = F.normalize(pos_emb, dim=-1)
                        neg_n = F.normalize(neg_emb, dim=-1)
                        # sims: (N, 2)
                        sims_pos = torch.sum(anchor_n * pos_n, dim=-1, keepdim=True)
                        sims_neg = torch.sum(anchor_n * neg_n, dim=-1, keepdim=True)
                        sims = torch.cat([sims_pos, sims_neg], dim=1) / float(temperature)
                        labels = torch.zeros(sims.size(0), dtype=torch.long, device=device)
                        ce_term = F.cross_entropy(sims, labels)
                        loss = ce_term + trip_loss_val * config.contrastive_weight
                    else:
                        # fallback to triplet
                        loss = trip_loss_val

                loss.backward()
                # gradient clipping if requested
                if config.max_grad_norm and config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), config.max_grad_norm)
                optimizer.step()

                total_loss += loss.item()
                processed_batches += 1
                # We can compute pos/neg sims for reporting
                with torch.no_grad():
                    sum_pos_sim += torch.mean(F.cosine_similarity(F.normalize(pos_emb, dim=-1), F.normalize(anchor_emb, dim=-1), dim=-1)).item()
                    sum_neg_sim += torch.mean(F.cosine_similarity(F.normalize(neg_emb, dim=-1), F.normalize(anchor_emb, dim=-1), dim=-1)).item()

                batch_iterator.set_postfix(loss=loss.item())
        else:
            random.shuffle(unique_batches)

            # Wrap unique_batches with tqdm for a progress bar
            batch_iterator = tqdm(unique_batches, desc=f"Epoch {epoch+1}/{config.epochs} Training", leave=False)

            for step, batch_pairs in enumerate(batch_iterator, start=1):
                # Use the collate_fn to process and batch the data
                dataset_batch = SignatureNameDataset(batch_pairs, train_transform)
                batch = collate_fn(dataset_batch, processor).to(device)

                with torch.amp.autocast('cuda', enabled=(config.use_amp and device.type == "cuda")):
                    optimizer.zero_grad()

                    outputs = model(**batch)
                    logits_per_image = outputs.logits_per_image
                    logits_per_text = outputs.logits_per_text

                    B = logits_per_image.size(0)
                    labels = torch.arange(B, device=device)

                    # CLIP CE Loss
                    ce_loss = cross_entropy_loss(logits_per_image, logits_per_text, labels)

                    # Triplet Loss (legacy batch mode)
                    trip_loss = triplet_loss(outputs.image_embeds, outputs.text_embeds, labels)

                    # Decide final loss based on config.loss_mode
                    loss_mode = getattr(config, "loss_mode", "ce")
                    if loss_mode == "ce":
                        loss = ce_loss
                    elif loss_mode == "triplet":
                        loss = trip_loss
                    elif loss_mode == "ce+triplet":
                        loss = ce_loss + trip_loss * config.contrastive_weight
                    else:
                        # fallback
                        loss = ce_loss

                loss.backward()
                # gradient clipping if requested
                if config.max_grad_norm and config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), config.max_grad_norm)
                optimizer.step()

                total_loss += loss.item()
                processed_batches += 1
                # Still calculate pos and neg sim for reporting, even if not used in loss
                with torch.no_grad():
                    _, pos_sims, neg_sims = contrastive_loss(outputs.image_embeds.detach(), outputs.text_embeds.detach(), B)
                    sum_pos_sim += pos_sims.mean().item()
                    sum_neg_sim += neg_sims.mean().item()

                # Update tqdm description with current batch loss (optional, but helpful)
                batch_iterator.set_postfix(loss=loss.item())


        epoch_avg_loss = total_loss/processed_batches
        epoch_avg_pos_sim = sum_pos_sim/processed_batches
        epoch_avg_neg_sim = sum_neg_sim/processed_batches


        # Validation and Accuracy Calculation
        current_val_accuracy = -1.0 # Initialize with a value that won't be the best initially
        if (epoch + 1) % config.save_every == 0: # Run validation every SAVE_EVERY epochs
            model.eval() # Set model to evaluation mode
            validation_metrics = validate_and_compute_metrics(model, processor, val_pairs, val_transform, num_negative_samples=getattr(config,'neg_samples',3))
            current_val_accuracy = validation_metrics["custom_accuracy"]
            validation_accuracies.append(current_val_accuracy) # Store accuracy

            # Scheduler step (monitoring validation accuracy)
            if scheduler is not None:
                scheduler.step(current_val_accuracy)  # ReduceLROnPlateau expects the metric value

            # Check for best validation accuracy and save model
            if current_val_accuracy > best_val_accuracy:
                best_val_accuracy = current_val_accuracy
                best_model_path = Path(output_dir) / "best" if not config.best_model_path else Path(config.best_model_path)
                best_model_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_model_path)
                processor.save_pretrained(best_model_path)
                print(f"🌟 Nova melhor acurácia de validação ({best_val_accuracy:.4f}). Melhor modelo salvo em: {best_model_path}")
                # Save / show validation visualization examples to output_dir for inspection
                try:
                    validate_and_visualize(model, processor, val_pairs, val_transform, num_pairs_to_display=10, num_negative_samples=getattr(config, 'neg_samples', 3), save_dir=output_dir)
                except Exception as e:
                    print(f"Warning: validate_and_visualize failed: {e}")
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            # Early stopping check
            if early_stopping_patience is not None and epochs_since_improvement >= early_stopping_patience:
                print(f"⏸️ Early stopping: sem melhoria por {early_stopping_patience} epochs. Parando treino.")
                break

            validate_and_visualize(model, processor, val_pairs, val_transform, num_pairs_to_display=10, num_negative_samples=3)
        # Print combined metrics for the epoch
        print(f"✅ Epoch {epoch+1}/{config.epochs} | Train Loss: {epoch_avg_loss:.4f} | Train Avg Pos Sim: {epoch_avg_pos_sim:.4f} | Train Avg Neg Sim: {epoch_avg_neg_sim:.4f} | Val Accuracy: {current_val_accuracy:.4f}")

    print("🏁 Fine-tuning finalizado.")

    # Retorna métricas para uso por runners/experiments
    metrics = {
        "best_val_accuracy": float(best_val_accuracy),
        "validation_accuracies": validation_accuracies,
        "final_epoch": epoch + 1
    }
    return metrics
