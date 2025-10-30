from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from collections import defaultdict
from src.dataset import SignatureNameDataset, collate_fn
import torch
import random
import numpy as np
from src.utils import cosine_similarity, paste_center_on_canvas
from PIL import Image


def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    # print(f"EER: {eer:.4f} (threshold={eer_threshold:.4f})") # Remove print here
    return eer, eer_threshold


def validate_and_compute_metrics(model, processor, val_pairs, val_transform, num_negative_samples=3):
    model.eval()
    all_similarity_scores = []
    all_labels = []
    correct_predictions = 0
    total_evaluations = 0

    # Group pairs by name for easier sampling of negative examples
    pairs_by_name = defaultdict(list)
    for img_path, name in val_pairs:
        pairs_by_name[name].append(img_path)

    unique_names = list(pairs_by_name.keys())

    # print(f"🔹 Total unique names in validation set: {len(unique_names)}") # Remove print here

    with torch.no_grad():
        # Evaluate each unique name once
        for name in unique_names:
            # Get all image paths for this name
            matching_img_paths = pairs_by_name[name]
            if not matching_img_paths:
                continue

            # Randomly select one matching image for the positive pair
            matching_img_path = random.choice(matching_img_paths)

            # Select non-matching images (signatures from different names)
            non_matching_img_paths = []
            other_names = [n for n in unique_names if n != name]
            if len(other_names) > 0:
                # Randomly select names for negative samples
                negative_names = random.sample(other_names, min(num_negative_samples, len(other_names)))
                for neg_name in negative_names:
                    # Select a random signature from the negative name
                    neg_img_path = random.choice(pairs_by_name[neg_name])
                    non_matching_img_paths.append(neg_img_path)

            # Create pairs for the current evaluation
            evaluation_pairs = [(matching_img_path, name)] + [(img_path, name) for img_path in non_matching_img_paths] # Use the correct name for all

            # Prepare batch for the current evaluation (one name, multiple images)
            eval_dataset = SignatureNameDataset(evaluation_pairs, val_transform)
            eval_batch = collate_fn(eval_dataset, processor).to(model.device)

            # Get embeddings
            eval_outputs = model(**eval_batch)
            image_embeds = eval_outputs.image_embeds
            text_embeds = eval_outputs.text_embeds # This will be the embedding for the current 'name'

            # Calculate similarities between the text embedding and all image embeddings in the batch
            # The first image embedding corresponds to the matching image
            similarities = cosine_similarity(image_embeds, text_embeds.squeeze(0)).tolist() # Compare each image embed with the single text embed

            # The similarity of the matching pair is the first one
            matching_similarity = similarities[0]
            # The similarities of the non-matching pairs are the rest
            non_matching_similarities = similarities[1:]

            # For EER and other metrics, we need all individual similarities and their true labels
            all_similarity_scores.append(matching_similarity)
            all_labels.append(1) # True label is 1 for the matching pair

            for non_matching_sim in non_matching_similarities:
                all_similarity_scores.append(non_matching_sim)
                all_labels.append(0) # True label is 0 for non-matching pairs

            # For custom accuracy: check if the matching similarity is the highest among all similarities in this evaluation batch
            if matching_similarity == max(similarities):
                correct_predictions += 1
            total_evaluations += 1 # One evaluation per unique name


    # Calculate EER using all collected similarity scores and labels
    eer, eer_threshold = compute_eer(all_labels, all_similarity_scores)

    # Calculate custom accuracy
    custom_accuracy = correct_predictions / total_evaluations if total_evaluations > 0 else 0

    # Return the metrics
    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "custom_accuracy": custom_accuracy,
        "all_similarity_scores": all_similarity_scores,
        "all_labels": all_labels
    }


def validate_and_visualize(model, processor, val_pairs, val_transform, num_pairs_to_display=10, num_negative_samples=3):

    # This function is now primarily for visualization after training
    metrics = validate_and_compute_metrics(model, processor, val_pairs, val_transform, num_negative_samples=num_negative_samples)

    print("\n### Validation Metrics ###")
    print(f"EER: {metrics['eer']:.4f} (threshold={metrics['eer_threshold']:.4f})")
    print(f"Custom Accuracy (Matching Sim > All Non-Matching Sims): {metrics['custom_accuracy']:.4f}")


    print(f"\n### Showing {num_pairs_to_display} Example Pairs ###")
    # Collect display pairs (similar logic as before, but separate from metric calculation)
    pairs_by_name = defaultdict(list)
    for img_path, name in val_pairs:
        pairs_by_name[name].append(img_path)
    unique_names = list(pairs_by_name.keys())

    with torch.no_grad():
         for name in random.sample(unique_names, min(num_pairs_to_display, len(unique_names))):
            matching_img_paths = pairs_by_name[name]
            if not matching_img_paths:
                continue
            matching_img_path = random.choice(matching_img_paths)

            non_matching_img_paths = []
            other_names = [n for n in unique_names if n != name]
            if len(other_names) > 0:
                negative_names = random.sample(other_names, min(num_negative_samples, len(other_names)))
                for neg_name in negative_names:
                    neg_img_path = random.choice(pairs_by_name[neg_name])
                    non_matching_img_paths.append(neg_img_path)

            evaluation_pairs = [(matching_img_path, name)] + [(img_path, name) for img_path in non_matching_img_paths]
            eval_dataset = SignatureNameDataset(evaluation_pairs, val_transform)
            eval_batch = collate_fn(eval_dataset, processor).to(model.device)

            eval_outputs = model(**eval_batch)
            image_embeds = eval_outputs.image_embeds
            text_embeds = eval_outputs.text_embeds
            similarities = cosine_similarity(image_embeds, text_embeds.squeeze(0)).tolist()

            matching_similarity = similarities[0]
            non_matching_similarities = similarities[1:]

            # Create list of images and their similarities for display
            display_images_data = [{
                "img_path": matching_img_path,
                "similarity": matching_similarity,
                "type": "Matching"
            }]
            for i, neg_img_path in enumerate(non_matching_img_paths):
                display_images_data.append({
                    "img_path": neg_img_path,
                    "similarity": non_matching_similarities[i],
                    "type": f"Non-Matching {i+1}"
                })

            # Display the images
            num_images_to_display = 1 + len(non_matching_img_paths)
            if num_images_to_display > 0:
                fig, axes = plt.subplots(1, num_images_to_display, figsize=(5 * num_images_to_display, 5))
                # Ensure axes is iterable even if there's only one subplot
                if num_images_to_display == 1:
                    axes = [axes]

                for i, img_data in enumerate(display_images_data):
                    img = Image.open(img_data['img_path'])
                    axes[i].imshow(paste_center_on_canvas(img, canvas_size=224, background=(255,255,255)))
                    axes[i].set_title(f"{img_data['type']}\nSimilarity: {img_data['similarity']:.4f}")
                    axes[i].axis('off')

                prediction_correct = matching_similarity == max(similarities)
                prediction_status = "Correct" if prediction_correct else "Incorrect"
                plt.suptitle(f"{name} - Prediction: {prediction_status}", y=1.02, fontsize=14)
                plt.tight_layout()
                plt.show()


                
