from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import defaultdict
from PIL import Image
import random
import cv2
import numpy as np
from transformers import CLIPProcessor
from src.utils import paste_center_on_canvas
import pandas as pd
from pathlib import Path

def build_unique_name_batches(pairs, batch_size=16):
    """
    Gera batches onde nenhum nome se repete.
    """
    # Group by name
    by_name = defaultdict(list)
    for img, name in pairs:
        by_name[name].append(img)

    # Create all image/text pairs
    all_items = [(img, name) for name, imgs in by_name.items() for img in imgs]
    random.shuffle(all_items)

    batches = []
    used_names = set()
    batch = []

    for img, name in all_items:
        if name in used_names:
            continue  # It's already in the batch
        batch.append((img, name))
        used_names.add(name)
        if len(batch) == batch_size:
            batches.append(batch)
            batch = []
            used_names.clear()
    if batch:
        batches.append(batch)
    return batches


def build_fixed_validation_batches(val_pairs: List[tuple], num_negative_samples: int = 3, seed: Optional[int] = None):
    """
    Gera uma lista determinística de evaluation_batches (uma lista por nome).
    Cada evaluation_batch é uma lista de pares (img_path, name): primeiro é o positivo, os seguintes são negativos.
    """
    rnd = random.Random(seed)
    pairs_by_name = defaultdict(list)
    for img_path, name in val_pairs:
        pairs_by_name[name].append(img_path)
    unique_names = list(pairs_by_name.keys())

    fixed_batches = []
    for name in unique_names:
        matching_img_paths = pairs_by_name[name]
        if not matching_img_paths:
            continue
        matching_img_path = rnd.choice(matching_img_paths)
        other_names = [n for n in unique_names if n != name]
        non_matching_img_paths = []
        if other_names:
            negative_names = rnd.sample(other_names, min(num_negative_samples, len(other_names)))
            for neg_name in negative_names:
                neg_img = rnd.choice(pairs_by_name[neg_name])
                non_matching_img_paths.append(neg_img)
        evaluation_pairs = [(matching_img_path, name)] + [(p, name) for p in non_matching_img_paths]
        fixed_batches.append(evaluation_pairs)
    return fixed_batches


class SignatureNameDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], transform: A.Compose):
        """
        pairs: lista de (caminho_imagem, texto_nome)
        transform: Albumentations Compose object for image transformations
        """
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, nome = self.pairs[idx]
        img = Image.open(img_path)
        img = paste_center_on_canvas(img, canvas_size=224, background=(255,255,255))
        img_np = np.array(img)
        img_transformed = self.transform(image=img_np)["image"]
        return {"image": img_transformed, "text": nome}


def collate_fn(batch, processor: CLIPProcessor):
    images = [x["image"] for x in batch]
    texts = [x["text"] for x in batch]
    inputs = processor(images=images, text=texts, return_tensors="pt", padding=True)
    return inputs


class TripletDataset(Dataset):
    """
    Dataset de triplets.

    Aceita:
      - lista de tuples: (anchor_img, positive_img, negative_img, anchor_name, pos_name, neg_name)
      - ou path para CSV com colunas esperadas produzidas por build_triplets:
        anchor, positive, negative, anchor_id, pos_id, neg_id, anchor_name, pos_name, neg_name

    __getitem__ retorna dict com chaves:
      "anchor_text" (str), "pos_image" (PIL/np/tensor conforme transform), "neg_image"
    """
    def __init__(self, triplets: Optional[List[Tuple[str,str,str,str,str,str]]] = None,
                 csv_path: Optional[Path] = None,
                 transform: Optional[A.Compose] = None):
        self.transform = transform or (A.Compose([ToTensorV2()]))
        if csv_path is not None:
            df = pd.read_csv(csv_path)
            # support different column orders/names but expect at least anchor,positive,negative and anchor_name,pos_name,neg_name if present
            required = ["anchor","positive","negative"]
            for c in required:
                if c not in df.columns:
                    raise ValueError(f"CSV missing required column '{c}'")
            self.rows = []
            for _, r in df.iterrows():
                anchor = str(r["anchor"])
                positive = str(r["positive"])
                negative = str(r["negative"])
                anchor_name = str(r["anchor_name"]) if "anchor_name" in df.columns else ""
                pos_name = str(r["pos_name"]) if "pos_name" in df.columns else ""
                neg_name = str(r["neg_name"]) if "neg_name" in df.columns else ""
                self.rows.append((anchor, positive, negative, anchor_name, pos_name, neg_name))
        else:
            # expect explicit list
            self.rows = triplets or []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        anchor_img, pos_img, neg_img, anchor_name, pos_name, neg_name = self.rows[idx]
        # load images
        pos = Image.open(pos_img)
        neg = Image.open(neg_img)

        pos = paste_center_on_canvas(pos, canvas_size=224, background=(255,255,255))
        neg = paste_center_on_canvas(neg, canvas_size=224, background=(255,255,255))

        pos_np = np.array(pos)
        neg_np = np.array(neg)

        pos_trans = self.transform(image=pos_np)["image"]
        neg_trans = self.transform(image=neg_np)["image"]

        return {
            "anchor_text": anchor_name,
            "pos_image": pos_trans,
            "neg_image": neg_trans,
            "anchor_id": None,
            "pos_id": None,
            "neg_id": None
        }


def collate_fn_triplets(batch: List[Dict[str, Any]], processor: CLIPProcessor):
    """
    Recebe batch de itens do TripletDataset e retorna dict com:
      - text_inputs: processor outputs for anchor texts (input_ids, attention_mask, ...)
      - pos_images: processor outputs for positive images (pixel_values, ...)
      - neg_images: processor outputs for negative images
    Observação: as chaves são prefixadas para evitar colisão com chamadas ao CLIPModel.
    """
    anchor_texts = [x["anchor_text"] for x in batch]
    pos_images = [x["pos_image"] for x in batch]
    neg_images = [x["neg_image"] for x in batch]

    text_inputs = processor(text=anchor_texts, return_tensors="pt", padding=True)
    pos_image_inputs = processor(images=pos_images, return_tensors="pt", padding=True)
    neg_image_inputs = processor(images=neg_images, return_tensors="pt", padding=True)

    # return combined dict; o treinamento/loop deve usar essas chaves explicitamente
    out = {
        "text_input_ids": text_inputs["input_ids"],
        "text_attention_mask": text_inputs.get("attention_mask"),
        "pos_pixel_values": pos_image_inputs["pixel_values"],
        "neg_pixel_values": neg_image_inputs["pixel_values"],
    }
    return out



def make_transform(config, is_train=True):
    if is_train:
        return A.Compose([
                    A.OneOf([
                        A.ShiftScaleRotate(
                        shift_limit=[-0.0625, 0.0625],
                        scale_limit=[-0.1, 0.1],
                        rotate_limit=[-10, 15],
                        interpolation=cv2.INTER_LINEAR,
                        border_mode=cv2.BORDER_CONSTANT,
                        rotate_method="ellipse",
                        mask_interpolation=cv2.INTER_NEAREST,
                        fill=255,
                        fill_mask=0
                    ),
                    A.Downscale(
                        scale_range=[0.4, 1],
                        interpolation_pair={"upscale":0,"downscale":0}
                    )
                    ], p=0.2),
                    A.OneOf([
                        A.MotionBlur(
                        blur_limit=[3, 5],
                        allow_shifted=False,
                        angle_range=[0, 0],
                        direction_range=[0, 0]
                        ),
                        A.GaussNoise(std_range=(0.1, 0.2), p=0.2),
                    ], p=0.2),
                    A.RandomBrightnessContrast(
                        brightness_limit=[-0.2, 0.2],
                        contrast_limit=[-0.2, 0.2],
                        brightness_by_max=True,
                        ensure_safe_range=False,
                        p=0.2
                    ),
                    A.ImageCompression(quality_range=(40, 100), p=0.2),
                    ToTensorV2()
                ])
    else:
        return A.Compose([
            ToTensorV2()
        ])