"""
Dataset para pares texto-assinatura.

Implementa SignatureNameDataset e funções de collate.
"""

from typing import List, Tuple, Dict, Any
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


def paste_center_on_canvas(
    img: Image.Image,
    canvas_size: int = 224,
    background: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """
    Centraliza imagem em canvas quadrado com background.
    
    Args:
        img: Imagem PIL a ser processada.
        canvas_size: Tamanho do canvas quadrado.
        background: Cor de fundo RGB.
        
    Returns:
        Imagem centralizada no canvas.
    """
    img = img.convert("RGBA")
    w, h = img.size
    scale = min(canvas_size / w, canvas_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGBA", (canvas_size, canvas_size), color=background + (255,))
    offset = ((canvas_size - new_w) // 2, (canvas_size - new_h) // 2)
    canvas.paste(img_resized, offset, img_resized)
    return canvas.convert("RGB")


class SignatureNameDataset(Dataset):
    """
    Dataset para pares (imagem de assinatura, nome textual).
    
    Attributes:
        pairs: Lista de tuplas (caminho_imagem, nome).
        transform: Transformações de Albumentations.
        image_size: Tamanho do canvas.
    """
    
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        transform: A.Compose,
        image_size: int = 224
    ):
        """
        Inicializa o dataset.
        
        Args:
            pairs: Lista de (caminho_imagem, texto_nome).
            transform: Albumentations Compose para transformações.
            image_size: Tamanho do canvas quadrado.
        """
        self.pairs = pairs
        self.transform = transform
        self.image_size = image_size
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, nome = self.pairs[idx]
        
        # Carregar e preprocessar imagem
        img = Image.open(img_path)
        img = paste_center_on_canvas(img, canvas_size=self.image_size)
        img_np = np.array(img)
        
        # Aplicar transformações
        img_transformed = self.transform(image=img_np)["image"]
        
        return {
            "image": img_transformed,
            "text": nome,
            "image_path": img_path
        }


def collate_fn(batch: List[Dict[str, Any]], processor) -> Dict[str, torch.Tensor]:
    """
    Função de collate para DataLoader.
    
    Processa batch de imagens e textos usando o processor do modelo.
    
    Args:
        batch: Lista de dicionários com 'image' e 'text'.
        processor: CLIPProcessor ou similar.
        
    Returns:
        Dicionário com tensores prontos para o modelo.
    """
    images = [x["image"] for x in batch]
    texts = [x["text"] for x in batch]
    
    inputs = processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding=True
    )
    
    return inputs


def collate_fn_with_paths(batch: List[Dict[str, Any]], processor) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """
    Função de collate que também retorna os caminhos das imagens.
    
    Args:
        batch: Lista de dicionários com 'image', 'text' e 'image_path'.
        processor: CLIPProcessor ou similar.
        
    Returns:
        Tupla (inputs processados, lista de caminhos).
    """
    images = [x["image"] for x in batch]
    texts = [x["text"] for x in batch]
    paths = [x["image_path"] for x in batch]
    
    inputs = processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding=True
    )
    
    return inputs, paths

