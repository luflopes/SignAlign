"""
Predictor para inferência com modelos treinados.

Permite carregar modelos e fazer predições de similaridade.
"""

from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.models.registry import load_model_from_experiment
from src.models.clip_models import load_from_checkpoint
from src.data.dataset import paste_center_on_canvas
from src.data.augmentations import get_inference_transform


class SignaturePredictor:
    """
    Preditor de similaridade texto-assinatura.
    
    Carrega um modelo treinado e permite fazer predições
    de similaridade entre nomes e assinaturas.
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "tinyclip",
        device: Optional[torch.device] = None,
        image_size: int = 224
    ):
        """
        Args:
            model_path: Caminho para o checkpoint do modelo.
            model_type: Tipo do modelo ("clip", "tinyclip", "siglip").
            device: Dispositivo de computação (default: auto-detect).
            image_size: Tamanho das imagens.
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.image_size = image_size
        self.model_type = model_type
        
        # Carregar modelo
        print(f"📦 Carregando modelo de: {model_path}")
        self.model = load_from_checkpoint(
            model_path,
            model_type=model_type,
            output_attentions=True
        )
        self.model.model = self.model.model.to(self.device)
        self.model.model.eval()
        
        self.processor = self.model.get_processor()
        self.transform = get_inference_transform()
        
        print(f"✅ Modelo carregado em {self.device}")
    
    @classmethod
    def from_experiment(
        cls,
        experiment_path: str,
        checkpoint: str = "best",
        model_type: str = "tinyclip",
        device: Optional[torch.device] = None,
        image_size: int = 224
    ) -> "SignaturePredictor":
        """
        Carrega predictor de um experimento salvo.
        
        Args:
            experiment_path: Caminho do experimento.
            checkpoint: Nome do checkpoint ("best" ou "epoch_XXX").
            model_type: Tipo do modelo.
            device: Dispositivo.
            image_size: Tamanho da imagem.
            
        Returns:
            Instância do predictor.
        """
        checkpoint_path = Path(experiment_path) / "checkpoints" / checkpoint
        return cls(
            str(checkpoint_path),
            model_type=model_type,
            device=device,
            image_size=image_size
        )
    
    def preprocess_image(self, image_path: str) -> Image.Image:
        """Preprocessa uma imagem."""
        img = Image.open(image_path)
        return paste_center_on_canvas(img, self.image_size)
    
    def get_image_embedding(
        self,
        image_path: str
    ) -> torch.Tensor:
        """
        Obtém embedding de uma imagem.
        
        Args:
            image_path: Caminho da imagem.
            
        Returns:
            Tensor de embedding normalizado.
        """
        img = self.preprocess_image(image_path)
        inputs = self.processor(images=[img], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            embed = self.model.get_image_features(**inputs)
        
        return F.normalize(embed, dim=-1)
    
    def get_text_embedding(self, text: str) -> torch.Tensor:
        """
        Obtém embedding de um texto.
        
        Args:
            text: Texto (nome).
            
        Returns:
            Tensor de embedding normalizado.
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            embed = self.model.get_text_features(**inputs)
        
        return F.normalize(embed, dim=-1)
    
    def compute_similarity(
        self,
        text: str,
        image_path: str
    ) -> float:
        """
        Computa similaridade entre texto e imagem.
        
        Args:
            text: Texto (nome).
            image_path: Caminho da imagem.
            
        Returns:
            Score de similaridade (cosine similarity).
        """
        text_embed = self.get_text_embedding(text)
        image_embed = self.get_image_embedding(image_path)
        
        similarity = (text_embed @ image_embed.t()).item()
        return similarity
    
    def rank_signatures(
        self,
        query_text: str,
        candidate_images: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Rankeia assinaturas candidatas para um nome.
        
        Args:
            query_text: Nome para busca.
            candidate_images: Lista de caminhos de imagens candidatas.
            
        Returns:
            Lista ordenada de (caminho, similaridade).
        """
        text_embed = self.get_text_embedding(query_text)
        
        results = []
        for img_path in candidate_images:
            img_embed = self.get_image_embedding(img_path)
            sim = (text_embed @ img_embed.t()).item()
            results.append((img_path, sim))
        
        # Ordenar por similaridade (maior primeiro)
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def batch_similarity(
        self,
        texts: List[str],
        images: List[str]
    ) -> np.ndarray:
        """
        Computa matriz de similaridade entre textos e imagens.
        
        Args:
            texts: Lista de textos.
            images: Lista de caminhos de imagens.
            
        Returns:
            Matriz (len(texts), len(images)) de similaridades.
        """
        # Embeddings de texto
        text_embeds = []
        for text in texts:
            embed = self.get_text_embedding(text)
            text_embeds.append(embed)
        text_embeds = torch.cat(text_embeds, dim=0)
        
        # Embeddings de imagem
        image_embeds = []
        for img_path in images:
            embed = self.get_image_embedding(img_path)
            image_embeds.append(embed)
        image_embeds = torch.cat(image_embeds, dim=0)
        
        # Matriz de similaridade
        similarity_matrix = (text_embeds @ image_embeds.t()).cpu().numpy()
        
        return similarity_matrix
    
    def predict_with_visualization(
        self,
        text: str,
        image_path: str,
        show: bool = True,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Faz predição e visualiza resultado com attention.
        
        Args:
            text: Texto (nome).
            image_path: Caminho da imagem.
            show: Se deve exibir.
            save_path: Caminho para salvar.
            
        Returns:
            Dicionário com similaridade e paths.
        """
        from src.explainability.attention_rollout import AttentionRollout
        
        similarity = self.compute_similarity(text, image_path)
        
        # Visualizar atenção
        rollout = AttentionRollout(self.model, self.processor, self.device)
        rollout.visualize(
            image_path, text, self.image_size,
            save_path=save_path, show=show
        )
        
        return {
            "text": text,
            "image_path": image_path,
            "similarity": similarity
        }
    
    def verify_signature(
        self,
        claimed_name: str,
        signature_image: str,
        threshold: float = 0.15
    ) -> Dict[str, Any]:
        """
        Verifica se uma assinatura corresponde ao nome alegado.
        
        Args:
            claimed_name: Nome alegado.
            signature_image: Caminho da imagem da assinatura.
            threshold: Limiar de similaridade para aceitar.
            
        Returns:
            Dicionário com resultado da verificação.
        """
        similarity = self.compute_similarity(claimed_name, signature_image)
        is_match = similarity >= threshold
        
        return {
            "claimed_name": claimed_name,
            "signature_image": signature_image,
            "similarity": similarity,
            "threshold": threshold,
            "is_match": is_match,
            "confidence": "high" if similarity > threshold + 0.1 else (
                "medium" if is_match else "low"
            )
        }

