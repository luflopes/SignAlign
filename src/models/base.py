"""
Classes base para modelos multimodais.

Define interface comum para todos os wrappers de modelo.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn


class BaseMultimodalModel(ABC, nn.Module):
    """
    Classe base abstrata para modelos multimodais.
    
    Define interface comum para CLIP, TinyCLIP, SigLIP, etc.
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None
    
    @abstractmethod
    def load_pretrained(self, model_name: str, **kwargs) -> None:
        """Carrega modelo pré-treinado."""
        pass
    
    @abstractmethod
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass do modelo.
        
        Returns:
            Dicionário contendo pelo menos:
            - image_embeds: Embeddings das imagens
            - text_embeds: Embeddings dos textos
            - logits_per_image: Logits imagem → texto
            - logits_per_text: Logits texto → imagem
        """
        pass
    
    @abstractmethod
    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Obtém features da imagem."""
        pass
    
    @abstractmethod
    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Obtém features do texto."""
        pass
    
    def freeze_vision_encoder(self) -> None:
        """Congela o encoder de visão."""
        if hasattr(self.model, 'vision_model'):
            for param in self.model.vision_model.parameters():
                param.requires_grad = False
            print("❄️ Vision encoder congelado.")
    
    def freeze_text_encoder(self) -> None:
        """Congela o encoder de texto."""
        if hasattr(self.model, 'text_model'):
            for param in self.model.text_model.parameters():
                param.requires_grad = False
            print("❄️ Text encoder congelado.")
    
    def unfreeze_all(self) -> None:
        """Descongela todos os parâmetros."""
        for param in self.model.parameters():
            param.requires_grad = True
        print("🔥 Todos os parâmetros descongelados.")
    
    def count_parameters(self) -> Tuple[int, int]:
        """
        Conta parâmetros totais e treináveis.
        
        Returns:
            Tupla (total, trainable).
        """
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total, trainable
    
    def save_pretrained(self, path: str) -> None:
        """Salva modelo e processor."""
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
    
    def get_processor(self):
        """Retorna o processor do modelo."""
        return self.processor

