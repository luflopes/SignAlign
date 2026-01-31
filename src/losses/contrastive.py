"""
Losses contrastivas: CrossEntropy e InfoNCE.

Implementações padrão usadas no CLIP original.
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss simétrica para CLIP.
    
    Calcula a perda média entre:
    - Imagem → Texto (qual texto corresponde a cada imagem)
    - Texto → Imagem (qual imagem corresponde a cada texto)
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: Escala para logits (já aplicada pelo modelo CLIP).
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        logits_per_image: torch.Tensor,
        logits_per_text: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula CrossEntropy simétrica.
        
        Args:
            logits_per_image: (B, B) logits de imagem para texto.
            logits_per_text: (B, B) logits de texto para imagem.
            labels: (B,) labels para diagonal (arange).
            
        Returns:
            Loss escalar.
        """
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        return (loss_i2t + loss_t2i) / 2


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss (Contrastive Loss com temperatura).
    
    Também conhecida como NT-Xent (Normalized Temperature-scaled Cross Entropy).
    Equivalente à loss original do CLIP.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperatura para escalar similaridades.
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        return_similarities: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Calcula InfoNCE loss.
        
        Args:
            image_embeds: (B, D) embeddings de imagem.
            text_embeds: (B, D) embeddings de texto.
            return_similarities: Se True, retorna também as similaridades.
            
        Returns:
            Dicionário com loss e opcionalmente similaridades.
        """
        # Normalizar embeddings
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        # Calcular matriz de similaridade
        # (B, D) @ (D, B) -> (B, B)
        logits = (image_embeds @ text_embeds.t()) / self.temperature
        
        # Labels: diagonal é o par correto
        batch_size = image_embeds.size(0)
        labels = torch.arange(batch_size, device=image_embeds.device)
        
        # Loss simétrica
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        loss = (loss_i2t + loss_t2i) / 2
        
        result = {"loss": loss}
        
        if return_similarities:
            # Extrair similaridades positivas e negativas
            pos_sims = logits.diagonal() * self.temperature  # Desfazer escala
            mask = torch.eye(batch_size, device=logits.device).bool()
            neg_sims = logits[~mask].view(batch_size, -1) * self.temperature
            
            result["pos_similarities"] = pos_sims
            result["neg_similarities"] = neg_sims
            result["logits"] = logits * self.temperature
        
        return result


class ContrastiveMarginLoss(nn.Module):
    """
    Contrastive Loss com margem.
    
    Penaliza quando a similaridade média dos negativos + margem
    excede a similaridade do positivo.
    """
    
    def __init__(self, margin: float = 0.2):
        """
        Args:
            margin: Margem de separação entre positivos e negativos.
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calcula Contrastive Margin Loss.
        
        Args:
            image_embeds: (B, D) embeddings de imagem.
            text_embeds: (B, D) embeddings de texto.
            
        Returns:
            Tupla (loss, pos_sims, neg_sims).
        """
        # Normalizar
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        batch_size = image_embeds.size(0)
        device = image_embeds.device
        
        # Matriz de similaridade
        sims = image_embeds @ text_embeds.t()
        
        # Positivos (diagonal)
        pos_sims = sims.diagonal()
        
        # Negativos (fora da diagonal)
        mask = torch.eye(batch_size, device=device).bool()
        neg_sims = sims[~mask].view(batch_size, -1)
        
        # Loss: margem + média_negativos - positivo
        loss = torch.clamp(
            self.margin + neg_sims.mean(dim=1) - pos_sims,
            min=0.0
        ).mean()
        
        return loss, pos_sims, neg_sims

