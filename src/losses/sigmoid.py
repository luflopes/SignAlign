"""
Sigmoid Loss para SigLIP.

Implementação baseada no paper "Sigmoid Loss for Language Image Pre-Training".
Usa binary cross entropy ao invés de softmax cross entropy.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidLoss(nn.Module):
    """
    Sigmoid Loss usada no SigLIP.
    
    Diferente do InfoNCE/softmax, trata cada par como uma predição
    binária independente (é par positivo ou não).
    
    Vantagens:
    - Escala melhor para batches grandes
    - Não assume que há exatamente 1 positivo por linha/coluna
    - Mais estável numericamente
    """
    
    def __init__(self, bias: float = -10.0, temperature: float = 1.0):
        """
        Args:
            bias: Bias inicial para logits (ajuda estabilidade).
            temperature: Escala para similaridades.
        """
        super().__init__()
        self.bias = bias
        self.temperature = temperature
    
    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        return_similarities: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Calcula Sigmoid Loss.
        
        Args:
            image_embeds: (B, D) embeddings de imagem.
            text_embeds: (B, D) embeddings de texto.
            return_similarities: Se True, retorna similaridades.
            
        Returns:
            Dicionário com loss e opcionalmente métricas adicionais.
        """
        # Normalizar embeddings
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        batch_size = image_embeds.size(0)
        device = image_embeds.device
        
        # Calcular logits
        # (B, B) matriz de similaridade
        logits = (image_embeds @ text_embeds.t()) / self.temperature + self.bias
        
        # Labels: diagonal = 1 (par positivo), resto = 0 (negativo)
        labels = torch.eye(batch_size, device=device)
        
        # Binary Cross Entropy com logits
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        result = {"loss": loss}
        
        if return_similarities:
            # Similaridades sem bias
            sims = (image_embeds @ text_embeds.t()) / self.temperature
            pos_sims = sims.diagonal()
            mask = torch.eye(batch_size, device=device).bool()
            neg_sims = sims[~mask].view(batch_size, -1)
            
            result["pos_similarities"] = pos_sims
            result["neg_similarities"] = neg_sims
            result["logits"] = logits
        
        return result


class SigmoidLossFromLogits(nn.Module):
    """
    Sigmoid Loss aplicada diretamente nos logits do modelo.
    
    Para uso quando o modelo já retorna logits escalados.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        logits_per_image: torch.Tensor,
        logits_per_text: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula Sigmoid Loss a partir dos logits do modelo.
        
        Args:
            logits_per_image: (B, B) logits imagem → texto.
            logits_per_text: (B, B) logits texto → imagem.
            
        Returns:
            Loss escalar.
        """
        batch_size = logits_per_image.size(0)
        device = logits_per_image.device
        
        # Labels diagonais
        labels = torch.eye(batch_size, device=device)
        
        # BCE para ambas direções
        loss_i2t = F.binary_cross_entropy_with_logits(logits_per_image, labels)
        loss_t2i = F.binary_cross_entropy_with_logits(logits_per_text, labels)
        
        return (loss_i2t + loss_t2i) / 2

