"""
Loss combinada e factory para criação de funções de perda.

Permite combinar diferentes losses com pesos configuráveis.
"""

from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn

from src.losses.contrastive import CrossEntropyLoss, InfoNCELoss
from src.losses.sigmoid import SigmoidLoss, SigmoidLossFromLogits
from src.losses.triplet import TripletLoss
from src.config.experiment_config import LossConfig


class CombinedLoss(nn.Module):
    """
    Loss combinada: Primary Loss + λ * Auxiliary Loss.
    
    Permite combinar InfoNCE/CrossEntropy/Sigmoid com TripletLoss.
    """
    
    def __init__(
        self,
        primary_loss: nn.Module,
        auxiliary_loss: Optional[nn.Module] = None,
        auxiliary_weight: float = 0.0
    ):
        """
        Args:
            primary_loss: Loss principal (InfoNCE, Sigmoid, etc.).
            auxiliary_loss: Loss auxiliar opcional (ex: TripletLoss).
            auxiliary_weight: Peso da loss auxiliar (λ).
        """
        super().__init__()
        self.primary_loss = primary_loss
        self.auxiliary_loss = auxiliary_loss
        self.auxiliary_weight = auxiliary_weight
    
    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        logits_per_image: Optional[torch.Tensor] = None,
        logits_per_text: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Calcula loss combinada.
        
        Args:
            image_embeds: Embeddings de imagem.
            text_embeds: Embeddings de texto.
            logits_per_image: Logits do modelo (para CrossEntropy).
            logits_per_text: Logits do modelo.
            return_components: Se True, retorna componentes individuais.
            
        Returns:
            Dicionário com loss total e opcionalmente componentes.
        """
        batch_size = image_embeds.size(0)
        device = image_embeds.device
        
        # Calcular loss primária
        if isinstance(self.primary_loss, CrossEntropyLoss):
            labels = torch.arange(batch_size, device=device)
            primary_result = self.primary_loss(
                logits_per_image, logits_per_text, labels
            )
            # CrossEntropyLoss retorna escalar direto
            if isinstance(primary_result, torch.Tensor):
                primary_loss_value = primary_result
            else:
                primary_loss_value = primary_result["loss"]
            
            # Extrair similaridades dos logits para tracking
            if logits_per_image is not None:
                # Os logits já são similaridades escaladas pela temperatura do modelo
                pos_sims = logits_per_image.diagonal()
                mask = torch.eye(batch_size, device=device).bool()
                neg_sims = logits_per_image[~mask].view(batch_size, -1)
            else:
                pos_sims = None
                neg_sims = None
        elif isinstance(self.primary_loss, (InfoNCELoss, SigmoidLoss)):
            primary_result = self.primary_loss(
                image_embeds, text_embeds, return_similarities=True
            )
            primary_loss_value = primary_result["loss"]
            pos_sims = primary_result.get("pos_similarities")
            neg_sims = primary_result.get("neg_similarities")
        elif isinstance(self.primary_loss, SigmoidLossFromLogits):
            primary_loss_value = self.primary_loss(logits_per_image, logits_per_text)
            pos_sims = None
            neg_sims = None
        else:
            raise ValueError(f"Loss primária não suportada: {type(self.primary_loss)}")
        
        total_loss = primary_loss_value
        auxiliary_loss_value = torch.tensor(0.0, device=device)
        
        # Calcular loss auxiliar se existir
        if self.auxiliary_loss is not None and self.auxiliary_weight > 0:
            auxiliary_loss_value = self.auxiliary_loss(image_embeds, text_embeds)
            total_loss = total_loss + self.auxiliary_weight * auxiliary_loss_value
        
        result = {"loss": total_loss}
        
        if return_components:
            result["primary_loss"] = primary_loss_value
            result["auxiliary_loss"] = auxiliary_loss_value
            if pos_sims is not None:
                result["pos_similarities"] = pos_sims
            if neg_sims is not None:
                result["neg_similarities"] = neg_sims
        
        return result


def create_loss_function(config: LossConfig) -> CombinedLoss:
    """
    Factory para criar função de perda a partir de configuração.
    
    Args:
        config: Configuração de loss.
        
    Returns:
        CombinedLoss configurada.
    """
    # Criar loss primária
    if config.primary_loss == "infonce":
        primary = InfoNCELoss(temperature=config.temperature)
    elif config.primary_loss == "sigmoid":
        primary = SigmoidLoss(
            bias=config.sigmoid_bias,
            temperature=config.temperature
        )
    elif config.primary_loss == "cross_entropy":
        primary = CrossEntropyLoss(temperature=config.temperature)
    else:
        raise ValueError(f"Loss primária não suportada: {config.primary_loss}")
    
    # Criar loss auxiliar (Triplet) se configurada
    auxiliary = None
    if config.use_triplet_loss:
        auxiliary = TripletLoss(margin=config.triplet_margin)
    
    loss_fn = CombinedLoss(
        primary_loss=primary,
        auxiliary_loss=auxiliary,
        auxiliary_weight=config.triplet_weight if config.use_triplet_loss else 0.0
    )
    
    print(f"🎯 Loss configurada:")
    print(f"   - Primária: {config.primary_loss} (temp={config.temperature})")
    if config.use_triplet_loss:
        print(f"   - Auxiliar: TripletLoss (margin={config.triplet_margin}, λ={config.triplet_weight})")
    
    return loss_fn

