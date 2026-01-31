"""
Triplet Loss para aprendizado métrico.

Força que pares positivos tenham similaridade maior que
negativos por uma margem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet Loss com margem.
    
    Para cada âncora (imagem), força:
    sim(imagem, texto_positivo) > sim(imagem, texto_negativo) + margem
    
    Usa todos os negativos do batch para cada âncora.
    """
    
    def __init__(self, margin: float = 0.2):
        """
        Args:
            margin: Margem de separação positivo-negativo.
        """
        super().__init__()
        self.margin = margin
        self.relu = nn.ReLU()
    
    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        labels: torch.Tensor = None  # Não usado, mantido para compatibilidade
    ) -> torch.Tensor:
        """
        Calcula Triplet Loss.
        
        Args:
            image_embeds: (B, D) embeddings de imagem.
            text_embeds: (B, D) embeddings de texto.
            labels: Ignorado (assume diagonal como positivos).
            
        Returns:
            Loss escalar.
        """
        # Normalizar
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        batch_size = image_embeds.size(0)
        device = image_embeds.device
        
        # Matriz de similaridade
        similarities = image_embeds @ text_embeds.t()  # (B, B)
        
        # Similaridade positiva (diagonal)
        positive_sims = similarities.diag()  # (B,)
        
        # Para cada âncora, calcular loss com todos os negativos
        total_loss = 0.0
        num_triplets = 0
        
        for i in range(batch_size):
            # Similaridade com positivo
            sim_positive = positive_sims[i]
            
            # Similaridades com negativos (todos exceto o positivo)
            # Seleciona linha i, remove posição i
            sim_negatives = torch.cat([
                similarities[i, :i],
                similarities[i, i+1:]
            ])  # (B-1,)
            
            # Loss: max(0, sim_neg - sim_pos + margin)
            loss_for_anchor = self.relu(sim_negatives - sim_positive + self.margin)
            
            total_loss += loss_for_anchor.sum()
            num_triplets += batch_size - 1
        
        # Média sobre todos os triplets
        if num_triplets > 0:
            return total_loss / num_triplets
        return torch.tensor(0.0, device=device)


class TripletLossHardMining(nn.Module):
    """
    Triplet Loss com Hard Negative Mining.
    
    Usa apenas o negativo mais difícil (maior similaridade)
    para cada âncora.
    """
    
    def __init__(self, margin: float = 0.2):
        """
        Args:
            margin: Margem de separação.
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula Triplet Loss com hard mining.
        
        Args:
            image_embeds: (B, D) embeddings de imagem.
            text_embeds: (B, D) embeddings de texto.
            
        Returns:
            Loss escalar.
        """
        # Normalizar
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        batch_size = image_embeds.size(0)
        device = image_embeds.device
        
        # Matriz de similaridade
        similarities = image_embeds @ text_embeds.t()  # (B, B)
        
        # Similaridade positiva
        positive_sims = similarities.diag()  # (B,)
        
        # Máscara para negativos (tudo exceto diagonal)
        mask = torch.eye(batch_size, device=device).bool()
        neg_similarities = similarities.masked_fill(mask, float('-inf'))
        
        # Hardest negative: maior similaridade negativa
        hardest_neg_sims = neg_similarities.max(dim=1)[0]  # (B,)
        
        # Loss: max(0, sim_hard_neg - sim_pos + margin)
        losses = F.relu(hardest_neg_sims - positive_sims + self.margin)
        
        return losses.mean()


class TripletLossSemiHard(nn.Module):
    """
    Triplet Loss com Semi-Hard Mining.
    
    Usa negativos que estão entre positivo e positivo+margin
    (mais informativos para o gradiente).
    """
    
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula Triplet Loss semi-hard.
        """
        # Normalizar
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        batch_size = image_embeds.size(0)
        device = image_embeds.device
        
        # Matriz de similaridade
        similarities = image_embeds @ text_embeds.t()
        positive_sims = similarities.diag().unsqueeze(1)  # (B, 1)
        
        # Máscara diagonal
        mask = torch.eye(batch_size, device=device).bool()
        
        # Semi-hard: negativos onde sim_pos - margin < sim_neg < sim_pos
        lower_bound = positive_sims - self.margin
        upper_bound = positive_sims
        
        semi_hard_mask = (
            (similarities > lower_bound) &
            (similarities < upper_bound) &
            ~mask
        )
        
        # Se não houver semi-hard, usa hard mining
        if not semi_hard_mask.any():
            neg_similarities = similarities.masked_fill(mask, float('-inf'))
            hardest_neg_sims = neg_similarities.max(dim=1)[0]
            losses = F.relu(hardest_neg_sims - positive_sims.squeeze() + self.margin)
        else:
            # Média dos semi-hard negatives
            semi_hard_sims = similarities.masked_fill(~semi_hard_mask, 0)
            counts = semi_hard_mask.sum(dim=1).clamp(min=1)
            avg_semi_hard = semi_hard_sims.sum(dim=1) / counts
            losses = F.relu(avg_semi_hard - positive_sims.squeeze() + self.margin)
        
        return losses.mean()

