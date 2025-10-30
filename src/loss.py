import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits_per_image, logits_per_text, labels):
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        ce_loss = (loss_i2t + loss_t2i) / 2
        return ce_loss


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5, margin=0.2):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.cos = nn.CosineSimilarity(dim=1)

    def is_finite_tensor(self, tensor):
        return torch.all(torch.isfinite(tensor))

    def forward(self, image_embeds, text_embeds, batch_size):
        img_embeds = F.normalize(image_embeds, dim=-1)
        txt_embeds = F.normalize(text_embeds, dim=-1)
        sims = img_embeds @ txt_embeds.t()  # (B, B)
        pos_sims = sims.diagonal()
        mask = torch.eye(batch_size, device=img_embeds.device).bool()
        neg_sims = sims[~mask].view(batch_size, -1)
        contrastive_loss = torch.clamp(self.margin + neg_sims.mean(dim=1) - pos_sims, min=0.0).mean()
        return contrastive_loss, pos_sims, neg_sims

class TripletLoss(nn.Module):
    """
    Triplet loss com suporte a:
      - modo explícito: forward(anchor_emb, positive_emb, negative_emb)
          Cada tensor: (N, D). Interpretação típica: anchor = texto, positive = imagem positiva, negative = imagem negativa.
          Loss: mean( ReLU( margin + cos(anchor, negative) - cos(anchor, positive) ) )
    """
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        # Se o terceiro argumento for tensor float -> interpretamos como modo explícito (triplet)
        if torch.is_floating_point(c):
            # a: anchor embeddings (ex.: texto), b: positive embeddings (ex.: imagem), c: negative embeddings (ex.: imagem)
            anchor = F.normalize(a, dim=-1)
            positive = F.normalize(b, dim=-1)
            negative = F.normalize(c, dim=-1)

            pos_sim = F.cosine_similarity(anchor, positive, dim=-1)   # (N,)
            neg_sim = F.cosine_similarity(anchor, negative, dim=-1)   # (N,)

            loss = self.relu(self.margin + neg_sim - pos_sim).mean()
            return loss