"""
Avaliação de retrieval multimodal.

Classes e funções para avaliação detalhada de recuperação.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.data.dataset import SignatureNameDataset, collate_fn


def compute_recall_at_k(
    ranks: List[int],
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Calcula Recall@k a partir de ranks.
    
    Args:
        ranks: Lista de ranks (posições do item correto, 1-indexed).
        k_values: Valores de k.
        
    Returns:
        Dicionário {recall@k: valor}.
    """
    results = {}
    for k in k_values:
        results[f"recall@{k}"] = sum(r <= k for r in ranks) / len(ranks)
    return results


def compute_mrr(ranks: List[int]) -> float:
    """
    Calcula Mean Reciprocal Rank.
    
    MRR = (1/|Q|) * Σ(1/rank_i)
    
    Args:
        ranks: Lista de ranks (1-indexed).
        
    Returns:
        MRR score.
    """
    return np.mean([1.0 / r for r in ranks])


def compute_ndcg(ranks: List[int], k: Optional[int] = None) -> float:
    """
    Calcula Normalized Discounted Cumulative Gain.
    
    Para relevância binária (um item correto por query):
    NDCG = DCG / IDCG = (1/log2(rank+1)) / 1
    
    Args:
        ranks: Lista de ranks (1-indexed).
        k: Cutoff opcional (considera apenas top-k).
        
    Returns:
        NDCG score.
    """
    ndcg_scores = []
    for rank in ranks:
        if k is not None and rank > k:
            ndcg_scores.append(0.0)
        else:
            dcg = 1.0 / np.log2(rank + 1)
            idcg = 1.0  # Item relevante na posição 1
            ndcg_scores.append(dcg / idcg)
    return np.mean(ndcg_scores)


class RetrievalEvaluator:
    """
    Avaliador de retrieval multimodal.
    
    Avalia a capacidade do modelo de recuperar a assinatura
    correta dado um nome textual.
    """
    
    def __init__(
        self,
        model,
        processor,
        device: torch.device,
        image_size: int = 224
    ):
        """
        Args:
            model: Wrapper do modelo.
            processor: Processor do modelo.
            device: Dispositivo de computação.
            image_size: Tamanho das imagens.
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.image_size = image_size
    
    def evaluate_single_query(
        self,
        query_text: str,
        candidate_images: List[str],
        correct_idx: int,
        transform
    ) -> Dict[str, float]:
        """
        Avalia uma única query de retrieval.
        
        Args:
            query_text: Texto da query (nome).
            candidate_images: Lista de caminhos de imagens candidatas.
            correct_idx: Índice da imagem correta.
            transform: Transform para imagens.
            
        Returns:
            Dicionário com métricas.
        """
        self.model.model.eval()
        
        # Criar pares (todas imagens com o mesmo texto)
        pairs = [(img, query_text) for img in candidate_images]
        
        # Preparar batch
        dataset = SignatureNameDataset(pairs, transform, self.image_size)
        batch_data = [dataset[i] for i in range(len(dataset))]
        batch = collate_fn(batch_data, self.processor).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.forward(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
            )
            
            # Normalizar embeddings
            image_embeds = F.normalize(outputs['image_embeds'], dim=-1)
            text_embeds = F.normalize(outputs['text_embeds'], dim=-1)
            
            # Similaridades (usar primeiro texto, todos são iguais)
            single_text = text_embeds[0].unsqueeze(0)
            similarities = (image_embeds @ single_text.t()).squeeze(-1)
            similarities = similarities.cpu().numpy()
        
        # Calcular rank
        sorted_indices = np.argsort(similarities)[::-1]
        rank = np.where(sorted_indices == correct_idx)[0][0] + 1
        
        return {
            "rank": rank,
            "mrr": 1.0 / rank,
            "ndcg": 1.0 / np.log2(rank + 1),
            "correct": rank == 1,
            "similarities": similarities,
        }
    
    def evaluate_dataset(
        self,
        eval_data: Dict,
        num_negative_samples: int,
        transform,
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """
        Avalia todo o dataset de retrieval.
        
        Args:
            eval_data: Dados de avaliação {nome: {positive_img, negative_img_pool}}.
            num_negative_samples: Número de negativos.
            transform: Transform.
            k_values: Valores de k para Recall@k.
            
        Returns:
            Métricas agregadas.
        """
        ranks = []
        all_correct = 0
        
        for name, data in tqdm(
            eval_data.items(),
            desc=f"Retrieval ({num_negative_samples} neg)"
        ):
            positive_img = data["positive_img"]
            negative_pool = data["negative_img_pool"][:num_negative_samples]
            
            if len(negative_pool) < num_negative_samples:
                continue
            
            # Candidatos: positivo primeiro
            candidates = [positive_img] + negative_pool
            
            result = self.evaluate_single_query(
                query_text=name,
                candidate_images=candidates,
                correct_idx=0,
                transform=transform
            )
            
            ranks.append(result["rank"])
            if result["correct"]:
                all_correct += 1
        
        # Métricas agregadas
        metrics = {
            "accuracy": all_correct / len(ranks),
            "mrr": compute_mrr(ranks),
            "ndcg": compute_ndcg(ranks),
            "mean_rank": np.mean(ranks),
            "median_rank": np.median(ranks),
        }
        
        # Recall@k
        recall_metrics = compute_recall_at_k(ranks, k_values)
        metrics.update(recall_metrics)
        
        return metrics
    
    def get_similarity_matrix(
        self,
        texts: List[str],
        images: List[str],
        transform
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Calcula matriz de similaridade texto × imagem.
        
        Args:
            texts: Lista de textos (nomes).
            images: Lista de caminhos de imagens.
            transform: Transform.
            
        Returns:
            Tupla (matriz de similaridade, textos, imagens).
        """
        self.model.model.eval()
        
        # Extrair embeddings de texto
        text_embeds = []
        for text in texts:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                embed = self.model.get_text_features(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask')
                )
            text_embeds.append(embed)
        
        text_embeds = torch.cat(text_embeds, dim=0)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        # Extrair embeddings de imagem
        from src.data.dataset import paste_center_on_canvas
        from PIL import Image
        
        image_embeds = []
        for img_path in images:
            img = Image.open(img_path)
            img = paste_center_on_canvas(img, self.image_size)
            
            inputs = self.processor(
                images=[img],
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                embed = self.model.get_image_features(
                    pixel_values=inputs['pixel_values']
                )
            image_embeds.append(embed)
        
        image_embeds = torch.cat(image_embeds, dim=0)
        image_embeds = F.normalize(image_embeds, dim=-1)
        
        # Matriz de similaridade
        similarity_matrix = (text_embeds @ image_embeds.t()).cpu().numpy()
        
        return similarity_matrix, texts, images

