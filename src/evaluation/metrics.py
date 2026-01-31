"""
Métricas de avaliação para modelos multimodais.

Inclui EER, AUC, Accuracy, Recall@k, MRR, NDCG.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm.auto import tqdm

from src.data.dataset import SignatureNameDataset, collate_fn
from src.config.experiment_config import EvaluationConfig


def compute_eer(y_true: List[int], y_scores: List[float]) -> tuple:
    """
    Calcula Equal Error Rate (EER).
    
    EER é o ponto onde FPR == FNR (False Positive Rate == False Negative Rate).
    
    Args:
        y_true: Labels verdadeiros (0 ou 1).
        y_scores: Scores de similaridade.
        
    Returns:
        Tupla (eer, threshold).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    # Encontrar ponto onde FPR ≈ FNR
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]
    
    return eer, eer_threshold


def compute_retrieval_metrics(
    similarities: np.ndarray,
    ground_truth_idx: int = 0,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Calcula métricas de retrieval para uma query.
    
    Args:
        similarities: Array de similaridades [positivo, neg1, neg2, ...].
        ground_truth_idx: Índice do item correto (default 0 = primeiro).
        k_values: Valores de k para Recall@k.
        
    Returns:
        Dicionário com métricas.
    """
    # Ordenar índices por similaridade (maior primeiro)
    sorted_indices = np.argsort(similarities)[::-1]
    
    # Posição do item correto (1-indexed)
    rank = np.where(sorted_indices == ground_truth_idx)[0][0] + 1
    
    metrics = {
        "rank": rank,
        "mrr": 1.0 / rank,  # Mean Reciprocal Rank
    }
    
    # Recall@k
    for k in k_values:
        metrics[f"recall@{k}"] = 1.0 if rank <= k else 0.0
    
    # NDCG (com relevância binária)
    # DCG = rel_i / log2(rank + 1)
    # IDCG = 1 / log2(2) = 1 (para item único relevante na posição 1)
    dcg = 1.0 / np.log2(rank + 1)
    idcg = 1.0  # Relevância 1 na posição 1
    metrics["ndcg"] = dcg / idcg
    
    return metrics


def compute_all_metrics(
    model,
    processor,
    fixed_eval_data: Dict,
    num_negative_samples: int,
    device: torch.device,
    transform,
    image_size: int = 224,
    config: Optional[EvaluationConfig] = None
) -> Dict[str, float]:
    """
    Calcula todas as métricas de avaliação.
    
    Args:
        model: Wrapper do modelo.
        processor: Processor do modelo.
        fixed_eval_data: Dados de avaliação fixos.
        num_negative_samples: Número de negativos por amostra.
        device: Dispositivo para computação.
        transform: Transform para imagens.
        image_size: Tamanho da imagem.
        config: Configuração de avaliação.
        
    Returns:
        Dicionário com todas as métricas.
    """
    if config is None:
        config = EvaluationConfig()
    
    # Verificar se há dados suficientes para avaliação
    if not fixed_eval_data:
        print(f"⚠️ Sem dados de avaliação para {num_negative_samples} negativos")
        return {"accuracy": 0.0, "eer": 0.0, "auc": 0.0, "mrr": 0.0, "ndcg": 0.0}
    
    model.model.eval()
    
    all_similarity_scores = []
    all_labels = []
    correct_predictions = 0
    total_evaluations = 0
    
    all_mrr = []
    all_ndcg = []
    all_recalls = {k: [] for k in config.recall_k_values}
    
    with torch.no_grad():
        for name, data in tqdm(
            fixed_eval_data.items(),
            desc=f"Avaliando ({num_negative_samples} neg)",
            leave=False
        ):
            positive_img_path = data["positive_img"]
            negative_img_pool = data["negative_img_pool"]
            
            # Selecionar negativos
            selected_negatives = negative_img_pool[:min(
                num_negative_samples, len(negative_img_pool)
            )]
            
            if len(selected_negatives) < num_negative_samples:
                continue  # Pular se não houver negativos suficientes
            
            # Criar pares de avaliação
            eval_pairs = [(positive_img_path, name)]
            for neg_img in selected_negatives:
                eval_pairs.append((neg_img, name))
            
            # Preparar batch
            eval_dataset = SignatureNameDataset(eval_pairs, transform, image_size)
            batch_data = [eval_dataset[i] for i in range(len(eval_dataset))]
            batch = collate_fn(batch_data, processor).to(device)
            
            # Forward
            outputs = model.forward(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
            )
            
            image_embeds = outputs['image_embeds']
            text_embeds = outputs['text_embeds']
            
            # Usar primeiro embedding de texto (todos são iguais)
            single_text_embed = text_embeds[0].unsqueeze(0)
            
            # Calcular similaridades
            image_embeds_norm = F.normalize(image_embeds, dim=-1)
            text_embed_norm = F.normalize(single_text_embed, dim=-1)
            similarities = (image_embeds_norm @ text_embed_norm.t()).squeeze(-1)
            similarities = similarities.cpu().numpy()
            
            # Métricas de retrieval
            retrieval_metrics = compute_retrieval_metrics(
                similarities,
                ground_truth_idx=0,
                k_values=config.recall_k_values
            )
            
            all_mrr.append(retrieval_metrics["mrr"])
            all_ndcg.append(retrieval_metrics["ndcg"])
            for k in config.recall_k_values:
                all_recalls[k].append(retrieval_metrics[f"recall@{k}"])
            
            # Accuracy (positivo tem maior similaridade)
            if similarities[0] == max(similarities):
                correct_predictions += 1
            total_evaluations += 1
            
            # Para EER/AUC
            all_similarity_scores.append(similarities[0])
            all_labels.append(1)
            for neg_sim in similarities[1:]:
                all_similarity_scores.append(neg_sim)
                all_labels.append(0)
    
    # Verificar se houve avaliações
    if total_evaluations == 0:
        print(f"⚠️ Nenhuma avaliação possível com {num_negative_samples} negativos (dados insuficientes)")
        return {"accuracy": 0.0}
    
    # Compilar métricas
    metrics = {
        "accuracy": correct_predictions / total_evaluations,
    }
    
    # EER e AUC
    if config.compute_eer and len(set(all_labels)) >= 2:
        eer, eer_threshold = compute_eer(all_labels, all_similarity_scores)
        metrics["eer"] = eer
        metrics["eer_threshold"] = eer_threshold
    
    if config.compute_auc and len(set(all_labels)) >= 2:
        metrics["auc"] = roc_auc_score(all_labels, all_similarity_scores)
    
    # Métricas de retrieval (verificar se há dados)
    if config.compute_mrr and len(all_mrr) > 0:
        metrics["mrr"] = np.mean(all_mrr)
    
    if config.compute_ndcg and len(all_ndcg) > 0:
        metrics["ndcg"] = np.mean(all_ndcg)
    
    if config.compute_recall_at_k:
        for k in config.recall_k_values:
            if len(all_recalls[k]) > 0:
                metrics[f"recall@{k}"] = np.mean(all_recalls[k])
    
    return metrics


def evaluate_model_comprehensive(
    model,
    processor,
    val_pairs: List[tuple],
    device: torch.device,
    transform,
    image_size: int = 224,
    negative_samples_list: List[int] = [1, 2, 3],
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Avaliação completa do modelo com diferentes configurações.
    
    Args:
        model: Wrapper do modelo.
        processor: Processor.
        val_pairs: Pares de validação.
        device: Dispositivo.
        transform: Transform.
        image_size: Tamanho da imagem.
        negative_samples_list: Lista de números de negativos.
        seed: Seed para reprodutibilidade.
        
    Returns:
        Dicionário aninhado {num_neg: {métrica: valor}}.
    """
    from src.data.batch_builder import create_fixed_evaluation_data
    
    # Criar dados fixos
    fixed_eval_data = create_fixed_evaluation_data(
        val_pairs,
        max_negative_samples=max(negative_samples_list),
        seed=seed
    )
    
    results = {}
    
    for num_neg in negative_samples_list:
        metrics = compute_all_metrics(
            model=model,
            processor=processor,
            fixed_eval_data=fixed_eval_data,
            num_negative_samples=num_neg,
            device=device,
            transform=transform,
            image_size=image_size
        )
        results[f"{num_neg}_negative_samples"] = metrics
        
        print(f"\n--- Métricas com {num_neg} Negativos ---")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    
    return results

