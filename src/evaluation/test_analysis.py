"""
Análise detalhada dos resultados de teste.

Salva acertos e erros em CSV para análise posterior.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.data.dataset import SignatureNameDataset, collate_fn
from src.data.batch_builder import create_fixed_evaluation_data


def evaluate_and_save_results(
    model,
    processor,
    test_pairs: List[Tuple[str, str]],
    output_dir: str,
    device: torch.device,
    transform,
    image_size: int = 224,
    num_negative_samples: int = 3,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Avalia modelo no conjunto de teste e salva resultados detalhados.
    
    Args:
        model: Wrapper do modelo.
        processor: Processor do modelo.
        test_pairs: Pares de teste (imagem, nome).
        output_dir: Diretório para salvar resultados.
        device: Dispositivo.
        transform: Transform para imagens.
        image_size: Tamanho da imagem.
        num_negative_samples: Número de negativos por avaliação.
        seed: Seed para reprodutibilidade.
        
    Returns:
        Dicionário com métricas agregadas.
    """
    model.model.eval()
    
    # Criar dados fixos de avaliação
    fixed_eval_data = create_fixed_evaluation_data(
        test_pairs,
        max_negative_samples=num_negative_samples,
        seed=seed
    )
    
    # Preparar diretório de saída
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Listas para resultados
    all_results = []
    correct_count = 0
    total_count = 0
    
    all_pos_similarities = []
    all_neg_similarities = []
    
    with torch.no_grad():
        for name, data in tqdm(
            fixed_eval_data.items(),
            desc=f"Avaliando teste ({num_negative_samples} neg)"
        ):
            positive_img_path = data["positive_img"]
            negative_img_pool = data["negative_img_pool"]
            
            # Selecionar negativos
            selected_negatives = negative_img_pool[:min(
                num_negative_samples, len(negative_img_pool)
            )]
            
            if len(selected_negatives) < num_negative_samples:
                continue
            
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
            
            # Usar primeiro embedding de texto
            single_text_embed = text_embeds[0].unsqueeze(0)
            
            # Calcular similaridades
            image_embeds_norm = F.normalize(image_embeds, dim=-1)
            text_embed_norm = F.normalize(single_text_embed, dim=-1)
            similarities = (image_embeds_norm @ text_embed_norm.t()).squeeze(-1)
            similarities = similarities.cpu().numpy()
            
            # Determinar se acertou
            positive_similarity = float(similarities[0])
            negative_similarities = similarities[1:].tolist()
            max_neg_similarity = float(max(negative_similarities))
            
            is_correct = positive_similarity > max_neg_similarity
            predicted_rank = 1 + sum(1 for neg_sim in negative_similarities if neg_sim >= positive_similarity)
            
            if is_correct:
                correct_count += 1
            total_count += 1
            
            # Guardar similaridades para estatísticas
            all_pos_similarities.append(positive_similarity)
            all_neg_similarities.extend(negative_similarities)
            
            # Salvar resultado
            result = {
                "expected_name": name,
                "image_path": positive_img_path,
                "positive_similarity": positive_similarity,
                "negative_similarities": negative_similarities,
                "max_negative_similarity": max_neg_similarity,
                "predicted_rank": predicted_rank,
                "is_correct": is_correct,
                "negative_image_paths": selected_negatives
            }
            all_results.append(result)
    
    # Calcular métricas agregadas
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    metrics = {
        "accuracy": accuracy,
        "correct": correct_count,
        "total": total_count,
        "num_negative_samples": num_negative_samples,
        "mean_positive_similarity": float(np.mean(all_pos_similarities)) if all_pos_similarities else 0.0,
        "std_positive_similarity": float(np.std(all_pos_similarities)) if all_pos_similarities else 0.0,
        "mean_negative_similarity": float(np.mean(all_neg_similarities)) if all_neg_similarities else 0.0,
        "std_negative_similarity": float(np.std(all_neg_similarities)) if all_neg_similarities else 0.0,
        "similarity_gap": float(np.mean(all_pos_similarities) - np.mean(all_neg_similarities)) if all_pos_similarities else 0.0,
    }
    
    # Calcular MRR
    ranks = [r["predicted_rank"] for r in all_results]
    metrics["mrr"] = float(np.mean([1.0/r for r in ranks])) if ranks else 0.0
    metrics["mean_rank"] = float(np.mean(ranks)) if ranks else 0.0
    
    # Separar acertos e erros
    correct_results = [r for r in all_results if r["is_correct"]]
    error_results = [r for r in all_results if not r["is_correct"]]
    
    # Salvar CSV com todos os resultados
    csv_path = output_path / f"test_results_{num_negative_samples}neg.csv"
    _save_results_csv(all_results, csv_path)
    
    # Salvar CSV apenas com erros
    errors_csv_path = output_path / f"test_errors_{num_negative_samples}neg.csv"
    _save_results_csv(error_results, errors_csv_path)
    
    # Salvar métricas em JSON
    metrics_path = output_path / f"test_metrics_{num_negative_samples}neg.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n📊 Resultados do Teste ({num_negative_samples} negativos):")
    print(f"   Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
    print(f"   MRR: {metrics['mrr']:.4f}")
    print(f"   Sim+: {metrics['mean_positive_similarity']:.4f} ± {metrics['std_positive_similarity']:.4f}")
    print(f"   Sim-: {metrics['mean_negative_similarity']:.4f} ± {metrics['std_negative_similarity']:.4f}")
    print(f"   Gap: {metrics['similarity_gap']:.4f}")
    print(f"\n📁 Arquivos salvos em: {output_path}")
    print(f"   - {csv_path.name} ({len(all_results)} casos)")
    print(f"   - {errors_csv_path.name} ({len(error_results)} erros)")
    
    return {
        "metrics": metrics,
        "all_results": all_results,
        "correct_results": correct_results,
        "error_results": error_results,
        "output_files": {
            "all_results_csv": str(csv_path),
            "errors_csv": str(errors_csv_path),
            "metrics_json": str(metrics_path)
        }
    }


def _save_results_csv(results: List[Dict], path: Path) -> None:
    """Salva resultados em CSV."""
    if not results:
        return
    
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "expected_name",
            "image_path",
            "positive_similarity",
            "max_negative_similarity",
            "predicted_rank",
            "is_correct",
            "all_similarities",
            "negative_image_paths"
        ])
        
        # Dados
        for r in results:
            all_sims = [r["positive_similarity"]] + r["negative_similarities"]
            writer.writerow([
                r["expected_name"],
                r["image_path"],
                f"{r['positive_similarity']:.6f}",
                f"{r['max_negative_similarity']:.6f}",
                r["predicted_rank"],
                r["is_correct"],
                json.dumps([f"{s:.6f}" for s in all_sims]),
                json.dumps(r["negative_image_paths"])
            ])


def evaluate_test_multiple_negatives(
    model,
    processor,
    test_pairs: List[Tuple[str, str]],
    output_dir: str,
    device: torch.device,
    transform,
    image_size: int = 224,
    negative_samples_list: List[int] = [1, 2, 3],
    seed: int = 42
) -> Dict[str, Any]:
    """
    Avalia teste com diferentes números de negativos.
    
    Args:
        model: Wrapper do modelo.
        processor: Processor do modelo.
        test_pairs: Pares de teste.
        output_dir: Diretório de saída.
        device: Dispositivo.
        transform: Transform.
        image_size: Tamanho da imagem.
        negative_samples_list: Lista de números de negativos.
        seed: Seed.
        
    Returns:
        Dicionário com resultados para cada configuração.
    """
    all_metrics = {}
    
    for num_neg in negative_samples_list:
        print(f"\n{'='*60}")
        print(f"🔬 Avaliando com {num_neg} negativos")
        print(f"{'='*60}")
        
        result = evaluate_and_save_results(
            model=model,
            processor=processor,
            test_pairs=test_pairs,
            output_dir=output_dir,
            device=device,
            transform=transform,
            image_size=image_size,
            num_negative_samples=num_neg,
            seed=seed
        )
        
        # Prefixar métricas
        for key, value in result["metrics"].items():
            all_metrics[f"{key}_{num_neg}_neg"] = value
    
    # Salvar métricas agregadas
    output_path = Path(output_dir)
    with open(output_path / "test_metrics_all.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    return all_metrics


def analyze_errors(
    error_results: List[Dict],
    output_dir: str
) -> Dict[str, Any]:
    """
    Analisa padrões nos erros.
    
    Args:
        error_results: Lista de resultados com erro.
        output_dir: Diretório de saída.
        
    Returns:
        Análise dos erros.
    """
    if not error_results:
        return {"message": "Nenhum erro para analisar"}
    
    analysis = {
        "total_errors": len(error_results),
        "similarity_stats": {
            "mean_positive": float(np.mean([r["positive_similarity"] for r in error_results])),
            "mean_max_negative": float(np.mean([r["max_negative_similarity"] for r in error_results])),
            "mean_gap": float(np.mean([r["positive_similarity"] - r["max_negative_similarity"] for r in error_results])),
        },
        "rank_distribution": {},
        "hardest_cases": []
    }
    
    # Distribuição de ranks
    ranks = [r["predicted_rank"] for r in error_results]
    for rank in sorted(set(ranks)):
        analysis["rank_distribution"][f"rank_{rank}"] = ranks.count(rank)
    
    # Casos mais difíceis (menor gap)
    sorted_by_gap = sorted(
        error_results,
        key=lambda x: x["positive_similarity"] - x["max_negative_similarity"]
    )
    
    for r in sorted_by_gap[:10]:
        analysis["hardest_cases"].append({
            "name": r["expected_name"],
            "image": r["image_path"],
            "gap": r["positive_similarity"] - r["max_negative_similarity"]
        })
    
    # Salvar análise
    output_path = Path(output_dir)
    with open(output_path / "error_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    return analysis

