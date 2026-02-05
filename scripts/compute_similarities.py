#!/usr/bin/env python3
"""
Script para calcular estatísticas de similaridade em experimentos já treinados.

Carrega os checkpoints salvos e calcula mean_pos_similarity, mean_neg_similarity
e similarity_gap no conjunto de validação, atualizando os arquivos de métricas.

Uso:
    python scripts/compute_similarities.py                    # Todos os experimentos
    python scripts/compute_similarities.py --experiment NAME  # Experimento específico
"""

import argparse
import json
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm

from src.config.experiment_config import load_config
from src.models.registry import load_model_from_experiment
from src.data.batch_builder import load_dataset_pairs, create_train_val_split
from src.data.dataset import SignatureNameDataset, collate_fn
from src.data.augmentations import get_val_transform
from src.utils.seed import set_seed, get_device


def compute_similarity_statistics(
    model,
    processor,
    val_pairs,
    device,
    batch_size: int = 32,
    image_size: int = 224
):
    """
    Calcula estatísticas de similaridade no conjunto de validação.
    
    Returns:
        Dicionário com similaridades médias positivas e negativas.
    """
    model.model.eval()
    transform = get_val_transform()
    
    all_pos_sims = []
    all_neg_sims = []
    
    dataset = SignatureNameDataset(val_pairs, transform, image_size)
    batch_size = min(batch_size, len(dataset))
    
    print(f"   Processando {len(dataset)} amostras de validação...")
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="   Calculando similaridades"):
        batch_data = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        batch = collate_fn(batch_data, processor).to(device)
        
        with torch.no_grad():
            outputs = model.forward(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
            )
            
            image_embeds = outputs['image_embeds']
            text_embeds = outputs['text_embeds']
            
            # Normalizar embeddings
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            # Calcular similaridades
            similarity_matrix = image_embeds @ text_embeds.T
            
            # Similaridades positivas (diagonal)
            pos_sims = similarity_matrix.diagonal()
            all_pos_sims.extend(pos_sims.cpu().tolist())
            
            # Similaridades negativas (fora da diagonal)
            bs = similarity_matrix.size(0)
            mask = ~torch.eye(bs, dtype=torch.bool, device=device)
            neg_sims = similarity_matrix[mask]
            all_neg_sims.extend(neg_sims.cpu().tolist())
    
    import numpy as np
    
    # Calcular estatísticas para similaridades positivas
    pos_stats = {}
    if all_pos_sims:
        pos_stats = {
            "mean_pos_similarity": float(np.mean(all_pos_sims)),
            "std_pos_similarity": float(np.std(all_pos_sims)),
            "median_pos_similarity": float(np.median(all_pos_sims)),
            "q1_pos_similarity": float(np.percentile(all_pos_sims, 25)),
            "q3_pos_similarity": float(np.percentile(all_pos_sims, 75)),
            "min_pos_similarity": float(np.min(all_pos_sims)),
            "max_pos_similarity": float(np.max(all_pos_sims)),
        }
    else:
        pos_stats = {k: 0.0 for k in [
            "mean_pos_similarity", "std_pos_similarity", "median_pos_similarity",
            "q1_pos_similarity", "q3_pos_similarity", "min_pos_similarity", "max_pos_similarity"
        ]}
    
    # Calcular estatísticas para similaridades negativas
    neg_stats = {}
    if all_neg_sims:
        neg_stats = {
            "mean_neg_similarity": float(np.mean(all_neg_sims)),
            "std_neg_similarity": float(np.std(all_neg_sims)),
            "median_neg_similarity": float(np.median(all_neg_sims)),
            "q1_neg_similarity": float(np.percentile(all_neg_sims, 25)),
            "q3_neg_similarity": float(np.percentile(all_neg_sims, 75)),
            "min_neg_similarity": float(np.min(all_neg_sims)),
            "max_neg_similarity": float(np.max(all_neg_sims)),
        }
    else:
        neg_stats = {k: 0.0 for k in [
            "mean_neg_similarity", "std_neg_similarity", "median_neg_similarity",
            "q1_neg_similarity", "q3_neg_similarity", "min_neg_similarity", "max_neg_similarity"
        ]}
    
    # Calcular gaps (diferença entre positivos e negativos)
    gap_stats = {
        "similarity_gap_mean": pos_stats["mean_pos_similarity"] - neg_stats["mean_neg_similarity"],
        "similarity_gap_median": pos_stats["median_pos_similarity"] - neg_stats["median_neg_similarity"],
    }
    
    return {**pos_stats, **neg_stats, **gap_stats}


def process_experiment(experiment_dir: Path, device: torch.device):
    """Processa um experimento e atualiza suas métricas."""
    from src.models.registry import create_model
    
    print(f"\n{'='*60}")
    print(f"📊 Processando: {experiment_dir.name}")
    print(f"{'='*60}")
    
    # Carregar config
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        print(f"   ⚠️ Config não encontrada, pulando...")
        return False
    
    with open(config_path) as f:
        config_dict = json.load(f)
    
    # Verificar se é experimento frozen (sem checkpoint)
    is_frozen = "frozen_eval" in experiment_dir.name or config_dict.get("training", {}).get("epochs", 1) == 0
    checkpoint_dir = experiment_dir / "checkpoints" / "best"
    has_checkpoint = checkpoint_dir.exists()
    
    # Carregar modelo
    print(f"   📦 Carregando modelo...")
    try:
        model_type = config_dict.get("model", {}).get("type", "tinyclip")
        model_name = config_dict.get("model", {}).get("name", "")
        
        if is_frozen or not has_checkpoint:
            # Experimento frozen ou sem checkpoint: carregar modelo pré-treinado
            print(f"   🔒 Experimento frozen/sem checkpoint - carregando modelo pré-treinado: {model_name}")
            model = create_model(
                model_type=model_type,
                pretrained_name=model_name,
                output_attentions=True,
                device=device
            )
        else:
            # Experimento com fine-tuning: carregar do checkpoint
            model = load_model_from_experiment(
                str(experiment_dir), 
                device=device,
                model_type=model_type
            )
        processor = model.processor
    except Exception as e:
        print(f"   ❌ Erro ao carregar modelo: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Carregar dados de validação
    print(f"   📂 Carregando dados de validação...")
    
    # Extrair paths do config
    dataset_csv = config_dict.get("data", {}).get("dataset_csv", "datasets/dataset-sign-align/dataset.csv")
    images_base_path = config_dict.get("data", {}).get("images_base_path", "datasets/dataset-sign-align")
    val_ratio = config_dict.get("data", {}).get("val_ratio", 0.15)
    exclude_unknown = config_dict.get("data", {}).get("exclude_unknown", True)
    seed = config_dict.get("seed", 42)
    
    set_seed(seed)
    
    all_pairs = load_dataset_pairs(
        csv_path=dataset_csv,
        images_base_path=images_base_path,
        exclude_unknown=exclude_unknown
    )
    
    _, val_pairs = create_train_val_split(
        all_pairs,
        val_ratio=val_ratio,
        seed=seed
    )
    
    print(f"   📊 {len(val_pairs)} amostras de validação")
    
    # Calcular similaridades
    similarity_stats = compute_similarity_statistics(
        model=model,
        processor=processor,
        val_pairs=val_pairs,
        device=device,
        batch_size=config_dict.get("data", {}).get("batch_size", 32),
        image_size=config_dict.get("model", {}).get("image_size", 224)
    )
    
    print(f"\n   📈 Resultados:")
    print(f"      Sim+ média:   {similarity_stats['mean_pos_similarity']:.4f} (±{similarity_stats['std_pos_similarity']:.4f})")
    print(f"      Sim+ mediana: {similarity_stats['median_pos_similarity']:.4f} [Q1={similarity_stats['q1_pos_similarity']:.4f}, Q3={similarity_stats['q3_pos_similarity']:.4f}]")
    print(f"      Sim- média:   {similarity_stats['mean_neg_similarity']:.4f} (±{similarity_stats['std_neg_similarity']:.4f})")
    print(f"      Sim- mediana: {similarity_stats['median_neg_similarity']:.4f} [Q1={similarity_stats['q1_neg_similarity']:.4f}, Q3={similarity_stats['q3_neg_similarity']:.4f}]")
    print(f"      Gap (média):   {similarity_stats['similarity_gap_mean']:.4f}")
    print(f"      Gap (mediana): {similarity_stats['similarity_gap_median']:.4f}")
    
    # Atualizar val_history.json
    val_history_path = experiment_dir / "metrics" / "val_history.json"
    if val_history_path.exists():
        with open(val_history_path) as f:
            val_history = json.load(f)
        
        # Adicionar todas as estatísticas a cada entrada
        if isinstance(val_history, list) and len(val_history) > 0:
            for entry in val_history:
                # Adicionar todas as métricas com prefixo val_
                for key, value in similarity_stats.items():
                    entry[f"val_{key}"] = value
        
        with open(val_history_path, "w") as f:
            json.dump(val_history, f, indent=2)
        
        print(f"   ✅ val_history.json atualizado")
    
    # Criar/atualizar similarity_stats.json separado
    stats_path = experiment_dir / "metrics" / "similarity_stats.json"
    with open(stats_path, "w") as f:
        json.dump(similarity_stats, f, indent=2)
    print(f"   ✅ similarity_stats.json criado")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Calcular similaridades em experimentos existentes")
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Nome do experimento específico (padrão: todos)"
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="experiments",
        help="Diretório com experimentos (padrão: experiments)"
    )
    
    args = parser.parse_args()
    
    device = get_device()
    experiments_dir = Path(args.experiments_dir)
    
    if not experiments_dir.exists():
        print(f"❌ Diretório não encontrado: {experiments_dir}")
        return
    
    # Listar experimentos
    if args.experiment:
        experiment_dirs = [experiments_dir / args.experiment]
        if not experiment_dirs[0].exists():
            print(f"❌ Experimento não encontrado: {args.experiment}")
            return
    else:
        experiment_dirs = sorted([
            d for d in experiments_dir.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        ])
    
    print(f"\n🔬 Processando {len(experiment_dirs)} experimento(s)...")
    
    success = 0
    failed = 0
    
    for exp_dir in experiment_dirs:
        try:
            if process_experiment(exp_dir, device):
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"   ❌ Erro: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Concluído: {success} sucesso, {failed} falhas")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

