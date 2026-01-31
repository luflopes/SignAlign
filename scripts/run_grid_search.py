#!/usr/bin/env python3
"""
Script para executar grid search de experimentos.

Permite testar múltiplas configurações automaticamente.

Uso:
    python scripts/run_grid_search.py --base-config configs/base.yaml
"""

import argparse
import itertools
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ExperimentConfig, load_config, save_config
from src.training import Trainer


# Definir grid de parâmetros
GRID_PARAMS = {
    # Modelos a testar
    "model.type": ["tinyclip"],  # ["tinyclip", "clip", "siglip"]
    "model.name": ["wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M"],
    
    # Losses
    "loss.primary_loss": ["cross_entropy", "infonce"],
    "loss.use_triplet_loss": [True, False],
    "loss.triplet_weight": [0.2, 0.3],
    
    # Fine-tuning
    "model.freeze_vision_encoder": [True, False],
    
    # Data augmentation
    "data.augmentation.enabled": [True],
}


def set_nested_attr(obj: Any, path: str, value: Any) -> None:
    """Define atributo aninhado usando notação de ponto."""
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def get_nested_attr(obj: Any, path: str) -> Any:
    """Obtém atributo aninhado usando notação de ponto."""
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def generate_experiment_name(params: Dict[str, Any]) -> str:
    """Gera nome único para experimento baseado nos parâmetros."""
    parts = []
    
    # Adicionar parâmetros relevantes ao nome
    if "model.type" in params:
        parts.append(params["model.type"])
    
    if "loss.primary_loss" in params:
        parts.append(params["loss.primary_loss"])
    
    if params.get("loss.use_triplet_loss", False):
        weight = params.get("loss.triplet_weight", 0.2)
        parts.append(f"triplet{weight}")
    
    if params.get("model.freeze_vision_encoder", False):
        parts.append("frozen")
    
    # Timestamp para unicidade
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts.append(timestamp)
    
    return "_".join(parts)


def run_grid_search(
    base_config_path: str,
    grid_params: Dict[str, List[Any]],
    output_dir: str = "experiments"
) -> List[Dict[str, Any]]:
    """
    Executa grid search de experimentos.
    
    Args:
        base_config_path: Caminho para config base.
        grid_params: Dicionário de parâmetros e seus valores.
        output_dir: Diretório de saída.
        
    Returns:
        Lista de resultados de cada experimento.
    """
    # Carregar config base
    base_config = load_config(base_config_path)
    
    # Gerar todas as combinações
    param_names = list(grid_params.keys())
    param_values = list(grid_params.values())
    combinations = list(itertools.product(*param_values))
    
    print(f"🔬 Grid Search: {len(combinations)} combinações")
    print("=" * 60)
    
    results = []
    
    for i, values in enumerate(combinations):
        # Criar configuração para esta combinação
        params = dict(zip(param_names, values))
        
        # Pular combinações inválidas
        if not params.get("loss.use_triplet_loss", True) and "loss.triplet_weight" in params:
            continue
        
        print(f"\n📋 Experimento {i+1}/{len(combinations)}")
        print(f"   Parâmetros: {params}")
        
        # Criar nova config
        config = load_config(base_config_path)
        
        for param_path, value in params.items():
            try:
                set_nested_attr(config, param_path, value)
            except AttributeError:
                print(f"⚠️ Parâmetro inválido: {param_path}")
                continue
        
        # Gerar nome do experimento
        config.name = generate_experiment_name(params)
        config.output_dir = output_dir
        
        # Salvar config do experimento
        exp_config_path = Path(output_dir) / config.name / "config.yaml"
        exp_config_path.parent.mkdir(parents=True, exist_ok=True)
        save_config(config, str(exp_config_path))
        
        try:
            # Executar treinamento
            trainer = Trainer(config)
            trainer.setup()
            result = trainer.train()
            
            results.append({
                "name": config.name,
                "params": params,
                "best_metric": result["best_metric"],
                "best_epoch": result["best_epoch"],
                "status": "success"
            })
            
        except Exception as e:
            print(f"❌ Erro no experimento: {e}")
            results.append({
                "name": config.name,
                "params": params,
                "error": str(e),
                "status": "failed"
            })
    
    # Resumo final
    print("\n" + "=" * 60)
    print("📊 RESUMO DO GRID SEARCH")
    print("=" * 60)
    
    successful = [r for r in results if r["status"] == "success"]
    if successful:
        best = max(successful, key=lambda x: x["best_metric"])
        print(f"\n🏆 Melhor experimento: {best['name']}")
        print(f"   Métrica: {best['best_metric']:.4f}")
        print(f"   Parâmetros: {best['params']}")
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Executa grid search de experimentos"
    )
    
    parser.add_argument(
        "--base-config",
        type=str,
        required=True,
        help="Caminho para configuração base"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Diretório de saída"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    results = run_grid_search(
        base_config_path=args.base_config,
        grid_params=GRID_PARAMS,
        output_dir=args.output_dir
    )
    
    # Salvar resultados
    import json
    results_path = Path(args.output_dir) / "grid_search_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Resultados salvos em: {results_path}")


if __name__ == "__main__":
    main()

