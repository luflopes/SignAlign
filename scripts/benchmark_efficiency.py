#!/usr/bin/env python3
"""
Script para benchmark de eficiência dos modelos.

Mede throughput (pares/s), latência e estima FLOPs para cada arquitetura.
"""

import argparse
import json
from pathlib import Path
import sys

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.models.registry import create_model
from src.config.experiment_config import ModelConfig
from src.evaluation.efficiency import benchmark_model, compare_models_efficiency
from src.utils.seed import get_device, set_seed


# Configurações dos modelos a serem testados
MODELS_TO_BENCHMARK = [
    {
        "name": "TinyCLIP",
        "config": ModelConfig(
            name="wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M",
            type="tinyclip",
            image_size=224
        )
    },
    {
        "name": "CLIP-ViT-B/32",
        "config": ModelConfig(
            name="openai/clip-vit-base-patch32",
            type="clip",
            image_size=224
        )
    },
    {
        "name": "SigLIP-B/16",
        "config": ModelConfig(
            name="google/siglip-base-patch16-224",
            type="siglip",
            image_size=224
        )
    },
]


def benchmark_single_model(
    model_name: str,
    model_config: ModelConfig,
    device: torch.device,
    batch_sizes: list = [1, 8, 16, 32],
    num_iterations: int = 50
) -> dict:
    """
    Executa benchmark de um único modelo.
    
    Args:
        model_name: Nome do modelo para display.
        model_config: Configuração do modelo.
        device: Dispositivo.
        batch_sizes: Tamanhos de batch para testar.
        num_iterations: Número de iterações para medição.
        
    Returns:
        Dicionário com resultados do benchmark.
    """
    print(f"\n{'='*60}")
    print(f"🔬 Benchmarking: {model_name}")
    print(f"{'='*60}")
    
    # Criar modelo
    model = create_model(model_config, device)
    processor = model.get_processor()
    
    # Executar benchmark
    results = benchmark_model(
        model=model,
        processor=processor,
        device=device,
        image_size=model_config.image_size,
        batch_sizes=batch_sizes,
        num_iterations=num_iterations
    )
    
    results["model_name"] = model_name
    results["model_config"] = {
        "name": model_config.name,
        "type": model_config.type,
        "image_size": model_config.image_size
    }
    
    # Limpar memória
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    return results


def benchmark_all_models(
    output_path: str,
    batch_sizes: list = [1, 8, 16, 32],
    num_iterations: int = 50
) -> dict:
    """
    Executa benchmark de todos os modelos configurados.
    
    Args:
        output_path: Caminho para salvar resultados.
        batch_sizes: Tamanhos de batch.
        num_iterations: Iterações por batch.
        
    Returns:
        Dicionário com todos os resultados.
    """
    set_seed(42)
    device = get_device()
    
    print(f"🖥️  Device: {device}")
    print(f"📊 Batch sizes: {batch_sizes}")
    print(f"🔄 Iterações: {num_iterations}")
    
    all_results = {}
    
    for model_info in MODELS_TO_BENCHMARK:
        name = model_info["name"]
        config = model_info["config"]
        
        try:
            results = benchmark_single_model(
                model_name=name,
                model_config=config,
                device=device,
                batch_sizes=batch_sizes,
                num_iterations=num_iterations
            )
            all_results[name] = results
        except Exception as e:
            print(f"❌ Erro ao benchmarkar {name}: {e}")
            all_results[name] = {"error": str(e)}
    
    # Criar tabela comparativa
    print("\n" + "="*80)
    print("📊 COMPARAÇÃO DE MODELOS")
    print("="*80)
    
    print(f"\n{'Modelo':<20} {'Params (M)':<12} {'Best Throughput':<18} {'GFLOPs':<12}")
    print("-"*62)
    
    for name, results in all_results.items():
        if "error" in results:
            print(f"{name:<20} {'ERROR':<12}")
            continue
        
        params = results["model_info"]["parameters"]["total_millions"]
        throughput = results["throughput"]["best_throughput"]["pairs_per_second"]
        gflops = results["flops"]["total_gflops"]
        
        print(f"{name:<20} {params:<12.1f} {throughput:<18.1f} {gflops:<12.2f}")
    
    # Salvar resultados
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Resultados salvos em: {output_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark de eficiência dos modelos"
    )
    parser.add_argument(
        "--output",
        default="experiments/efficiency_benchmark.json",
        help="Caminho para salvar resultados"
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 8, 16, 32],
        help="Tamanhos de batch para testar"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Número de iterações para medição"
    )
    parser.add_argument(
        "--model",
        choices=["tinyclip", "clip", "siglip", "all"],
        default="all",
        help="Modelo específico para benchmarkar"
    )
    
    args = parser.parse_args()
    
    if args.model == "all":
        benchmark_all_models(
            output_path=args.output,
            batch_sizes=args.batch_sizes,
            num_iterations=args.iterations
        )
    else:
        # Benchmarkar modelo específico
        model_map = {
            "tinyclip": MODELS_TO_BENCHMARK[0],
            "clip": MODELS_TO_BENCHMARK[1],
            "siglip": MODELS_TO_BENCHMARK[2],
        }
        
        model_info = model_map[args.model]
        set_seed(42)
        device = get_device()
        
        results = benchmark_single_model(
            model_name=model_info["name"],
            model_config=model_info["config"],
            device=device,
            batch_sizes=args.batch_sizes,
            num_iterations=args.iterations
        )
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Resultados salvos em: {output_path}")


if __name__ == "__main__":
    main()

