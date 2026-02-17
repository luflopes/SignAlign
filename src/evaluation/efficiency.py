"""
Métricas de eficiência para modelos multimodais.

Inclui throughput (pares/segundo), latência e FLOPs.
"""

import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from src.models.base import BaseMultimodalModel


def measure_throughput(
    model: BaseMultimodalModel,
    processor,
    device: torch.device,
    batch_sizes: List[int] = [1, 8, 16, 32],
    num_warmup: int = 10,
    num_iterations: int = 50,
    image_size: int = 224
) -> Dict[str, Any]:
    """
    Mede throughput do modelo em pares/segundo.
    
    Args:
        model: Wrapper do modelo.
        processor: Processor do modelo.
        device: Dispositivo (cuda/cpu).
        batch_sizes: Lista de tamanhos de batch para testar.
        num_warmup: Iterações de warmup.
        num_iterations: Iterações para medição.
        image_size: Tamanho da imagem.
        
    Returns:
        Dicionário com métricas de throughput.
    """
    model.model.eval()
    results = {}
    
    for batch_size in batch_sizes:
        # Criar dados sintéticos
        dummy_images = [
            Image.new("RGB", (image_size, image_size), color=(255, 255, 255))
            for _ in range(batch_size)
        ]
        dummy_texts = [f"JOHN DOE {i}" for i in range(batch_size)]
        
        # Processar
        inputs = processor(
            images=dummy_images,
            text=dummy_texts,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model.forward(
                    pixel_values=inputs['pixel_values'],
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                )
        
        # Sincronizar GPU
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Medir
        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = model.forward(
                    pixel_values=inputs['pixel_values'],
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append(end - start)
        
        latencies = np.array(latencies)
        
        results[f"batch_{batch_size}"] = {
            "batch_size": batch_size,
            "pairs_per_second": batch_size / np.mean(latencies),
            "latency_mean_ms": np.mean(latencies) * 1000,
            "latency_std_ms": np.std(latencies) * 1000,
            "latency_p50_ms": np.percentile(latencies, 50) * 1000,
            "latency_p95_ms": np.percentile(latencies, 95) * 1000,
            "latency_p99_ms": np.percentile(latencies, 99) * 1000,
        }
    
    # Encontrar melhor batch size
    best_batch = max(
        results.values(),
        key=lambda x: x["pairs_per_second"]
    )
    
    return {
        "by_batch_size": results,
        "best_throughput": {
            "batch_size": best_batch["batch_size"],
            "pairs_per_second": best_batch["pairs_per_second"],
            "fps": best_batch["pairs_per_second"],  # Equivalente para imagens
        }
    }


def estimate_flops(
    model: BaseMultimodalModel,
    processor,
    device: torch.device,
    image_size: int = 224
) -> Dict[str, Any]:
    """
    Estima FLOPs do modelo usando ptflops.
    
    Args:
        model: Wrapper do modelo.
        processor: Processor do modelo.
        device: Dispositivo.
        image_size: Tamanho da imagem.
        
    Returns:
        Dicionário com FLOPs estimados.
    """
    try:
        from ptflops import get_model_complexity_info
        HAS_PTFLOPS = True
    except ImportError:
        HAS_PTFLOPS = False
    
    results = {
        "method": "ptflops" if HAS_PTFLOPS else "estimation",
        "image_size": image_size,
    }
    
    if HAS_PTFLOPS:
        try:
            # Medir encoder visual
            vision_model = model.model.vision_model
            macs_vision, params_vision = get_model_complexity_info(
                vision_model,
                (3, image_size, image_size),
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False
            )
            results["vision_encoder"] = {
                "gflops": macs_vision * 2 / 1e9,  # MACs * 2 = FLOPs
                "params_millions": params_vision / 1e6
            }
        except Exception as e:
            results["vision_encoder"] = {"error": str(e)}
        
        try:
            # Medir encoder textual (mais complexo por causa do embedding)
            text_model = model.model.text_model
            
            # Criar input dummy para texto
            dummy_text = ["JOHN DOE"]
            text_inputs = processor(
                text=dummy_text,
                return_tensors="pt",
                padding="max_length",
                max_length=77
            )
            
            # Não é trivial medir FLOPs do text encoder com ptflops
            # Usamos estimativa baseada em parâmetros
            text_params = sum(p.numel() for p in text_model.parameters())
            # Estimativa grosseira: 2 * params * seq_len para transformers
            seq_len = 77
            estimated_flops = 2 * text_params * seq_len
            
            results["text_encoder"] = {
                "gflops_estimated": estimated_flops / 1e9,
                "params_millions": text_params / 1e6
            }
        except Exception as e:
            results["text_encoder"] = {"error": str(e)}
    
    else:
        # Estimativa baseada em parâmetros
        total_params, trainable_params = model.count_parameters()
        
        # Estimativa grosseira para Vision Transformer:
        # FLOPs ≈ 2 * params * (image_size/patch_size)^2 * embed_dim
        # Para ViT-B/32: patch_size=32, embed_dim=768
        patch_size = 32
        num_patches = (image_size // patch_size) ** 2
        embed_dim = 768
        
        vision_flops = 2 * (total_params * 0.7) * num_patches  # ~70% params in vision
        text_flops = 2 * (total_params * 0.3) * 77  # ~30% params in text, seq_len=77
        
        results["vision_encoder"] = {
            "gflops_estimated": vision_flops / 1e9,
        }
        results["text_encoder"] = {
            "gflops_estimated": text_flops / 1e9,
        }
    
    # Total
    vision_gflops = results.get("vision_encoder", {}).get("gflops") or \
                    results.get("vision_encoder", {}).get("gflops_estimated", 0)
    text_gflops = results.get("text_encoder", {}).get("gflops") or \
                  results.get("text_encoder", {}).get("gflops_estimated", 0)
    
    results["total_gflops"] = vision_gflops + text_gflops
    
    return results


def get_model_info(model: BaseMultimodalModel) -> Dict[str, Any]:
    """
    Obtém informações gerais do modelo.
    
    Args:
        model: Wrapper do modelo.
        
    Returns:
        Dicionário com informações do modelo.
    """
    total_params, trainable_params = model.count_parameters()
    
    # Tentar obter configuração
    config = {}
    if hasattr(model.model, 'config'):
        cfg = model.model.config
        config = {
            "hidden_size": getattr(cfg, 'hidden_size', None) or getattr(cfg, 'projection_dim', None),
            "vision_config": {
                "hidden_size": getattr(cfg.vision_config, 'hidden_size', None) if hasattr(cfg, 'vision_config') else None,
                "patch_size": getattr(cfg.vision_config, 'patch_size', None) if hasattr(cfg, 'vision_config') else None,
                "num_hidden_layers": getattr(cfg.vision_config, 'num_hidden_layers', None) if hasattr(cfg, 'vision_config') else None,
            },
            "text_config": {
                "hidden_size": getattr(cfg.text_config, 'hidden_size', None) if hasattr(cfg, 'text_config') else None,
                "num_hidden_layers": getattr(cfg.text_config, 'num_hidden_layers', None) if hasattr(cfg, 'text_config') else None,
            }
        }
    
    return {
        "parameters": {
            "total": total_params,
            "trainable": trainable_params,
            "total_millions": total_params / 1e6,
            "trainable_millions": trainable_params / 1e6,
        },
        "config": config
    }


def benchmark_model(
    model: BaseMultimodalModel,
    processor,
    device: torch.device,
    image_size: int = 224,
    batch_sizes: List[int] = [1, 8, 16, 32],
    num_iterations: int = 50
) -> Dict[str, Any]:
    """
    Executa benchmark completo do modelo.
    
    Args:
        model: Wrapper do modelo.
        processor: Processor do modelo.
        device: Dispositivo.
        image_size: Tamanho da imagem.
        batch_sizes: Tamanhos de batch para testar.
        num_iterations: Iterações para medição.
        
    Returns:
        Dicionário com todas as métricas de eficiência.
    """
    print(f"🔬 Executando benchmark de eficiência...")
    
    # Info do modelo
    model_info = get_model_info(model)
    print(f"   📦 Parâmetros: {model_info['parameters']['total_millions']:.1f}M")
    
    # Throughput
    print(f"   ⏱️ Medindo throughput...")
    throughput = measure_throughput(
        model, processor, device,
        batch_sizes=batch_sizes,
        num_iterations=num_iterations,
        image_size=image_size
    )
    print(f"   ✅ Melhor throughput: {throughput['best_throughput']['pairs_per_second']:.1f} pares/s")
    
    # FLOPs
    print(f"   🔢 Estimando FLOPs...")
    flops = estimate_flops(model, processor, device, image_size)
    print(f"   ✅ Total: {flops['total_gflops']:.2f} GFLOPs")
    
    return {
        "model_info": model_info,
        "throughput": throughput,
        "flops": flops,
        "device": str(device),
        "image_size": image_size
    }


def compare_models_efficiency(
    models_configs: List[Dict[str, Any]],
    device: torch.device,
    image_size: int = 224
) -> Dict[str, Dict]:
    """
    Compara eficiência de múltiplos modelos.
    
    Args:
        models_configs: Lista de {name, model, processor}.
        device: Dispositivo.
        image_size: Tamanho da imagem.
        
    Returns:
        Dicionário com benchmarks por modelo.
    """
    results = {}
    
    for config in tqdm(models_configs, desc="Benchmarking modelos"):
        name = config["name"]
        model = config["model"]
        processor = config["processor"]
        
        print(f"\n📊 Benchmarking: {name}")
        results[name] = benchmark_model(
            model, processor, device, image_size
        )
    
    # Criar tabela comparativa
    comparison = {
        "models": [],
        "params_millions": [],
        "best_throughput": [],
        "total_gflops": []
    }
    
    for name, data in results.items():
        comparison["models"].append(name)
        comparison["params_millions"].append(data["model_info"]["parameters"]["total_millions"])
        comparison["best_throughput"].append(data["throughput"]["best_throughput"]["pairs_per_second"])
        comparison["total_gflops"].append(data["flops"]["total_gflops"])
    
    results["comparison"] = comparison
    
    return results

