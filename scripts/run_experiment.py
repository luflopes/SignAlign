#!/usr/bin/env python3
"""
Script principal para executar experimentos SignAlign.

Uso:
    python scripts/run_experiment.py --config configs/tinyclip_base.yaml
    python scripts/run_experiment.py --config configs/siglip_sigmoid.yaml --name meu_experimento
"""

import argparse
import sys
from pathlib import Path

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, ExperimentConfig
from src.training import Trainer
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Executa um experimento SignAlign"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Caminho para arquivo de configuração YAML"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Nome do experimento (sobrescreve config)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed para reprodutibilidade (sobrescreve config)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Número de épocas (sobrescreve config)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Tamanho do batch (sobrescreve config)"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (sobrescreve config)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Diretório de saída (sobrescreve config)"
    )
    
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Desabilita data augmentation"
    )
    
    parser.add_argument(
        "--freeze-vision",
        action="store_true",
        help="Congela vision encoder"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limita número de amostras (modo teste rápido)"
    )
    
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Modo teste: usa apenas 50 amostras e 1 época"
    )
    
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Apenas avaliação (sem treino) - útil para modelos frozen"
    )
    
    parser.add_argument(
        "--triplet-weight",
        type=float,
        default=None,
        help="Peso da Triplet Loss (sobrescreve config)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Carregar configuração
    print(f"📄 Carregando configuração: {args.config}")
    config = load_config(args.config)
    
    # Sobrescrever com argumentos da CLI
    if args.name:
        config.name = args.name
    
    if args.seed:
        config.seed = args.seed
    
    if args.epochs:
        config.training.epochs = args.epochs
    
    if args.batch_size:
        config.data.batch_size = args.batch_size
    
    if args.lr:
        config.training.learning_rate = args.lr
    
    if args.output_dir:
        config.output_dir = args.output_dir
    
    if args.no_augmentation:
        config.data.augmentation.enabled = False
    
    if args.freeze_vision:
        config.model.freeze_vision_encoder = True
    
    if args.max_samples:
        config.data.max_samples = args.max_samples
    
    # Modo teste rápido: poucas amostras e 1 época
    if args.test_mode:
        config.data.max_samples = 50
        config.training.epochs = 1
        config.name = f"TEST_{config.name}"
        print("🧪 MODO TESTE ATIVADO: 50 amostras, 1 época")
    
    # Triplet weight override
    if args.triplet_weight is not None:
        config.loss.triplet_weight = args.triplet_weight
        config.loss.use_triplet_loss = True
    
    # Modo avaliação apenas
    if args.eval_only:
        config.training.epochs = 0
        print("📊 MODO AVALIAÇÃO: Sem treino, apenas métricas")
    
    # Criar e executar trainer
    trainer = Trainer(config)
    trainer.setup()
    
    # Se eval_only, apenas avaliar sem treinar
    if args.eval_only or config.training.epochs == 0:
        results = trainer.evaluate_only()
    else:
        results = trainer.train()
    
    print("\n" + "=" * 60)
    print("📊 Resultados Finais:")
    print(f"   Melhor época: {results['best_epoch']}")
    print(f"   Melhor métrica: {results['best_metric']:.4f}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()

