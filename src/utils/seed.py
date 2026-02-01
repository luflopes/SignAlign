"""Utilitários para reprodutibilidade de experimentos."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Define seeds para reprodutibilidade completa.
    
    Args:
        seed: Valor inteiro para inicialização dos geradores de números aleatórios.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f'🎲 Random seed definido como {seed}')


def get_device() -> torch.device:
    """Retorna o dispositivo disponível (CUDA se disponível, senão CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🖥️ GPU disponível: {torch.cuda.get_device_name(0)}")
        print(f"   Memória total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("⚠️ GPU não disponível, usando CPU")
    print(f"🖥️ Dispositivo selecionado: {device}")
    return device

