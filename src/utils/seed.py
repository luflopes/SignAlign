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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Usando dispositivo: {device}")
    return device

