"""
Configuração de Learning Rate Schedulers.

Suporta ReduceLROnPlateau, Cosine, Step, etc.
"""

from typing import Optional
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingLR,
    StepLR,
    _LRScheduler,
)

from src.config.experiment_config import SchedulerConfig


def create_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig,
    num_epochs: Optional[int] = None
) -> Optional[_LRScheduler]:
    """
    Cria scheduler de learning rate.
    
    Args:
        optimizer: Otimizador a ser usado.
        config: Configuração do scheduler.
        num_epochs: Número de épocas (para cosine).
        
    Returns:
        Scheduler configurado ou None.
    """
    if config.name == "none":
        print("📈 Sem scheduler de learning rate.")
        return None
    
    if config.name == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            patience=config.patience,
            min_lr=config.min_lr,
        )
        print(f"📈 Scheduler: ReduceLROnPlateau")
        print(f"   mode={config.mode}, factor={config.factor}, patience={config.patience}")
    
    elif config.name == "cosine":
        T_max = config.T_max or num_epochs or 100
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=config.eta_min
        )
        print(f"📈 Scheduler: CosineAnnealing (T_max={T_max})")
    
    elif config.name == "step":
        scheduler = StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma
        )
        print(f"📈 Scheduler: StepLR (step={config.step_size}, gamma={config.gamma})")
    
    else:
        raise ValueError(f"Scheduler não suportado: {config.name}")
    
    return scheduler


class WarmupScheduler:
    """
    Wrapper para adicionar warmup a qualquer scheduler.
    
    Aumenta linearmente o LR durante os primeiros N steps.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        base_scheduler: Optional[_LRScheduler] = None
    ):
        """
        Args:
            optimizer: Otimizador.
            warmup_steps: Número de steps de warmup.
            base_scheduler: Scheduler principal após warmup.
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        self.current_step = 0
        
        # Salvar LR inicial
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, metric=None):
        """Executa um step do scheduler."""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Warmup: aumentar linearmente
            scale = self.current_step / self.warmup_steps
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * scale
        elif self.base_scheduler is not None:
            # Após warmup: usar scheduler base
            if isinstance(self.base_scheduler, ReduceLROnPlateau):
                if metric is not None:
                    self.base_scheduler.step(metric)
            else:
                self.base_scheduler.step()
    
    def get_last_lr(self):
        """Retorna último LR."""
        return [group['lr'] for group in self.optimizer.param_groups]

