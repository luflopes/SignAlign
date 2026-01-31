"""
SignAlign - Framework Modular para Alinhamento Multimodal Texto-Assinatura

Este framework fornece uma arquitetura extensível para treinar e avaliar
modelos de aprendizado multimodal (CLIP, TinyCLIP, SigLIP) na tarefa de
recuperação de assinaturas manuscritas a partir de nomes textuais.
"""

__version__ = "1.0.0"
__author__ = "Lucas"

from src.config.experiment_config import (
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    LossConfig,
    EvaluationConfig,
)
from src.utils.seed import set_seed

__all__ = [
    "ExperimentConfig",
    "ModelConfig", 
    "TrainingConfig",
    "DataConfig",
    "LossConfig",
    "EvaluationConfig",
    "set_seed",
]

