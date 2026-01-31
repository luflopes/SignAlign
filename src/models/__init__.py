"""Módulo de modelos multimodais."""

from src.models.base import BaseMultimodalModel
from src.models.clip_models import (
    CLIPWrapper,
    TinyCLIPWrapper,
    SigLIPWrapper,
)
from src.models.registry import ModelRegistry, create_model

__all__ = [
    "BaseMultimodalModel",
    "CLIPWrapper",
    "TinyCLIPWrapper", 
    "SigLIPWrapper",
    "ModelRegistry",
    "create_model",
]

