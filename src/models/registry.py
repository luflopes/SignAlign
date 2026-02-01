"""
Registry de modelos - Factory Pattern para criação de modelos.

Permite adicionar novos modelos de forma extensível.
"""

from typing import Dict, Type, Optional
import torch

from src.models.base import BaseMultimodalModel
from src.models.clip_models import (
    CLIPWrapper,
    TinyCLIPWrapper,
    SigLIPWrapper,
    load_from_checkpoint,
)
from src.config.experiment_config import ModelConfig


class ModelRegistry:
    """
    Registry central de modelos.
    
    Permite registrar novos tipos de modelos e criar instâncias
    a partir de configurações.
    """
    
    _models: Dict[str, Type[BaseMultimodalModel]] = {
        "clip": CLIPWrapper,
        "tinyclip": TinyCLIPWrapper,
        "siglip": SigLIPWrapper,
    }
    
    # Modelos padrão do HuggingFace por tipo
    _default_models: Dict[str, str] = {
        "clip": "openai/clip-vit-base-patch32",
        "tinyclip": "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M",
        "siglip": "google/siglip-base-patch16-224",
    }
    
    @classmethod
    def register(cls, name: str, model_class: Type[BaseMultimodalModel]) -> None:
        """
        Registra um novo tipo de modelo.
        
        Args:
            name: Nome identificador do modelo.
            model_class: Classe do wrapper do modelo.
        """
        cls._models[name] = model_class
        print(f"✅ Modelo '{name}' registrado.")
    
    @classmethod
    def get(cls, name: str) -> Type[BaseMultimodalModel]:
        """
        Obtém classe do modelo pelo nome.
        
        Args:
            name: Nome do tipo de modelo.
            
        Returns:
            Classe do wrapper.
        """
        if name not in cls._models:
            raise ValueError(
                f"Modelo '{name}' não registrado. "
                f"Disponíveis: {list(cls._models.keys())}"
            )
        return cls._models[name]
    
    @classmethod
    def list_available(cls) -> list:
        """Lista modelos disponíveis."""
        return list(cls._models.keys())
    
    @classmethod
    def get_default_model_name(cls, model_type: str) -> str:
        """Retorna nome padrão do modelo HuggingFace para um tipo."""
        return cls._default_models.get(model_type, "")


def create_model(
    config: ModelConfig,
    device: Optional[torch.device] = None
) -> BaseMultimodalModel:
    """
    Cria modelo a partir de configuração.
    
    Args:
        config: Configuração do modelo.
        device: Dispositivo para carregar o modelo.
        
    Returns:
        Wrapper do modelo configurado.
    """
    # Obter classe do wrapper
    wrapper_class = ModelRegistry.get(config.type)
    wrapper = wrapper_class()
    
    # Carregar modelo pré-treinado
    model_name = config.name or ModelRegistry.get_default_model_name(config.type)
    wrapper.load_pretrained(
        model_name=model_name,
        output_attentions=config.output_attentions
    )
    
    # Aplicar freezing conforme configuração
    if config.freeze_vision_encoder:
        wrapper.freeze_vision_encoder()
    
    if config.freeze_text_encoder:
        wrapper.freeze_text_encoder()
    
    # Mover para dispositivo
    if device:
        wrapper.model = wrapper.model.to(device)
        # Confirmar que está no dispositivo correto
        param_device = next(wrapper.model.parameters()).device
        print(f"✅ Modelo carregado no dispositivo: {param_device}")
    
    return wrapper


def load_model_from_experiment(
    experiment_path: str,
    checkpoint: str = "best",
    model_type: str = "tinyclip",
    device: Optional[torch.device] = None
) -> BaseMultimodalModel:
    """
    Carrega modelo de um experimento salvo.
    
    Args:
        experiment_path: Caminho para o diretório do experimento.
        checkpoint: Nome do checkpoint ("best" ou "epoch_XXX").
        model_type: Tipo do modelo.
        device: Dispositivo para carregar.
        
    Returns:
        Wrapper do modelo carregado.
    """
    from pathlib import Path
    
    checkpoint_path = Path(experiment_path) / "checkpoints" / checkpoint
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")
    
    wrapper = load_from_checkpoint(
        str(checkpoint_path),
        model_type=model_type,
        output_attentions=True
    )
    
    if device:
        wrapper.model = wrapper.model.to(device)
    
    wrapper.model.eval()
    return wrapper

