"""
Configurações de experimentos usando dataclasses.

Permite validação de tipos e serialização automática para YAML.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Literal, Optional, Dict, Any
from pathlib import Path
import yaml


@dataclass
class AugmentationConfig:
    """Configuração de data augmentation."""
    
    enabled: bool = True
    
    # ShiftScaleRotate
    shift_limit: List[float] = field(default_factory=lambda: [-0.0625, 0.0625])
    scale_limit: List[float] = field(default_factory=lambda: [-0.1, 0.1])
    rotate_limit: List[int] = field(default_factory=lambda: [-10, 15])
    
    # Downscale
    downscale_range: List[float] = field(default_factory=lambda: [0.4, 1.0])
    
    # Motion blur
    motion_blur_limit: List[int] = field(default_factory=lambda: [3, 5])
    
    # Gaussian noise
    noise_std_range: List[float] = field(default_factory=lambda: [0.1, 0.2])
    
    # Brightness/Contrast
    brightness_limit: List[float] = field(default_factory=lambda: [-0.2, 0.2])
    contrast_limit: List[float] = field(default_factory=lambda: [-0.2, 0.2])
    
    # Image compression
    compression_quality_range: List[int] = field(default_factory=lambda: [40, 100])
    
    # Probabilidades
    geometric_p: float = 0.2
    blur_noise_p: float = 0.2
    color_p: float = 0.2
    compression_p: float = 0.2


@dataclass
class SchedulerConfig:
    """Configuração do Learning Rate Scheduler."""
    
    name: Literal["reduce_on_plateau", "cosine", "step", "none"] = "reduce_on_plateau"
    
    # ReduceLROnPlateau params
    factor: float = 0.5
    patience: int = 3
    min_lr: float = 1e-8
    mode: Literal["min", "max"] = "max"  # max para accuracy, min para loss
    
    # Cosine params
    T_max: Optional[int] = None
    eta_min: float = 1e-8
    
    # Step params
    step_size: int = 10
    gamma: float = 0.1


@dataclass
class ModelConfig:
    """Configuração do modelo multimodal."""
    
    # Nome do modelo no HuggingFace ou caminho local
    name: str = "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M"
    
    # Tipo de modelo para seleção de wrapper
    type: Literal["clip", "tinyclip", "siglip"] = "tinyclip"
    
    # Tamanho da imagem de entrada
    image_size: int = 224
    
    # Fine-tuning strategy
    freeze_vision_encoder: bool = False
    freeze_text_encoder: bool = False
    
    # Configurações específicas
    output_attentions: bool = True  # Para attention rollout


@dataclass
class LossConfig:
    """Configuração das funções de perda."""
    
    # Tipo de loss principal
    primary_loss: Literal["infonce", "sigmoid", "cross_entropy"] = "cross_entropy"
    
    # Loss auxiliar (opcional)
    use_triplet_loss: bool = True
    triplet_weight: float = 0.2  # λ para loss combinada
    triplet_margin: float = 0.2
    
    # Temperatura para contrastive losses
    temperature: float = 0.07
    
    # Sigmoid loss params (SigLIP)
    sigmoid_bias: float = -10.0


@dataclass
class DataConfig:
    """Configuração do dataset."""
    
    # Caminhos
    dataset_csv: str = "datasets/dataset-sign-align/dataset.csv"
    images_base_path: str = ""  # Se vazio, usa caminhos absolutos do CSV
    
    # Split fixo (v2) - preferir usar split_path
    split_path: Optional[str] = "datasets/dataset-sign-align/splits/split_v2.json"
    
    # Split dinâmico (legado) - usado apenas se split_path for None
    train_ratio: float = 0.75
    val_ratio: float = 0.10
    test_ratio: float = 0.15
    
    # Batching
    batch_size: int = 8
    num_workers: int = 4
    
    # Filtros
    exclude_unknown: bool = True
    
    # Modo de teste (limita número de amostras)
    max_samples: Optional[int] = None  # Se definido, usa apenas N amostras
    
    # Augmentation
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class EvaluationConfig:
    """Configuração de avaliação."""
    
    # Número de exemplos negativos para teste
    negative_samples: List[int] = field(default_factory=lambda: [1, 2, 3])
    
    # Métricas a calcular
    compute_eer: bool = True
    compute_auc: bool = True
    compute_recall_at_k: bool = True
    recall_k_values: List[int] = field(default_factory=lambda: [1, 5, 10])
    compute_mrr: bool = True
    compute_ndcg: bool = True
    
    # Visualização
    num_visualization_samples: int = 10
    save_attention_maps: bool = True


@dataclass
class TrainingConfig:
    """Configuração de treinamento."""
    
    # Hiperparâmetros
    epochs: int = 30  # Atualizado de 20 para 30
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Mixed precision
    use_amp: bool = True
    
    # Checkpointing
    save_every: int = 1
    save_top_k: int = 3
    save_only_best: bool = True  # Se True, salva apenas o melhor modelo
    
    # Logging
    log_interval: int = 50
    
    # Early stopping (opcional)
    early_stopping_patience: Optional[int] = None
    
    # Scheduler
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class ExperimentConfig:
    """Configuração completa de um experimento."""
    
    # Identificação
    name: str = "experiment"
    description: str = ""
    seed: int = 5932
    
    # Diretórios
    output_dir: str = "experiments"
    
    # Componentes
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte configuração para dicionário."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Cria configuração a partir de dicionário."""
        # Converter subdicionários para dataclasses
        if "model" in data and isinstance(data["model"], dict):
            data["model"] = ModelConfig(**data["model"])
        
        if "training" in data and isinstance(data["training"], dict):
            if "scheduler" in data["training"] and isinstance(data["training"]["scheduler"], dict):
                data["training"]["scheduler"] = SchedulerConfig(**data["training"]["scheduler"])
            data["training"] = TrainingConfig(**data["training"])
        
        if "data" in data and isinstance(data["data"], dict):
            if "augmentation" in data["data"] and isinstance(data["data"]["augmentation"], dict):
                data["data"]["augmentation"] = AugmentationConfig(**data["data"]["augmentation"])
            data["data"] = DataConfig(**data["data"])
        
        if "loss" in data and isinstance(data["loss"], dict):
            data["loss"] = LossConfig(**data["loss"])
        
        if "evaluation" in data and isinstance(data["evaluation"], dict):
            data["evaluation"] = EvaluationConfig(**data["evaluation"])
        
        return cls(**data)


def load_config(path: str) -> ExperimentConfig:
    """
    Carrega configuração de um arquivo YAML.
    
    Args:
        path: Caminho para o arquivo YAML.
        
    Returns:
        Objeto ExperimentConfig.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return ExperimentConfig.from_dict(data)


def save_config(config: ExperimentConfig, path: str) -> None:
    """
    Salva configuração em um arquivo YAML.
    
    Args:
        config: Objeto ExperimentConfig.
        path: Caminho para salvar o arquivo YAML.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
    print(f"📝 Configuração salva em: {path}")

