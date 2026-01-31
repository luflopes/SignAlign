"""
Data Augmentation para imagens de assinaturas.

Baseado nas augmentations do experimento original com configuração flexível.
"""

from typing import Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from src.config.experiment_config import AugmentationConfig


def get_train_transform(config: Optional[AugmentationConfig] = None) -> A.Compose:
    """
    Retorna transformações de treinamento.
    
    Args:
        config: Configuração de augmentation. Se None, usa padrão.
        
    Returns:
        Compose de Albumentations para treinamento.
    """
    if config is None:
        config = AugmentationConfig()
    
    if not config.enabled:
        return get_val_transform()
    
    transforms = [
        # Transformações geométricas
        A.OneOf([
            A.Affine(
                translate_percent={
                    "x": tuple(config.shift_limit),
                    "y": tuple(config.shift_limit)
                },
                scale=tuple([1 + s for s in config.scale_limit]),
                rotate=tuple(config.rotate_limit),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                fill=255,
            ),
            A.Downscale(
                scale_range=tuple(config.downscale_range),
            )
        ], p=config.geometric_p),
        
        # Blur e ruído
        A.OneOf([
            A.MotionBlur(
                blur_limit=tuple(config.motion_blur_limit),
                allow_shifted=False,
            ),
            A.GaussNoise(
                std_range=tuple(config.noise_std_range),
            ),
        ], p=config.blur_noise_p),
        
        # Ajustes de cor
        A.RandomBrightnessContrast(
            brightness_limit=tuple(config.brightness_limit),
            contrast_limit=tuple(config.contrast_limit),
            brightness_by_max=True,
            p=config.color_p
        ),
        
        # Compressão JPEG
        A.ImageCompression(
            quality_range=tuple(config.compression_quality_range),
            p=config.compression_p
        ),
        
        # Converter para tensor
        ToTensorV2()
    ]
    
    return A.Compose(transforms)


def get_val_transform() -> A.Compose:
    """
    Retorna transformações de validação (apenas conversão para tensor).
    
    Returns:
        Compose de Albumentations para validação.
    """
    return A.Compose([ToTensorV2()])


def get_inference_transform() -> A.Compose:
    """
    Retorna transformações para inferência (idêntico a validação).
    
    Returns:
        Compose de Albumentations para inferência.
    """
    return get_val_transform()

