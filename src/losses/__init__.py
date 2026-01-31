"""Módulo de funções de perda para treinamento contrastivo."""

from src.losses.contrastive import CrossEntropyLoss, InfoNCELoss
from src.losses.sigmoid import SigmoidLoss
from src.losses.triplet import TripletLoss
from src.losses.combined import CombinedLoss, create_loss_function

__all__ = [
    "CrossEntropyLoss",
    "InfoNCELoss",
    "SigmoidLoss",
    "TripletLoss",
    "CombinedLoss",
    "create_loss_function",
]

