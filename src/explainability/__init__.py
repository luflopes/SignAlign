"""Módulo de explicabilidade e visualização de atenção."""

from src.explainability.attention_rollout import (
    AttentionRollout,
    visualize_attention,
    generate_attention_map,
)

__all__ = [
    "AttentionRollout",
    "visualize_attention",
    "generate_attention_map",
]

