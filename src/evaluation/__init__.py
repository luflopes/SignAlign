"""Módulo de avaliação e métricas."""

from src.evaluation.metrics import (
    compute_eer,
    compute_all_metrics,
    compute_retrieval_metrics,
)
from src.evaluation.retrieval import (
    RetrievalEvaluator,
    compute_recall_at_k,
    compute_mrr,
    compute_ndcg,
)

__all__ = [
    "compute_eer",
    "compute_all_metrics",
    "compute_retrieval_metrics",
    "RetrievalEvaluator",
    "compute_recall_at_k",
    "compute_mrr",
    "compute_ndcg",
]

