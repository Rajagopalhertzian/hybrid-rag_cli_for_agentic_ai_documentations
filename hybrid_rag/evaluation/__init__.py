"""
Evaluation modules for hybrid-rag.
"""

from hybrid_rag.evaluation.baseline import closed_book_answer
from hybrid_rag.evaluation.metrics import RetrievalMetrics, GenerationMetrics, compute_retrieval_metrics, compute_generation_metrics
from hybrid_rag.evaluation.runner import EvaluationRunner, run_evaluation

__all__ = [
    "closed_book_answer",
    "RetrievalMetrics",
    "GenerationMetrics",
    "compute_retrieval_metrics",
    "compute_generation_metrics",
    "EvaluationRunner",
    "run_evaluation",
]