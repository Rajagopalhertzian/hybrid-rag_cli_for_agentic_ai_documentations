"""
Evaluation metrics for retrieval and generation.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Optional imports for RAGAS
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    ragas_evaluate = None
    faithfulness = answer_relevancy = context_precision = context_recall = None
    Dataset = None
    RAGAS_AVAILABLE = False

from hybrid_rag.models import RetrievedChunk, Evidence, EvalQuestion, RetrievalMetrics, GenerationMetrics


def compute_retrieval_metrics(
    retrieved: list[RetrievedChunk],
    evidence: list[Evidence],
    k_values: list[int] = [5, 10],
) -> RetrievalMetrics:
    """
    Compute retrieval metrics: Recall@k, MRR, nDCG@k.

    Evidence is matched by URL and heading overlap with retrieved chunks.
    """
    if not evidence or not retrieved:
        return RetrievalMetrics(
            recall_at_5=0.0,
            recall_at_10=0.0,
            mrr=0.0,
            ndcg_at_10=0.0,
        )

    # Build set of relevant (url, heading) pairs from evidence
    relevant_pairs = set()
    for ev in evidence:
        relevant_pairs.add((ev.url, ev.heading))

    # Check which retrieved chunks match evidence
    retrieved_pairs = []
    for chunk in retrieved:
        meta = chunk.chunk.metadata
        heading = meta.heading_path[-1] if meta.heading_path else ""
        retrieved_pairs.append((meta.url, heading))

    # Recall@k
    recall_scores = {}
    for k in k_values:
        top_k_pairs = set(retrieved_pairs[:k])
        hits = len(top_k_pairs & relevant_pairs)
        recall_scores[k] = hits / len(relevant_pairs) if relevant_pairs else 0.0

    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for i, pair in enumerate(retrieved_pairs, 1):
        if pair in relevant_pairs:
            mrr = 1.0 / i
            break

    # nDCG@10
    def dcg_at_k(relevance: list[int], k: int) -> float:
        return sum(
            rel / math.log2(idx + 2) for idx, rel in enumerate(relevance[:k])
        )

    relevance = [1 if pair in relevant_pairs else 0 for pair in retrieved_pairs]
    ideal_relevance = sorted(relevance, reverse=True)
    dcg = dcg_at_k(relevance, 10)
    idcg = dcg_at_k(ideal_relevance, 10)
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return RetrievalMetrics(
        recall_at_5=recall_scores.get(5, 0.0),
        recall_at_10=recall_scores.get(10, 0.0),
        mrr=mrr,
        ndcg_at_10=ndcg,
    )


async def compute_generation_metrics(
    question: str,
    answer: str,
    retrieved_chunks: list[RetrievedChunk],
) -> GenerationMetrics:
    """
    Compute generation metrics using RAGAS.

    Metrics: faithfulness, answer_relevancy, context_precision, context_recall
    """
    if not retrieved_chunks or not RAGAS_AVAILABLE:
        return GenerationMetrics(
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_precision=0.0,
            context_recall=0.0,
        )

    # Prepare data for RAGAS
    contexts = [chunk.chunk.text for chunk in retrieved_chunks]

    dataset = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [""],  # Not used for these metrics
    })

    try:
        result = ragas_evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )

        return GenerationMetrics(
            faithfulness=float(result["faithfulness"]),
            answer_relevancy=float(result["answer_relevancy"]),
            context_precision=float(result["context_precision"]),
            context_recall=float(result["context_recall"]),
        )
    except Exception as e:
        # Fallback if RAGAS fails
        return GenerationMetrics(
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_precision=0.0,
            context_recall=0.0,
        )