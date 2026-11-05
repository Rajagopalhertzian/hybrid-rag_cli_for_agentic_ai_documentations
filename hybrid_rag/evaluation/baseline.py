"""
Closed-book baseline (no retrieval) for evaluation.
"""

import time
from dataclasses import dataclass
from typing import Optional

from groq import AsyncGroq

from hybrid_rag.config import settings
from hybrid_rag.models import Answer


SYSTEM_PROMPT_CLOSED_BOOK = """You are a knowledgeable assistant answering questions about agentic AI protocols (MCP, A2A, LangGraph).
Answer based on your training knowledge. If you're unsure, say so.
Do not make up specific version numbers or API details."""


@dataclass
class BaselineConfig:
    model: str = settings.groq_model
    temperature: float = 0.1
    max_tokens: int = 2048


async def closed_book_answer(
    question: str,
    config: Optional[BaselineConfig] = None,
) -> Answer:
    """
    Generate answer without any retrieval context (closed-book).
    """
    config = config or BaselineConfig()
    client = AsyncGroq(api_key=settings.groq_api_key)
    start_time = time.perf_counter()

    response = await client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_CLOSED_BOOK},
            {"role": "user", "content": question},
        ],
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    latency_ms = (time.perf_counter() - start_time) * 1000
    answer_text = response.choices[0].message.content or ""

    return Answer(
        text=answer_text,
        citations=[],
        retrieved_chunks=[],
        model=config.model,
        latency_ms=latency_ms,
    )