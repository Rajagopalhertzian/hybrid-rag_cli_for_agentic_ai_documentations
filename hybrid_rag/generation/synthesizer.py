"""
Answer synthesis using Groq LLM.
"""

import time
from dataclasses import dataclass
from typing import Optional

from groq import AsyncGroq

from hybrid_rag.config import settings
from hybrid_rag.models import Answer, RetrievedChunk
from hybrid_rag.retrieval.hybrid import hybrid_search
from hybrid_rag.retrieval.qdrant import get_qdrant_client


# Prompt templates
SYSTEM_PROMPT = """You are a precise technical assistant answering questions about agentic AI protocols and frameworks.
Use ONLY the provided context to answer. If the context doesn't contain the answer, say so.
Cite sources inline using [url#heading] format."""

USER_PROMPT_TEMPLATE = """Question: {question}

Context:
{context}

Answer the question based on the context above. Include inline citations as [url#heading]."""


@dataclass
class SynthesizerConfig:
    """Configuration for answer synthesis."""

    model: str = settings.groq_model
    temperature: float = 0.1
    max_tokens: int = 2048
    top_p: float = 0.95


class Synthesizer:
    """Generates answers using Groq LLM with retrieved context."""

    def __init__(
        self,
        config: Optional[SynthesizerConfig] = None,
        client: Optional[AsyncGroq] = None,
    ):
        self.config = config or SynthesizerConfig()
        self._client = client

    async def _get_client(self) -> AsyncGroq:
        if self._client is None:
            self._client = AsyncGroq(api_key=settings.groq_api_key)
        return self._client

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into context string."""
        parts = []
        for i, chunk in enumerate(chunks):
            metadata = chunk.chunk.metadata
            heading = metadata.heading_path[-1] if metadata.heading_path else "Document"
            citation = f"{metadata.url}#{heading}"
            parts.append(f"[Source {i+1}] {citation}\n{chunk.chunk.text}")
        return "\n\n---\n\n".join(parts)

    def _extract_citations(self, chunks: list[RetrievedChunk]) -> list[str]:
        """Extract citation strings from chunks."""
        citations = []
        for chunk in chunks:
            metadata = chunk.chunk.metadata
            heading = metadata.heading_path[-1] if metadata.heading_path else "Document"
            citations.append(f"{metadata.url}#{heading}")
        return citations

    async def synthesize(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> Answer:
        """Generate answer from question and retrieved chunks."""
        client = await self._get_client()
        start_time = time.perf_counter()

        context = self._format_context(retrieved_chunks)
        citations = self._extract_citations(retrieved_chunks)

        prompt = USER_PROMPT_TEMPLATE.format(question=question, context=context)

        response = await client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000
        answer_text = response.choices[0].message.content or ""

        return Answer(
            text=answer_text,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            model=self.config.model,
            latency_ms=latency_ms,
        )


async def synthesize_answer(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    config: Optional[SynthesizerConfig] = None,
) -> Answer:
    """Convenience function for single-shot answer synthesis."""
    synthesizer = Synthesizer(config)
    return await synthesizer.synthesize(question, retrieved_chunks)