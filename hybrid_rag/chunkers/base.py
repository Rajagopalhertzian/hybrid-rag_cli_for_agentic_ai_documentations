"""
Base chunker protocol and shared utilities.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from hybrid_rag.models import Chunk, ChunkMetadata, ChunkerType


@dataclass(frozen=True, slots=True)
class ChunkerConfig:
    """Configuration for a chunker."""

    chunker_type: ChunkerType
    target_tokens: int = 512
    overlap_tokens: int = 50
    max_tokens: int = 1024


@runtime_checkable
class Chunker(Protocol):
    """Protocol for text chunkers."""

    config: ChunkerConfig

    def chunk(self, text: str, metadata: ChunkMetadata) -> list[Chunk]:
        """Split text into chunks with metadata."""
        ...

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        ...


# Token counting utility (uses tiktoken cl100k_base for BGE compatibility)
try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _ENCODER = None


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base)."""
    if _ENCODER is None:
        # Fallback: rough estimate (~4 chars per token)
        return len(text) // 4
    return len(_ENCODER.encode(text))


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (simple regex-based)."""
    # Simple sentence splitting - could be improved with nltk/spacy
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]