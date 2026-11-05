"""
Core data models for hybrid-rag.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4


class Source(str, Enum):
    """Document source identifiers."""

    MCP = "mcp"
    A2A = "a2a"
    LANGGRAPH = "langgraph"


class ChunkerType(str, Enum):
    """Chunking strategy identifiers."""

    FIXED = "fixed"
    HEADING = "heading"


@dataclass(frozen=True, slots=True)
class DocumentRef:
    """Reference to a source document."""

    source: Source
    url: str
    local_path: Path
    sha256: str
    fetched_at: datetime
    heading_path: tuple[str, ...] = field(default_factory=tuple)

    def collection_name(self, chunker: ChunkerType) -> str:
        """Generate Qdrant collection name for this document's chunks."""
        return f"hybrid_rag_{chunker.value}_{self.source.value}"


@dataclass(frozen=True, slots=True)
class ChunkMetadata:
    """Metadata attached to each chunk."""

    source: Source
    url: str
    heading_path: tuple[str, ...]
    char_start: int
    char_end: int
    token_count: int
    chunker_type: ChunkerType
    document_sha256: str
    fetched_at: datetime


@dataclass(frozen=True, slots=True)
class Chunk:
    """A text chunk with metadata and optional embedding."""

    id: UUID = field(default_factory=uuid4)
    text: str = ""
    metadata: Optional[ChunkMetadata] = None
    embedding: Optional[list[float]] = None

    def __post_init__(self):
        if self.metadata is not None and self.id is None:
            object.__setattr__(self, "id", uuid4())


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk returned from retrieval with score."""

    chunk: Chunk
    score: float
    retrieval_method: str  # "vector", "bm25", "rrf", "reranked"

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def metadata(self) -> Optional[ChunkMetadata]:
        return self.chunk.metadata


@dataclass(frozen=True, slots=True)
class Evidence:
    """Ground truth evidence for evaluation."""

    url: str
    heading: str
    char_range: tuple[int, int]  # (start, end) character offsets in source doc
    source: Source


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    """An evaluation question with ground truth evidence."""

    id: str
    question: str
    evidence: list[Evidence]
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Answer:
    """Generated answer with citations."""

    text: str
    citations: list[str]  # Format: "url#heading"
    retrieved_chunks: list[RetrievedChunk]
    model: str
    latency_ms: float


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Retrieval quality metrics."""

    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """Generation quality metrics (RAGAS)."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Complete evaluation result for one question."""

    question_id: str
    chunker_type: ChunkerType
    retrieval: RetrievalMetrics
    generation: GenerationMetrics
    answer: Answer
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """Entry in the fetch manifest."""

    source: Source
    url: str
    local_path: Path
    sha256: str
    fetched_at: datetime
    status: str = "ok"  # ok, skipped, error
    error: Optional[str] = None


@dataclass(frozen=True, slots=True)
class Manifest:
    """Fetch manifest containing all downloaded documents."""

    entries: list[ManifestEntry]
    created_at: datetime = field(default_factory=datetime.now)
    version: int = 1

    def get_by_source(self, source: Source) -> list[ManifestEntry]:
        return [e for e in self.entries if e.source == source]

    def get_entry(self, url: str) -> Optional[ManifestEntry]:
        for entry in self.entries:
            if entry.url == url:
                return entry
        return None