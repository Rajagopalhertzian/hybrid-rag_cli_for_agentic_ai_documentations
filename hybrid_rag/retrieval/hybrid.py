"""
Hybrid search using Qdrant's native vector + BM25 + RRF.
"""

from dataclasses import dataclass
from typing import Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Prefetch,
    FusionQuery,
    Fusion,
    SearchParams,
)

from hybrid_rag.config import settings
from hybrid_rag.models import Chunk, ChunkMetadata, ChunkerType, RetrievedChunk, Source
from hybrid_rag.retrieval.qdrant import QdrantClient, get_qdrant_client


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""

    top_k: int = settings.retrieval_top_k
    vector_k: int = settings.retrieval_vector_k
    bm25_k: int = settings.retrieval_bm25_k
    rrf_k: int = settings.retrieval_rrf_k
    use_reranker: bool = settings.reranker_enabled
    rerank_top_k: int = settings.retrieval_rerank_top_k
    hnsw_ef: int = 128


async def hybrid_search(
    query_text: str,
    query_vector: list[float],
    source: Optional[Source] = None,
    chunker: ChunkerType = ChunkerType.FIXED,
    config: Optional[HybridSearchConfig] = None,
    client: Optional[AsyncQdrantClient] = None,
) -> list[RetrievedChunk]:
    """
    Perform hybrid search (vector + BM25 + RRF) using Qdrant's native API.

    Args:
        query_text: Raw query text for BM25
        query_vector: Query embedding for vector search
        source: Optional source filter
        chunker: Which chunker's collection to search
        config: Search configuration
        client: Optional Qdrant client

    Returns:
        List of RetrievedChunk with scores
    """
    config = config or HybridSearchConfig()
    qdrant = QdrantClient(client)

    # Build filter
    filter_conditions = []
    if source:
        filter_conditions.append(
            FieldCondition(key="source", match=MatchValue(value=source.value))
        )
    filter_conditions.append(
        FieldCondition(key="chunker_type", match=MatchValue(value=chunker.value))
    )

    query_filter = Filter(must=filter_conditions) if filter_conditions else None

    # Build prefetch for vector search
    vector_prefetch = Prefetch(
        query=query_vector,
        using="embedding",
        limit=config.vector_k,
        filter=query_filter,
        params=SearchParams(hnsw_ef=config.hnsw_ef, exact=False),
    )

    # Build prefetch for BM25
    bm25_prefetch = Prefetch(
        query=query_text,
        using="text",
        limit=config.bm25_k,
        filter=query_filter,
    )

    # Execute hybrid search with RRF
    collections = [qdrant.collection_name(s, chunker) for s in Source] if source is None else [qdrant.collection_name(source, chunker)]

    all_results = []
    for collection_name in collections:
        try:
            results = await qdrant._get_client().query_points(
                collection_name=collection_name,
                prefetch=[vector_prefetch, bm25_prefetch],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=config.top_k,
                with_payload=True,
                with_vectors=False,
            )
            all_results.extend(results.points)
        except Exception as e:
            # Collection might not exist
            pass

    # Sort by RRF score (already sorted by Qdrant, but merge across collections)
    all_results.sort(key=lambda p: p.score or 0, reverse=True)
    all_results = all_results[: config.top_k]

    # Convert to RetrievedChunk
    retrieved = []
    for point in all_results:
        payload = point.payload or {}
        metadata = ChunkMetadata(
            source=Source(payload.get("source", "mcp")),
            url=payload.get("url", ""),
            heading_path=tuple(payload.get("heading_path", [])),
            char_start=payload.get("char_start", 0),
            char_end=payload.get("char_end", 0),
            token_count=payload.get("token_count", 0),
            chunker_type=ChunkerType(payload.get("chunker_type", "fixed")),
            document_sha256=payload.get("document_sha256", ""),
            fetched_at=payload.get("fetched_at", ""),
        )

        chunk = Chunk(
            id=point.id,
            text=payload.get("text", ""),
            metadata=metadata,
        )

        retrieved.append(
            RetrievedChunk(
                chunk=chunk,
                score=point.score or 0.0,
                retrieval_method="rrf",
            )
        )

    return retrieved


async def vector_search_only(
    query_vector: list[float],
    source: Optional[Source] = None,
    chunker: ChunkerType = ChunkerType.FIXED,
    top_k: int = 10,
    client: Optional[AsyncQdrantClient] = None,
) -> list[RetrievedChunk]:
    """Vector-only search (for comparison)."""
    config = HybridSearchConfig(top_k=top_k, vector_k=top_k, bm25_k=0)
    # Use empty text query for BM25 (effectively disabled)
    return await hybrid_search(
        query_text="",
        query_vector=query_vector,
        source=source,
        chunker=chunker,
        config=config,
        client=client,
    )


async def bm25_search_only(
    query_text: str,
    source: Optional[Source] = None,
    chunker: ChunkerType = ChunkerType.FIXED,
    top_k: int = 10,
    client: Optional[AsyncQdrantClient] = None,
) -> list[RetrievedChunk]:
    """BM25-only search (for comparison)."""
    config = HybridSearchConfig(top_k=top_k, vector_k=0, bm25_k=top_k)
    # Use zero vector for vector search (effectively disabled)
    zero_vector = [0.0] * settings.embedding_dim
    return await hybrid_search(
        query_text=query_text,
        query_vector=zero_vector,
        source=source,
        chunker=chunker,
        config=config,
        client=client,
    )