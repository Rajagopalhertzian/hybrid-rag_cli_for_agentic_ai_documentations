"""
Qdrant client wrapper for hybrid-rag.
"""

from contextlib import asynccontextmanager
from typing import Optional

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import Distance, VectorParams

from hybrid_rag.config import settings
from hybrid_rag.models import ChunkerType, Source

_client: Optional[AsyncQdrantClient] = None


async def get_qdrant_client() -> AsyncQdrantClient:
    """Get or create the global Qdrant client."""
    global _client
    if _client is None:
        _client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            prefer_grpc=settings.qdrant_prefer_grpc,
        )
    return _client


async def close_qdrant_client():
    """Close the global Qdrant client."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None


@asynccontextmanager
async def qdrant_client():
    """Context manager for Qdrant client."""
    client = await get_qdrant_client()
    try:
        yield client
    finally:
        pass  # Keep client alive for reuse


class QdrantClient:
    """High-level Qdrant operations for hybrid-rag."""

    def __init__(self, client: Optional[AsyncQdrantClient] = None):
        self._client = client

    async def _get_client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = await get_qdrant_client()
        return self._client

    def collection_name(self, source: Source, chunker: ChunkerType) -> str:
        """Generate collection name."""
        return f"hybrid_rag_{chunker.value}_{source.value}"

    async def create_collections(self):
        """Create all 6 collections with proper schema."""
        client = await self._get_client()

        for source in Source:
            for chunker in ChunkerType:
                name = self.collection_name(source, chunker)
                exists = await client.collection_exists(name)
                if not exists:
                    await client.create_collection(
                        collection_name=name,
                        vectors_config=VectorParams(
                            size=settings.embedding_dim,
                            distance=Distance.COSINE,
                            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
                        ),
                        # Enable full-text search (BM25) on text field
                        sparse_vectors_config={
                            "text": models.SparseVectorParams(
                                index=models.SparseIndexParams(
                                    on_disk=False,
                                )
                            )
                        },
                    )
                    console.print(f"Created collection: {name}")

                    # Create payload indexes for filtering
                    await self._create_payload_indexes(client, name)
                else:
                    console.print(f"Collection exists: {name}")

    async def _create_payload_indexes(self, client: AsyncQdrantClient, collection_name: str):
        """Create payload indexes for efficient filtering."""
        indexes = [
            ("source", models.PayloadSchemaType.KEYWORD),
            ("url", models.PayloadSchemaType.KEYWORD),
            ("heading_path", models.PayloadSchemaType.KEYWORD),
            ("token_count", models.PayloadSchemaType.INTEGER),
            ("chunker_type", models.PayloadSchemaType.KEYWORD),
        ]

        for field_name, field_type in indexes:
            try:
                await client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
            except Exception:
                # Index might already exist
                pass

    async def upsert_chunks(
        self,
        source: Source,
        chunker: ChunkerType,
        chunks: list,
        batch_size: int = 256,
    ):
        """Upsert chunks into Qdrant collection."""
        client = await self._get_client()
        collection_name = self.collection_name(source, chunker)

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            points = []

            for chunk in batch:
                if chunk.embedding is None:
                    continue

                points.append(
                    models.PointStruct(
                        id=str(chunk.id),
                        vector={
                            "embedding": chunk.embedding,
                            "text": chunk.text,  # For BM25
                        },
                        payload={
                            "text": chunk.text,
                            "source": chunk.metadata.source.value,
                            "url": chunk.metadata.url,
                            "heading_path": list(chunk.metadata.heading_path),
                            "char_start": chunk.metadata.char_start,
                            "char_end": chunk.metadata.char_end,
                            "token_count": chunk.metadata.token_count,
                            "chunker_type": chunk.metadata.chunker_type.value,
                            "document_sha256": chunk.metadata.document_sha256,
                            "fetched_at": chunk.metadata.fetched_at.isoformat(),
                        },
                    )
                )

            if points:
                await client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True,
                )

    async def get_collection_info(self, source: Source, chunker: ChunkerType) -> dict:
        """Get collection info (count, etc.)."""
        client = await self._get_client()
        name = self.collection_name(source, chunker)
        info = await client.get_collection(name)
        return {
            "name": name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
        }

    async def delete_collection(self, source: Source, chunker: ChunkerType):
        """Delete a collection."""
        client = await self._get_client()
        name = self.collection_name(source, chunker)
        await client.delete_collection(name)


# Import console for logging
from rich.console import Console
console = Console()