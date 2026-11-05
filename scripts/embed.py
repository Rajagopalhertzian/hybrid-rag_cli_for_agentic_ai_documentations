#!/usr/bin/env python3
"""
Embed chunks and upsert to Qdrant.
"""

import asyncio
from pathlib import Path

import orjson
from rich.console import Console
from rich.progress import Progress, TaskProgressColumn, TextColumn
from sentence_transformers import SentenceTransformer

from hybrid_rag.config import settings
from hybrid_rag.models import Chunk, ChunkMetadata, ChunkerType, Source
from hybrid_rag.retrieval.qdrant import QdrantClient

console = Console()


async def main():
    console.print("[bold]hybrid-rag embed[/bold] - Embedding chunks and upserting to Qdrant\n")

    # Load embedding model
    console.print(f"Loading embedding model: {settings.embedding_model}")
    model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
    console.print(f"Model loaded (dim={model.get_sentence_embedding_dimension()})")

    # Initialize Qdrant
    qdrant = QdrantClient()
    await qdrant.create_collections()

    # Process each source and chunker
    total_chunks = 0

    for source in Source:
        for chunker in ChunkerType:
            chunker_dir = settings.processed_data_dir / chunker.value / source.value
            if not chunker_dir.exists():
                console.print(f"[yellow]Skipping {chunker.value}/{source.value} - no data[/yellow]")
                continue

            # Collect chunk files
            chunk_files = list(chunker_dir.glob("*.json"))
            console.print(f"\n[bold cyan]{chunker.value}/{source.value}[/bold cyan]: {len(chunk_files)} chunks")

            if not chunk_files:
                continue

            # Process in batches
            chunks_to_embed = []

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                load_task = progress.add_task("[green]Loading", total=len(chunk_files))

                for chunk_file in chunk_files:
                    data = orjson.loads(chunk_file.read_bytes())
                    chunk = Chunk(
                        id=data["id"],
                        text=data["text"],
                        metadata=ChunkMetadata(
                            source=Source(data["metadata"]["source"]),
                            url=data["metadata"]["url"],
                            heading_path=tuple(data["metadata"]["heading_path"]),
                            char_start=data["metadata"]["char_start"],
                            char_end=data["metadata"]["char_end"],
                            token_count=data["metadata"]["token_count"],
                            chunker_type=ChunkerType(data["metadata"]["chunker_type"]),
                            document_sha256=data["metadata"]["document_sha256"],
                            fetched_at=data["metadata"]["fetched_at"],
                        ),
                    )
                    chunks_to_embed.append(chunk)
                    progress.advance(load_task)

            # Embed in batches
            embed_task = progress.add_task("[blue]Embedding", total=len(chunks_to_embed))

            for i in range(0, len(chunks_to_embed), settings.embedding_batch_size):
                batch = chunks_to_embed[i : i + settings.embedding_batch_size]
                texts = [c.text for c in batch]

                embeddings = model.encode(
                    texts,
                    batch_size=settings.embedding_batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                for chunk, embedding in zip(batch, embeddings):
                    chunk.embedding = embedding.tolist()

                progress.advance(embed_task, len(batch))

            # Upsert to Qdrant
            upsert_task = progress.add_task("[magenta]Upserting", total=len(chunks_to_embed))

            for i in range(0, len(chunks_to_embed), 256):
                batch = chunks_to_embed[i : i + 256]
                await qdrant.upsert_chunks(source, chunker, batch)
                progress.advance(upsert_task, len(batch))

            total_chunks += len(chunks_to_embed)

    console.print(f"\n[green]Done! Embedded and upserted {total_chunks} total chunks[/green]")


if __name__ == "__main__":
    asyncio.run(main())