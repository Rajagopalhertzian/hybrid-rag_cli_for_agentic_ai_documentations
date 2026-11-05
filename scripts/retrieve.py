#!/usr/bin/env python3
"""
Test retrieval on a sample question.
Usage: python -m scripts.retrieve "your question here"
"""

import asyncio
import sys

from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer

from hybrid_rag.config import settings
from hybrid_rag.models import ChunkerType, Source
from hybrid_rag.retrieval.hybrid import hybrid_search
from hybrid_rag.retrieval.qdrant import get_qdrant_client

console = Console()


async def main(question: str):
    console.print(f"[bold]Question:[/bold] {question}\n")

    # Embed query
    console.print("Embedding query...")
    model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
    query_vector = model.encode([question], normalize_embeddings=True)[0].tolist()

    # Search both chunkers
    client = await get_qdrant_client()

    for chunker in ChunkerType:
        console.print(f"\n[bold cyan]=== {chunker.value.upper()} chunker ===[/bold cyan]")

        results = await hybrid_search(
            query_text=question,
            query_vector=query_vector,
            source=None,  # All sources
            chunker=chunker,
            client=client,
        )

        if not results:
            console.print("[yellow]No results found[/yellow]")
            continue

        table = Table(title=f"Top {len(results)} results ({chunker.value})")
        table.add_column("Score", style="green", width=8)
        table.add_column("Source", style="cyan", width=10)
        table.add_column("Heading", style="white", width=40)
        table.add_column("Preview", style="dim", width=80)

        for r in results:
            meta = r.chunk.metadata
            heading = " > ".join(meta.heading_path) if meta.heading_path else "(root)"
            preview = r.chunk.text[:120].replace("\n", " ") + "..."
            table.add_row(f"{r.score:.4f}", meta.source.value, heading, preview)

        console.print(table)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.retrieve \"your question here\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    asyncio.run(main(question))