#!/usr/bin/env python3
"""
Test full RAG pipeline (retrieve + generate) on a sample question.
Usage: python -m scripts.generate "your question here"
"""

import asyncio
import sys

from rich.console import Console
from rich.panel import Panel
from sentence_transformers import SentenceTransformer

from hybrid_rag.config import settings
from hybrid_rag.models import ChunkerType, Source
from hybrid_rag.retrieval.hybrid import hybrid_search
from hybrid_rag.retrieval.qdrant import get_qdrant_client
from hybrid_rag.generation.synthesizer import synthesize_answer

console = Console()


async def main(question: str):
    console.print(f"[bold]Question:[/bold] {question}\n")

    # Embed query
    console.print("Embedding query...")
    model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
    query_vector = model.encode([question], normalize_embeddings=True)[0].tolist()

    client = await get_qdrant_client()

    for chunker in ChunkerType:
        console.print(f"\n[bold cyan]=== {chunker.value.upper()} chunker ===[/bold cyan]")

        # Retrieve
        results = await hybrid_search(
            query_text=question,
            query_vector=query_vector,
            source=None,
            chunker=chunker,
            client=client,
        )

        if not results:
            console.print("[yellow]No results found[/yellow]")
            continue

        console.print(f"Retrieved {len(results)} chunks")

        # Generate
        console.print("Generating answer...")
        answer = await synthesize_answer(question, results)

        # Display
        console.print(Panel(
            answer.text,
            title=f"Answer ({chunker.value})",
            subtitle=f"Latency: {answer.latency_ms:.0f}ms | Model: {answer.model}",
            border_style="green",
        ))

        if answer.citations:
            console.print("\n[bold]Citations:[/bold]")
            for cite in answer.citations:
                console.print(f"  • {cite}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.generate \"your question here\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    asyncio.run(main(question))