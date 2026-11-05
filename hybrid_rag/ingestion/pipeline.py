"""
Ingestion pipeline orchestrating clean → chunk → embed.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import orjson
from rich.console import Console
from rich.progress import Progress, TaskProgressColumn, TextColumn

from hybrid_rag.chunkers import FixedChunker, HeadingChunker
from hybrid_rag.config import settings
from hybrid_rag.ingestion.cleaner import clean_markdown
from hybrid_rag.models import Chunk, ChunkMetadata, Manifest, ManifestEntry, Source

console = Console()


class IngestionPipeline:
    """Orchestrates the full ingestion pipeline: clean → chunk (both strategies)."""

    def __init__(
        self,
        fixed_chunker: Optional[FixedChunker] = None,
        heading_chunker: Optional[HeadingChunker] = None,
    ):
        self.fixed_chunker = fixed_chunker or FixedChunker(
            target_tokens=settings.chunk_fixed_tokens,
            overlap_tokens=settings.chunk_fixed_overlap,
        )
        self.heading_chunker = heading_chunker or HeadingChunker(
            max_tokens=settings.chunk_heading_max_tokens,
        )

    def process_manifest(self, manifest: Manifest) -> dict[Source, list[Chunk]]:
        """Process all entries in manifest, returning chunks by source."""
        all_chunks = {}

        for source in Source:
            console.print(f"\n[bold cyan]Processing {source.value.upper()}[/bold cyan]")
            entries = manifest.get_by_source(source)
            source_chunks = []

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task(f"[green]{source.value}", total=len(entries))

                for entry in entries:
                    if entry.status != "ok":
                        progress.advance(task_id)
                        continue

                    chunks = self.process_entry(entry)
                    source_chunks.extend(chunks)
                    progress.advance(task_id)

            all_chunks[source] = source_chunks
            console.print(f"  Generated {len(source_chunks)} chunks")

        return all_chunks

    def process_entry(self, entry: ManifestEntry) -> list[Chunk]:
        """Process a single manifest entry through clean → chunk (both strategies)."""
        # Read raw content
        content = entry.local_path.read_text(encoding="utf-8")

        # Clean
        clean_result = clean_markdown(content)

        # Build base metadata
        base_metadata = ChunkMetadata(
            source=entry.source,
            url=entry.url,
            heading_path=clean_result.heading_path,
            char_start=0,
            char_end=len(clean_result.content),
            token_count=0,  # Will be set by chunker
            chunker_type=None,  # Will be set by chunker
            document_sha256=entry.sha256,
            fetched_at=entry.fetched_at,
        )

        # Chunk with both strategies
        all_chunks = []

        # Fixed chunking
        fixed_chunks = self.fixed_chunker.chunk(clean_result.content, base_metadata)
        all_chunks.extend(fixed_chunks)

        # Heading-aware chunking
        heading_chunks = self.heading_chunker.chunk(clean_result.content, base_metadata)
        all_chunks.extend(heading_chunks)

        return all_chunks

    def save_chunks(self, chunks_by_source: dict[Source, list[Chunk]]):
        """Save chunks to data/processed/{fixed,heading}/<source>/."""
        for source, chunks in chunks_by_source.items():
            for chunk in chunks:
                chunker_dir = "fixed" if chunk.metadata.chunker_type.value == "fixed" else "heading"
                out_dir = settings.processed_data_dir / chunker_dir / source.value
                out_dir.mkdir(parents=True, exist_ok=True)

                out_file = out_dir / f"{chunk.id}.json"
                out_file.write_bytes(
                    orjson.dumps(
                        {
                            "id": str(chunk.id),
                            "text": chunk.text,
                            "metadata": {
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
                            "embedding": chunk.embedding,
                        },
                        option=orjson.OPT_INDENT_2,
                    )
                )


def load_manifest() -> Manifest:
    """Load manifest from data/manifest.json."""
    manifest_path = settings.data_dir / "manifest.json"
    with open(manifest_path, "rb") as f:
        data = orjson.loads(f.read())

    entries = [
        ManifestEntry(
            source=Source(e["source"]),
            url=e["url"],
            local_path=Path(e["local_path"]),
            sha256=e["sha256"],
            fetched_at=datetime.fromisoformat(e["fetched_at"]),
            status=e.get("status", "ok"),
            error=e.get("error"),
        )
        for e in data.get("entries", [])
    ]
    return Manifest(entries=entries, version=data.get("version", 1))


def main():
    """Main pipeline entry point."""
    console.print("[bold]hybrid-rag ingestion pipeline[/bold]")

    manifest = load_manifest()
    console.print(f"Loaded manifest with {len(manifest.entries)} entries")

    pipeline = IngestionPipeline()
    chunks_by_source = pipeline.process_manifest(manifest)

    pipeline.save_chunks(chunks_by_source)

    total = sum(len(c) for c in chunks_by_source.values())
    console.print(f"\n[green]Saved {total} total chunks to data/processed/[/green]")


if __name__ == "__main__":
    main()