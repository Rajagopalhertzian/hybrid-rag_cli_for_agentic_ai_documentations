#!/usr/bin/env python3
"""
Clean raw markdown files (strip MDX/JSX).
Outputs cleaned markdown to data/processed/cleaned/<source>/.
"""

import json
from pathlib import Path

import orjson
from rich.console import Console
from rich.progress import Progress, TaskProgressColumn, TextColumn

from hybrid_rag.config import settings
from hybrid_rag.ingestion.cleaner import clean_markdown
from hybrid_rag.models import Manifest, ManifestEntry, Source

console = Console()


def load_manifest() -> Manifest:
    manifest_path = settings.data_dir / "manifest.json"
    with open(manifest_path, "rb") as f:
        data = orjson.loads(f.read())

    entries = [
        ManifestEntry(
            source=Source(e["source"]),
            url=e["url"],
            local_path=Path(e["local_path"]),
            sha256=e["sha256"],
            fetched_at=e["fetched_at"],
            status=e.get("status", "ok"),
            error=e.get("error"),
        )
        for e in data.get("entries", [])
    ]
    return Manifest(entries=entries, version=data.get("version", 1))


def main():
    console.print("[bold]hybrid-rag clean_md[/bold] - Stripping MDX/JSX from raw markdown\n")

    manifest = load_manifest()
    console.print(f"Loaded manifest with {len(manifest.entries)} entries")

    out_base = settings.processed_data_dir / "cleaned"
    out_base.mkdir(parents=True, exist_ok=True)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("[green]Cleaning", total=len(manifest.entries))

        for entry in manifest.entries:
            if entry.status != "ok":
                progress.advance(task_id)
                continue

            content = entry.local_path.read_text(encoding="utf-8")
            result = clean_markdown(content)

            # Save cleaned content
            rel_path = entry.local_path.relative_to(settings.raw_data_dir / entry.source.value)
            out_path = out_base / entry.source.value / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(result.content, encoding="utf-8")

            # Save heading path as sidecar
            heading_path_file = out_path.with_suffix(".headings.json")
            heading_path_file.write_text(
                json.dumps(list(result.heading_path), indent=2),
                encoding="utf-8",
            )

            progress.advance(task_id)

    console.print(f"\n[green]Cleaned markdown saved to {out_base}[/green]")


if __name__ == "__main__":
    main()