#!/usr/bin/env python3
"""
Fetch documentation from llms.txt sources.

Downloads all linked pages as markdown, saves to data/raw/<source>/,
and writes data/manifest.json with URL, fetch date, SHA256.
Idempotent: never re-fetches once snapshot is frozen.
"""

import asyncio
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
import orjson
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from hybrid_rag.config import settings
from hybrid_rag.models import Manifest, ManifestEntry, Source

console = Console()

# =============================================================================
# Source Configurations
# =============================================================================

SOURCES = {
    Source.MCP: {
        "llms_txt": "https://modelcontextprotocol.io/llms.txt",
        "base_url": "https://modelcontextprotocol.io",
        "version_filter": "2026-07-28",  # Use latest stable version
    },
    Source.A2A: {
        "llms_txt": "https://a2a-protocol.org/llms.txt",
        "base_url": "https://a2a-protocol.org",
        "version_filter": None,
    },
    Source.LANGGRAPH: {
        "llms_txt": "https://docs.langchain.com/_llms/agent-development-lifecycle/build/python.txt",
        "base_url": "https://docs.langchain.com",
        "version_filter": None,
        "path_filter": "/langgraph/",  # Only keep LangGraph pages
    },
}

# Rate limiter: 2 requests/second per source
RATE_LIMIT = settings.fetch_rate_limit
SEMAPHORES: dict[Source, asyncio.Semaphore] = {
    source: asyncio.Semaphore(2) for source in Source
}
LAST_REQUEST: dict[Source, float] = {source: 0.0 for source in Source}


# =============================================================================
# HTTP Client with Retry & Rate Limiting
# =============================================================================

class RateLimitedClient:
    """HTTP client with per-source rate limiting and retries."""

    def __init__(self, timeout: float = 30.0, max_retries: int = 3):
        self.timeout = httpx.Timeout(timeout)
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            headers={"User-Agent": "hybrid-rag/0.1 (+https://github.com/your-org/hybrid-rag)"},
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def _rate_limit(self, source: Source):
        """Enforce rate limit per source."""
        semaphore = SEMAPHORES[source]
        async with semaphore:
            now = asyncio.get_event_loop().time()
            elapsed = now - LAST_REQUEST[source]
            min_interval = 1.0 / RATE_LIMIT
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            LAST_REQUEST[source] = asyncio.get_event_loop().time()

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        reraise=True,
    )
    async def get(self, source: Source, url: str) -> httpx.Response:
        await self._rate_limit(source)
        assert self._client is not None
        response = await self._client.get(url)
        response.raise_for_status()
        return response


# =============================================================================
# llms.txt Parsers
# =============================================================================

MARKDOWN_LINK_PATTERN = re.compile(r"\[.*?\]\((https?://[^)]+)\)")

def extract_markdown_urls(content: str) -> list[str]:
    """Extract URLs from markdown links [text](url)."""
    return MARKDOWN_LINK_PATTERN.findall(content)


def parse_mcp_llms_txt(content: str, version_filter: str) -> list[str]:
    """Parse MCP's llms.txt (flat list of markdown links for latest version)."""
    # MCP llms.txt is a flat list without version headers - all links are for latest
    return extract_markdown_urls(content)


def parse_a2a_llms_txt(content: str) -> list[str]:
    """Parse A2A's flat llms.txt (markdown links)."""
    return extract_markdown_urls(content)


def parse_langgraph_llms_txt(content: str, path_filter: str) -> list[str]:
    """Parse LangGraph's nested index, filtering to /langgraph/ paths."""
    urls = extract_markdown_urls(content)
    # Handle relative URLs
    filtered = []
    for url in urls:
        if url.startswith("/"):
            url = f"https://docs.langchain.com{url}"
        if path_filter in url:
            filtered.append(url)
    return filtered


PARSERS = {
    Source.MCP: parse_mcp_llms_txt,
    Source.A2A: parse_a2a_llms_txt,
    Source.LANGGRAPH: parse_langgraph_llms_txt,
}


# =============================================================================
# Fetch Logic
# =============================================================================

async def fetch_llms_txt(client: RateLimitedClient, source: Source) -> str:
    """Fetch and return llms.txt content for a source."""
    config = SOURCES[source]
    response = await client.get(source, config["llms_txt"])
    return response.text


def parse_urls(source: Source, content: str) -> list[str]:
    """Parse URLs from llms.txt content based on source type."""
    config = SOURCES[source]
    parser = PARSERS[source]

    if source == Source.MCP:
        return parser(content, config["version_filter"])
    elif source == Source.LANGGRAPH:
        return parser(content, config["path_filter"])
    else:
        return parser(content)


def url_to_local_path(source: Source, url: str, base_url: str) -> Path:
    """Convert a URL to a local file path under data/raw/<source>/."""
    parsed = urlparse(url)
    path = parsed.path.lstrip("/")
    if not path or path.endswith("/"):
        path += "index.md"
    if not path.endswith(".md"):
        path += ".md"
    return settings.raw_data_dir / source.value / path


async def fetch_page(
    client: RateLimitedClient,
    source: Source,
    url: str,
    progress: Progress,
    task_id: int,
) -> ManifestEntry:
    """Fetch a single page and save as markdown."""
    config = SOURCES[source]
    local_path = url_to_local_path(source, url, config["base_url"])

    # Check if already fetched (by manifest)
    manifest_path = settings.data_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "rb") as f:
            manifest_data = orjson.loads(f.read())
        for entry_data in manifest_data.get("entries", []):
            if entry_data["url"] == url:
                # Verify file exists and SHA matches
                existing_path = Path(entry_data["local_path"])
                if existing_path.exists():
                    with open(existing_path, "rb") as pf:
                        sha256 = hashlib.sha256(pf.read()).hexdigest()
                    if sha256 == entry_data["sha256"]:
                        return ManifestEntry(
                            source=Source(entry_data["source"]),
                            url=entry_data["url"],
                            local_path=existing_path,
                            sha256=sha256,
                            fetched_at=datetime.fromisoformat(entry_data["fetched_at"]),
                            status="skipped",
                        )

    # Fetch the page
    try:
        response = await client.get(source, url)
        content = response.text

        # Save to local path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(content, encoding="utf-8")

        # Compute SHA256
        sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        fetched_at = datetime.now(timezone.utc)

        progress.advance(task_id)
        return ManifestEntry(
            source=source,
            url=url,
            local_path=local_path,
            sha256=sha256,
            fetched_at=fetched_at,
            status="ok",
        )
    except Exception as e:
        progress.advance(task_id)
        return ManifestEntry(
            source=source,
            url=url,
            local_path=local_path,
            sha256="",
            fetched_at=datetime.now(timezone.utc),
            status="error",
            error=str(e),
        )


async def fetch_source(
    client: RateLimitedClient,
    source: Source,
    progress: Progress,
) -> list[ManifestEntry]:
    """Fetch all pages for a single source."""
    console.print(f"\n[bold cyan]Fetching {source.value.upper()}...[/bold cyan]")

    # Fetch llms.txt
    llms_content = await fetch_llms_txt(client, source)
    urls = parse_urls(source, llms_content)
    console.print(f"  Found {len(urls)} URLs in llms.txt")

    # Create progress task
    task_id = progress.add_task(f"[green]{source.value}", total=len(urls))

    # Fetch all pages concurrently (rate limited by semaphore)
    tasks = [
        fetch_page(client, source, url, progress, task_id)
        for url in urls
    ]
    entries = await asyncio.gather(*tasks)

    # Summary
    ok = sum(1 for e in entries if e.status == "ok")
    skipped = sum(1 for e in entries if e.status == "skipped")
    errors = sum(1 for e in entries if e.status == "error")
    console.print(f"  [green]OK[/green] {ok} fetched, [yellow]SKIP[/yellow] {skipped} skipped, [red]ERR[/red] {errors} errors")

    return entries


def load_existing_manifest() -> Optional[Manifest]:
    """Load existing manifest if present."""
    manifest_path = settings.data_dir / "manifest.json"
    if not manifest_path.exists():
        return None
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


def save_manifest(manifest: Manifest):
    """Save manifest to JSON."""
    manifest_path = settings.data_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "version": manifest.version,
        "created_at": manifest.created_at.isoformat(),
        "entries": [
            {
                "source": e.source.value,
                "url": e.url,
                "local_path": str(e.local_path),
                "sha256": e.sha256,
                "fetched_at": e.fetched_at.isoformat(),
                "status": e.status,
                "error": e.error,
            }
            for e in manifest.entries
        ],
    }
    manifest_path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    console.print(f"\n[green]Manifest saved to {manifest_path}[/green]")


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    """Main fetch routine."""
    console.print("[bold]hybrid-rag fetch_docs[/bold] - Downloading agentic-AI documentation\n")

    # Load existing manifest to skip already-fetched files
    existing_manifest = load_existing_manifest()
    if existing_manifest:
        console.print(f"Loaded existing manifest with {len(existing_manifest.entries)} entries")

    all_entries = list(existing_manifest.entries) if existing_manifest else []

    async with RateLimitedClient(
        timeout=settings.fetch_timeout,
        max_retries=settings.fetch_max_retries,
    ) as client:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            DownloadColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            for source in Source:
                entries = await fetch_source(client, source, progress)
                all_entries.extend(entries)

    # Create new manifest (deduplicate by URL)
    seen_urls = set()
    unique_entries = []
    for entry in all_entries:
        if entry.url not in seen_urls:
            seen_urls.add(entry.url)
            unique_entries.append(entry)

    manifest = Manifest(entries=unique_entries)
    save_manifest(manifest)

    # Summary
    total = len(unique_entries)
    by_source = {}
    for e in unique_entries:
        by_source.setdefault(e.source.value, {"ok": 0, "skipped": 0, "error": 0})
        by_source[e.source.value][e.status] = by_source[e.source.value].get(e.status, 0) + 1

    console.print("\n[bold]Summary:[/bold]")
    for source, counts in by_source.items():
        console.print(f"  {source}: {counts}")

    console.print(f"\n[green]Total unique documents: {total}[/green]")

    # Exit with error code if any errors
    errors = sum(1 for e in unique_entries if e.status == "error")
    if errors > 0:
        console.print(f"[red]{errors} errors occurred[/red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())