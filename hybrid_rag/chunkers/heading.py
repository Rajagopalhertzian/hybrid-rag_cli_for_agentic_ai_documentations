"""
Heading-aware chunker that respects document structure.
"""

import re
from dataclasses import dataclass

from hybrid_rag.chunkers.base import Chunker, ChunkerConfig, count_tokens
from hybrid_rag.models import Chunk, ChunkMetadata, ChunkerType


@dataclass
class HeadingSection:
    """A document section defined by a heading."""

    level: int  # 1, 2, 3
    title: str
    content: str
    char_start: int
    char_end: int
    heading_path: tuple[str, ...]


class HeadingChunker:
    """Heading-aware chunker that splits on H1/H2/H3 boundaries."""

    def __init__(self, max_tokens: int = 1024):
        self.config = ChunkerConfig(
            chunker_type=ChunkerType.HEADING,
            target_tokens=max_tokens // 2,
            overlap_tokens=0,
            max_tokens=max_tokens,
        )
        self._heading_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

    def count_tokens(self, text: str) -> int:
        return count_tokens(text)

    def chunk(self, text: str, metadata: ChunkMetadata) -> list[Chunk]:
        """Split text into heading-aware chunks."""
        if not text.strip():
            return []

        # Parse sections
        sections = self._parse_sections(text, metadata.heading_path)

        chunks = []
        for section in sections:
            section_chunks = self._chunk_section(section, metadata)
            chunks.extend(section_chunks)

        return chunks

    def _parse_sections(
        self,
        text: str,
        base_heading_path: tuple[str, ...],
    ) -> list[HeadingSection]:
        """Parse markdown into heading sections."""
        sections = []
        matches = list(self._heading_pattern.finditer(text))

        if not matches:
            # No headings found - treat entire text as one section
            return [
                HeadingSection(
                    level=0,
                    title="",
                    content=text,
                    char_start=0,
                    char_end=len(text),
                    heading_path=base_heading_path,
                )
            ]

        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            char_start = match.start()

            # Find end of this section (start of next heading of same or higher level)
            char_end = len(text)
            for next_match in matches[i + 1 :]:
                next_level = len(next_match.group(1))
                if next_level <= level:
                    char_end = next_match.start()
                    break

            content = text[char_start:char_end].strip()

            # Build heading path
            heading_path = self._build_heading_path(base_heading_path, matches[: i + 1], level)

            sections.append(
                HeadingSection(
                    level=level,
                    title=title,
                    content=content,
                    char_start=char_start,
                    char_end=char_end,
                    heading_path=heading_path,
                )
            )

        return sections

    def _build_heading_path(
        self,
        base_path: tuple[str, ...],
        matches: list[re.Match],
        current_level: int,
    ) -> tuple[str, ...]:
        """Build heading path from base path and current heading hierarchy."""
        path = list(base_path)
        heading_stack: list[tuple[int, str]] = []

        for match in matches:
            level = len(match.group(1))
            title = match.group(2).strip()

            # Pop higher or equal level headings
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()

            heading_stack.append((level, title))

        # Add all headings in stack to path
        path.extend(title for _, title in heading_stack)

        return tuple(path)

    def _chunk_section(
        self,
        section: HeadingSection,
        metadata: ChunkMetadata,
    ) -> list[Chunk]:
        """Chunk a single section, splitting if too large."""
        section_tokens = count_tokens(section.content)

        if section_tokens <= self.config.max_tokens:
            # Section fits in one chunk
            return [
                Chunk(
                    text=section.content,
                    metadata=ChunkMetadata(
                        source=metadata.source,
                        url=metadata.url,
                        heading_path=section.heading_path,
                        char_start=section.char_start,
                        char_end=section.char_end,
                        token_count=section_tokens,
                        chunker_type=self.config.chunker_type,
                        document_sha256=metadata.document_sha256,
                        fetched_at=metadata.fetched_at,
                    ),
                )
            ]

        # Section too large - split by sub-headings or paragraphs
        return self._split_large_section(section, metadata)

    def _split_large_section(
        self,
        section: HeadingSection,
        metadata: ChunkMetadata,
    ) -> list[Chunk]:
        """Split a large section into smaller chunks."""
        # Try to split by sub-headings (H4+)
        sub_sections = self._split_by_subheadings(section.content)

        if len(sub_sections) > 1:
            chunks = []
            char_offset = section.char_start
            for sub_content, sub_start, sub_end in sub_sections:
                sub_tokens = count_tokens(sub_content)
                if sub_tokens <= self.config.max_tokens:
                    chunks.append(
                        Chunk(
                            text=sub_content,
                            metadata=ChunkMetadata(
                                source=metadata.source,
                                url=metadata.url,
                                heading_path=section.heading_path,
                                char_start=char_offset + sub_start,
                                char_end=char_offset + sub_end,
                                token_count=sub_tokens,
                                chunker_type=self.config.chunker_type,
                                document_sha256=metadata.document_sha256,
                                fetched_at=metadata.fetched_at,
                            ),
                        )
                    )
                else:
                    # Still too large - fall back to fixed chunking
                    from hybrid_rag.chunkers.fixed import FixedChunker
                    fixed_chunker = FixedChunker(
                        target_tokens=self.config.max_tokens // 2,
                        overlap_tokens=50,
                    )
                    fixed_chunks = fixed_chunker.chunk(
                        sub_content,
                        ChunkMetadata(
                            source=metadata.source,
                            url=metadata.url,
                            heading_path=section.heading_path,
                            char_start=0,
                            char_end=len(sub_content),
                            token_count=sub_tokens,
                            chunker_type=ChunkerType.FIXED,
                            document_sha256=metadata.document_sha256,
                            fetched_at=metadata.fetched_at,
                        ),
                    )
                    # Adjust char offsets - create new chunks with updated metadata
                    for fc in fixed_chunks:
                        new_chunk = Chunk(
                            id=fc.id,
                            text=fc.text,
                            metadata=ChunkMetadata(
                                source=fc.metadata.source,
                                url=fc.metadata.url,
                                heading_path=fc.metadata.heading_path,
                                char_start=char_offset + sub_start + fc.metadata.char_start,
                                char_end=char_offset + sub_start + fc.metadata.char_end,
                                token_count=fc.metadata.token_count,
                                chunker_type=self.config.chunker_type,  # Override to HEADING
                                document_sha256=fc.metadata.document_sha256,
                                fetched_at=fc.metadata.fetched_at,
                            ),
                        )
                        chunks.append(new_chunk)
            return chunks

        # No sub-headings - split by paragraphs
        paragraphs = [p.strip() for p in section.content.split("\n\n") if p.strip()]
        chunks = []
        current_chunk = []
        current_tokens = 0
        char_offset = section.char_start

        for para in paragraphs:
            para_tokens = count_tokens(para)
            if current_tokens + para_tokens > self.config.max_tokens and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata=ChunkMetadata(
                            source=metadata.source,
                            url=metadata.url,
                            heading_path=section.heading_path,
                            char_start=char_offset - len(chunk_text),
                            char_end=char_offset,
                            token_count=count_tokens(chunk_text),
                            chunker_type=self.config.chunker_type,
                            document_sha256=metadata.document_sha256,
                            fetched_at=metadata.fetched_at,
                        ),
                    )
                )
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata=ChunkMetadata(
                        source=metadata.source,
                        url=metadata.url,
                        heading_path=section.heading_path,
                        char_start=char_offset - len(chunk_text),
                        char_end=char_offset,
                        token_count=count_tokens(chunk_text),
                        chunker_type=self.config.chunker_type,
                        document_sha256=metadata.document_sha256,
                        fetched_at=metadata.fetched_at,
                    ),
                )
            )

        return chunks

    def _split_by_subheadings(self, content: str) -> list[tuple[str, int, int]]:
        """Split content by H4+ headings (####, #####, ######)."""
        sub_pattern = re.compile(r"^(#{4,6})\s+(.+)$", re.MULTILINE)
        matches = list(sub_pattern.finditer(content))

        if not matches:
            return [(content, 0, len(content))]

        sections = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            sections.append((content[start:end], start, end))

        # Add content before first sub-heading
        if matches[0].start() > 0:
            sections.insert(0, (content[: matches[0].start()], 0, matches[0].start()))

        return [(s.strip(), start, end) for s, start, end in sections if s.strip()]