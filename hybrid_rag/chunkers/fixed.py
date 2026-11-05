"""
Fixed-size token chunker with overlap.
"""

from hybrid_rag.chunkers.base import Chunker, ChunkerConfig, count_tokens, split_sentences
from hybrid_rag.models import Chunk, ChunkMetadata, ChunkerType


class FixedChunker:
    """Fixed-size chunker with configurable overlap."""

    def __init__(self, target_tokens: int = 512, overlap_tokens: int = 50):
        self.config = ChunkerConfig(
            chunker_type=ChunkerType.FIXED,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            max_tokens=target_tokens + overlap_tokens,
        )

    def count_tokens(self, text: str) -> int:
        return count_tokens(text)

    def chunk(self, text: str, metadata: ChunkMetadata) -> list[Chunk]:
        """Split text into fixed-size chunks with overlap."""
        if not text.strip():
            return []

        tokens = count_tokens(text)
        if tokens <= self.config.target_tokens:
            # Single chunk fits
            return [
                Chunk(
                    text=text.strip(),
                    metadata=ChunkMetadata(
                        source=metadata.source,
                        url=metadata.url,
                        heading_path=metadata.heading_path,
                        char_start=0,
                        char_end=len(text),
                        token_count=tokens,
                        chunker_type=self.config.chunker_type,
                        document_sha256=metadata.document_sha256,
                        fetched_at=metadata.fetched_at,
                    ),
                )
            ]

        # Split into sentences first for cleaner boundaries
        sentences = split_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        char_offset = 0

        for sentence in sentences:
            sentence_tokens = count_tokens(sentence)

            # If adding this sentence exceeds target, finalize current chunk
            if current_tokens + sentence_tokens > self.config.target_tokens and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(self._make_chunk(chunk_text, metadata, char_offset - len(chunk_text), char_offset))
                current_tokens = count_tokens(chunk_text)

                # Apply overlap: keep last N tokens from current chunk
                if self.config.overlap_tokens > 0 and len(current_chunk) > 1:
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = [overlap_text] if overlap_text else []
                    current_tokens = count_tokens(overlap_text) if overlap_text else 0
                else:
                    current_chunk = []
                    current_tokens = 0

            current_chunk.append(sentence)
            char_offset += len(sentence) + 1  # +1 for space

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(self._make_chunk(chunk_text, metadata, char_offset - len(chunk_text), char_offset))

        return chunks

    def _make_chunk(
        self,
        text: str,
        metadata: ChunkMetadata,
        char_start: int,
        char_end: int,
    ) -> Chunk:
        """Create a Chunk with proper metadata."""
        token_count = count_tokens(text)
        return Chunk(
            text=text.strip(),
            metadata=ChunkMetadata(
                source=metadata.source,
                url=metadata.url,
                heading_path=metadata.heading_path,
                char_start=max(0, char_start),
                char_end=char_end,
                token_count=token_count,
                chunker_type=self.config.chunker_type,
                document_sha256=metadata.document_sha256,
                fetched_at=metadata.fetched_at,
            ),
        )

    def _get_overlap_text(self, sentences: list[str]) -> str:
        """Get overlap text from end of sentence list."""
        overlap = ""
        tokens = 0
        for sent in reversed(sentences):
            sent_tokens = count_tokens(sent)
            if tokens + sent_tokens > self.config.overlap_tokens:
                break
            overlap = sent + " " + overlap
            tokens += sent_tokens
        return overlap.strip()