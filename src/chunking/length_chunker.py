from typing import List

from .base_chunker import BaseChunker, Chunk


class LengthChunker(BaseChunker):
    """Simple fixed-length chunker that ignores document structure."""

    def chunk(self, text: str) -> List[Chunk]:
        """Slice the document into size-limited windows with optional overlap."""
        clean_text = self._sanitize(text, remove_tags=True)
        windows = self._token_windows(clean_text)
        return [Chunk(chunk_text) for chunk_text in windows]
