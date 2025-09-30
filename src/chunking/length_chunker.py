# chunkers/length_chunker.py
import re
from typing import List

from .base_chunker import BaseChunker


class LengthChunker(BaseChunker):
    def chunk(self, text: str) -> List[str]:
        # Remove XML/HTML tags → keep only visible text
        clean_text = re.sub(r"<[^>]+>", "", text)

        step = self.chunk_size - self.chunk_overlap
        chunks = [
            clean_text[i : i + self.chunk_size] for i in range(0, len(clean_text), step)
        ]

        # Apply overlap if needed
        if self.chunk_overlap > 0 and len(chunks) > 1:
            adjusted = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    overlap = chunks[i - 1][-self.chunk_overlap :]
                    chunk = overlap + " " + chunk
                adjusted.append(chunk)
            return adjusted

        # ✅ Always return something
        return chunks
