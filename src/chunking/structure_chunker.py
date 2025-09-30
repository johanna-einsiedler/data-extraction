# chunkers/structure_chunker.py
import re
from typing import List

from .base_chunker import BaseChunker
from .text_chunker import TextStructureChunker


class StructureChunker(BaseChunker):
    def chunk(self, text: str) -> List[str]:
        # Regex covers:
        # - Markdown headers (#, ##, ### etc.)
        # - HTML headers (<h1>...</h1>)
        # - XML/HTML structural tags (<section>...</section>, <article>...</article>, etc.)
        section_pattern = re.compile(
            r"(?:^|\n)"
            r"(#+ .+|"  # Markdown headings
            r"\<h\d\>.+?\</h\d\>|"  # HTML headings
            r"\<(section|article|chapter|div|part)[^>]*\>.*?\</\1\>)",  # XML/HTML structural tags
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Split the text based on structure markers
        sections = re.split(section_pattern, text)
        sections = [s.strip() for s in sections if s and s.strip()]
        chunks = []

        for sec in sections:
            if len(sec) <= self.chunk_size:
                chunks.append(sec)
            else:
                # Fallback: split further into smaller chunks
                sub_chunker = TextStructureChunker(self.chunk_size, self.chunk_overlap)
                chunks.extend(sub_chunker.chunk(sec))

        # Apply overlap across sections
        if self.chunk_overlap > 0 and len(chunks) > 1:
            adjusted = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    overlap = chunks[i - 1][-self.chunk_overlap :]
                    chunk = overlap + " " + chunk
                adjusted.append(chunk)
            return adjusted

        return chunks
