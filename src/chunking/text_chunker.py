# chunkers/text_chunker.py
import re
import xml.etree.ElementTree as ET
from typing import List

from .base_chunker import BaseChunker


class TextStructureChunker(BaseChunker):
    def chunk(self, text: str) -> List[str]:
        chunks = []

        # Try XML parsing first
        try:
            root = ET.fromstring(text.strip())
            paragraphs = []

            # Collect text from paragraph-like elements
            for elem in root.iter():
                if elem.tag.lower() in {"p", "section", "article", "div"}:
                    if elem.text and elem.text.strip():
                        paragraphs.append(elem.text.strip())

            # If no <p>/<section> found, fall back to all non-empty text nodes
            if not paragraphs:
                for elem in root.iter():
                    if elem.text and elem.text.strip():
                        paragraphs.append(elem.text.strip())

        except ET.ParseError:
            # Fallback: assume markdown/plain text → split on blank lines
            paragraphs = re.split(r"\n\s*\n", text)

        # Build chunks from paragraphs
        buf = ""
        for para in paragraphs:
            if len(buf) + len(para) + 2 <= self.chunk_size:
                buf += para + "\n\n"
            else:
                if buf.strip():
                    chunks.append(buf.strip())
                buf = para + "\n\n"

        if buf.strip():
            chunks.append(buf.strip())

        # Apply overlap if needed
        if self.chunk_overlap > 0 and len(chunks) > 1:
            adjusted = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    overlap = chunks[i - 1][-self.chunk_overlap :]
                    chunk = overlap + " " + chunk
                adjusted.append(chunk)
            return adjusted

        return chunks
