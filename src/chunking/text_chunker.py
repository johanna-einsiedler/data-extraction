import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import List

sys.path.append(os.path.dirname(__file__))

from base_chunker import BaseChunker


class TextStructureChunker(BaseChunker):
    def chunk(self, text: str) -> List[str]:
        # Detect structure (XML or plain text)
        if text.strip().startswith("<"):
            try:
                root = ET.fromstring(text.strip())
                paragraphs = [
                    elem.text.strip()
                    for elem in root.iter()
                    if elem.tag.lower() in {"p", "section", "article", "div"}
                    and elem.text
                ]
            except ET.ParseError:
                paragraphs = re.split(r"\n\s*\n", text.strip())
        else:
            paragraphs = re.split(r"\n\s*\n", text.strip())

        def split_para(para: str) -> List[str]:
            """Split paragraph into chunks roughly by sentence boundaries."""
            sentences = re.split(r"(?<=[.!?])\s+", para)
            result, current = [], ""
            for sent in sentences:
                if len(current) + len(sent) + 1 <= self.chunk_size:
                    current += (" " if current else "") + sent
                else:
                    if current:
                        result.append(current.strip())
                    current = sent
            if current:
                result.append(current.strip())
            return result

        def force_cut(text: str) -> List[str]:
            """Hard cut into pieces <= chunk_size, preserving overlap."""
            chunks = []
            step = self.chunk_size - self.chunk_overlap
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunks.append(text[start:end])
                start += step
            return chunks

        chunks: List[str] = []

        # Paragraph-level processing
        for para in paragraphs:
            if len(para) > self.chunk_size:
                chunks.extend(split_para(para))
            elif chunks and len(chunks[-1]) + len(para) + 2 <= self.chunk_size:
                chunks[-1] += "\n\n" + para
            else:
                chunks.append(para.strip())

        # ✅ Post-process: ensure no chunk exceeds chunk_size + overlap
        final_chunks: List[str] = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size + self.chunk_overlap:
                # Try sentence-based secondary split
                subchunks = split_para(chunk)
                # If still too long (maybe no punctuation), fall back to hard cut
                for sc in subchunks:
                    if len(sc) > self.chunk_size + self.chunk_overlap:
                        final_chunks.extend(force_cut(sc))
                    else:
                        final_chunks.append(sc)
            else:
                final_chunks.append(chunk)

        # ✅ Apply global overlap across all final chunks
        if self.chunk_overlap > 0 and len(final_chunks) > 1:
            adjusted = []
            for i, chunk in enumerate(final_chunks):
                if i > 0:
                    overlap_text = final_chunks[i - 1][-self.chunk_overlap :]
                    # Ensure overlap_text is prefixed to current chunk
                    chunk = overlap_text + chunk
                adjusted.append(chunk)
            final_chunks = adjusted

        return final_chunks
