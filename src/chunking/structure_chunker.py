# chunkers/structure_chunker.py
import json
import os
import re
import sys
from typing import List

sys.path.append(os.path.dirname(__file__))
from text_chunker import TextStructureChunker


class StructureChunker(BaseChunker):
    """
    Document-structure-based chunking.

    If the document is in a structured format (Markdown, HTML, XML, or JSON),
    the structure is exploited to split it by headers or sections.
    If a resulting chunk exceeds `chunk_size`, recursive text-based splitting
    (as in TextStructureChunker) is applied, with a fallback to hard cutting.
    """

    def chunk(self, text: str) -> List[str]:
        text = text.strip()
        chunks: List[str] = []

        # --- 1️⃣ Detect document type ------------------------------------------
        if text.startswith("{") or text.startswith("["):
            # JSON: split roughly by top-level items
            try:
                obj = json.loads(text)
                formatted = json.dumps(obj, indent=2, ensure_ascii=False)
                sections = formatted.split("\n\n")
            except json.JSONDecodeError:
                sections = [text]
        elif text.startswith("<"):
            # HTML/XML
            section_pattern = re.compile(
                r"(?:(?:^|\n)(?:"
                r"(?:<h\d>.*?</h\d>)|"  # HTML headings
                r"(<(?:section|article|chapter|div|part)[^>]*>.*?</(?:section|article|chapter|div|part)>)"
                r"))",
                flags=re.DOTALL | re.IGNORECASE,
            )
            sections = re.split(section_pattern, text)
        else:
            # Markdown or plain text
            section_pattern = re.compile(
                r"(?:(?:^|\n)(#+ .+))",  # Markdown headers
                flags=re.DOTALL,
            )
            sections = re.split(section_pattern, text)

        sections = [s.strip() for s in sections if s and s.strip()]

        # --- 2️⃣ Helper: sentence-aware and hard cut fallback ------------------
        def split_sentences(text_block: str) -> List[str]:
            """Split into chunks by sentence boundaries."""
            sentences = re.split(r"(?<=[.!?])\s+", text_block)
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

        def force_cut(text_block: str) -> List[str]:
            """Hard cut text into fixed-size slices (preserving overlap)."""
            chunks = []
            step = self.chunk_size - self.chunk_overlap
            start = 0
            while start < len(text_block):
                end = start + self.chunk_size
                chunks.append(text_block[start:end])
                start += step
            return chunks

        text_chunker = TextStructureChunker(self.chunk_size, self.chunk_overlap)

        # --- 3️⃣ Recursive partitioning for large sections ---------------------
        for section in sections:
            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                # Recursive text-based split
                subchunks = text_chunker.chunk(section)
                for sc in subchunks:
                    if len(sc) > self.chunk_size + self.chunk_overlap:
                        # Try sentence-based split
                        subsent = split_sentences(sc)
                        for ss in subsent:
                            if len(ss) > self.chunk_size + self.chunk_overlap:
                                chunks.extend(force_cut(ss))
                            else:
                                chunks.append(ss)
                    else:
                        chunks.append(sc)

            # --- 4️⃣ Global overlap (final adjustment) -----------------------------

        # ✅ Apply global overlap across all final chunks
        if self.chunk_overlap > 0 and len(chunks) > 1:
            adjusted = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    overlap_text = chunks[i - 1][-self.chunk_overlap :]
                    # Ensure overlap_text is prefixed to current chunk
                    chunk = overlap_text + chunk
                adjusted.append(chunk)
            final_chunks = adjusted

        return chunks
