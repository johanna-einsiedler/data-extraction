import re
import xml.etree.ElementTree as ET
from typing import List

from .base_chunker import BaseChunker, Chunk


class TextStructureChunker(BaseChunker):
    """Chunker that respects paragraph/section boundaries where possible."""

    def chunk(self, text: str) -> List[Chunk]:
        """Break text or lightweight markup into paragraph- and sentence-level chunks."""
        raw = text.strip()

        if raw.startswith("<"):
            try:
                root = ET.fromstring(raw)
                paragraphs = []
                for elem in root.iter():
                    if elem.tag.lower() in {"p", "section", "article", "div"}:
                        text_content = "".join(elem.itertext()).strip()
                        if text_content:
                            paragraphs.append(
                                (
                                    self._sanitize(text_content, remove_tags=True),
                                    elem.tag.lower(),
                                )
                            )
                if not paragraphs:
                    raise ET.ParseError("No structural elements with text")
            except ET.ParseError:
                paragraphs = [
                    (p, "paragraph")
                    for p in re.split(r"\n\s*\n", self._sanitize(raw, remove_tags=True))
                ]
        else:
            paragraphs = [
                (p, "paragraph")
                for p in re.split(r"\n\s*\n", self._sanitize(raw, remove_tags=True))
            ]

        paragraphs = [(p.strip(), tag) for p, tag in paragraphs if p and p.strip()]

        def split_para(para: str, tag: str) -> List[Chunk]:
            """Sentence-aware splitting to avoid chopping within sentences."""
            sentences = re.split(r"(?<=[.!?])\s+", para)
            result: List[Chunk] = []
            current = ""
            for sent in sentences:
                if len(current) + len(sent) + 1 <= self.chunk_size:
                    current += (" " if current else "") + sent
                else:
                    if current:
                        result.append(Chunk(current.strip(), {"source_tag": tag}))
                    current = sent
            if current:
                result.append(Chunk(current.strip(), {"source_tag": tag}))
            return result

        chunks: List[Chunk] = []

        for para, tag in paragraphs:
            if len(para) > self.chunk_size:
                chunks.extend(split_para(para, tag))
            elif (
                chunks
                and len(chunks[-1].text) + len(para) + 2 <= self.chunk_size
                and chunks[-1].metadata.get("source_tag") == tag
            ):
                chunks[-1].text += "\n\n" + para
            else:
                chunks.append(Chunk(para, {"source_tag": tag}))

        final_chunks: List[Chunk] = []
        for chunk in chunks:
            if len(chunk.text) > self.chunk_size + self.chunk_overlap:
                subchunks = split_para(
                    chunk.text, chunk.metadata.get("source_tag", "paragraph")
                )
                for sc in subchunks:
                    if len(sc.text) > self.chunk_size + self.chunk_overlap:
                        forced = self._force_cut(sc.text)
                        for piece in forced:
                            piece.metadata.update(sc.metadata)
                        final_chunks.extend(forced)
                    else:
                        final_chunks.append(sc)
            else:
                final_chunks.append(chunk)

        cleaned: List[Chunk] = []
        for chunk in final_chunks:
            sanitized = self._sanitize(chunk.text, remove_tags=True)
            if sanitized:
                cleaned.append(Chunk(sanitized, chunk.metadata))

        return self._apply_overlap(cleaned)
