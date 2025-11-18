import json
import re
from typing import Dict, List, Tuple

from .base_chunker import BaseChunker, Chunk
from .text_chunker import TextStructureChunker


class StructureChunker(BaseChunker):
    """Structure-aware chunker that yields metadata-rich chunks when possible."""

    def chunk(self, text: str) -> List[Chunk]:
        raw = text.strip()
        sections: List[Tuple[str, Dict[str, str]]] = []

        if raw.startswith("{") or raw.startswith("["):
            try:
                obj = json.loads(raw)
                formatted = json.dumps(obj, indent=2, ensure_ascii=False)
                for block in formatted.split("\n\n"):
                    if block.strip():
                        sections.append((block, {"section_type": "json"}))
            except json.JSONDecodeError:
                sections.append((raw, {"section_type": "json"}))
        elif raw.startswith("<"):
            sections.extend(self._split_html_sections(raw))
        else:
            sections.extend(self._split_markdown_sections(raw))

        text_chunker = TextStructureChunker(self.chunk_size, self.chunk_overlap)
        chunks: List[Chunk] = []

        for section_text, base_meta in sections:
            sanitized_section = self._sanitize(section_text, remove_tags=True)
            if not sanitized_section:
                continue

            if len(sanitized_section) <= self.chunk_size:
                chunks.append(Chunk(sanitized_section, dict(base_meta)))
                continue

            subchunks = text_chunker.chunk(section_text)
            for sc in subchunks:
                chunk_meta = {**base_meta, **sc.metadata}
                chunk_text = sc.text

                if len(chunk_text) > self.chunk_size + self.chunk_overlap:
                    sentence_chunks = self._split_sentences(chunk_text, chunk_meta)
                    for sent_chunk in sentence_chunks:
                        if len(sent_chunk.text) > self.chunk_size + self.chunk_overlap:
                            forced = self._force_cut(sent_chunk.text)
                            for piece in forced:
                                piece.metadata.update(chunk_meta)
                            chunks.extend(forced)
                        else:
                            chunks.append(sent_chunk)
                else:
                    chunks.append(Chunk(chunk_text, chunk_meta))

        cleaned = [
            Chunk(self._sanitize(chunk.text, remove_tags=True), chunk.metadata)
            for chunk in chunks
            if chunk.text
        ]
        cleaned = [chunk for chunk in cleaned if chunk.text]
        return self._apply_overlap(cleaned)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _split_markdown_sections(self, text: str) -> List[Tuple[str, Dict[str, str]]]:
        """Return markdown sections paired with basic metadata."""
        pattern = re.compile(r"^(#+ .+)$", flags=re.MULTILINE)
        matches = list(pattern.finditer(text))
        sections: List[Tuple[str, Dict[str, str]]] = []
        cursor = 0

        if not matches:
            return [(text, {"section_type": "markdown"})]

        for idx, match in enumerate(matches):
            start = match.start()
            if start > cursor:
                leading = text[cursor:start]
                if leading.strip():
                    sections.append((leading, {"section_type": "markdown"}))

            next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            section_text = text[start:next_start]
            heading_line = match.group(0)
            heading = self._sanitize(re.sub(r"^#+", "", heading_line), remove_tags=True)
            metadata = {"section_type": "markdown"}
            if heading:
                metadata["section_heading"] = heading
            sections.append((section_text, metadata))
            cursor = next_start

        if cursor < len(text):
            trailing = text[cursor:]
            if trailing.strip():
                sections.append((trailing, {"section_type": "markdown"}))

        return sections

    def _split_html_sections(self, text: str) -> List[Tuple[str, Dict[str, str]]]:
        """Return HTML sections paired with metadata about the source element."""
        pattern = re.compile(
            r"(<h\d[^>]*>.*?</h\d>|<(?:section|article|chapter|div|part)[^>]*>.*?</(?:section|article|chapter|div|part)>)",
            flags=re.DOTALL | re.IGNORECASE,
        )
        matches = list(pattern.finditer(text))
        sections: List[Tuple[str, Dict[str, str]]] = []
        cursor = 0

        if not matches:
            return [(text, {"section_type": "html"})]

        for idx, match in enumerate(matches):
            start, end = match.span()
            if start > cursor:
                between = text[cursor:start]
                if between.strip():
                    sections.append((between, {"section_type": "html"}))

            token = match.group(0)
            metadata: Dict[str, str] = {"section_type": "html"}
            lower_token = token.lower()
            if lower_token.startswith("<h"):
                heading = self._sanitize(re.sub(r"<[^>]+>", " ", token), remove_tags=True)
                if heading:
                    metadata["section_heading"] = heading
                next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
                section_text = text[start:next_start]
                sections.append((section_text, metadata))
                cursor = next_start
            else:
                heading_match = re.search(
                    r"<h\d[^>]*>(.*?)</h\d>", token, flags=re.DOTALL | re.IGNORECASE
                )
                if heading_match:
                    heading = self._sanitize(heading_match.group(1), remove_tags=True)
                    if heading:
                        metadata["section_heading"] = heading
                sections.append((token, metadata))
                cursor = end

        if cursor < len(text):
            tail = text[cursor:]
            if tail.strip():
                sections.append((tail, {"section_type": "html"}))

        return sections

    def _split_sentences(self, text_block: str, metadata: Dict[str, str]) -> List[Chunk]:
        """Split a large section on sentence boundaries, preserving metadata."""
        sanitized = self._sanitize(text_block, remove_tags=True)
        sentences = re.split(r"(?<=[.!?])\s+", sanitized)
        result: List[Chunk] = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) + 1 <= self.chunk_size:
                current += (" " if current else "") + sent
            else:
                if current:
                    result.append(Chunk(current.strip(), dict(metadata)))
                current = sent
        if current:
            result.append(Chunk(current.strip(), dict(metadata)))
        return result
