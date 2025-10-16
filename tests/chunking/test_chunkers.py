import os
import re
import sys
import xml.etree.ElementTree as ET

import pytest
from dotenv import find_dotenv, load_dotenv

# ---------- Setup ----------
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
PARSERS_DIR = os.path.join(PROJECT_ROOT, "src/chunking/")
sys.path.insert(0, PARSERS_DIR)

from length_chunker import LengthChunker
from structure_chunker import StructureChunker
from text_chunker import TextStructureChunker

# ---------- Debug helper ----------
DEBUG_DIR = os.path.join(CURRENT_DIR, "_debug", "chunks")
os.makedirs(DEBUG_DIR, exist_ok=True)


def save_chunks(chunks, name: str):
    """Save chunk output to a debug file for inspection."""
    file_path = os.path.join(DEBUG_DIR, f"{name}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- CHUNK {i + 1} ---\n{chunk.strip()}\n\n")
    print(f"[DEBUG] Saved chunks to: {file_path}")


# ---------- Helper for overlap check ----------
def check_overlap(chunks, overlap):
    for i in range(1, len(chunks)):
        assert chunks[i].startswith(chunks[i - 1][-overlap:]), (
            "Overlap not applied correctly"
        )


# ---------- LengthChunker tests ----------
def test_length_chunker_no_overlap():
    text = "a" * 2500
    chunker = LengthChunker(chunk_size=1000, chunk_overlap=0)
    chunks = chunker.chunk(text)
    save_chunks(chunks, "length_no_overlap")

    assert len(chunks) == 3
    for chunk in chunks:
        assert len(chunk) <= 1000


def test_length_chunker_with_overlap():
    text = "a" * 2500
    chunker = LengthChunker(chunk_size=1000, chunk_overlap=100)
    chunks = chunker.chunk(text)
    save_chunks(chunks, "length_with_overlap")

    assert len(chunks) == 3
    check_overlap(chunks, 100)


# ---------- TextStructureChunker tests ----------
def test_text_structure_chunker_plain_text():
    text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
    chunker = TextStructureChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk(text)
    save_chunks(chunks, "text_plain")

    for chunk in chunks:
        assert chunk.strip() != ""
        assert len(chunk) <= 50 + 10
    check_overlap(chunks, 10)


def test_text_structure_chunker_xml_text():
    text = """
    <article>
        <section>Section 1 text.</section>
        <section>Section 2 text longer than chunk size, so it should be split accordingly.</section>
    </article>
    """
    chunker = TextStructureChunker(chunk_size=30, chunk_overlap=5)
    chunks = chunker.chunk(text)
    save_chunks(chunks, "text_xml")

    for chunk in chunks:
        print(chunk)
        print(len(chunk))
        assert chunk.strip() != ""
        assert len(chunk) <= 30 + 5
    check_overlap(chunks, 5)


# ---------- StructureChunker tests ----------
def test_structure_chunker_markdown_headers():
    text = "# Header 1\nContent 1\n## Subheader 1.1\nContent 2\n# Header 2\nContent 3"
    chunker = StructureChunker(chunk_size=20, chunk_overlap=5)
    chunks = chunker.chunk(text)
    print(chunks)
    save_chunks(chunks, "structure_markdown")

    for chunk in chunks:
        assert chunk.strip() != ""
        assert len(chunk) <= 20 + 5
    check_overlap(chunks, 5)


def test_structure_chunker_fallback_long_text():
    text = "x" * 100
    chunker = StructureChunker(chunk_size=30, chunk_overlap=5)
    chunks = chunker.chunk(text)
    save_chunks(chunks, "structure_fallback")

    for chunk in chunks:
        assert chunk.strip() != ""
        assert len(chunk) <= 30 + 5
    check_overlap(chunks, 5)
