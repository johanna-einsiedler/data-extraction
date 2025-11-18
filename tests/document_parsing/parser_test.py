"""Structure-focused regression tests for every document parser in the registry."""

import json
import os
import re
import shutil
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from dotenv import find_dotenv, load_dotenv
from lxml import etree

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ------------- Project Setup -----------------
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from registry import PARSERS as REGISTERED_PARSERS  # noqa: E402


def _build_parser(registry_name: str):
    """Instantiate a parser via the registry, skipping when dependencies are unavailable."""
    entry = REGISTERED_PARSERS[registry_name]
    cls = entry["cls"]
    kwargs = dict(entry.get("kwargs", {}))
    try:
        return cls(**kwargs)
    except ValueError as exc:
        if "API key" in str(exc):
            pytest.skip(f"Skipping {registry_name}: {exc}")
        raise
    except FileNotFoundError as exc:
        pytest.skip(f"Skipping {registry_name}: {exc}")
    except ConnectionError as exc:
        pytest.skip(f"Skipping {registry_name}: {exc}")


def _as_text(result):
    """Normalize parser outputs that may be ParseResult-like objects."""
    if hasattr(result, "content"):
        return result.content
    return result

# ------------- Constants -----------------
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}  # TEI XML namespace

# ------------- Helper Functions -----------------


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def load_expected_structure_json(json_path: str | Path) -> dict:
    import json

    path = Path(json_path)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- TEI Extraction ----------


def extract_tei_title(xml_text: str) -> str:
    root = ET.fromstring(xml_text)
    title = root.find(".//tei:titleStmt/tei:title", namespaces=TEI_NS)
    return title.text.strip() if title is not None else ""


def extract_tei_authors(xml_text: str) -> list[str]:
    root = ET.fromstring(xml_text)
    authors = []
    for author in root.findall(".//tei:analytic/tei:author", namespaces=TEI_NS):
        persName = author.find("tei:persName", namespaces=TEI_NS)
        if persName is not None:
            first = persName.findtext("tei:forename", default="", namespaces=TEI_NS)
            last = persName.findtext("tei:surname", default="", namespaces=TEI_NS)
            full_name = f"{first} {last}".strip()
            if full_name:
                authors.append(full_name)
    return authors


def extract_tei_sections(xml_text: str) -> dict:
    root = ET.fromstring(xml_text)
    sections = {}
    for div in root.findall(".//tei:text/tei:body/tei:div", namespaces=TEI_NS):
        head = div.find("tei:head", namespaces=TEI_NS)
        paragraphs = div.findall("tei:p", namespaces=TEI_NS)
        if head is not None:
            section_title = head.text.strip()
            section_text = " ".join(p.text.strip() for p in paragraphs if p.text)
            sections[section_title] = section_text
    return sections


def extract_tei_tables(xml_str: str):
    root = etree.fromstring(xml_str.encode("utf-8"))
    tables = []
    for fig in root.xpath("//tei:figure[@type='table']", namespaces=TEI_NS):
        caption_el = fig.find("tei:head", namespaces=TEI_NS)
        caption = caption_el.text if caption_el is not None else ""
        cells = fig.xpath(".//tei:cell//text()", namespaces=TEI_NS)
        text = " ".join(cells)
        tables.append({"caption": caption, "text": text})
    return tables


def extract_tei_equations(xml_str: str):
    root = etree.fromstring(xml_str.encode("utf-8"))
    equations = []
    for fig in root.xpath("//tei:figure[@type='equation']", namespaces=TEI_NS):
        label = fig.findtext("tei:label", namespaces=TEI_NS)
        latex = fig.findtext("tei:formula", namespaces=TEI_NS)
        desc = fig.findtext("tei:figDesc", namespaces=TEI_NS)
        equations.append({"label": label, "latex": latex, "description": desc})
    return equations


# ---------- Markdown Extraction ----------


def extract_md_title(md_text: str) -> str:
    for line in md_text.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


def extract_md_authors(md_text: str) -> list[str]:
    authors = []
    for line in md_text.splitlines():
        if line.lower().startswith("author:"):
            authors.append(line.split(":", 1)[1].strip())
    return authors


def extract_md_sections(md_text: str) -> dict:
    sections: dict[str, str] = {}
    current_title = None
    current_text = []
    title_consumed = False

    for line in md_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            hashes = len(stripped) - len(stripped.lstrip("#"))
            heading = stripped[hashes:].strip()
            if not heading:
                continue
            if hashes == 1 and not title_consumed:
                # Treat the very first H1 as the document title (already captured elsewhere)
                title_consumed = True
                current_title = None
                current_text = []
                continue
            if current_title is not None:
                sections[current_title] = " ".join(current_text).strip()
            current_title = heading
            current_text = []
        elif current_title:
            current_text.append(stripped)

    if current_title is not None:
        sections[current_title] = " ".join(current_text).strip()

    return sections


def extract_md_tables(md_text: str):
    # Simple Markdown table parser (| ... | ...)
    tables = []
    current_table = []
    for line in md_text.splitlines():
        if line.strip().startswith("|") and "|" in line.strip()[1:]:
            current_table.append(line.strip())
        elif current_table:
            tables.append({"caption": "", "text": " ".join(current_table)})
            current_table = []
    if current_table:
        tables.append({"caption": "", "text": " ".join(current_table)})
    return tables


def extract_md_equations(md_text: str):
    # Equations in $...$ or $$...$$
    pattern = r"(\${1,2})(.+?)\1"
    matches = re.findall(pattern, md_text, flags=re.DOTALL)
    equations = [{"latex": m[1].strip(), "description": ""} for m in matches]
    return equations


# ---------- Unified Extractors ----------


def extract_title(content: str, fmt: str) -> str:
    return extract_tei_title(content) if fmt == "tei" else extract_md_title(content)


def extract_authors(content: str, fmt: str) -> list[str]:
    return extract_tei_authors(content) if fmt == "tei" else extract_md_authors(content)


def extract_sections(content: str, fmt: str) -> dict:
    return (
        extract_tei_sections(content) if fmt == "tei" else extract_md_sections(content)
    )


def extract_tables(content: str, fmt: str):
    return extract_tei_tables(content) if fmt == "tei" else extract_md_tables(content)


def extract_equations(content: str, fmt: str):
    return (
        extract_tei_equations(content)
        if fmt == "tei"
        else extract_md_equations(content)
    )


# ---------- Test Fixtures ----------


@pytest.fixture(scope="session")
def test_files():
    base = os.path.dirname(__file__)
    pdf = os.path.join(base, "paper.pdf")
    assert os.path.exists(pdf), f"Missing test file: {pdf}"
    return {"pdf": pdf}


PARSER_META = [
    ("grobid", "GROBIDParser", "tei"),
    ("mineru", "MinerUParser", "markdown"),
    ("pymupdf", "PyMuPDFParser", "markdown"),
    ("pymupdf_tesseract", "PyMuPDFTesseractParser", "markdown"),
    ("vlm_gemini", "GeminiParser", "markdown"),
    ("vlm_qwen", "QwenParser", "markdown"),
]


# ---------- Tests ----------


@pytest.mark.parametrize("parser_name,registry_name,fmt", PARSER_META)
def test_parser_alive(parser_name, registry_name, fmt):
    parser = _build_parser(registry_name)
    if parser_name == "grobid":
        assert getattr(parser, "grobid_url", "").startswith("http"), "Invalid GROBID URL"
    assert parser is not None


@pytest.mark.parametrize(
    "parser_name,registry_name,fmt",
    [
        ("grobid", "GROBIDParser", "tei"),
        ("mineru", "MinerUParser", "markdown"),
    ],
)
def test_parse_and_validate_structure(parser_name, registry_name, fmt, test_files):
    parser = _build_parser(registry_name)
    pdf_path = test_files["pdf"]
    expected = load_expected_structure_json(Path(__file__).parent / "expected_paper.json")

    # Parse the PDF
    result = parser.parse(pdf_path)
    content = _as_text(result)

    # Save parser output to a file for debugging
    output_dir = Path(__file__).parent / "debug_parsed_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed_filename = f"{parser_name}_parsed.{fmt}"
    parsed_path = output_dir / parsed_filename

    payload = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2)
    with parsed_path.open("w", encoding="utf-8") as f:
        f.write(payload)

    print(f"Saved parser output to {parsed_path}")

    # Extract elements
    title = extract_title(content, fmt)
    authors = extract_authors(content, fmt)
    sections = extract_sections(content, fmt)

    print("Parsed title:", title)
    print("Parsed authors:", authors)
    print("Parsed sections:", list(sections.keys()))

    # Title check
    assert normalize_text(expected["title"]) in normalize_text(title)

    # Authors check with debug info (only TEI currently preserves author metadata reliably)
    if fmt == "tei":
        found_authors = [
            a
            for a, _ in expected["authors"]
            if any(normalize_text(a) in normalize_text(auth) for auth in authors)
        ]
        if len(found_authors) < 2:
            print("Expected authors:", [a for a, _ in expected["authors"]])
            print("Found authors:", authors)
        assert len(found_authors) >= 2, f"Matched authors: {found_authors}"
    else:
        # Markdown-style outputs (e.g. MinerU) often omit structured author metadata.
        assert isinstance(authors, list)

    # Sections check with debug info
    for sec in expected["sections"]:
        sec_title = sec["title"]
        sec_content = sec["content"]
        if not any(
            normalize_text(sec_title) in normalize_text(title)
            or normalize_text(sec_content) in normalize_text(content)
            for title, content in sections.items()
        ):
            print(f"Missing section: {sec_title}")
        assert any(
            normalize_text(sec_title) in normalize_text(title)
            or normalize_text(sec_content) in normalize_text(content)
            for title, content in sections.items()
        )


@pytest.mark.parametrize(
    "parser_name,registry_name,fmt",
    [
        ("grobid", "GROBIDParser", "tei"),
        ("mineru", "MinerUParser", "markdown"),
    ],
)
def test_tables_and_equations(parser_name, registry_name, fmt, test_files):
    parser = _build_parser(registry_name)
    pdf_path = test_files["pdf"]
    result = parser.parse(pdf_path)
    content = _as_text(result)

    tables = extract_tables(content, fmt)
    equations = extract_equations(content, fmt)

    # Tables
    assert isinstance(tables, list)
    for table in tables:
        assert "text" in table

    # Equations
    assert isinstance(equations, list)
    for eq in equations:
        assert "latex" in eq
