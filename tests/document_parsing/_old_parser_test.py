import json
import os
import re
import shutil
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

import pytest
from lxml import etree

# ------------- Project Setup -----------------
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
PARSERS_DIR = os.path.join(PROJECT_ROOT, "src/document_parsing/")
sys.path.insert(0, PARSERS_DIR)

from grobid_parser import GROBIDParser  # noqa: E402
from mineru_parser import MinerUParser  # noqa: E402
from pymupdf_parser import PyMuPDFParser  # noqa: E402
from pymupdf_tesseract import PyMuPDFTesseractParser  # noqa: E402
from vlm_gemini import GeminiParser  # noqa: E402
from vlm_qwen import QwenParser  # noqa: E402

# ------------- Constants -----------------
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}  # TEI XML namespace

# ------------- Helper Functions -----------------


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def load_expected_structure_json(json_path: str) -> dict:
    import json

    with open(json_path, "r", encoding="utf-8") as f:
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
    sections = {}
    current_title = None
    current_text = []
    for line in md_text.splitlines():
        if line.startswith("## "):
            if current_title:
                sections[current_title] = " ".join(current_text)
            current_title = line[3:].strip()
            current_text = []
        elif current_title:
            current_text.append(line.strip())
    if current_title:
        sections[current_title] = " ".join(current_text)
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


@pytest.fixture(scope="session")
def parsers():
    return [
        ("grobid", GROBIDParser(grobid_url="https://kermitt2-grobid.hf.space"), "tei"),
        ("mineru", MinerUParser(output_dir="../data/_intermediate/"), "markdown"),
    ]


# ---------- Tests ----------


@pytest.mark.parametrize(
    "parser_name,parser,fmt",
    [
        ("grobid", GROBIDParser(grobid_url="https://kermitt2-grobid.hf.space"), "tei"),
        ("mineru", MinerUParser(output_dir="../data/_intermediate/"), "markdown"),
        ("pymupdf", PyMuPDFParser(), "markdown"),
        ("pymupdf_tesseract", PyMuPDFTesseractParser(), "markdown"),
        ("vlm_gemini", GeminiParser(api_key=GOOGLE_API_KEY), "markdown"),
        ("vlm_qwen", QwenParser(api_key=TOGETHER_API_KEY), "markdown"),
    ],
)
def test_parser_alive(parser_name, parser, fmt):
    if parser_name == "grobid":
        assert parser.grobid_url.startswith("http"), "Invalid GROBID URL"
    else:
        assert parser is not None


@pytest.mark.parametrize(
    "parser_name,parser,fmt",
    [
        ("grobid", GROBIDParser(grobid_url="https://kermitt2-grobid.hf.space"), "tei"),
        ("mineru", MinerUParser(output_dir="../data/_intermediate/"), "markdown"),
    ],
)
def test_parse_and_validate_structure(parser_name, parser, fmt, test_files):
    pdf_path = test_files["pdf"]
    expected = load_expected_structure_json("expected_paper.json")

    # Parse the PDF
    output = parser.parse(pdf_path)

    # Save parser output to a file for debugging
    output_dir = "debug_parsed_outputs"
    os.makedirs(output_dir, exist_ok=True)
    parsed_filename = f"{parser_name}_parsed.{fmt}"
    parsed_path = os.path.join(output_dir, parsed_filename)

    if fmt == "markdown":
        with open(parsed_path, "w", encoding="utf-8") as f:
            f.write(output)
    elif fmt == "tei":
        with open(parsed_path, "w", encoding="utf-8") as f:
            f.write(output)  # TEI is XML text
    else:
        # Save as JSON if unknown format
        with open(parsed_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved parser output to {parsed_path}")

    # Extract elements
    title = extract_title(output, fmt)
    authors = extract_authors(output, fmt)
    sections = extract_sections(output, fmt)

    print("Parsed title:", title)
    print("Parsed authors:", authors)
    print("Parsed sections:", list(sections.keys()))

    # Title check
    assert normalize_text(expected["title"]) in normalize_text(title)

    # Authors check with debug info
    found_authors = [
        a
        for a, _ in expected["authors"]
        if any(normalize_text(a) in normalize_text(auth) for auth in authors)
    ]
    if len(found_authors) < 2:
        print("Expected authors:", [a for a, _ in expected["authors"]])
        print("Found authors:", authors)
    assert len(found_authors) >= 2, f"Matched authors: {found_authors}"

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
    "parser_name,parser,fmt",
    [
        ("grobid", GROBIDParser(grobid_url="https://kermitt2-grobid.hf.space"), "tei"),
        ("mineru", MinerUParser(output_dir="../data/_intermediate/"), "markdown"),
    ],
)
def test_tables_and_equations(parser_name, parser, fmt, test_files):
    pdf_path = test_files["pdf"]
    output = parser.parse(pdf_path)

    tables = extract_tables(output, fmt)
    equations = extract_equations(output, fmt)

    # Tables
    assert isinstance(tables, list)
    for table in tables:
        assert "text" in table

    # Equations
    assert isinstance(equations, list)
    for eq in equations:
        assert "latex" in eq
