import os
import sys
import xml.etree.ElementTree as ET

import pytest
from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ------------- Project Setup -----------------
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
PARSERS_DIR = os.path.join(PROJECT_ROOT, "src/document_parsing/")
sys.path.insert(0, PARSERS_DIR)
# Import your parsers and keys
from grobid_parser import GROBIDParser
from mineru_parser import MinerUParser
from pymupdf_parser import PyMuPDFParser
from pymupdf_tesseract import PyMuPDFTesseractParser
from vlm_gemini import GeminiParser
from vlm_qwen import QwenParser

# Example dynamic API keys (replace with actual)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "dummy_key")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "dummy_key")

# Parser registry
PARSERS = {
    "GROBIDParser": {
        "cls": GROBIDParser,
        "kwargs": {},
    },
    "PyMuPDFParser": {
        "cls": PyMuPDFParser,
        "kwargs": {},
    },
    "PyMuPDFTesseractParser": {
        "cls": PyMuPDFTesseractParser,
        "kwargs": {},
    },
    "GeminiParser": {
        "cls": GeminiParser,
        "kwargs": {"api_key": GOOGLE_API_KEY},
    },
    "QwenParser": {
        "cls": QwenParser,
        "kwargs": {"api_key": TOGETHER_API_KEY},
    },
    "MinerUParser": {
        "cls": MinerUParser,
        "kwargs": {},
    },
}


@pytest.fixture(scope="session")
def test_pdf_path():
    """Return the path to the test PDF."""
    pdf_path = os.path.join(os.path.dirname(__file__), "paper.pdf")
    assert os.path.exists(pdf_path), f"Missing test PDF: {pdf_path}"
    return pdf_path


@pytest.fixture(params=PARSERS.keys())
def parser_instance(request):
    """Return parser instance + name."""
    parser_name = request.param
    cls = PARSERS[parser_name]["cls"]
    kwargs = PARSERS[parser_name]["kwargs"]
    instance = cls(**kwargs)
    return parser_name, instance


# ---------- Test ----------


def test_parser_output_nonempty(parser_instance, test_pdf_path):
    parser_name, parser = parser_instance

    # Parse PDF
    output = parser.parse(test_pdf_path)

    # Save output for debugging
    output_dir = os.path.join(os.path.dirname(__file__), "debug_parsed_outputs")
    os.makedirs(output_dir, exist_ok=True)
    parsed_file = f"{parser_name}_parsed.txt"
    parsed_path = os.path.join(output_dir, parsed_file)
    with open(parsed_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"Saved {parser_name} output to {parsed_path}")

    # Validation
    if parser_name == "GROBIDParser":
        # Check valid XML
        try:
            ET.fromstring(output)
        except ET.ParseError as e:
            pytest.fail(f"GROBID output is not valid XML: {e}")
    else:
        # All others: just check non-empty
        assert output.strip(), f"{parser_name} output is empty"
