from pathlib import Path
from tempfile import NamedTemporaryFile

import google.generativeai as genai
from PIL import Image

from .base_parser import BaseParser, ParseResult
from .utils import pdf_to_images


class GeminiParser(BaseParser):
    """Run Gemini 2.5 on PDF page images, returning Markdown."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def parse_page(self, page_image: Image.Image) -> str:
        """Parse a single page image via Gemini."""
        with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img_path = tmp.name
            page_image.save(img_path, format="PNG")

        prompt = """Convert the following scientific article page into well-structured Markdown:
- Use # and ## for headings
- Preserve bullet points, tables, math, and code blocks
- Format citations as (Author, Year)
- Output Markdown only, no explanation
Input (image):"""

        try:
            response = self.model.generate_content([prompt, img_path])
            return response.text
        finally:
            Path(img_path).unlink(missing_ok=True)

    def parse(self, file_path: str) -> ParseResult:
        """Convert a PDF to Markdown using Gemini via image OCR."""
        images = pdf_to_images(file_path)
        results = []

        for img in images:
            page_text = self.parse_page(img)
            results.append(page_text)

        content = "\n\n".join(results)
        return ParseResult(
            content=content,
            metadata={"parser": "gemini", "model": self.model_name},
        )
