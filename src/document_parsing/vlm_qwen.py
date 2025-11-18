import base64
import io
from typing import Optional

from together import Together

from .base_parser import BaseParser, ParseResult
from .utils import pdf_to_images


def _encode_image_to_data_url(image) -> str:
    """Serialize a PIL image to a data URL for Together's image_url payload."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


class QwenParser(BaseParser):
    """Call Together's Qwen2.5-VL 72B model on page images to produce Markdown."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct",
        max_output_tokens: Optional[int] = 2048,
    ):
        if not api_key:
            raise ValueError("Together API key is required for QwenParser.")
        self.client = Together(api_key=api_key)
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens

    def parse_page(self, page_image) -> str:
        """Upload a page image to Qwen2.5-VL via chat.completions and return Markdown."""
        data_url = _encode_image_to_data_url(page_image)
        prompt = """Convert the following scientific article page into polished Markdown:
- Use # / ## headings that mirror the document structure.
- Preserve tables using Markdown table syntax.
- Keep bullet or numbered lists intact.
- Retain mathematical expressions, figure references, and inline citations.
- Respond with Markdown only."""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_output_tokens=self.max_output_tokens,
        )
        message = response.choices[0].message
        return message.content if isinstance(message.content, str) else str(message.content)

    def parse(self, file_path: str) -> ParseResult:
        """Parse a PDF into Markdown by processing each page image with Qwen VL."""
        images = pdf_to_images(file_path)
        if not images:
            raise ValueError(f"No pages found when converting {file_path} to images.")

        page_markdown = []
        for img in images:
            page_text = self.parse_page(img)
            if page_text:
                page_markdown.append(page_text.strip())

        content = "\n\n".join(page_markdown)
        return ParseResult(
            content=content,
            metadata={"parser": "qwen", "model": self.model_name},
        )
