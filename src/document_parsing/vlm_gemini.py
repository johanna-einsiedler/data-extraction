from pathlib import Path

import google.generativeai as genai

from .base_parser import BaseParser


class GeminiParser(BaseParser):
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def parse(self, file_path: str) -> str:
        # Upload the PDF file
        uploaded_file = genai.upload_file(file_path)

        # Ask Gemini to convert to Markdown
        response = self.model.generate_content(
            [
                """Convert the following scientific article text into well - structured Markdown .
                - Use # and ## headings to match the section and subsection titles .
                - Format inline citations as ( Author , Year ) if available .
                Use bullet points for lists .
                - Preserve all mathematical notation and code blocks in Markdown syntax .
                - Render tables using Markdown syntax (| ... |)
                - Output clean Markdown only no explanation .
                Input :""",
                uploaded_file,
            ]
        )

        return response.text
