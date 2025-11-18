import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from .base_parser import BaseParser, ParseResult

logger = logging.getLogger(__name__)


class MinerUParser(BaseParser):
    """Wrapper around the MinerU CLI tool to extract deeply structured content."""

    def __init__(self, output_dir: str = "../data/_intermediate/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if shutil.which("mineru") is None:
            raise FileNotFoundError(
                "The 'mineru' executable was not found on PATH."
            )

    def parse(self, file_path: str) -> ParseResult:
        """Run MinerU on the target PDF and return Markdown (or fenced fallback)."""
        run_dir = self.output_dir / f"mineru_{uuid.uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info("Running MinerU on %s", file_path)
            result = subprocess.run(
                ["mineru", "-p", file_path, "-o", str(run_dir)],
                capture_output=True,
                text=True,
            )
            if result.stdout:
                logger.debug("mineru stdout for %s:\n%s", file_path, result.stdout)
            if result.returncode != 0:
                logger.error("mineru stderr for %s:\n%s", file_path, result.stderr)
                raise RuntimeError(f"MinerU failed: {result.stderr}")

            # Find subdir (usually <file_stem>/auto/)
            subdirs = os.listdir(run_dir)
            if not subdirs:
                raise RuntimeError("MinerU produced no output.")

            doc_dir = run_dir / subdirs[0] / "auto"
            if not doc_dir.is_dir():
                raise RuntimeError(f"Expected auto/ directory, found: {doc_dir}")

            # Prefer Markdown, fallback to JSON
            preferred = [".md", "_content_list.json", "_middle.json", "_model.json"]
            content = None
            source_ext = None
            for ext in preferred:
                for f in os.listdir(doc_dir):
                    if f.endswith(ext):
                        source_ext = ext.lstrip(".")
                        with open(doc_dir / f, "r", encoding="utf-8") as fh:
                            content = fh.read()
                        break
                if content is not None:
                    break

            if content is None:
                raise RuntimeError(f"No suitable output found in {doc_dir}")

            if source_ext != "md":
                fenced_lang = "json" if "json" in source_ext else source_ext or ""
                logger.debug(
                    "Wrapping MinerU %s output for %s in fenced Markdown", source_ext, file_path
                )
                content = f"```{fenced_lang}\n{content}\n```"

            return ParseResult(
                content=content,
                metadata={
                    "parser": "mineru",
                    "source_format": source_ext or "unknown",
                },
            )

        finally:
            # Clean up intermediate run folder
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)
