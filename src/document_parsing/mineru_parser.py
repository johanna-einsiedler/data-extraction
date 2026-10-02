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

            # Locate the produced output. MinerU's on-disk layout has changed
            # across versions (e.g. <stem>/auto/ in <=2.x, <stem>/<backend>/ in
            # 3.x), so search recursively for the best available file instead of
            # assuming a fixed subdirectory.
            preferred = [".md", "_content_list.json", "_middle.json", "_model.json"]
            content = None
            source_ext = None
            for ext in preferred:
                matches = sorted(run_dir.rglob(f"*{ext}"))
                if matches:
                    source_ext = ext.lstrip(".")
                    with open(matches[0], "r", encoding="utf-8") as fh:
                        content = fh.read()
                    break

            if content is None:
                produced = [str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file()]
                raise RuntimeError(
                    f"MinerU produced no suitable output in {run_dir}; files: {produced[:12]}"
                )

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
