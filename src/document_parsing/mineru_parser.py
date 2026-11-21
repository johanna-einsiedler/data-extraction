import importlib.metadata as _im
import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from .base_parser import BaseParser, ParseResult

logger = logging.getLogger(__name__)

# The backend is pinned to the one that produced every cached parse the search
# used: the pipeline backend (PDF-Extract-Kit models). Its model set was fetched
# on 2026-03-11 (MinerU 2.2.1, 20 tuning papers) and on 2026-07-08 (MinerU 3.4.3,
# 15 tuning papers); the VLM model of the `hybrid-engine`/`vlm-engine` backends was
# never present on the machine. MinerU 3.x defaults to `hybrid-engine` when no
# backend is given, so the flag has to be explicit for new parses to match the
# old ones. Override with MINERU_BACKEND only for a deliberate, recorded change.
DEFAULT_BACKEND = "pipeline"


class MinerUParser(BaseParser):
    """Wrapper around the MinerU CLI tool to extract deeply structured content."""

    def __init__(self, output_dir: str = "../data/_intermediate/", backend: str | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if shutil.which("mineru") is None:
            raise FileNotFoundError(
                "The 'mineru' executable was not found on PATH."
            )
        self.backend = backend or os.getenv("MINERU_BACKEND", DEFAULT_BACKEND)
        try:
            self.version = _im.version("mineru")
        except Exception:  # noqa: BLE001 - version is provenance only
            self.version = None

    def command(self, file_path: str, run_dir: Path) -> list[str]:
        """The exact CLI invocation (also written to the parse meta)."""
        return ["mineru", "-p", str(file_path), "-o", str(run_dir), "-b", self.backend]

    def parse(self, file_path: str) -> ParseResult:
        """Run MinerU on the target PDF and return Markdown (or fenced fallback)."""
        run_dir = self.output_dir / f"mineru_{uuid.uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = self.command(file_path, run_dir)

        try:
            logger.info("Running MinerU %s (backend %s) on %s", self.version, self.backend, file_path)
            result = subprocess.run(
                cmd,
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
                    "mineru_version": self.version,
                    "mineru_backend": self.backend,
                    # run_dir is a throw-away temp folder; record the flags only
                    "mineru_command": " ".join(["mineru", "-p", "<pdf>", "-o", "<tmp>", "-b", self.backend]),
                },
            )

        finally:
            # Clean up intermediate run folder
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)
