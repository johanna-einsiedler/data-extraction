"""Top-level CLI entry for the paper extraction pipeline."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from document_parsing.base_parser import ParseResult  # noqa: E402
from query_creation import create_prompts, get_outcome, prompts_to_html  # noqa: E402
from query_creation.collect_item_info import collect_item_info  # noqa: E402
from registry import LLMS, LLM_META_MAP, OPENAI_API_KEY, TOGETHER_API_KEY  # noqa: E402

QUERY_CREATION_DIR = PROJECT_ROOT / "query_creation"

app = typer.Typer(help="Manage pipeline tasks such as parsing, chunking, and prompt generation.")


def _resolve_path(path: Path, base_dir: Path) -> Path:
    """Return an absolute path; if relative, resolve it against the given base."""
    return path if path.is_absolute() else base_dir / path


def _load_queries() -> dict:
    queries_path = PROJECT_ROOT / "query_creation" / "queries_with_prompts.json"
    if not queries_path.exists():
        raise FileNotFoundError(f"queries_with_prompts.json not found at {queries_path}")
    with open(queries_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_system_prompt(single: bool = True) -> str:
    prompt_file = (
        PROJECT_ROOT / "query_creation" / ("system_prompt_single_outcome.txt" if single else "system_prompt_multiple_outcomes.txt")
    )
    if not prompt_file.exists():
        raise FileNotFoundError(f"System prompt file not found at {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")


def _load_full_document_text(parsed_path: Path) -> str:
    """Return a string representation of a parsed document."""
    if parsed_path.suffix == ".json":
        with open(parsed_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, str):
            return data
        if isinstance(data, list):
            return "\n\n".join(str(item) for item in data)
        return json.dumps(data, ensure_ascii=False, indent=2)
    return parsed_path.read_text(encoding="utf-8")


def _import_pytest():
    try:
        import pytest  # type: ignore
    except ImportError as exc:  # pragma: no cover - guard clause
        typer.echo(
            "pytest is required but not installed. "
            "Install test dependencies (e.g. `pip install -r requirements.txt`)."
        )
        raise typer.Exit(code=1) from exc
    return pytest


class _DetailedSummaryPlugin:
    """Collect detailed pass/fail information from a pytest session."""

    def __init__(self) -> None:
        self.passed: List[str] = []
        self.failed: List[str] = []
        self.skipped: List[str] = []
        self.deselected: List[str] = []
        self._failed_seen: set[str] = set()
        self._skipped_seen: set[str] = set()

    def pytest_runtest_logreport(self, report):  # pragma: no cover - pytest hook
        nodeid = getattr(report, "nodeid", None)
        if not nodeid:
            return

        outcome = getattr(report, "outcome", None)
        when = getattr(report, "when", "")

        if outcome == "failed":
            if nodeid not in self._failed_seen:
                self.failed.append(nodeid)
                self._failed_seen.add(nodeid)
        elif outcome == "skipped":
            if nodeid not in self._skipped_seen:
                self.skipped.append(nodeid)
                self._skipped_seen.add(nodeid)
        elif outcome == "passed" and when == "call":
            self.passed.append(nodeid)

    def pytest_deselected(self, items):  # pragma: no cover - pytest hook
        for item in items:
            nodeid = getattr(item, "nodeid", None)
            if nodeid:
                self.deselected.append(nodeid)

    def stats_line(self) -> str:
        return (
            f"passed: {len(self.passed)} | "
            f"failed: {len(self.failed)} | "
            f"skipped: {len(self.skipped)} | "
            f"deselected: {len(self.deselected)}"
        )

    def to_payload(self, exit_code: int, invoked_args: List[str]) -> dict:
        timestamp = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        return {
            "timestamp": timestamp,
            "exit_code": exit_code,
            "pytest_args": invoked_args,
            "stats": {
                "passed": len(self.passed),
                "failed": len(self.failed),
                "skipped": len(self.skipped),
                "deselected": len(self.deselected),
            },
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "deselected": self.deselected,
        }


@app.command()
def create_queries(
    codebook: Path = typer.Option(
        Path("coding_scheme.xlsx"),
        help="Path to the coding scheme Excel workbook.",
    ),
    only_base: bool = typer.Option(
        False,
        help="If set, generate only the base prompts (skip rewriting and few-shot steps).",
    ),
) -> None:
    """Generate prompts and supporting artefacts from the coding scheme."""

    base_dir = QUERY_CREATION_DIR
    codebook_path = _resolve_path(codebook, base_dir)

    if not codebook_path.exists():
        typer.echo(f"Codebook not found at {codebook_path}")
        raise typer.Exit(code=1)

    typer.echo("📄 Running collect_item_info to build query metadata…")
    item_info = collect_item_info(str(codebook_path))
    query_info_path = base_dir / "query_info.json"
    with open(query_info_path, "w", encoding="utf-8") as fh:
        json.dump(item_info, fh, ensure_ascii=False, indent=4)
    typer.echo(f"   • Wrote structured query info to {query_info_path}")

    typer.echo("🧠 Running create_prompts to generate prompt variants…")
    prompt_templates = create_prompts.load_templates()
    rewriting_prompt = create_prompts.load_prompt("rewriting_prompt.txt")
    if rewriting_prompt is None and not only_base:
        typer.echo("Rewriting prompt template not found; cannot proceed.")
        raise typer.Exit(code=1)

    create_prompts.BINARIZATION_PROMPT = create_prompts.load_prompt(
        "binarization_prompt.txt"
    )
    create_prompts.FOLLOW_UP_PROMPT = create_prompts.load_prompt(
        "follow_up_prompt_other.txt"
    )
    create_prompts.BINARY_BASE_PROMPT = create_prompts.load_prompt(
        "binary_base_prompt.txt"
    )
    if not only_base and any(
        prompt is None
        for prompt in (
            create_prompts.BINARIZATION_PROMPT,
            create_prompts.FOLLOW_UP_PROMPT,
            create_prompts.BINARY_BASE_PROMPT,
        )
    ):
        typer.echo("One or more prompt templates are missing; aborting.")
        raise typer.Exit(code=1)

    prompt_payload = create_prompts.generate_prompts_with_rewrite(
        item_info,
        prompt_templates,
        rewriting_prompt,
        only_base=only_base,
    )

    prompts_json_path = base_dir / "queries_with_prompts.json"
    with open(prompts_json_path, "w", encoding="utf-8") as fh:
        json.dump(prompt_payload, fh, ensure_ascii=False, indent=4)
    typer.echo(f"   • Wrote prompt catalog to {prompts_json_path}")

    typer.echo("📊 Running get_outcome to export outcomes.json…")
    get_outcome.main()

    typer.echo("🌐 Running prompts_to_html to render HTML preview…")
    prompts_to_html.main()

    typer.echo("✅ Full pipeline completed successfully.")


def _resolve_parser(name: str):
    from registry import PARSERS  # local import to avoid heavy dependencies at startup

    if name not in PARSERS:
        raise KeyError(name)
    entry = PARSERS[name]
    kwargs = dict(entry.get("kwargs", {}))
    if "api_key" in kwargs and not kwargs["api_key"]:
        raise ValueError("Missing API key")
    parser_cls = entry["cls"]
    return parser_cls(**kwargs)


def _write_parse_output(output_dir: Path, doc_path: Path, result) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {}
    format_hint = "markdown"
    if isinstance(result, ParseResult):
        content = result.content
        metadata = result.metadata or {}
        format_hint = result.format
    else:
        content = str(result)

    ext = {"tei_xml": ".xml", "markdown": ".md"}.get(format_hint, ".md")
    out_path = output_dir / f"{doc_path.stem}{ext}"
    out_path.write_text(content, encoding="utf-8")
    if metadata:
        meta_path = output_dir / f"{doc_path.stem}.meta.json"
        meta_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return out_path


@app.command("test")
def run_tests(
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Run pytest in quiet mode (equivalent to passing -q).",
    ),
    keyword: Optional[str] = typer.Option(
        None,
        "--keyword",
        "-k",
        help="Only run tests whose node names match the given expression.",
    ),
    marker: Optional[str] = typer.Option(
        None,
        "--marker",
        "-m",
        help="Only run tests marked with the given marker expression.",
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        help="Write the JSON test report to this path (defaults to trial_logs/test_runs/<timestamp>.json).",
    ),
) -> None:
    """Run the full test suite and produce a concise summary."""

    pytest = _import_pytest()
    plugin = _DetailedSummaryPlugin()

    tests_dir = PROJECT_ROOT / "tests"
    if not tests_dir.exists():
        typer.echo(f"Tests directory not found at {tests_dir}")
        raise typer.Exit(code=1)

    pytest_args: List[str] = []
    if quiet:
        pytest_args.append("-q")
    if keyword:
        pytest_args.extend(["-k", keyword])
    if marker:
        pytest_args.extend(["-m", marker])
    pytest_args.append(str(tests_dir))

    exit_code = pytest.main(pytest_args, plugins=[plugin])

    typer.echo("\n=== Test Summary ===")
    typer.echo(plugin.stats_line())

    if plugin.failed:
        typer.echo("\n❌ Failing tests:")
        for nodeid in plugin.failed:
            typer.echo(f"  - {nodeid}")
    else:
        typer.echo("\n✅ No failing tests.")

    if plugin.skipped:
        typer.echo("\n⚠️ Skipped tests:")
        for nodeid in plugin.skipped:
            typer.echo(f"  - {nodeid}")

    if plugin.passed:
        sample_count = min(len(plugin.passed), 10)
        sample = plugin.passed[:sample_count]
        typer.echo(f"\n✅ Passing tests ({len(plugin.passed)} total):")
        for nodeid in sample:
            typer.echo(f"  - {nodeid}")
        if len(plugin.passed) > sample_count:
            typer.echo(f"  ... and {len(plugin.passed) - sample_count} more.")

    report_payload = plugin.to_payload(exit_code, pytest_args)

    if log_file is None:
        log_dir = PROJECT_ROOT / "trial_logs" / "test_runs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = report_payload["timestamp"].replace(":", "").replace("-", "")
        default_name = f"pytest_{timestamp}.json"
        log_path = log_dir / default_name
    else:
        log_path = log_file if log_file.is_absolute() else PROJECT_ROOT / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)

    log_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        display_path = log_path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = log_path
    typer.echo(f"\n📝 Wrote test report to {display_path}")

    if exit_code != 0:
        raise typer.Exit(code=int(exit_code))


@app.command("parse-docs")
def parse_docs(
    folder: Optional[Path] = typer.Option(
        None,
        "--folder",
        "-f",
        help="Folder containing documents to parse (defaults to data/raw).",
    ),
    parser_name: Optional[str] = typer.Option(
        None,
        "--parser",
        "-p",
        help="Specific parser to run (defaults to all available parsers).",
    ),
) -> None:
    """Run one or more parsers on every document in a folder."""

    target_folder = folder if folder is not None else PROJECT_ROOT / "data" / "raw"
    if not target_folder.exists():
        typer.echo(f"Folder {target_folder} does not exist.")
        raise typer.Exit(code=1)

    documents = sorted(
        [p for p in target_folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
    )
    if not documents:
        typer.echo(f"No PDF documents found in {target_folder}.")
        raise typer.Exit(code=0)

    from registry import PARSERS  # defer import

    parser_names = [parser_name] if parser_name else list(PARSERS.keys())

    for name in parser_names:
        typer.echo(f"\n🚀 Running parser {name} on {len(documents)} document(s)…")
        try:
            parser = _resolve_parser(name)
        except KeyError:
            typer.echo(f"  ⚠️ Unknown parser '{name}'. Skipping.")
            continue
        except ValueError as err:
            typer.echo(f"  ⚠️ Skipping {name}: {err}")
            continue
        except Exception as exc:
            typer.echo(f"  ⚠️ Failed to initialise {name}: {exc}")
            continue

        output_dir = PROJECT_ROOT / "data" / "parsed" / name

        for doc in documents:
            cached_targets = [
                output_dir / f"{doc.stem}.md",
                output_dir / f"{doc.stem}.xml",
            ]
            if any(path.exists() for path in cached_targets):
                typer.echo(
                    f"  • [{name}] {doc.name} → skipped (cached)",
                )
                continue

            typer.echo(f"  • [{name}] {doc.name} … ", nl=False)
            try:
                result = parser.parse(str(doc))
                out_path = _write_parse_output(output_dir, doc, result)
                typer.echo(f"saved to {out_path.name}")
            except Exception as exc:
                typer.echo(f"failed ({exc})")

    typer.echo("\n✅ Parsing complete.")


def _iter_parsed_documents(root: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in root.iterdir()
            if p.is_file() and not p.name.startswith('.')
        ]
    )


def _default_parsed_root(folder: Optional[Path]) -> Path:
    return folder if folder is not None else PROJECT_ROOT / "data" / "parsed"


def _chunk_document(
    parser_name: str,
    chunker_name: str,
    parsed_path: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> Path:
    chunk_dir = PROJECT_ROOT / "data" / "chunks" / parser_name / chunker_name
    chunk_dir.mkdir(parents=True, exist_ok=True)
    out_path = chunk_dir / f"{parsed_path.stem}_s{chunk_size}_o{chunk_overlap}.json"

    if parsed_path.suffix.lower() == ".json":
        with open(parsed_path, "r", encoding="utf-8") as fh:
            parsed_content = json.load(fh)
    else:
        parsed_content = parsed_path.read_text(encoding="utf-8")

    if not isinstance(parsed_content, str):
        parsed_content = json.dumps(parsed_content, ensure_ascii=False)

    from registry import CHUNKERS  # local import

    chunker_inst = CHUNKERS[chunker_name](
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunk_objects = chunker_inst.chunk(parsed_content)
    payload = [
        {"text": chunk.text, "metadata": chunk.metadata}
        for chunk in chunk_objects
    ]

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    return out_path


@app.command("chunking")
def chunking(
    folder: Optional[Path] = typer.Option(
        None,
        "--folder",
        "-f",
        help="Root folder containing parsed documents (defaults to data/parsed).",
    ),
    parser: Optional[str] = typer.Option(
        None,
        "--parser",
        "-p",
        help="Only process parsed outputs from this parser (defaults to all).",
    ),
    chunker: Optional[List[str]] = typer.Option(
        None,
        "--chunker",
        "-c",
        help="Specific chunker(s) to run (may be used multiple times).",
    ),
    chunk_sizes: Optional[List[int]] = typer.Option(
        None,
        "--chunk-size",
        help="Chunk size(s) to use (defaults to 100,250,500,750,1000,2500,5000).",
    ),
    chunk_overlaps: Optional[List[int]] = typer.Option(
        None,
        "--chunk-overlap",
        help="Chunk overlap(s) to use (defaults to 0,250,750).",
    ),
) -> None:
    """Chunk previously parsed documents using one or more chunkers."""

    default_sizes = [50, 100, 250, 500, 750, 1000, 2500, 5000]
    default_overlaps = [0, 250, 750]
    sizes = chunk_sizes if chunk_sizes else default_sizes
    overlaps = chunk_overlaps if chunk_overlaps else default_overlaps

    parsed_root = _default_parsed_root(folder)
    if not parsed_root.exists():
        typer.echo(f"Parsed folder {parsed_root} does not exist.")
        raise typer.Exit(code=1)

    parser_dirs = [d for d in parsed_root.iterdir() if d.is_dir()]
    if parser:
        parser_dirs = [d for d in parser_dirs if d.name == parser]

    if not parser_dirs:
        typer.echo("No parsed documents found.")
        raise typer.Exit(code=0)

    from registry import CHUNKERS  # local import

    valid_chunkers = [name for name in CHUNKERS.keys() if name != "BaseChunker"]
    if chunker:
        unknown = [name for name in chunker if name not in valid_chunkers]
        if unknown:
            for name in unknown:
                typer.echo(f"  ⚠️ Unsupported chunker '{name}' (skipping).")
        chunker_names = [name for name in chunker if name in valid_chunkers]
    else:
        chunker_names = valid_chunkers

    if not chunker_names:
        typer.echo("No valid chunkers selected.")
        raise typer.Exit(code=0)

    for parser_dir in sorted(parser_dirs, key=lambda p: p.name):
        parser_name = parser_dir.name
        documents = _iter_parsed_documents(parser_dir)
        if not documents:
            typer.echo(f"\n⚠️ No parsed documents for parser {parser_name}, skipping.")
            continue

        typer.echo(
            f"\n🚀 Chunking parser {parser_name} ({len(documents)} document(s)) with {len(chunker_names)} chunker(s)…"
        )

        for chunker_name in chunker_names:
            typer.echo(f"  ▸ Chunker {chunker_name}:")
            chunk_folder = (
                PROJECT_ROOT
                / "data"
                / "chunks"
                / parser_name
                / chunker_name
            )

            for doc in documents:
                for size in sizes:
                    for overlap in overlaps:
                        if overlap >= size:
                            continue
                        chunk_file = chunk_folder / f"{doc.stem}_s{size}_o{overlap}.json"
                        if chunk_file.exists():
                            typer.echo(
                                f"    • {doc.name} (s={size}, o={overlap}) → skipped (cached)",
                            )
                            continue

                        typer.echo(
                            f"    • {doc.name} (s={size}, o={overlap}) … ",
                            nl=False,
                        )
                        try:
                            result_path = _chunk_document(
                                parser_name,
                                chunker_name,
                                doc,
                                size,
                                overlap,
                            )
                            typer.echo(f"saved to {result_path.name}")
                        except Exception as exc:
                            typer.echo(f"failed ({exc})")

    typer.echo("\n✅ Chunking complete.")


@app.command("full-infer")
def full_inference(
    parser_name: str = typer.Option(..., "--parser", "-p", help="Parser name (must match parsed folder under data/parsed)."),
    llm_model: Optional[str] = typer.Option(
        None,
        "--llm-model",
        "-m",
        help="LLM model identifier (defaults to first configured model).",
    ),
    prompt_type: str = typer.Option(
        "base_prompt",
        "--prompt",
        "-q",
        help="Prompt variant key to use from queries_with_prompts.json.",
    ),
    document: Optional[str] = typer.Option(
        None,
        "--doc",
        "-d",
        help="Specific document stem (without extension) to process. Defaults to all.",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help="Maximum number of documents to process.",
    ),
    output_dir: Path = typer.Option(
        Path("trial_logs/full_runs"),
        "--output-dir",
        help="Where to write JSON outputs with model responses.",
    ),
) -> None:
    """Run full-document inference (no retrieval) for parsed papers."""

    parsed_root = PROJECT_ROOT / "data" / "parsed" / parser_name
    if not parsed_root.exists():
        typer.echo(f"Parsed folder not found at {parsed_root}")
        raise typer.Exit(code=1)

    queries = _load_queries()
    system_prompt = _load_system_prompt(single=True)

    available_llms = list(LLMS.keys())
    if not available_llms:
        typer.echo("No LLMs configured in registry.")
        raise typer.Exit(code=1)
    chosen_llm = llm_model or available_llms[0]
    if chosen_llm not in LLMS:
        typer.echo(f"LLM '{chosen_llm}' is not configured. Available: {', '.join(available_llms)}")
        raise typer.Exit(code=1)
    generator = LLMS[chosen_llm]

    doc_paths = sorted(
        p for p in parsed_root.iterdir() if p.is_file() and not p.name.startswith(".")
    )
    if document:
        doc_paths = [p for p in doc_paths if p.stem == document]
        if not doc_paths:
            typer.echo(f"No parsed document matching '{document}' under {parsed_root}")
            raise typer.Exit(code=1)
    if limit:
        doc_paths = doc_paths[:limit]
    if not doc_paths:
        typer.echo("No parsed documents to process.")
        raise typer.Exit(code=0)

    output_dir = _resolve_path(output_dir, PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(
        f"Running full-document inference using parser={parser_name}, llm_model={chosen_llm}, prompt={prompt_type}, docs={len(doc_paths)}"
    )

    successes = 0
    failures = 0

    for doc_path in doc_paths:
        try:
            full_text = load_full_document_text(doc_path)
        except Exception as exc:  # pragma: no cover - file IO guard
            typer.echo(f"⚠️ Failed to load {doc_path.name}: {exc}")
            failures += 1
            continue

        responses = {}
        for qid, qdata in queries.items():
            prompt = qdata.get("prompts", {}).get(prompt_type)
            if not prompt:
                typer.echo(f"  ⚠️ Query {qid} lacks prompt '{prompt_type}', skipping.")
                continue

            llm_output = generator.generate(
                query=prompt,
                chunks=[(full_text, 1.0)],
                system_prompt=system_prompt,
                return_logprobs=True,
            )
            responses[qid] = {
                "question": qdata.get("label"),
                "type": qdata.get("type"),
                "prompt_type": prompt_type,
                "llm_text": llm_output.get("text"),
                "logprobs": llm_output.get("logprobs"),
            }

        if not responses:
            typer.echo(f"⚠️ No prompts produced output for {doc_path.name}")
            failures += 1
            continue

        out_payload = {
            "document": doc_path.name,
            "parser": parser_name,
            "llm_model": chosen_llm,
            "prompt_type": prompt_type,
            "responses": responses,
        }
        out_path = output_dir / f"{doc_path.stem}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(out_payload, fh, ensure_ascii=False, indent=2)
        typer.echo(f"✅ Saved {out_path.name}")
        successes += 1

    typer.echo(f"\nFull-document inference complete. {successes} success, {failures} failed.")


@app.command("embed")
def embed_chunks(
    chunks_root: Path = typer.Option(
        Path("data/chunks"),
        "--chunks-root",
        help="Root folder containing chunked documents (defaults to data/chunks).",
    ),
    embeddings_root: Path = typer.Option(
        Path("data/embeddings"),
        "--embeddings-root",
        help="Destination folder for generated embeddings (defaults to data/embeddings).",
    ),
    parser: Optional[List[str]] = typer.Option(
        None,
        "--parser",
        "-p",
        help="Only process chunk outputs from these parser(s).",
    ),
    chunker: Optional[List[str]] = typer.Option(
        None,
        "--chunker",
        "-c",
        help="Only process these chunker(s).",
    ),
    embedder: Optional[List[str]] = typer.Option(
        None,
        "--embedder",
        "-e",
        help="Only run the listed embedder(s). Defaults to all available.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Recompute embeddings even if the output file already exists.",
    ),
) -> None:
    """Embed every chunk file and store vectors per parser/chunker/embedder combination."""

    chunks_root = _resolve_path(chunks_root, PROJECT_ROOT)
    embeddings_root = _resolve_path(embeddings_root, PROJECT_ROOT)

    if not chunks_root.exists():
        typer.echo(f"Chunks folder {chunks_root} not found.")
        raise typer.Exit(code=1)

    from registry import EMBEDDERS as REGISTERED_EMBEDDERS  # local import
    from vectorstore.numpy_store import NumpyVectorStore

    available_embedders = list(REGISTERED_EMBEDDERS.keys())
    selected_embedders = embedder if embedder else available_embedders
    unknown = [name for name in selected_embedders if name not in available_embedders]
    if unknown:
        for name in unknown:
            typer.echo(f"⚠️ Unknown embedder '{name}' (skipping).")
        selected_embedders = [name for name in selected_embedders if name in available_embedders]
    if not selected_embedders:
        typer.echo("No valid embedders selected.")
        raise typer.Exit(code=1)

    parser_filters = set(parser) if parser else None
    chunker_filters = set(chunker) if chunker else None

    jobs = 0
    saved = 0
    skipped = 0

    parser_dirs = [p for p in sorted(chunks_root.iterdir()) if p.is_dir()]
    if not parser_dirs:
        typer.echo(f"No parser directories found under {chunks_root}.")
        raise typer.Exit(code=0)

    for parser_dir in parser_dirs:
        parser_name = parser_dir.name
        if parser_filters and parser_name not in parser_filters:
            continue

        chunker_dirs = [c for c in sorted(parser_dir.iterdir()) if c.is_dir()]
        if not chunker_dirs:
            typer.echo(f"⚠️ No chunker outputs for parser {parser_name}, skipping.")
            continue

        typer.echo(f"\n📁 Parser {parser_name} ({len(chunker_dirs)} chunker(s)):")

        for chunker_dir in chunker_dirs:
            chunker_name = chunker_dir.name
            if chunker_filters and chunker_name not in chunker_filters:
                continue

            chunk_files = sorted(chunker_dir.glob("*.json"))
            if not chunk_files:
                typer.echo(f"  ⚠️ No chunk files for chunker {chunker_name}.")
                continue

            typer.echo(
                f"  ▸ Chunker {chunker_name}: {len(chunk_files)} file(s) × {len(selected_embedders)} embedder(s)"
            )

            for chunk_file in chunk_files:
                with open(chunk_file, "r", encoding="utf-8") as fh:
                    try:
                        data = json.load(fh)
                    except json.JSONDecodeError:
                        fh.seek(0)
                        data = fh.read().split("\n\n")

                if isinstance(data, list) and data and isinstance(data[0], dict):
                    chunk_texts = [item.get("text", "") for item in data]
                else:
                    chunk_texts = data if isinstance(data, list) else [str(data)]

                for embedder_name in selected_embedders:
                    jobs += 1
                    embedder_inst = REGISTERED_EMBEDDERS[embedder_name]
                    target_dir = embeddings_root / parser_name / chunker_name / embedder_name
                    target_dir.mkdir(parents=True, exist_ok=True)
                    store_path = target_dir / chunk_file.stem
                    npz_path = Path(f"{store_path}.npz")

                    if npz_path.exists() and not force:
                        skipped += 1
                        continue

                    typer.echo(
                        f"    • {chunk_file.name} → {embedder_name}",
                        nl=False,
                    )
                    try:
                        embeddings = embedder_inst.embed(chunk_texts)
                    except Exception as exc:  # pragma: no cover - runtime guard
                        typer.echo(f" failed ({exc})")
                        continue

                    store = NumpyVectorStore(save_path=store_path)
                    store.add(embeddings, chunk_texts)
                    store.save()
                    typer.echo(" saved")
                    saved += 1

    typer.echo(
        f"\n✅ Embedding complete. Processed {jobs} job(s), saved {saved}, skipped {skipped} (cached)."
    )


def main() -> None:
    """Run the Typer application."""
    app()


if __name__ == "__main__":
    main()
