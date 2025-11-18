"""Top-level CLI entry for the paper extraction pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer

from document_parsing.base_parser import ParseResult
from query_creation import create_prompts, get_outcome, prompts_to_html
from query_creation.collect_item_info import collect_item_info
from registry import CHUNKERS, PARSERS

CLI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CLI_DIR
QUERY_CREATION_DIR = PROJECT_ROOT / "query_creation"

app = typer.Typer(help="Manage pipeline tasks such as parsing, chunking, and prompt generation.")


def _resolve_path(path: Path, base_dir: Path) -> Path:
    """Return an absolute path; if relative, resolve it against the given base."""
    return path if path.is_absolute() else base_dir / path


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
    return sorted([p for p in root.iterdir() if p.is_file()])


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
    chunk_size: int = typer.Option(1000, help="Chunk size to use."),
    chunk_overlap: int = typer.Option(100, help="Chunk overlap to use."),
) -> None:
    """Chunk previously parsed documents using one or more chunkers."""

    if chunk_overlap >= chunk_size:
        typer.echo("chunk_overlap must be smaller than chunk_size")
        raise typer.Exit(code=1)

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

    chunker_names = list(chunker) if chunker else list(CHUNKERS.keys())

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
            if chunker_name not in CHUNKERS:
                typer.echo(f"  ⚠️ Unknown chunker '{chunker_name}'. Skipping.")
                continue

            typer.echo(f"  ▸ Chunker {chunker_name}:")
            chunk_folder = (
                PROJECT_ROOT
                / "data"
                / "chunks"
                / parser_name
                / chunker_name
            )

            for doc in documents:
                stem = doc.stem
                chunk_file = chunk_folder / f"{stem}_s{chunk_size}_o{chunk_overlap}.json"
                if chunk_file.exists():
                    typer.echo(
                        f"    • {doc.name} → skipped (cached)",
                    )
                    continue

                typer.echo(f"    • {doc.name} … ", nl=False)
                try:
                    result_path = _chunk_document(
                        parser_name,
                        chunker_name,
                        doc,
                        chunk_size,
                        chunk_overlap,
                    )
                    typer.echo(f"saved to {result_path.name}")
                except Exception as exc:
                    typer.echo(f"failed ({exc})")

    typer.echo("\n✅ Chunking complete.")


def main() -> None:
    """Run the Typer application."""
    app()


if __name__ == "__main__":
    main()
