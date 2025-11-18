# Paper Extraction Pipeline

Tools for parsing PDFs, creating structured prompts, chunking text, embedding content, and running LLM-based coding/data-extraction experiments for social science articles (OSF prereg: https://osf.io/quxdb).

## How the pipeline fits together
- **Prompt preparation** (`query_creation/`): Build query metadata from `coding_scheme.xlsx`, generate prompt variants, and render HTML views of prompts. Run `python -m pipeline_cli create-queries` (or `pipe create-queries` after installation).
- **Document parsing** (`src/document_parsing/`): Multiple parsers (GROBID, PyMuPDF, Tesseract OCR, Gemini/Qwen VLMs, MinerU) convert PDFs in `data/raw/` to normalised text/Markdown in `data/parsed/<parser>/`. Run `python -m pipeline_cli parse-docs --folder data/raw --parser GROBIDParser`.
- **Chunking** (`src/chunking/`): Slice parsed text into overlapping windows using several strategies (length-based, structure-aware, text heuristics). Writes JSON chunks under `data/chunks/<parser>/<chunker>/`. Example: `python -m pipeline_cli chunking --parser GROBIDParser --chunker LengthChunker --chunk-size 500 --chunk-overlap 50`.
- **Embedding + vector store** (`src/embedding/`, `src/vectorstore/`): Generate embeddings with BGE/E5/OpenAI and optionally save/query them via the lightweight NumPy store. Run `python -m pipeline_cli embed --parser GROBIDParser --chunker LengthChunker --embedder BGEEmbedder`.
- **Answer generation** (`src/answer_generation/`): Wraps OpenAI/Together chat APIs, builds prompts, and returns text + logprobs.
- **End-to-end inference** (`pipeline_cli.py full-infer`): Loads parsed docs, applies a chosen prompt variant from `query_creation/queries_with_prompts.json`, and records LLM responses in `trial_logs/full_runs/`. Example: `python -m pipeline_cli full-infer --parser GROBIDParser --llm-model gpt-4.1 --prompt base_prompt`.
- **Testing** (`python -m pipeline_cli test`): Runs pytest and stores a JSON report under `trial_logs/test_runs/`.

## Repository layout
- `pipeline_cli.py` – Primary Typer CLI orchestrating parsing, chunking, embedding, full-document inference, and test runs.
- `cli.py` – Earlier/simple CLI for prompt generation and parsing.
- `query_creation/` – Coding scheme Excel file, prompt templates, few-shot examples, and helpers (`collect_item_info.py`, `create_prompts.py`, `get_outcome.py`, `prompts_to_html.py`).
- `src/document_parsing/` – Parsers for PDFs and images (`base_parser`, `pymupdf`, `pymupdf_tesseract`, `grobid`, `vlm_gemini`, `vlm_qwen`, `mineru`).
- `src/chunking/` – Base chunker plus length-, structure-, and text-aware chunking strategies.
- `src/embedding/` – Embedder interfaces and concrete implementations (BGE/E5/OpenAI) registered in `src/registry.py`.
- `src/retrieval/` – Retriever implementations (top-k, cross-encoder reranking, token-budget, RCS) used to pair queries with relevant chunks.
- `src/answer_generation/` – LLM wrapper for OpenAI/Together plus model metadata in `filtered_models.json`.
- `src/vectorstore/numpy_store.py` – Minimal on-disk vector store for embeddings.
- `tests/` – Pytest suite; markers identify integration/API-dependent tests.
- `data/` – Expected locations for raw PDFs (`data/raw`), parsed outputs, chunks, and embeddings; logs live in `trial_logs/`.

## Running it locally
- Python 3.11+; install dependencies with `pip install -r requirements.txt` (and optionally `pip install -e .` to expose the `pipe` entry point).
- Environment variables: `OPENAI_API_KEY`, `TOGETHER_API_KEY`, and `GOOGLE_API_KEY` (for Gemini parser) are loaded from `.env` if present.
- Typical flow: place PDFs in `data/raw/` → `create-queries` → `parse-docs` → `chunking` → `embed` (if using retrieval) → `full-infer` to collect model outputs. Adjust parser/chunker/embedder names to match those registered in `src/registry.py`.
