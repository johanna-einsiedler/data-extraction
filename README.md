# Paper Extraction Pipeline

Tools for parsing PDFs, creating structured prompts, chunking text, embedding content, and running LLM-based coding/data-extraction experiments for social science articles (OSF prereg: https://osf.io/quxdb).

## Status (2026-10)
- Phase-1 Optuna search: `optuna_study.db` (repo root, frozen 2026-08-18, 96 COMPLETE). Read-only;
  `analysis/hpo_analysis/paths.py`, `stage2_analyses/phase2_study/copy_trials.py` and `src/objective.py`
  (default when `OPTUNA_DB` is unset) all point at it. **Never run `src/objective.py` without `OPTUNA_DB`**
  (it still holds 3 WAITING trials that would execute).
- Phase-2 searches: `stage2_analyses/phase2_study/{phase2,open_tpe,consumer_tpe}.db` (finished 2026-09-28/29/30).
  Design and seeding provenance: `stage2_analyses/phase2_study/README.md`. The four DBs are git-tracked: a
  `git status` change on any of them means something wrote to a finished study.
- Analysis layer: `analysis/hpo_analysis/` -> `analysis/cache/` (parquets), `analysis/tables/hpo/`,
  `analysis/figures/hpo/`. `analysis/tables/hpo/mode_by_type_*.csv` and `temperature_probe.md` are cited by the
  Phase-2 preregistration (`../prereg2_overleaf/phase2_additions/`).
- Next stage: frozen configurations on the calibration and test papers (the tuning set was `data/tuning/`, 35 papers).
- Superseded material lives in `archive/` (git-ignored except READMEs; see `archive/README.md`).

## Running tests
`pipeline_env/bin/python -m pytest tests -q --ignore=tests/caching` (pytest puts `src/` on `sys.path` via
`pyproject.toml`; `tests/test_repo_imports.py` fails if the venv ever resolves `registry` outside this repo again;
`tests/caching/test_objective_caching.py` overwrites real files at import and must not be run).

## Running a study
`OPTUNA_DB=sqlite:///$PWD/<study>.db PAPER_CONCURRENCY=4 N_JOBS=1 pipeline_env/bin/python src/objective.py <n_trials> --models=none`
or `stage2_analyses/phase2_study/launch.sh <db-file> <n>` (detached). Papers are read from `data/tuning/`;
per-trial outputs go to `trial_logs/<cfghash>_trial_<n>/`, parsed/chunk/embedding caches to `data/` and `trial_cache/`.

## How the pipeline fits together
- **Prompt preparation** (`query_creation/`): Build query metadata from `coding_scheme.xlsx`, generate prompt variants, and render HTML views of prompts. Run `python -m pipeline_cli create-queries` (or `pipe create-queries` after installation).
- **Document parsing** (`src/document_parsing/`): Multiple parsers (GROBID, PyMuPDF, Tesseract OCR, Gemini/Qwen VLMs, MinerU) convert PDFs in `data/tuning/` (symlinks into `data/all_articles/`) to normalised text/Markdown in `data/parsed/<parser>/`. Run `python -m pipeline_cli parse-docs --folder data/tuning --parser GROBIDParser`.
- **Chunking** (`src/chunking/`): Slice parsed text into overlapping windows using several strategies (length-based, structure-aware, text heuristics). Writes JSON chunks under `data/chunks/<parser>/<chunker>/`. Example: `python -m pipeline_cli chunking --parser GROBIDParser --chunker LengthChunker --chunk-size 500 --chunk-overlap 50`.
- **Embedding + vector store** (`src/embedding/`, `src/vectorstore/`): Generate embeddings with BGE/E5/OpenAI and optionally save/query them via the lightweight NumPy store. Run `python -m pipeline_cli embed --parser GROBIDParser --chunker LengthChunker --embedder BGEEmbedder`.
- **Answer generation** (`src/answer_generation/`): Wraps OpenAI/Together chat APIs, builds prompts, and returns text + logprobs.
- **End-to-end inference** (`pipeline_cli.py full-infer`): Loads parsed docs, applies a chosen prompt variant from `query_creation/queries_with_prompts.json`, and records LLM responses in `trial_logs/<cfghash>_trial_<n>/`. Example: `python -m pipeline_cli full-infer --parser GROBIDParser --llm-model gpt-4.1 --prompt base_prompt`.
- **Testing** (`python -m pipeline_cli test`): Runs pytest and stores a JSON report under `trial_logs/test_runs/`.

## Repository layout
- `pipeline_cli.py` – Primary Typer CLI orchestrating parsing, chunking, embedding, full-document inference, and test runs.
- `query_creation/` – Coding scheme Excel file, prompt templates, few-shot examples, and helpers (`collect_item_info.py`, `create_prompts.py`, `get_outcome.py`, `prompts_to_html.py`).
- `src/document_parsing/` – Parsers for PDFs and images (`base_parser`, `pymupdf`, `pymupdf_tesseract`, `grobid`, `vlm_gemini`, `vlm_qwen`, `mineru`).
- `src/chunking/` – Base chunker plus length-, structure-, and text-aware chunking strategies.
- `src/embedding/` – Embedder interfaces and concrete implementations (BGE/E5/OpenAI) registered in `src/registry.py`.
- `src/retrieval/` – Retriever implementations (top-k, cross-encoder reranking, token-budget, RCS) used to pair queries with relevant chunks.
- `src/answer_generation/` – LLM wrapper for OpenAI/Together plus model metadata in `filtered_models.json`.
- `src/vectorstore/numpy_store.py` – Minimal on-disk vector store for embeddings.
- `tests/` – Pytest suite; markers identify integration/API-dependent tests.
- `data/` – PDFs (`data/all_articles/`, with `data/tuning/`, `data/calibration/`, `data/test/` symlink folders), parsed outputs, chunks, embeddings, gold labels (`data/true_labels/`) and split id lists (`data/dataset_splits/`); per-trial logs live in `trial_logs/`.

## Running it locally
- Python 3.11; the venv is `pipeline_env/` (uv-created). `pip install -e .` registers only the `pipe` CLI; `src/` is a flat layout reached by explicit `sys.path.insert` (as `pipeline_cli.py` does) or via pytest's `pythonpath`.
- Environment variables: `OPENAI_API_KEY`, `TOGETHER_API_KEY`, and `GOOGLE_API_KEY` (for Gemini parser) are loaded from `.env` if present.
- Typical flow: place PDFs in `data/all_articles/` (and symlink the paper set) → `create-queries` → `parse-docs` → `chunking` → `embed` (if using retrieval) → `full-infer` to collect model outputs. Adjust parser/chunker/embedder names to match those registered in `src/registry.py`.
