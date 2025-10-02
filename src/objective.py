import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import optuna
from dotenv import find_dotenv, load_dotenv

from answer_generation.answer_generator import AnswerGenerator
from evaluate import evaluate_answer
from registry import CHUNKERS, EMBEDDERS, LLMS, PARSERS, RETRIEVERS
from vectorstore.numpy_store import NumpyVectorStore

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)


def get_parser(name, **overrides):
    """Factory to lazily instantiate parsers with optional overrides."""
    entry = PARSERS[name]
    cls = entry["cls"]
    kwargs = {**entry["kwargs"], **overrides}
    return cls(**kwargs)


QUERIES_JSON_PATH = Path(
    "../query_creation/queries_with_prompts.json"
)  # Path to your JSON file


def load_queries():
    with open(QUERIES_JSON_PATH, "r", encoding="utf-8") as f:
        query_dict = json.load(f)

    queries = {}
    for qid, q in query_dict.items():
        queries[qid] = q  # keep original structure
    return queries


QUERIES = load_queries()


# ---------------------------------------------------------------------
# helper path builders (change base folders if you like)
# ---------------------------------------------------------------------
def parsed_dir(parser_name: str, base: str = "../data/parsed") -> Path:
    return Path(base) / parser_name


def chunks_dir(
    parser_name: str, chunker_name: str, base: str = "../data/chunks"
) -> Path:
    return Path(base) / parser_name / chunker_name


def embeddings_dir(
    parser_name: str,
    chunker_name: str,
    embedder_name: str,
    base: str = "../data/embeddings",
) -> Path:
    return Path(base) / parser_name / chunker_name / embedder_name


# ---------------------------------------------------------------------
# Ensure functions: run component only if output missing
# ---------------------------------------------------------------------
def ensure_parsed(file_path: Path, parser_name: str):
    """Ensure a file is parsed. Return path to cached or newly parsed output."""
    print(os.getcwd())
    pdir = parsed_dir(parser_name)
    pdir.mkdir(parents=True, exist_ok=True)

    json_path = pdir / f"{file_path.stem}.json"
    txt_path = pdir / f"{file_path.stem}.txt"

    print(json_path, txt_path)

    # 1️⃣ Use cached version if it exists
    if json_path.exists():
        return json_path
    if txt_path.exists():
        return txt_path

    # 2️⃣ Run parser
    parser = get_parser(parser_name)
    result = parser.parse(str(file_path))  # make sure your parser uses `out_dir`
    # 3️⃣ Save result depending on type
    if isinstance(result, (dict, list)):
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        return json_path

    elif isinstance(result, str):
        # detect if it looks like XML
        if result.strip().startswith("<"):
            xml_path = pdir / f"{file_path.stem}.xml"
            with open(xml_path, "w", encoding="utf-8") as fh:
                fh.write(result)
            return xml_path
        else:
            with open(txt_path, "w", encoding="utf-8") as fh:
                fh.write(result)
            return txt_path

    elif isinstance(result, Path):
        return result

    # 4️⃣ fallback: save repr
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(repr(result))
    return txt_path


def ensure_chunks(
    parsed_path: Path,
    parser_name: str,
    chunker_name: str,
    chunk_size: int,
    overlap: int,
) -> Path:
    """Ensure parsed file is chunked with specific parameters.
    Cached by docname + chunk size + overlap."""
    cdir = chunks_dir(parser_name, chunker_name)
    cdir.mkdir(parents=True, exist_ok=True)

    # include size + overlap in filename
    out_path = cdir / f"{parsed_path.stem}_s{chunk_size}_o{overlap}.json"

    # Reuse if already exists
    if out_path.exists():
        return out_path

    # Load parsed content
    if parsed_path.suffix == ".json":
        with open(parsed_path, "r", encoding="utf-8") as fh:
            parsed = json.load(fh)
    else:
        parsed = parsed_path.read_text(encoding="utf-8")

    # Run chunker (pass params if supported)
    chunker = CHUNKERS[chunker_name](chunk_size=chunk_size, chunk_overlap=overlap)
    if hasattr(chunker, "chunk"):
        chunks = chunker.chunk(parsed)
    else:
        raise ValueError(f"Chunker {chunker_name} does not support chunking interface.")

    # Save chunks
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)
    return out_path


def ensure_embedding(
    chunks_path: Path,
    parser_name: str,
    chunker_name: str,
    embedder_name: str,
    chunk_size: int,
    overlap: int,
) -> Path:
    """Ensure embeddings exist for specific chunks (size + overlap)."""
    edir = embeddings_dir(parser_name, chunker_name, embedder_name)
    edir.mkdir(parents=True, exist_ok=True)

    out_path = edir / f"{chunks_path.stem}"

    if out_path.exists():
        return out_path

    # Load chunks
    with open(chunks_path, "r", encoding="utf-8") as fh:
        try:
            chunks = json.load(fh)
        except json.JSONDecodeError:
            chunks = fh.read().split("\n\n")

    # Embed
    embeddings = EMBEDDERS[embedder_name].embed(chunks)

    # Save in NumpyVectorStore
    store = NumpyVectorStore(save_path=out_path)
    store.add(embeddings, chunks)
    store.save()

    return out_path


def ensure_query_embeddings(
    embedder_name: str,
    doc_name: str,
    query_id: str,
    query_mode: str = "raw",  # "raw" or "label_def"
    base: str = "../data/query_embeddings",
    queries_base: str = "../query_creation",
) -> Path:
    """
    Ensure a single query embedding is cached.
    - Embeddings stored under {base}/{embedder_name}/{doc_name}_{query_id}_{query_mode}.npy
    - Query info and prompts are read directly from query_info.json
    """
    edir = Path(base) / embedder_name
    edir.mkdir(parents=True, exist_ok=True)

    out_path = edir / f"{doc_name}_{query_id}_{query_mode}.npy"

    # 1️⃣ If cached, reuse
    if out_path.exists():
        return out_path

    # 2️⃣ Load query_info.json
    qdir = Path(queries_base)
    info_file = qdir / "queries_with_prompts.json"  # single JSON file for all queries
    if not info_file.exists():
        raise FileNotFoundError(f"No queries.json found in {qdir}")

    with open(info_file, "r", encoding="utf-8") as fh:
        all_queries = json.load(fh)

    # Find the query by ID
    qinfo = all_queries.get(query_id)
    if qinfo is None:
        raise ValueError(f"Query ID {query_id} not found in queries.json")

    # 3️⃣ Prepare text to embed
    if query_mode == "label_def":
        parts = [qinfo.get("label", ""), qinfo.get("description", "")]
        if qinfo.get("choices"):
            choices_text = ", ".join(
                f"{k}: {v['value']}" for k, v in qinfo["choices"].items()
            )
            parts.append("Choices: " + choices_text)
        text = " ".join(p for p in parts if p)
    else:
        # "raw" mode: pick base_prompt if available
        text = qinfo.get("prompts", {}).get("base_prompt")
        if not text:
            raise ValueError(f"No 'base_prompt' found for query ID {query_id}")

    # 4️⃣ Embed
    vector = EMBEDDERS[embedder_name].embed([text])[0]
    np.save(out_path, np.asarray(vector))

    return out_path


# ---------------------------------------------------------------------
# Optuna objective (stepwise cached pipeline)
# ---------------------------------------------------------------------
CACHE_PATH = Path("../trial_cache")
CACHE_PATH.mkdir(exist_ok=True)


def _hash_trial_key(params: dict, doc_idx: int, query_id: str) -> str:
    """
    Create a stable hash key for caching based on params+doc+query.
    """
    key_data = json.dumps(
        {"params": params, "doc_idx": doc_idx, "query_id": query_id},
        sort_keys=True,
    )
    return hashlib.md5(key_data.encode("utf-8")).hexdigest()


def load_cache():
    cache_file = CACHE_PATH / "cache.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    cache_file = CACHE_PATH / "cache.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def load_doc_id_map(path: str = "../data/doc_id_map.json") -> dict:
    """Load or create a persistent mapping from document name to ID."""
    path = Path(path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_doc_id_map(doc_id_map: dict, path: str = "../data/doc_id_map.json"):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc_id_map, f, indent=2, ensure_ascii=False)


def objective(trial, verbose: bool = False):
    trial_choices = {}
    all_results = []
    llm_choices = []

    # Load global cache
    cache = load_cache()

    def log(msg: str):
        """Helper: print only if verbose=True, always flushed."""
        if verbose:
            print(msg, file=sys.stderr, flush=True)

    try:
        parser_name = trial.suggest_categorical("parser", list(PARSERS.keys()))
        trial_choices["parser"] = parser_name
        mode = "rag"
        trial_choices["mode"] = mode
        log(f"[Trial {trial.number}] Starting with parser={parser_name}, mode={mode}")

        folder = Path("../data/raw/")
        all_docs = [f for f in folder.glob("*") if f.suffix.lower() == ".pdf"]
        log(f"  Found {len(all_docs)} documents")
        parsed_docs = [ensure_parsed(f, parser_name) for f in all_docs]
        log("  Parsing complete")

        doc_aggregates = {}
        query_aggregates = {}

        doc_id_map = load_doc_id_map()
        next_id = max(doc_id_map.values(), default=-1) + 1

        for parsed_path in parsed_docs:
            doc_name = parsed_path.stem
            if doc_name not in doc_id_map:
                doc_id_map[doc_name] = next_id
                next_id += 1
            doc_id = doc_id_map[doc_name]

            log(f"--- Document {doc_id}: {parsed_path.name} ---")
            doc_results = []

            for qid, qdata in QUERIES.items():
                log(f"  > Query {qid}")
                # ---------------- Retrieval ----------------
                start_stage = time.time()
                if mode == "rag":
                    chunk_size = trial.suggest_categorical(
                        "chunk_size", [100, 250, 500, 750, 1000, 2500, 5000]
                    )
                    overlap = trial.suggest_categorical(
                        "chunk_overlap",
                        [o for o in [0, 250, 750] if o < chunk_size] or [0],
                    )
                    chunker_name = trial.suggest_categorical(
                        "chunker", list(CHUNKERS.keys())
                    )
                    embedder_name = trial.suggest_categorical(
                        "embedder", list(EMBEDDERS.keys())
                    )
                    trial_choices.update(
                        {
                            "chunk_size": chunk_size,
                            "chunk_overlap": overlap,
                            "chunker": chunker_name,
                            "embedder": embedder_name,
                        }
                    )
                    log(
                        f"    Retrieval: chunker={chunker_name}, embedder={embedder_name}, size={chunk_size}, overlap={overlap}"
                    )

                    chunk_path = ensure_chunks(
                        parsed_path, parser_name, chunker_name, chunk_size, overlap
                    )
                    vector_store_path = ensure_embedding(
                        chunk_path,
                        parser_name,
                        chunker_name,
                        embedder_name,
                        chunk_size,
                        overlap,
                    )
                    vector_store = NumpyVectorStore(
                        save_path=str(vector_store_path) + ".npz"
                    )
                    vector_store.load()

                    retriever_name = trial.suggest_categorical(
                        "retriever", list(RETRIEVERS.keys())
                    )
                    retriever_cls = RETRIEVERS[retriever_name]
                    k = trial.suggest_categorical("k", [1, 3, 5, 10, 20])
                    params = {"k": k}
                    if hasattr(retriever_cls, "token_budget"):
                        params["token_budget"] = trial.suggest_categorical(
                            "token_budget", [200, 500, 1000, 2000]
                        )
                    if getattr(retriever_cls, "supports_rerank", False):
                        params["rerank"] = trial.suggest_categorical(
                            "rerank", [True, False]
                        )
                        if params["rerank"] and getattr(
                            retriever_cls, "supports_top_m", False
                        ):
                            params["top_m"] = trial.suggest_categorical(
                                "top_m", [5, 10, 20, 50]
                            )

                    retriever = retriever_cls(**params)
                    log(f"    Retriever: {retriever_name}, k={k}, params={params}")

                    query_embedding_mode = trial.suggest_categorical(
                        "query_embedding", ["raw", "label_def"]
                    )
                    query_emb_path = ensure_query_embeddings(
                        embedder_name, parsed_path.stem, qid, query_embedding_mode
                    )

                    # Load embeddings
                    query_embed = np.load(query_emb_path)

                    # Retrieve chunks
                    retrieved_chunks = retriever.retrieve(query_embed, vector_store)

                log(f"    Retrieval done in {time.time() - start_stage:.2f}s")

                # ---------------- Caching ----------------
                trial_key = _hash_trial_key(trial_choices, doc_id, qid)
                if trial_key in cache:
                    result = cache[trial_key]
                    log("    Using cached result")
                else:
                    # ---------------- LLM inference ----------------
                    llm_model = trial.suggest_categorical(
                        "llm_model", list(LLMS.keys())
                    )

                    # Instead of selecting a file, select a prompt type
                    prompt_type = trial.suggest_categorical(
                        "prompt_type", list(qdata["prompts"].keys())
                    )
                    prompt_text = qdata["prompts"].get(prompt_type)
                    if not prompt_text:
                        raise ValueError(
                            f"No prompt found for query {qid} under '{prompt_type}'"
                        )

                    trial_choices.update(
                        {"llm_model": llm_model, "prompt_type": prompt_type}
                    )
                    log(f"    Calling LLM {llm_model} with prompt={prompt_type}")

                    ans_gen = LLMS[llm_model]

                    # log(
                    #     f"qdata {qdata['prompts'][prompt_file]},chunks {retrieved_chunks}, prompt {qdata['system_prompt']}"
                    # )

                    llm_output = ans_gen.generate(
                        query=qdata["prompts"][prompt_type],
                        chunks=retrieved_chunks,
                        return_logprobs=True,
                    )

                    def handle_other(llm_text):
                        """
                        Called when 'Other' (Z) is selected in multiple choice.
                        Uses qdata to generate a follow-up LLM query.
                        """
                        other_prompt = (
                            f"Question required 'Other' answer. Original LLM output: {llm_text}\n"
                            f"Please provide the specific 'Other' response."
                        )

                        # Call your ans_gen.generate with qdata arguments
                        other_response = ans_gen.generate(
                            query=other_prompt,
                            chunks=retrieved_chunks,  # same as original
                            return_logprobs=True,
                        )

                        return other_response["text"]

                    # ---------------- Evaluation ----------------
                    eval_result = evaluate_answer(
                        llm_output["text"],
                        question_type=qdata["type"],
                        doc_name=doc_name,
                        query_id=qid,
                        answer_file="human_codes_test.xlsx",
                        other_callback=handle_other,
                    )
                    result = {"doc_id": doc_id, "query_id": qid, **eval_result}
                    cache[trial_key] = result

                llm_choices.append(result)
                doc_results.append(result)
                all_results.append(result)

                doc_avg = np.mean([r["accuracy"] for r in doc_results])
                doc_aggregates[f"doc_{doc_id}"] = {
                    "average_accuracy": doc_avg,
                    "doc_name": doc_name,
                }
                log(f"--- Document {doc_id} avg acc={doc_avg:.3f} ---")

            # persist updated mapping
            save_doc_id_map(doc_id_map)

        for qid in QUERIES.keys():
            query_scores = [r["accuracy"] for r in all_results if r["query_id"] == qid]
            query_aggregates[qid] = {"average_accuracy": np.mean(query_scores)}

        overall_avg = np.mean([r["accuracy"] for r in all_results])
        log(f"[Trial {trial.number}] Finished. Overall acc={overall_avg:.3f}")

        # ---------------- Structured log ----------------
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "trial_number": trial.number,
            "parameters": trial_choices,
            "overall_average_accuracy": overall_avg,
            "document_aggregates": doc_aggregates,
            "query_aggregates": query_aggregates,
            "all_results": all_results,
        }

        log_path = Path("../trial_logs")
        log_path.mkdir(exist_ok=True)
        with open(log_path / f"trial_{trial.number}.json", "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        save_cache(cache)
        return float(overall_avg)

    except Exception as e:
        log(f"Trial failed with choices: {trial_choices}")
        log(f"LLM outputs: {llm_choices}")
        log(f"Error: {e}")
        raise


if __name__ == "__main__":
    folder = Path("../data/raw/")
    all_docs = list(folder.glob("*"))

    # ans_gen = AnswerGenerator(api_type="openai", model="gpt-3.5")
    # print(ans_gen.client)
    # Create a study

    fixed_params = {
        "parser": "PyMuPDFParser",
        "mode": "rag",
        "chunk_size": 500,
        "chunk_overlap": 0,
        "chunker": "LengthChunker",
        "embedder": "BGEEmbedder",
        "retriever": "TopKRetriever",
        "k": 5,
        "query_embedding": "raw",
        "llm_model": "gpt-3.5-turbo-instruct",
        "prompt_type": "base_prompt",
    }

    # Wrap trial as FixedTrial
    trial = optuna.trial.FixedTrial(fixed_params)
    score = objective(trial, verbose=True)
    # study = optuna.create_study(
    #     direction="maximize"
    # )  # or "minimize" depending on your metric

    # # Run trials
    # study.optimize(lambda t: objective(t, verbose=True), n_trials=1)

    ans_gen = LLMS["gpt-3.5-turbo-instruct"]
