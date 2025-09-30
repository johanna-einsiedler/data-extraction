import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import optuna
from dotenv import load_dotenv

from answer_generation.answer_generator import AnswerGenerator
from evaluate import evaluate_answer
from registry import CHUNKERS, EMBEDDERS, LLMS, PARSERS, RETRIEVERS
from vectorstore.numpy_store import NumpyVectorStore

load_dotenv()
QUERIES_FOLDER = Path("../queries")  # Folder containing subfolders 1.1, 1.2, etc.
PROMPT_FILES = ["base_prompt.txt", "few_shot_prompt.txt"]


def load_queries():
    """Load all queries and their metadata."""
    queries = {}
    for qid_folder in QUERIES_FOLDER.iterdir():
        if not qid_folder.is_dir():
            continue
        qid = qid_folder.name
        # Load question info
        with open(qid_folder / "question_info.json") as f:
            question_info = json.load(f)
        # load system prompt
        with open(qid_folder / "system_prompt.txt") as f:
            system_prompt = json.load(f)
        # Load available prompt variations
        prompts = {}
        for pf in PROMPT_FILES:
            path = qid_folder / pf
            if path.exists():
                with open(path) as f:
                    prompts[pf] = f.read()
        queries[qid] = {
            "question_info": question_info,
            "prompts": prompts,
            "system_prompt": system_prompt,
        }
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
    parser = PARSERS[parser_name]
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
    query_mode: str = "raw",  # "raw" or "label_def"
    base: str = "../data/query_embeddings",
) -> Path:
    """
    Ensure queries are embedded and cached.
    query_mode = "raw" (full prompt) or "label_def" (label + definition + choices).
    """
    edir = Path(base) / embedder_name
    edir.mkdir(parents=True, exist_ok=True)

    out_path = edir / f"queries_{query_mode}.npy"
    ids_path = out_path.with_suffix(".json")

    # 1️⃣ If cached, reuse
    if out_path.exists() and ids_path.exists():
        return out_path

    # 2️⃣ Otherwise embed
    with open("../data/queries.json", "r", encoding="utf-8") as fh:
        queries = json.load(fh)

    chunks = []
    qids = []
    for qid, q in queries.items():
        if query_mode == "label_def":
            parts = [q.get("label", ""), q.get("definition", "")]
            if q.get("choices"):
                parts.append("Choices: " + ", ".join(q["choices"]))
            text = " ".join(p for p in parts if p)
        else:
            text = q["prompt"]

        chunks.append(text)
        qids.append(qid)

    vectors = EMBEDDERS[embedder_name].embed(chunks)
    vectors_arr = np.asarray(vectors)

    # 3️⃣ Save embeddings + query IDs
    np.save(out_path, vectors_arr)
    with open(ids_path, "w", encoding="utf-8") as fh:
        json.dump(qids, fh, indent=2)

    return out_path


# ---------------------------------------------------------------------
# Optuna objective (stepwise cached pipeline)
# ---------------------------------------------------------------------
def objective(trial):
    trial_choices = {}
    all_results = []  # all doc-query results
    llm_choices = []  # keep all detailed evaluation results

    try:
        parser_name = trial.suggest_categorical("parser", list(PARSERS.keys()))
        trial_choices["parser"] = parser_name
        mode = "rag"
        trial_choices["mode"] = mode

        folder = Path("../data/raw/")
        all_docs = list(folder.glob("*"))
        parsed_docs = [ensure_parsed(f, parser_name) for f in all_docs]

        doc_aggregates = {}
        query_aggregates = {}

        for doc_idx, parsed_path in enumerate(parsed_docs):
            doc_results = []

            for qid, qdata in QUERIES.items():
                # ---------------- Retrieval ----------------
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

                    query_embedding_mode = trial.suggest_categorical(
                        "query_embedding", ["raw", "label_def"]
                    )
                    if query_embedding_mode == "raw":
                        text_to_embed = qdata["prompts"][PROMPT_FILES[0]]
                    else:
                        ld = qdata["question_info"]
                        text_to_embed = f"{ld['label']}: {ld['definition']}"

                    query_embed = EMBEDDERS[embedder_name].embed([text_to_embed])[0]
                    retrieved_chunks = retriever.retrieve(query_embed, vector_store)

                else:
                    if parsed_path.suffix == ".json":
                        with open(parsed_path, "r", encoding="utf-8") as fh:
                            retrieved_chunks = json.load(fh)
                    else:
                        retrieved_chunks = [parsed_path.read_text(encoding="utf-8")]

                # ---------------- LLM inference ----------------
                llm_model = trial.suggest_categorical("llm_model", list(LLMS.keys()))
                prompt_file = trial.suggest_categorical("prompt_file", PROMPT_FILES)
                trial_choices.update(
                    {"llm_model": llm_model, "prompt_file": prompt_file}
                )

                ans_gen = LLMS[llm_model]
                llm_output = ans_gen.generate(
                    query=qdata["prompts"][prompt_file],
                    chunks=retrieved_chunks,
                    return_logprobs=True,
                    system_prompt=qdata["system_prompt"],
                )

                # ---------------- Evaluation ----------------
                eval_result = evaluate_answer(llm_output, qdata.get("answer", ""))
                llm_choices.append(eval_result)

                result = {
                    "doc_idx": doc_idx,
                    "query_id": qid,
                    **eval_result,
                }
                doc_results.append(result)
                all_results.append(result)

            # Document-level aggregation
            doc_avg = np.mean([r["accuracy"] for r in doc_results])
            doc_aggregates[f"doc_{doc_idx}"] = {
                "average_accuracy": doc_avg,
                "results": doc_results,
            }

        # Query-level aggregation
        for qid in QUERIES.keys():
            query_scores = [r["accuracy"] for r in all_results if r["query_id"] == qid]
            query_aggregates[qid] = {"average_accuracy": np.mean(query_scores)}

        # Overall average
        overall_avg = np.mean([r["accuracy"] for r in all_results])

        # ---------------- Structured log ----------------
        log = {
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
            json.dump(log, f, indent=2, ensure_ascii=False)

        return float(overall_avg)

    except Exception as e:
        print("Trial failed with choices:", trial_choices)
        print("LLM outputs:", llm_choices)
        print("Error:", e)
        raise


if __name__ == "__main__":
    print(os.getcwd())
    folder = Path("../data/raw/")
    all_docs = list(folder.glob("*"))

    print(all_docs)
    # ans_gen = AnswerGenerator(api_type="openai", model="gpt-3.5")
    # print(ans_gen.client)
    # Create a study
    study = optuna.create_study(
        direction="maximize"
    )  # or "minimize" depending on your metric

    # Run trials
    study.optimize(objective, n_trials=1)
