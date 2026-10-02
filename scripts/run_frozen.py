"""Run ONE frozen configuration on ONE paper set, without Optuna.

usage: pipeline_env/bin/python scripts/run_frozen.py --config {primary|open|consumer|<json file>}
         --paper-set {calibration|test|tuning|rq2_subset|<ids file>} --run-id <label>
         [--dry-run] [--resume] [--concurrency N] [--splits-version v2_2026-10-02]
         [--runs-root runs] [--query-ids 1.8,3.2,...] [--allow-judge-override] [--allow-test-stub]

Contract (preregistered run policy):
  * single touch: a (run-id, config) output folder that already holds doc_*.json
    is refused unless --resume (which continues the SAME run; it never re-generates
    a finished paper, since the per-paper resume matches hash + doc_name);
  * nothing is parsed and no model is called outside the real run (--dry-run is
    read-only and prints the preflight as JSON);
  * every guard of the search is off (FROZEN_RUN=1); retries stay on;
  * outputs go to <runs-root>/<run-id>/<config>/ with their own cache.json and a
    hash-chained calls.jsonl; a manifest.json records config, hashes, paper set,
    missing papers with reasons, ledger head, judge, parser versions, usage, git.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.metadata as _im
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "configs" / "frozen_pipelines.json"
PREREG_JUDGE = {"JUDGE_BACKEND": "local", "JUDGE_MODEL_NAME": "microsoft/Phi-3-mini-4k-instruct",
                "_JUDGE_VERSION": "v2-supersetrelaxed"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    diff = run("diff", "HEAD")
    return {"commit": run("rev-parse", "HEAD") or None, "branch": run("rev-parse", "--abbrev-ref", "HEAD") or None,
            "dirty": bool(run("status", "--porcelain")), "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None}


def load_entry(name_or_path: str, allow_stub: bool) -> tuple[str, dict]:
    if name_or_path.endswith(".json") and Path(name_or_path).exists():
        e = json.load(open(name_or_path)); return Path(name_or_path).stem, e
    reg = json.load(open(REG))["pipelines"]
    if name_or_path not in reg:
        raise SystemExit(f"unknown config {name_or_path!r}; registry has {sorted(reg)}")
    e = reg[name_or_path]
    if e.get("test_only") and not allow_stub:
        raise SystemExit(f"{name_or_path} is test-only; pass --allow-test-stub")
    return name_or_path, e


def resolve_paper_set(spec: str, version: str) -> tuple[str, list[str], Path, Path]:
    """-> (name, ids, ids_file, pdf_folder)."""
    p = Path(spec)
    if p.exists() and p.is_file():
        ids = [l.strip() for l in open(p) if l.strip()]
        return p.stem, ids, p, ROOT / "data" / "all_articles"
    ids_file = ROOT / "configs" / "splits" / version / f"{spec}_ids.txt"
    if not ids_file.exists():
        raise SystemExit(f"no id list for paper set {spec!r} at {ids_file}")
    ids = [l.strip() for l in open(ids_file) if l.strip()]
    folder = ROOT / "data" / "splits_v2" / spec
    if spec == "tuning":
        folder = ROOT / "data" / "tuning"
    if not folder.exists():
        folder = ROOT / "data" / "all_articles"
    return spec, ids, ids_file, folder


def parser_dir(parser_name: str) -> Path:
    return ROOT / "data" / "parsed" / parser_name


def preflight(cname: str, entry: dict, pset: str, ids: list[str], ids_file: Path, folder: Path,
              allow_judge_override: bool) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    import evaluate as E  # noqa: E402
    roster = {m["id"]: m for m in json.load(open(ROOT / "src/answer_generation/filtered_models.json"))}
    params = entry["optuna_params"]; model = roster.get(params["llm_model"])
    if model is None or model.get("disabled"):
        raise SystemExit(f"model {params['llm_model']} missing/disabled in roster")
    key_env = {"openai": "OPENAI_API_KEY", "together": "TOGETHER_API_KEY", "openrouter": "OPENROUTER_API_KEY",
               "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(model["api_type"])
    from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
    api_key_present = bool(os.getenv(key_env)) if key_env else None
    if model["api_type"] == "gemini" and not api_key_present:
        api_key_present = bool(os.getenv("GOOGLE_API_KEY"))
    parser = params["parser"]
    parser_ok, parser_version = True, None
    if parser == "MinerUParser":
        parser_ok = shutil.which("mineru") is not None or (ROOT / "pipeline_env/bin/mineru").exists()
        try:
            parser_version = _im.version("mineru")
        except Exception:
            parser_version = None
    judge = {"JUDGE_BACKEND": E.JUDGE_BACKEND, "JUDGE_MODEL_NAME": E.JUDGE_MODEL_NAME, "_JUDGE_VERSION": E._JUDGE_VERSION,
             "JUDGE_DTYPE": os.getenv("JUDGE_DTYPE", "float32"), "OPEN_ENDED_METRIC": E.OPEN_ENDED_METRIC}
    judge_ok = all(judge[k] == v for k, v in PREREG_JUDGE.items())
    if not judge_ok and not allow_judge_override:
        raise SystemExit(f"judge differs from the preregistered one: {judge} vs {PREREG_JUDGE} (pass --allow-judge-override)")
    pdfs = {}
    for f in folder.glob("*.pdf"):
        import re
        m = re.match(r"^(\d+)", f.stem)
        if m:
            pdfs[m.group(1)] = f
    preparsed = {pid: any((parser_dir(parser) / f"{pdfs[pid].stem}{ext}").exists() for ext in (".md", ".xml", ".txt", ".json"))
                 for pid in ids if pid in pdfs}
    import objective
    try:
        usd_pp = objective.project_cost_per_paper(entry["trial_choices"])
    except Exception:
        usd_pp = None
    return {"config": cname, "hash10": entry["trial_hash10"], "paper_set": pset, "n_ids": len(ids),
            "ids_sha256": sha256_file(ids_file), "pdf_folder": str(folder), "pdfs_found": sum(1 for i in ids if i in pdfs),
            "pdfs_missing": [i for i in ids if i not in pdfs], "preparsed_count": sum(preparsed.values()),
            "preparsed": sorted(k for k, v in preparsed.items() if v),
            "model": {"id": params["llm_model"], "api_type": model["api_type"], "api_key_present": api_key_present},
            "parser": {"name": parser, "available": parser_ok, "version": parser_version},
            "judge": judge, "judge_matches_prereg": judge_ok,
            "projected_usd_per_paper": usd_pp, "projected_usd_total": (usd_pp * len(ids)) if usd_pp else None,
            "git": git_info()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--paper-set", required=True)
    ap.add_argument("--run-id", required=True); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--splits-version", default="v2_2026-10-02"); ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--query-ids"); ap.add_argument("--allow-judge-override", action="store_true")
    ap.add_argument("--allow-test-stub", action="store_true")
    a = ap.parse_args()

    cname, entry = load_entry(a.config, a.allow_test_stub)
    out_dir = (ROOT / a.runs_root / a.run_id / cname).resolve()
    # ---- environment BEFORE importing objective (module constants read env at import) ----
    os.environ["FROZEN_RUN"] = "1"
    # objective.py creates its cache dir at import; a dry run must not create out_dir.
    import tempfile
    os.environ["TRIAL_CACHE_DIR"] = tempfile.mkdtemp(prefix="run_frozen_dry_") if a.dry_run else str(out_dir)
    os.environ["CALL_LEDGER_PATH"] = str(out_dir / "calls.jsonl")
    os.environ["RUN_ID"] = a.run_id
    os.environ["PAPER_CONCURRENCY"] = str(a.concurrency)
    for k in ("SMOKE_MAX_DOCS", "SMOKE_QUERY_IDS", "RESUME_FROM_FOLDER", "SAMPLE_ONLY", "GEMINI_BUDGET_USD"):
        os.environ.pop(k, None)
    os.environ.update({"QWEN_TIMEOUT_EXCLUSION": "0", "TRIAL_WALL_BUDGET_S": "1000000000",
                       "TIME_CAP_S_PER_PAPER": "1000000000", "COST_CAP_USD_PER_PAPER": "1e9", "MIN_PAPER_FRAC": "0"})
    sys.path.insert(0, str(ROOT / "src"))

    pset, ids, ids_file, folder = resolve_paper_set(a.paper_set, a.splits_version)
    pf = preflight(cname, entry, pset, ids, ids_file, folder, a.allow_judge_override)
    if a.dry_run:
        print(json.dumps(pf, indent=1, default=str)); return
    # ---- single-touch gate + lock ----
    if any(out_dir.glob("doc_[0-9]*.json")) and not a.resume:
        raise SystemExit(f"single-touch: {out_dir} already holds outputs; pass --resume to continue the same run")
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / ".lock"
    if lock.exists():
        try:
            pid = int(json.load(open(lock)).get("pid", -1))
            os.kill(pid, 0); raise SystemExit(f"run in progress (pid {pid}) per {lock}")
        except (ProcessLookupError, ValueError, PermissionError):
            pass
    json.dump({"pid": os.getpid(), "host": platform.node(), "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat()},
              open(lock, "w"))
    fh = logging.FileHandler(out_dir / "run.log"); fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh); logging.getLogger().setLevel(logging.INFO)
    manifest_path = out_dir / "manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else {
        "run_id": a.run_id, "config_name": cname, "registry_entry": entry, "paper_set": {"name": pset, "ids_file": str(ids_file),
        "sha256": pf["ids_sha256"], "n": len(ids), "ids": ids}, "out_dir": str(out_dir), "attempts": []}
    attempt = {"started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(), "resume": a.resume}
    manifest["attempts"].append(attempt); manifest["status"] = "running"; manifest["preflight"] = pf
    json.dump(manifest, open(manifest_path, "w"), indent=1, default=str)

    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
    import objective
    from answer_generation import call_ledger
    inputs_sha = objective.init_globals()
    trial = optuna.trial.FixedTrial(entry["optuna_params"])
    qids = [q.strip() for q in a.query_ids.split(",")] if a.query_ids else None
    t0 = time.time(); acc = None; err = None
    try:
        acc = objective.objective(trial, verbose=True, query_ids=qids, paper_dir=folder, paper_ids=ids,
                                  out_dir=out_dir, frozen=True)
    except BaseException as e:  # noqa: BLE001 - manifest must record the failure
        err = f"{type(e).__name__}: {e}"
        raise
    finally:
        attempt["ended_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat(); attempt["seconds"] = round(time.time() - t0)
        ua = dict(trial.user_attrs)
        ledger = call_ledger.verify(out_dir / "calls.jsonl") if (out_dir / "calls.jsonl").exists() else {"ok": None, "n": 0, "head": None}
        papers = {}
        for f in sorted(out_dir.glob("doc_[0-9]*.json")):
            d = json.load(open(f)); pid = str(d.get("doc_name"))
            n_items = len(d.get("results", []))
            papers[pid] = {"status": "ok", "doc_file": f.name, "n_items": n_items,
                           "hash_ok": d.get("parameters_hash") == entry["parameters_hash_md5"], "in_set": pid in set(ids)}
            meta = parser_dir(entry["optuna_params"]["parser"]) / f"{next((p.stem for p in folder.glob(pid + '_*.pdf')), pid)}.meta.json"
            if meta.exists():
                try:
                    papers[pid]["parser_meta"] = json.load(open(meta))
                except Exception:
                    pass
        missing = dict(ua.get("paper_failures", {}))
        for pid in ids:
            if pid not in papers and pid not in missing:
                missing[pid] = "no output file"
        manifest.update({
            "status": "failed" if err else "complete", "error": err, "accuracy": acc,
            "totals": {k: ua.get(k) for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "llm_calls", "cost_usd",
                                               "batched_split", "batched_parse_failed", "batched_truncated", "n_papers_ok",
                                               "n_papers_requested")},
            "papers": papers, "missing": missing, "ledger": {"path": str(out_dir / "calls.jsonl"), **ledger},
            "judge": pf["judge"], "parser": pf["parser"], "model": pf["model"],
            "env": {k: os.getenv(k) for k in ("PAPER_CONCURRENCY", "REQUEST_TIMEOUT", "VLM_ZOOM", "JUDGE_DTYPE")},
            "versions": {"python": platform.python_version(), **{p: _v(p) for p in ("optuna", "openai", "google-genai", "transformers", "torch", "mineru")}},
            "inputs_sha256": inputs_sha, "git": git_info(), "ended_utc": attempt["ended_utc"]})
        json.dump(manifest, open(manifest_path, "w"), indent=1, default=str)
        lock.unlink(missing_ok=True)
        print(f"{manifest['status']}: accuracy={acc} papers_ok={len(papers)} missing={len(missing)} ledger_n={ledger.get('n')} -> {manifest_path}")


def _v(pkg: str):
    try:
        return _im.version(pkg)
    except Exception:
        return None


if __name__ == "__main__":
    main()
