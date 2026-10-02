"""scripts/run_frozen.py on two tuning papers with the LLM call stubbed at the class
level (the real generate() body, ledger hook and scoring run; no API, no judge:
the three items used are single-choice / numeric / multiple-choice)."""
import json, os, pathlib, subprocess, sys, textwrap
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PDFS = ROOT / "data" / "tuning"
pytestmark = pytest.mark.skipif(not (PDFS / "006_Ahnert_2023.pdf").exists() or not (ROOT / ".env").exists(),
                                reason="needs the tuning PDFs and .env")
QIDS = "1.8,1.2.1,3.2"
DRIVER = textwrap.dedent("""
    import json, runpy, sys, types, os
    ROOT = {root!r}
    sys.path.insert(0, ROOT + "/src")
    from answer_generation.answer_generator import AnswerGenerator
    REPLY = {{"1.8": "1", "1.2.1": "5", "3.2": "A"}}
    n = {{"calls": 0}}
    fail_doc = os.environ.get("STUB_FAIL_DOC")
    def _call(self, call_fn, fallback_kwargs=None, client=None):
        n["calls"] += 1
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=json.dumps(REPLY)), finish_reason="stop", logprobs=None)],
            usage=types.SimpleNamespace(prompt_tokens=100, completion_tokens=12), id=f"stub-{{n['calls']}}", created=1, model="stub/model")
    AnswerGenerator._ensure_client = lambda self: object()
    AnswerGenerator._call_with_retry = _call
    if os.environ.get("STUB_PARSE_FAIL"):
        import objective
        _orig = objective.ensure_parsed
        def _ep(f, parser_name, strict=False):
            if f.stem.startswith(os.environ["STUB_PARSE_FAIL"]):
                raise RuntimeError("stub parse failure")
            return _orig(f, parser_name, strict)
        objective.ensure_parsed = _ep
    sys.argv = ["run_frozen.py"] + sys.argv[1:]
    runpy.run_path(ROOT + "/scripts/run_frozen.py", run_name="__main__")
""")


def run(tmp, ids, run_id, *extra, env=None, expect_ok=True):
    drv = tmp / "driver.py"; drv.write_text(DRIVER.format(root=str(ROOT)))
    idf = tmp / f"{run_id}_ids.txt"; idf.write_text("\n".join(ids) + "\n")
    cmd = [sys.executable, str(drv), "--config", "test_stub", "--allow-test-stub", "--paper-set", str(idf),
           "--run-id", run_id, "--runs-root", str(tmp / "runs"), "--query-ids", QIDS, "--concurrency", "1", *extra]
    e = dict(os.environ); e.update(env or {}); e["PYTHONWARNINGS"] = "ignore"
    r = subprocess.run(cmd, cwd=ROOT, env=e, capture_output=True, text=True, timeout=900)
    if expect_ok:
        assert r.returncode == 0, r.stderr[-2000:]
    return r, tmp / "runs" / run_id / "test_stub"


def test_layout_stable_ids_manifest_and_ledger(tmp_path):
    before_logs = {p.name for p in (ROOT / "trial_logs").glob("*")}
    r, out = run(tmp_path, ["006", "079"], "r1")
    assert sorted(p.name for p in out.glob("*")) == sorted(
        ["cache.json", "calls.jsonl", "doc_0006.json", "doc_0079.json", "doc_id_map.json", "manifest.json", "run.log", "trial_summary.json"])
    d = json.load(open(out / "doc_0006.json")); assert d["doc_id"] == 6 and d["doc_name"] == "006" and len(d["results"]) == 3
    assert json.load(open(out / "doc_id_map.json")) == {"006": 6, "079": 79}
    m = json.load(open(out / "manifest.json"))
    assert m["status"] == "complete" and set(m["papers"]) == {"006", "079"} and m["missing"] == {}
    assert all(p["hash_ok"] and p["in_set"] for p in m["papers"].values())
    assert m["ledger"]["ok"] and m["ledger"]["n"] >= 2
    sys.path.insert(0, str(ROOT / "src")); from answer_generation import call_ledger as cl
    assert cl.verify(out / "calls.jsonl")["head"] == m["ledger"]["head"]
    assert m["judge"]["JUDGE_MODEL_NAME"] == "microsoft/Phi-3-mini-4k-instruct" and len(m["git"]["commit"]) == 40
    assert m["registry_entry"]["parameters_hash_md5"] == json.load(open(out / "doc_0006.json"))["parameters_hash"]
    assert {p.name for p in (ROOT / "trial_logs").glob("*")} == before_logs, "frozen runs must not write trial_logs/"


def test_single_touch_then_resume_on_hash_and_doc_name(tmp_path):
    r, out = run(tmp_path, ["006", "079"], "r2")
    n1 = json.load(open(out / "manifest.json"))["ledger"]["n"]
    r2, _ = run(tmp_path, ["006", "079"], "r2", expect_ok=False)
    assert r2.returncode != 0 and "single-touch" in (r2.stderr + r2.stdout)
    (out / "doc_0079.json").unlink()
    m6 = (out / "doc_0006.json").stat().st_mtime
    r3, _ = run(tmp_path, ["006", "079"], "r2", "--resume")
    m = json.load(open(out / "manifest.json"))
    assert (out / "doc_0079.json").exists() and (out / "doc_0006.json").stat().st_mtime == m6
    assert m["ledger"]["ok"] and m["ledger"]["n"] > n1 and len(m["attempts"]) == 2
    # a doc file that belongs to another paper is recomputed, not reused
    d = json.load(open(out / "doc_0006.json")); d["doc_name"] = "999"; json.dump(d, open(out / "doc_0006.json", "w"))
    run(tmp_path, ["006", "079"], "r2", "--resume")
    assert json.load(open(out / "doc_0006.json"))["doc_name"] == "006"


def test_strict_parser_failure_is_recorded_not_fatal(tmp_path):
    r, out = run(tmp_path, ["006", "079"], "r3", env={"STUB_PARSE_FAIL": "079"})
    m = json.load(open(out / "manifest.json"))
    assert m["status"] == "complete" and set(m["papers"]) == {"006"}
    assert list(m["missing"]) == ["079"] and m["missing"]["079"].startswith("parse_failed")


def test_dry_run_writes_nothing(tmp_path):
    r, out = run(tmp_path, ["006", "079"], "r4", "--dry-run")
    assert not out.exists() and '"preparsed_count": 2' in r.stdout


def test_guards_disabled_under_hostile_env(tmp_path):
    env = {"TRIAL_WALL_BUDGET_S": "1", "TIME_CAP_S_PER_PAPER": "0", "COST_CAP_USD_PER_PAPER": "0",
           "MIN_PAPER_FRAC": "1.0", "GEMINI_BUDGET_USD": "0"}
    r, out = run(tmp_path, ["006", "079"], "r5", env=env)
    assert json.load(open(out / "manifest.json"))["status"] == "complete" and (out / "trial_summary.json").exists()
