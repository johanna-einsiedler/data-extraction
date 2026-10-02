"""Call-provenance ledger (2026-10-02): hash chain, tamper detection, provider
provenance extraction, and the generate() hook's entry builder."""
import json, os, pathlib, sys, types
import pytest
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from answer_generation import call_ledger as cl  # noqa: E402


def _entry(i):
    return {"ts_utc": f"2026-10-02T00:00:0{i}Z", "api_type": "openrouter", "model_requested": "m",
            "response_id": f"gen-{i}", "usage": {"prompt_tokens": 10 * i}, "prompt_sha256": "a" * 64}


def test_chain_verifies_and_detects_tampering(tmp_path):
    p = tmp_path / "calls.jsonl"
    heads = [cl.record(_entry(i), p) for i in range(1, 4)]
    assert all(heads) and len(set(heads)) == 3
    v = cl.verify(p)
    assert v["ok"] and v["n"] == 3 and v["head"] == heads[-1]
    # tamper with the middle line's usage -> chain breaks at that line
    lines = p.read_text().splitlines()
    obj = json.loads(lines[1]); obj["usage"]["prompt_tokens"] = 999
    lines[1] = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    p.write_text("\n".join(lines) + "\n")
    v = cl.verify(p)
    assert not v["ok"] and v["first_bad_line"] == 2
    # dropping a line also breaks the chain
    p.write_text("\n".join([lines[0], lines[2]]) + "\n")
    assert not cl.verify(p)["ok"]


def test_chain_continues_across_processes(tmp_path):
    """A fresh process (empty in-memory head) must chain onto the last line on disk."""
    p = tmp_path / "calls.jsonl"
    h1 = cl.record(_entry(1), p)
    cl._HEAD.clear()
    h2 = cl.record(_entry(2), p)
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    assert rows[1]["prev_hash"] == h1 and cl.verify(p)["ok"] and cl.verify(p)["head"] == h2


def test_provenance_shapes():
    oai = types.SimpleNamespace(id="chatcmpl-1", created=1727800000, model="gpt-4.1-2025-04-14",
                                system_fingerprint="fp_abc", provider="DeepInfra")
    pv = cl.provenance(oai, "openrouter")
    assert pv == {"response_id": "chatcmpl-1", "created": 1727800000, "model_resolved": "gpt-4.1-2025-04-14",
                  "provider_routed": "DeepInfra", "system_fingerprint": "fp_abc"}
    assert cl.provenance(oai, "openai")["provider_routed"] is None
    gem = types.SimpleNamespace(response_id="resp-9", model_version="gemini-3.1-pro-preview-0924")
    pg = cl.provenance(gem, "gemini")
    assert pg["response_id"] == "resp-9" and pg["model_resolved"] == "gemini-3.1-pro-preview-0924"
    assert cl.provenance({"id": "x", "model": "m"}, "together")["response_id"] == "x"   # dict shape


def test_entry_for_call_hashes_prompt_and_output():
    r = types.SimpleNamespace(id="id1", created=1, model="m")
    e = cl.entry_for_call(api_type="together", model_requested="m", response=r,
                          usage={"prompt_tokens": 5, "completion_tokens": 2, "reasoning_tokens": 0},
                          finish_reason="stop", prompt_material="P", output_text="O")
    assert e["prompt_sha256"] == cl.sha256("P") and e["output_sha256"] == cl.sha256("O") and e["output_chars"] == 1
    assert e["response_id"] == "id1" and e["ts_utc"].endswith("+00:00")


def test_record_never_raises(tmp_path, monkeypatch):
    """A ledger failure must not propagate into the API call."""
    bad = tmp_path / "nodir" / "x" / "calls.jsonl"
    monkeypatch.setattr(cl.Path, "mkdir", lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
    assert cl.record(_entry(1), bad) is None


def test_two_ledgers_in_one_process_chain_independently(tmp_path):
    """Two targets written concurrently (as two runs in one process would) must each
    verify and must not receive each other's lines."""
    import threading
    pa, pb = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    def writer(p, tag):
        for i in range(200):
            cl.record({"ts_utc": "t", "api_type": tag, "response_id": f"{tag}-{i}", "usage": {}}, p)
    ta, tb = threading.Thread(target=writer, args=(pa, "A")), threading.Thread(target=writer, args=(pb, "B"))
    ta.start(); tb.start(); ta.join(); tb.join()
    va, vb = cl.verify(pa), cl.verify(pb)
    assert va["ok"] and vb["ok"] and va["n"] == 200 and vb["n"] == 200
    assert all(json.loads(l)["api_type"] == "A" for l in pa.read_text().splitlines())
    assert all(json.loads(l)["api_type"] == "B" for l in pb.read_text().splitlines())


def test_generate_writes_to_explicit_ledger_path_not_env(tmp_path, monkeypatch):
    """The generate() hook must honour ledger_path and ledger_extra, ignoring CALL_LEDGER_PATH."""
    from answer_generation.answer_generator import AnswerGenerator
    env_ledger, explicit = tmp_path / "env.jsonl", tmp_path / "explicit.jsonl"
    monkeypatch.setenv("CALL_LEDGER_PATH", str(env_ledger))
    g = AnswerGenerator(api_type="together", api_key="x", model="stub/model")
    resp = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"1.8": "1"}'),
                                       finish_reason="stop", logprobs=None)],
        usage=types.SimpleNamespace(prompt_tokens=7, completion_tokens=3), id="stub-1", created=1, model="stub/model")
    monkeypatch.setattr(g, "_ensure_client", lambda: object())
    monkeypatch.setattr(g, "_call_with_retry", lambda call_fn, **k: resp)
    out = g.generate(query="q", chunks=[("ctx",)], ledger_path=explicit,
                     ledger_extra={"doc_name": "006", "query_id": "batched", "run_id": "r1"})
    assert out["provenance"]["response_id"] == "stub-1"
    assert explicit.exists() and not env_ledger.exists()
    line = json.loads(explicit.read_text().splitlines()[0])
    assert line["doc_name"] == "006" and line["run_id"] == "r1" and line["usage"]["prompt_tokens"] == 7
    assert cl.verify(explicit)["ok"]
