"""Verify a call ledger's hash chain; optionally re-query OpenRouter for a sample of
generation ids so the provider confirms model/provider/time/tokens from its side.
usage: python scripts/verify_call_ledger.py trial_logs/<folder>/calls.jsonl [--openrouter N]"""
import sys, json, os, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from answer_generation import call_ledger as cl
path = sys.argv[1]
if Path(path).is_dir():
    path = str(Path(path) / "calls.jsonl")
v = cl.verify(path)
mf = Path(path).parent / "manifest.json"
if mf.exists():
    m = json.load(open(mf)).get("ledger", {})
    print(f"manifest ledger: n={m.get('n')} head={str(m.get('head'))[:16]}… ({'MATCH' if m.get('head') == v['head'] and m.get('n') == v['n'] else 'DIFFERS'})")
print(f"chain ok={v['ok']} n={v['n']} head={v['head'][:16]}…" + (f" FIRST BAD LINE {v['first_bad_line']} ({v['why']})" if not v["ok"] else ""))
if "--openrouter" in sys.argv:
    import requests
    from dotenv import load_dotenv; load_dotenv(".env")
    k = int(sys.argv[sys.argv.index("--openrouter") + 1]) if len(sys.argv) > sys.argv.index("--openrouter") + 1 else 5
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if r.get("api_type") == "openrouter" and r.get("response_id")]
    for r in random.sample(rows, min(k, len(rows))):
        g = {}
        for _ in range(12):   # OpenRouter's generation record appears ~1 min after the call
            resp = requests.get("https://openrouter.ai/api/v1/generation", params={"id": r["response_id"]},
                                headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]}, timeout=30)
            if resp.status_code == 200:
                g = resp.json().get("data", {}); break
            import time; time.sleep(10)
        lm = str(r.get("model_resolved") or "").lower(); pm = str(g.get("model") or "").lower()
        fam = bool(lm and pm) and (pm.startswith(lm) or lm.startswith(pm) or pm.split("-20")[0] == lm.split("-20")[0])
        lp = r["usage"].get("prompt_tokens") or 0
        tok_ok = any(abs((g.get(k) or 0) - lp) <= max(5, 0.02 * lp) for k in ("tokens_prompt", "native_tokens_prompt"))
        verdict = "MATCHES ledger (model family + tokens)" if (fam and tok_ok) else ("model family ok; token counts differ" if fam else "DIFFERS from ledger")
        print(f"  {r['response_id']}: provider says model={g.get('model')} provider={g.get('provider_name')} created={g.get('created_at')} "
              f"tokens_prompt={g.get('tokens_prompt')} native={g.get('native_tokens_prompt')} | ledger model={r.get('model_resolved')} prompt_tokens={lp} -> {verdict}")
