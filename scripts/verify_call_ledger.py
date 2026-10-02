"""Verify a call ledger's hash chain; optionally re-query OpenRouter for a sample of
generation ids so the provider confirms model/provider/time/tokens from its side.
usage: python scripts/verify_call_ledger.py trial_logs/<folder>/calls.jsonl [--openrouter N]"""
import sys, json, os, random
sys.path.insert(0, "src")
from answer_generation import call_ledger as cl
path = sys.argv[1]
v = cl.verify(path); print(f"chain ok={v['ok']} n={v['n']} head={v['head'][:16]}…" + (f" FIRST BAD LINE {v['first_bad_line']} ({v['why']})" if not v["ok"] else ""))
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
        same = (g.get("model") == r.get("model_resolved")) and (g.get("tokens_prompt") == r["usage"].get("prompt_tokens"))
        print(f"  {r['response_id']}: provider says model={g.get('model')} provider={g.get('provider_name')} created={g.get('created_at')} "
              f"prompt_tokens={g.get('tokens_prompt')} -> {'MATCHES ledger' if same else 'DIFFERS from ledger'}")
