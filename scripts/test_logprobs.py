"""Probe every model in the roster for usable answer-token logprobs.

Sends one tiny generation per model (run in non-thinking mode where relevant)
and records whether populated logprobs come back. Writes a CSV that can be fed
back into the roster's ``exposes_logprobs`` flag.

  python scripts/test_logprobs.py                       # probe whole roster
  python scripts/test_logprobs.py --only Qwen/... gpt-4.1-2025-04-14
  python scripts/test_logprobs.py --out data/logprobs_probe.csv

Reasoning/thinking models frequently return null logprobs on the reasoning
stream; this is the single biggest risk to the RQ2 calibration design, so probe
before committing a model to the calibration analyses.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import find_dotenv, load_dotenv  # noqa: E402

load_dotenv(find_dotenv())

import pandas as pd  # noqa: E402

from registry import LLM_META_MAP, LLMS  # noqa: E402

PROMPT = "What is 2+2? Answer with just the number.\n\nContext: {context}"


def probe(model_id: str) -> dict:
    gen = LLMS[model_id]
    try:
        result = gen.generate(query=PROMPT, chunks=[("irrelevant context",)],
                              return_logprobs=True)
        lp = result.get("logprobs")
        has = isinstance(lp, dict) and len(lp) > 0
        return {"id": model_id, "has_logprobs": has,
                "text": str(result.get("text", ""))[:30], "error": ""}
    except Exception as e:  # noqa: BLE001
        return {"id": model_id, "has_logprobs": False, "text": "",
                "error": str(e)[:160]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="probe only these model ids (default: whole roster)")
    ap.add_argument("--out", default="data/logprobs_probe.csv")
    args = ap.parse_args()

    ids = args.only or list(LLM_META_MAP.keys())
    rows = []
    for mid in ids:
        if mid not in LLMS:
            print(f"SKIP  {mid} (not in roster)")
            continue
        r = probe(mid)
        rows.append(r)
        flag = "OK  " if r["has_logprobs"] else "NONE"
        extra = r["error"] or r["text"]
        print(f"{flag} logprobs | {mid}: {extra}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    n_ok = int(df["has_logprobs"].sum()) if len(df) else 0
    print(f"\nWrote {args.out}  ({n_ok}/{len(df)} expose logprobs)")


if __name__ == "__main__":
    main()
