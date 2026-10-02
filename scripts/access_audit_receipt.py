"""Regenerate the test-data access receipts (Hofman B.3):
  parsed_documents_receipt.txt  -- every stem under data/parsed/<Parser>/ with counts
  trial_doc_manifest.txt        -- every distinct doc_name in trial_logs/**/doc_*.json AND runs/**/doc_*.json

usage: pipeline_env/bin/python scripts/access_audit_receipt.py --out <dir> [--expect-absent <ids file> ...]
--expect-absent exits non-zero if any listed id appears in either receipt (evidence of first touch)."""
from __future__ import annotations

import argparse, datetime, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parsed_receipt() -> tuple[str, set[str]]:
    lines, ids = [], set()
    for pdir in sorted((ROOT / "data" / "parsed").glob("*")):
        if not pdir.is_dir():
            continue
        stems = sorted({f.stem.replace(".meta", "") for f in pdir.glob("*") if f.suffix in (".md", ".xml", ".txt", ".json")})
        for s in stems:
            m = re.match(r"^(\d+)", s)
            if m:
                ids.add(m.group(1))
        lines.append(f"{pdir.name}: {len(stems)} documents")
        lines += [f"  {s}" for s in stems]
    return "\n".join(lines), ids


def trial_receipt() -> tuple[str, set[str]]:
    seen: dict[str, set[str]] = {}
    for root in (ROOT / "trial_logs", ROOT / "runs"):
        for f in root.glob("**/doc_[0-9]*.json"):
            try:
                d = json.load(open(f)); name = str(d.get("doc_name"))
            except Exception:
                continue
            seen.setdefault(name, set()).add(str(f.parent.relative_to(ROOT)))
    lines = [f"{k}: {len(v)} folders" for k, v in sorted(seen.items())]
    return "\n".join(lines), set(seen)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--expect-absent", nargs="*", default=[])
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    p_txt, p_ids = parsed_receipt(); t_txt, t_ids = trial_receipt()
    (out / "parsed_documents_receipt.txt").write_text(f"# generated {stamp} by scripts/access_audit_receipt.py\n{p_txt}\n")
    (out / "trial_doc_manifest.txt").write_text(f"# generated {stamp} by scripts/access_audit_receipt.py\n{t_txt}\n")
    print(f"parsed ids: {len(p_ids)}; doc_names ever processed: {len(t_ids)}")
    bad = {}
    for f in a.expect_absent:
        ids = {l.strip() for l in open(f) if l.strip()}
        hit = sorted((ids & p_ids) | (ids & t_ids))
        if hit:
            bad[f] = hit
    if bad:
        print("EXPECTED-ABSENT IDS PRESENT:", json.dumps(bad)); sys.exit(1)
    if a.expect_absent:
        print("expect-absent check passed")


if __name__ == "__main__":
    main()
