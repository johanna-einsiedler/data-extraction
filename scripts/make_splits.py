"""Materialise a split VERSION as tracked id lists + manifest (no PDFs touched).

usage: pipeline_env/bin/python scripts/make_splits.py --version v2_2026-10-02 [--sizes 35,50] [--rq2-size 60]
         [--rq2-seed 42] [--out-root configs/splits] [--mirror data/dataset_splits] [--legacy-check]

v2 (preregistered re-split): slices [:35] / [35:85] / [85:] of the seed-42 permutation of the
248 eligible (>= 2-coder) studies -> tuning 35 (identical to v1), calibration 50, test 163;
plus a 60-paper RQ2 subset drawn from the test ids with its own stated seed.
--legacy-check recomputes v1 ([:35]/[35:50]/[50:]) and compares with configs/splits/v1_2026-07-03/.
"""
from __future__ import annotations

import argparse, datetime, hashlib, json, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "dataset_splits"))
from create_splits import CSV_PATH, SEED, eligible_ids, permute  # noqa: E402


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_list(path: Path, ids: list[str]) -> str:
    text = "\n".join(ids) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
    return sha(text)


def build(version: str, sizes: tuple[int, int], rq2_size: int, rq2_seed: int, out_root: Path, mirror: Path | None):
    elig = eligible_ids(); perm = permute(elig, SEED)
    n_t, n_c = sizes
    sets = {"tuning": sorted(perm[:n_t]), "calibration": sorted(perm[n_t:n_t + n_c]), "test": sorted(perm[n_t + n_c:])}
    rng = np.random.default_rng(rq2_seed)
    sets["rq2_subset"] = sorted(rng.choice(sets["test"], size=rq2_size, replace=False).tolist())
    out = out_root / version; shas = {}
    for name, ids in sets.items():
        shas[name] = write_list(out / f"{name}_ids.txt", ids)
        if mirror:
            write_list(mirror / version / f"{name}_ids.txt", ids)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    manifest = {"version": version, "seed": SEED, "rq2_seed": rq2_seed, "rq2_size": rq2_size,
                "eligibility_rule": "study_ID coded by >= 2 raters in 01_study_data_long_raw.csv",
                "slices": {"tuning": f"[:{n_t}]", "calibration": f"[{n_t}:{n_t + n_c}]", "test": f"[{n_t + n_c}:]"},
                "source_csv": str(CSV_PATH.relative_to(ROOT)), "source_csv_sha256": hashlib.sha256(open(CSV_PATH, "rb").read()).hexdigest(),
                "n_eligible": len(elig), "counts": {k: len(v) for k, v in sets.items()}, "sha256": shas,
                "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "git_commit": commit}
    json.dump(manifest, open(out / "splits_manifest.json", "w"), indent=1)
    if mirror:
        json.dump(manifest, open(mirror / version / "splits_manifest.json", "w"), indent=1)
    return sets, manifest


def legacy_check(v1_dir: Path) -> bool:
    elig = eligible_ids(); perm = permute(elig, SEED)
    v1 = {"tuning_set_ids.txt": sorted(perm[:35]), "calibration_set_ids.txt": sorted(perm[35:50]), "test_set_ids.txt": sorted(perm[50:])}
    ok = True
    for fname, ids in v1.items():
        p = v1_dir / fname
        got = [l.strip() for l in open(p) if l.strip()] if p.exists() else None
        same = got == ids
        print(f"  legacy {fname}: {'OK' if same else 'MISMATCH'} ({len(ids)} ids)")
        ok &= same
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v3_2026-10-30"); ap.add_argument("--sizes", default="35,63")
    ap.add_argument("--rq2-size", type=int, default=60); ap.add_argument("--rq2-seed", type=int, default=42)
    ap.add_argument("--out-root", default="configs/splits"); ap.add_argument("--mirror", default="data/dataset_splits")
    ap.add_argument("--legacy-check", action="store_true"); ap.add_argument("--v1-dir", default="configs/splits/v1_2026-07-03")
    a = ap.parse_args()
    if a.legacy_check:
        sys.exit(0 if legacy_check(ROOT / a.v1_dir) else 1)
    n_t, n_c = (int(x) for x in a.sizes.split(","))
    sets, m = build(a.version, (n_t, n_c), a.rq2_size, a.rq2_seed, ROOT / a.out_root, (ROOT / a.mirror) if a.mirror else None)
    print(json.dumps({k: v for k, v in m.items() if k != "sha256"}, indent=1)); print("sha256:", m["sha256"])


if __name__ == "__main__":
    main()
