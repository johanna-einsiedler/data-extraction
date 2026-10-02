"""Physically separate the dataset splits into folders.

Reads the three id lists in data/dataset_splits/ and populates
  data/tuning/       (35 papers)
  data/calibration/  (15 papers)
  data/test/         (198 papers)
by symlinking each matching ``NNN_*.pdf`` from the master data/all_articles/.

Symlinks (relative) avoid duplicating ~250 MB and keep data/all_articles/ as the
single source of truth. The parsed/chunk/embedding caches are keyed on the file
stem (NNN), so they work regardless of which folder the PDF is read from.

Enforces the held-out-test discipline: the tuning study only ever globs
data/tuning/, so test papers cannot leak into a tuning run.

Usage:
  python scripts/materialize_splits.py            # symlink (default)
  python scripts/materialize_splits.py --copy     # copy files instead
"""

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "data" / "all_articles"
SPLIT_IDS = ROOT / "data" / "dataset_splits"
SPLITS = {
    "tuning": "tuning_set_ids.txt",
    "calibration": "calibration_set_ids.txt",
    "test": "test_set_ids.txt",
}


def _pdf_for(study_id: str) -> Path | None:
    """Find the master PDF whose leading digits match study_id (e.g. '006')."""
    matches = sorted(MASTER.glob(f"{study_id}_*.pdf")) + sorted(MASTER.glob(f"{study_id}.pdf"))
    return matches[0] if matches else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    # 2026-10: versioned splits. Defaults reproduce the historical behaviour
    # (v1 lists in data/dataset_splits -> data/{tuning,calibration,test}).
    ap.add_argument("--splits-dir", default=None, help="folder with <set>_ids.txt (e.g. configs/splits/v2_2026-10-02)")
    ap.add_argument("--dest", default=None, help="destination root (e.g. data/splits_v2); default data/")
    ap.add_argument("--sets", default=None, help="comma list of sets to materialise (default: all in SPLITS or *_ids.txt)")
    ap.add_argument("--prune", action="store_true", help="remove symlinks in the destination that are not in the id list (never regular files)")
    ap.add_argument("--verify", action="store_true", help="exit non-zero unless each destination holds exactly the ids")
    args = ap.parse_args()

    if not MASTER.exists():
        raise SystemExit(f"Master PDF dir not found: {MASTER}")

    splits = dict(SPLITS)
    ids_root = SPLIT_IDS
    if args.splits_dir:
        ids_root = (ROOT / args.splits_dir) if not Path(args.splits_dir).is_absolute() else Path(args.splits_dir)
        splits = {p.name[:-len("_ids.txt")]: p.name for p in sorted(ids_root.glob("*_ids.txt"))}
    if args.sets:
        want = set(args.sets.split(","))
        splits = {k: v for k, v in splits.items() if k in want}
    dest_root = (ROOT / args.dest) if args.dest else (ROOT / "data")
    bad = False
    for split, id_file in splits.items():
        ids = [
            x.strip()
            for x in (ids_root / id_file).read_text().split()
            if x.strip()
        ]
        out_dir = dest_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.prune:
            keep = set(ids)
            for f in out_dir.glob("*.pdf"):
                sid = f.name.split("_")[0].split(".")[0]
                if sid not in keep:
                    if f.is_symlink():
                        f.unlink(); print(f"  pruned stale link {f.name}")
                    else:
                        print(f"  REFUSING to delete regular file {f.name}"); bad = True

        linked, missing = 0, []
        for sid in ids:
            src = _pdf_for(sid)
            if src is None:
                missing.append(sid)
                continue
            dst = out_dir / src.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()  # idempotent refresh
            if args.copy:
                shutil.copy2(src, dst)
            else:
                # relative symlink: data/<split>/NNN.pdf -> ../all_articles/NNN.pdf
                os.symlink(Path("..") / "all_articles" / src.name, dst)
            linked += 1

        print(f"{split:12s}: {linked}/{len(ids)} {'copied' if args.copy else 'linked'}"
              + (f"  MISSING {missing}" if missing else ""))
        if args.verify:
            have = sorted({f.name.split("_")[0].split(".")[0] for f in out_dir.glob("*.pdf")})
            if have != sorted(ids):
                print(f"  VERIFY FAIL {split}: {len(have)} present vs {len(ids)} expected"); bad = True
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
