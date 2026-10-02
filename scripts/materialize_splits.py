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
    args = ap.parse_args()

    if not MASTER.exists():
        raise SystemExit(f"Master PDF dir not found: {MASTER}")

    for split, id_file in SPLITS.items():
        ids = [
            x.strip()
            for x in (SPLIT_IDS / id_file).read_text().split()
            if x.strip()
        ]
        out_dir = ROOT / "data" / split
        out_dir.mkdir(parents=True, exist_ok=True)

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


if __name__ == "__main__":
    main()
