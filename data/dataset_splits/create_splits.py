"""
Split double-coded studies into tuning / calibration / test sets.

Reads the finalized per-rater long file (01_study_data_long_raw), selects study
IDs with >= 2 coders, then randomly partitions them (seed 42) into:
  - tuning_set_ids.txt      (35 IDs)
  - calibration_set_ids.txt (15 IDs)
  - test_set_ids.txt        (remainder; 198 IDs)

Note: the eligible >=2-coder set is identical to the one previously derived from
mar_03.csv, so re-running reproduces the exact same, already-committed splits.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "true_labels"
    / "01_study_data_long_raw.csv"
)
OUT_DIR = Path(__file__).resolve().parent


def eligible_ids(csv_path: Path = CSV_PATH) -> list[str]:
    """Study IDs coded by >= 2 raters, sorted (the preregistered eligibility rule)."""
    df = pd.read_csv(csv_path)
    coding_counts = df.groupby("study_ID")["rater_ID"].nunique()
    return sorted(str(i) for i in coding_counts[coding_counts >= 2].index.tolist())


def permute(ids: list[str], seed: int = SEED) -> list[str]:
    """The seed-42 permutation every split version slices (v1: [:35]/[35:50]/[50:];
    v2 2026-10: [:35]/[35:85]/[85:])."""
    rng = np.random.default_rng(seed)
    return rng.permutation(ids).tolist()


def main():
    eligible = eligible_ids()
    print(f"Studies with >= 2 coders: {len(eligible)}")
    shuffled = permute(eligible)

    tuning_ids = sorted(shuffled[:35])
    calibration_ids = sorted(shuffled[35:50])
    test_ids = sorted(shuffled[50:])

    print(f"Tuning set:      {len(tuning_ids)}")
    print(f"Calibration set: {len(calibration_ids)}")
    print(f"Test set:        {len(test_ids)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, ids in [
        ("tuning_set_ids.txt", tuning_ids),
        ("calibration_set_ids.txt", calibration_ids),
        ("test_set_ids.txt", test_ids),
    ]:
        path = OUT_DIR / name
        path.write_text("\n".join(str(i) for i in ids) + "\n")
        print(f"  Wrote {path}")


if __name__ == "__main__":
    main()
