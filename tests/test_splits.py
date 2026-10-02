"""Versioned splits: the v2 lists are reproducible from the seed-42 rule, disjoint,
pinned by sha256, and keep v1's tuning set; materialize --prune never deletes files."""
import hashlib, json, os, pathlib, subprocess, sys
import pytest
ROOT = pathlib.Path(__file__).resolve().parents[1]
V2 = ROOT / "configs" / "splits" / "v2_2026-10-02"
V1 = ROOT / "configs" / "splits" / "v1_2026-07-03"
PIN = {"tuning": "3b8025cf6b222ddeec57d2382891fa1f005b38830efd928447def9ce8b5ab061",
       "calibration": "09a8524a5e86aff408d96c53de82e17742d0a9d74a1766876ad078f04f6fd677",
       "test": "327ba3bcdbef366f11e15319a0b88bf170c02738a4d954779c4cf7af7fafb025",
       "rq2_subset": "60fa7422ffa9624a94cbf3210d584f90de65a936e7cd079af2702105fe90272c"}
pytestmark = pytest.mark.skipif(not V2.exists(), reason="v2 split lists not built")


def ids(p):
    return [l.strip() for l in open(p) if l.strip()]


def test_counts_disjoint_and_pinned():
    sets = {k: ids(V2 / f"{k}_ids.txt") for k in PIN}
    assert {k: len(v) for k, v in sets.items()} == {"tuning": 35, "calibration": 50, "test": 163, "rq2_subset": 60}
    assert not (set(sets["tuning"]) & set(sets["calibration"])) and not (set(sets["tuning"]) & set(sets["test"]))
    assert not (set(sets["calibration"]) & set(sets["test"])) and set(sets["rq2_subset"]) <= set(sets["test"])
    for k, h in PIN.items():
        assert hashlib.sha256(open(V2 / f"{k}_ids.txt", "rb").read()).hexdigest() == h, k
    m = json.load(open(V2 / "splits_manifest.json"))
    assert m["sha256"] == PIN and m["n_eligible"] == 248 and m["seed"] == 42


def test_v2_tuning_equals_v1_tuning_and_legacy_check():
    assert ids(V2 / "tuning_ids.txt") == ids(V1 / "tuning_set_ids.txt")
    assert ids(V2 / "calibration_ids.txt")[:0] == []  # v2 calibration is a superset of v1 calibration:
    assert set(ids(V1 / "calibration_set_ids.txt")) <= set(ids(V2 / "calibration_ids.txt"))
    r = subprocess.run([sys.executable, "scripts/make_splits.py", "--legacy-check"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_make_splits_reproduces_into_tmp(tmp_path):
    r = subprocess.run([sys.executable, "scripts/make_splits.py", "--version", "vtest", "--out-root", str(tmp_path),
                        "--mirror", ""], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for k, h in PIN.items():
        assert hashlib.sha256(open(tmp_path / "vtest" / f"{k}_ids.txt", "rb").read()).hexdigest() == h


def test_materialize_prune_semantics(tmp_path):
    splits = tmp_path / "splits"; splits.mkdir(); (splits / "demo_ids.txt").write_text("006\n")
    dest = tmp_path / "dest" / "demo"; dest.mkdir(parents=True)
    stale = dest / "079_stale.pdf"; stale.symlink_to(ROOT / "data" / "all_articles" / "nonexistent.pdf")
    regular = dest / "123_regular.pdf"; regular.write_bytes(b"%PDF-1.4 not a link")
    r = subprocess.run([sys.executable, "scripts/materialize_splits.py", "--splits-dir", str(splits), "--dest", str(tmp_path / "dest"),
                        "--prune", "--verify"], cwd=ROOT, capture_output=True, text=True)
    assert not stale.exists() and not stale.is_symlink(), "stale symlink must be pruned"
    assert regular.exists(), "regular files must never be deleted"
    assert r.returncode != 0 and "REFUSING" in r.stdout
