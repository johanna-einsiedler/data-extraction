"""src/objective.py must refuse to run without OPTUNA_DB (the old default pointed at
the frozen Phase-1 DB, which still holds WAITING trials)."""
import os, pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]

def test_objective_refuses_without_optuna_db():
    env = {k: v for k, v in os.environ.items() if k != "OPTUNA_DB"}
    env["SAMPLE_ONLY"] = "1"
    r = subprocess.run([sys.executable, "src/objective.py", "1", "--models=none"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode != 0
    assert "OPTUNA_DB is required" in (r.stderr + r.stdout)
