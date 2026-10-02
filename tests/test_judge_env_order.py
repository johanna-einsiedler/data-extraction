"""The judge declared in .env must be the judge the scorer executes: .env is loaded
before `evaluate` is imported (import-order defect found 2026-10-01)."""
import os, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

def test_dotenv_loaded_before_evaluate_import():
    src = (ROOT / "src/objective.py").read_text()
    assert src.index("load_dotenv(find_dotenv())") < src.index("from evaluate import")

def test_scorer_sees_env_judge():
    code = ("import os,sys; sys.path.insert(0,'src'); os.environ['SAMPLE_ONLY']='1'; "
            "import objective, evaluate; print(evaluate.JUDGE_BACKEND, evaluate.JUDGE_MODEL_NAME)")
    env = {k: v for k, v in os.environ.items() if k not in ("JUDGE_BACKEND", "EQUIV_JUDGE_MODEL")}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True).stdout.strip().splitlines()[-1]
    declared = dict(re.findall(r"^(JUDGE_BACKEND|EQUIV_JUDGE_MODEL)=(\S+)", (ROOT / ".env").read_text(), re.M))
    assert out == f"{declared['JUDGE_BACKEND']} {declared['EQUIV_JUDGE_MODEL']}", out
