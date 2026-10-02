#!/bin/bash
# Detached runner for a Phase-2 study db.  usage: launch.sh <db-file> <n-trials>
# setsid so the run survives this session ending; caffeinate so the machine
# does not idle-sleep under it. Extra env (e.g. SMOKE_MAX_DOCS=2) passes through.
cd "$(dirname "$0")/../.."
ROOT="$PWD"; DB="$ROOT/stage2_analyses/phase2_study/$1"; N="$2"
mkdir -p stage2_analyses/phase2_study/logs
LOG="$ROOT/stage2_analyses/phase2_study/logs/$(basename "$1" .db)_$(date +%Y%m%d_%H%M%S).log"
exec pipeline_env/bin/python -c '
import os, sys, subprocess
os.setsid()
root, db, n, log = sys.argv[1:5]
env = dict(os.environ, OPTUNA_DB="sqlite:///" + db, PYTHONUNBUFFERED="1")
env.setdefault("PAPER_CONCURRENCY", "4"); env.setdefault("N_JOBS", "1")
with open(log, "ab") as fh:
    p = subprocess.Popen(["caffeinate", "-i", root + "/pipeline_env/bin/python",
                          "src/objective.py", n, "--models=none"],
                         cwd=root, env=env, stdout=fh, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL)
print(p.pid, log)
' "$ROOT" "$DB" "$N" "$LOG"
