"""The preregistered stopping rule (15 consecutive non-improving completed trials),
stateless across workers and restarts, blind to pruned trials."""
import os, pathlib, sys
os.environ.setdefault("SAMPLE_ONLY", "1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import optuna
from optuna.trial import TrialState, create_trial
from objective import PatienceCallback

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _study(values, best_at=None):
    st = optuna.create_study(direction="maximize")
    for i, v in enumerate(values):
        st.add_trial(create_trial(value=v, state=TrialState.COMPLETE))
    return st


def test_streak_counts_after_the_best():
    cb = PatienceCallback(patience=15, min_trials=5)
    st = _study([0.5, 0.9] + [0.6] * 7)
    streak, n, best = cb.streak(st)
    assert (streak, n, best) == (7, 9, 0.9)


def test_improvement_resets_and_stop_fires_at_15():
    cb = PatienceCallback(patience=15, min_trials=5)
    st = _study([0.9] + [0.5] * 14)
    assert cb.streak(st)[0] == 14
    st.add_trial(create_trial(value=0.5, state=TrialState.COMPLETE))
    fired = []
    st.stop = lambda: fired.append(True)   # stop() only takes effect inside optimize()
    cb(st, st.trials[-1])
    assert fired  # fired at exactly 15
    # one short of the bar must NOT fire
    st2 = _study([0.9] + [0.5] * 13)
    fired2 = []
    st2.stop = lambda: fired2.append(True)
    cb(st2, st2.trials[-1])
    assert not fired2


def test_pruned_and_failed_do_not_run_down_the_clock():
    cb = PatienceCallback(patience=15, min_trials=1)
    st = _study([0.9] + [0.5] * 5)
    for _ in range(20):
        st.add_trial(create_trial(state=TrialState.PRUNED))
    assert cb.streak(st)[0] == 5


def test_fresh_instance_sees_full_study_history():
    # a worker restart constructs a new callback: streak must survive it
    st = _study([0.9] + [0.5] * 12)
    assert PatienceCallback().streak(st)[0] == 12


def test_from_number_ignores_seed_tail():
    # seeds 0..9 with best at 3 and six non-improving seeds after it;
    # a stage starting at 10 must begin with streak 0, bar intact
    st = _study([0.5, 0.5, 0.5, 0.9] + [0.6] * 6)
    cb = PatienceCallback(patience=15, min_trials=1, from_number=10)
    assert cb.streak(st) == (0, 10, 0.9)
    st.add_trial(create_trial(value=0.7, state=TrialState.COMPLETE))  # number 10
    assert cb.streak(st)[0] == 1


def test_min_trials_follows_startup_budget(monkeypatch):
    """A seeded study must be able to stop: the floor is the sampler's startup
    budget, not a hard 100 (consumer search logged 18/15 and kept running)."""
    monkeypatch.delenv("PATIENCE_MIN_TRIALS", raising=False)
    monkeypatch.setenv("TPE_N_STARTUP", "34")
    cb = PatienceCallback(patience=15)
    assert cb.min_trials == 34
    st = _study([0.9] + [0.5] * 40)          # 41 completions (>= the 34 floor), streak 40
    fired = []
    st.stop = lambda: fired.append(True)
    cb(st, st.trials[-1])
    assert fired, "seeded study with streak>patience must stop"
    monkeypatch.delenv("TPE_N_STARTUP")
    assert PatienceCallback(patience=15).min_trials == 100   # unseeded default
