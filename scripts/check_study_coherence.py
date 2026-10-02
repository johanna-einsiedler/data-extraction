"""Read-only verifier of the study record (the four Optuna DBs vs their raw artefacts).

usage: pipeline_env/bin/python scripts/check_study_coherence.py
         [--db all|phase2|open_tpe|consumer_tpe] [--expected scripts/study_expected.json]
         [--json OUT.json] [--pre-aug-write] [--skip-batch-api]

Every invariant prints PASS/FAIL with a detail line; exit code 1 on any FAIL.
The script hashes every DB before and after and fails itself if one changed
(proof that a run of this script writes nothing). --pre-aug-write relaxes the
two invariants that are expected to fail before the Aug-batch seed copies are
re-scored (I6/I7/I11). Expectations live in scripts/study_expected.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _study_record import (AUG_BATCH, DB_PATHS, JUDGE_GPTOSS, JUDGE_PHI3, PHASE2_DBS, ROOT,  # noqa: E402
                           TRIAL_LOGS, db_md5, folder_stats, load_studies, matched_folder,
                           resolve_source, trial_by_number, tuning_set, union_folder_stats,
                           value_chain, jsonio)

os.environ.setdefault("SAMPLE_ONLY", "1")
sys.path.insert(0, str(ROOT / "src"))
from objective import PatienceCallback  # noqa: E402

TOL = 1e-9


class Ctx:
    def __init__(self, exp, dbs, pre_aug, skip_batch):
        self.exp, self.dbs, self.pre_aug, self.skip_batch = exp, dbs, pre_aug, skip_batch
        self.studies = load_studies(("phase1",) + tuple(dbs))
        self.tuning = tuning_set()
        self.results = []
        self._fs = {}
        self._cells = None
        self._resc = {}

    # ---- caches ----
    def fstats(self, name, t):
        key = (name, t.number)
        if key not in self._fs:
            ua = t.user_attrs
            agg = self.exp["dbs"][name].get("aggregated", {}).get(str(t.number))
            if agg:
                self._fs[key] = union_folder_stats(agg, self.tuning)
            else:
                origin, num, params = resolve_source(name, t, self.studies)
                f = matched_folder(num, params)
                self._fs[key] = folder_stats(f, self.tuning) if f is not None else None
                if self._fs[key] is not None:
                    self._fs[key]["origin"] = (origin, num)
        return self._fs[key]

    def unified_cells(self):
        if self._cells is None:
            self._cells = pd.read_parquet(ROOT / "analysis/cache/unified_rescore_cells.parquet")
        return self._cells

    def resc(self, p1):
        if p1 not in self._resc:
            p = ROOT / self.exp["aug_batch"]["parquet_dir"] / f"{p1}.parquet"
            self._resc[p1] = pd.read_parquet(p) if p.exists() else None
        return self._resc[p1]

    def per_paper_means(self, name, t):
        """Series doc_name -> mean accuracy on the unified basis for this trial."""
        origin, num, _ = resolve_source(name, t, self.studies)
        if origin == "phase1" and num in AUG_BATCH:
            d = self.resc(num)
            if d is not None:
                return d.groupby("doc_name").accuracy_unified.mean()
        p2num = None
        if name == "phase2":
            p2num = t.number
        elif "seeded_from_trial" in t.user_attrs:
            p2num = int(t.user_attrs["seeded_from_trial"])
        if p2num is not None:
            c = self.unified_cells()
            sub = c[c.p2_number == p2num]
            if len(sub):
                return sub.groupby("doc_name").accuracy_unified.mean()
        fs = self.fstats(name, t)
        f = TRIAL_LOGS / fs["folder"]
        ts = jsonio.load(f / "trial_summary.json")
        rows = [(str(c.get("doc_name")), float(c["accuracy"])) for c in ts["per_query_doc_results"]
                if c.get("accuracy") is not None and c.get("doc_name") is not None]
        return pd.DataFrame(rows, columns=["doc_name", "acc"]).groupby("doc_name").acc.mean()

    def add(self, inv, ok, detail=""):
        self.results.append({"inv": inv, "ok": bool(ok), "detail": detail})


def complete(study):
    return [t for t in study.trials if t.state.name == "COMPLETE" and t.value is not None]


# ---------------------------------------------------------------- invariants
def inv_I1(ctx):
    for name in ctx.dbs:
        st = ctx.studies[name]; e = ctx.exp["dbs"][name]
        bad = [t.number for t in st.trials if t.state.name in ("RUNNING", "WAITING")]
        ok = not bad and len(st.trials) == e["n_trials"] and len(complete(st)) == e["n_complete"]
        ctx.add("I1 states", ok, f"{name}: trials={len(st.trials)} complete={len(complete(st))} running/waiting={bad}")
    p1 = ctx.studies["phase1"]; counts = defaultdict(int)
    for t in p1.trials: counts[t.state.name] += 1
    e = ctx.exp["phase1_db"]
    ok = dict(counts) == e["states"] and db_md5(DB_PATHS["phase1"]) == e["md5"]
    ctx.add("I1 states", ok, f"phase1: {dict(counts)} md5={db_md5(DB_PATHS['phase1'])}")


def _batched_transform(p1params: dict, copy_params: dict) -> list[str]:
    """Differences between a Phase-1 trial's params and its phase2 copy beyond the
    documented copy transforms."""
    diffs = []
    for k, v in p1params.items():
        if k in ("question_mode", "default_prompt_type", "use_default_prompt"):
            continue
        if k in copy_params and copy_params[k] != v:
            diffs.append(f"{k}: p1={v!r} copy={copy_params[k]!r}")
    return diffs


def inv_I2(ctx):
    for name in ctx.dbs:
        st = ctx.studies[name]; e = ctx.exp["dbs"][name]; n_seed = 0; bad = []
        for t in complete(st):
            ua = t.user_attrs
            if name != "phase2" and "seeded_from_trial" in ua:
                n_seed += 1
                src = trial_by_number(ctx.studies["phase2"], int(ua["seeded_from_trial"]))
                if src is None or abs(src.value - t.value) > 1e-12:
                    bad.append(f"#{t.number}<-p2#{ua['seeded_from_trial']} value")
                elif any(src.params.get(k) != v for k, v in t.params.items()):
                    bad.append(f"#{t.number}<-p2#{ua['seeded_from_trial']} params")
            elif name == "phase2" and "phase1_trial" in ua:
                n_seed += 1
                src = trial_by_number(ctx.studies["phase1"], int(ua["phase1_trial"]))
                chain = value_chain(t)
                if src is None or abs(src.value - chain[0]) > 1e-12:
                    bad.append(f"#{t.number}<-p1#{ua['phase1_trial']} value_chain[0]")
                else:
                    d = _batched_transform(dict(src.params), dict(t.params))
                    if d: bad.append(f"#{t.number}<-p1#{ua['phase1_trial']} {d}")
                    if t.params.get("mode") == "full" and "batched_prompt_type" in t.params:
                        if ua.get("phase1_inert_default_prompt_type") != src.params.get("default_prompt_type"):
                            bad.append(f"#{t.number} phase1_inert_default_prompt_type")
        ok = not bad and n_seed == e["n_seeds"]
        ctx.add("I2 seed_copies_equal_sources", ok, f"{name}: seeds={n_seed}/{e['n_seeds']} bad={bad[:6]}")


def inv_I3(ctx):
    for name in ctx.dbs:
        st = ctx.studies[name]; bad = []
        for t in complete(st):
            fs = ctx.fstats(name, t)
            if fs is None:
                bad.append(f"#{t.number}: no unique folder"); continue
            ua = t.user_attrs
            h10, md5 = ua.get("config_hash10"), ua.get("config_md5")
            if fs["config_md5"] is None:
                continue  # union folder (#123): no single parameters hash
            if h10 != fs["config_md5"][:10] or md5 != fs["config_md5"]:
                bad.append(f"#{t.number}: attrs {h10}/{str(md5)[:10]} vs folder {fs['config_md5'][:10]}")
            if not fs["folder"].startswith(fs["config_md5"][:10]):
                bad.append(f"#{t.number}: folder prefix {fs['folder'][:10]} != md5 {fs['config_md5'][:10]}")
        ctx.add("I3 unique_folder_and_hash_attrs", not bad, f"{name}: bad={bad[:5]} ({len(bad)})")


def inv_I4(ctx):
    cells = ctx.unified_cells()
    for name in ctx.dbs:
        st = ctx.studies[name]; bad = []
        for t in complete(st):
            ua = t.user_attrs; fs = ctx.fstats(name, t)
            if fs is None or fs["mean"] is None:
                bad.append(f"#{t.number}: no folder mean"); continue
            if "value_imputation" in ua:
                # old basis: raw 31-paper mean + 0.0017 gap == value_before_unified_rescore;
                # unified basis: value == unified raw31 + its own gap (parquet, if present)
                raw = ua.get("partial_raw_mean_31papers"); vb = ua.get("value_before_unified_rescore")
                if raw is None or vb is None or abs(float(vb) - (float(raw) + 0.0017)) > 1e-4:
                    bad.append(f"#{t.number}: old-basis imputation chain mismatch")
                sub = cells[cells.p2_number == t.number] if name == "phase2" else cells.iloc[0:0]
                if len(sub):
                    gap = t.value - sub.accuracy_unified.mean()
                    if not (0.0005 <= gap <= 0.0035):
                        bad.append(f"#{t.number}: unified gap {gap:+.4f} outside the documented range")
                continue
            chain = value_chain(t)
            if abs(fs["mean"] - chain[0]) > 1e-6:
                bad.append(f"#{t.number}: folder {fs['mean']:.6f} != chain[0] {chain[0]:.6f}")
            if "unified_rescore" in ua:
                origin, num, _ = resolve_source(name, t, ctx.studies)
                is_aug = origin == "phase1" and num in AUG_BATCH
                if is_aug and ctx.pre_aug:
                    continue
                if int(ua.get("unified_rescore_cells", 0)) <= 0:
                    bad.append(f"#{t.number}: unified_rescore with 0 cells")
                if "value_before_unified_rescore" not in ua:
                    bad.append(f"#{t.number}: no value_before_unified_rescore")
                if name == "phase2" or "seeded_from_trial" in ua:
                    p2num = t.number if name == "phase2" else int(ua["seeded_from_trial"])
                    if is_aug:
                        d = ctx.resc(num); m = d.accuracy_unified.mean() if d is not None else None
                    else:
                        sub = cells[cells.p2_number == p2num]; m = sub.accuracy_unified.mean() if len(sub) else None
                    if m is None or abs(m - t.value) > 1e-6:
                        bad.append(f"#{t.number}: value {t.value:.6f} not reproduced by cells ({m})")
            else:
                if abs(fs["mean"] - t.value) > 1e-6:
                    bad.append(f"#{t.number}: no rescore attr but value != folder mean")
        ctx.add("I4 value_vs_folder", not bad, f"{name}: bad={bad[:5]} ({len(bad)})")


def inv_I5(ctx):
    n_tune = ctx.exp["tuning_papers"]
    for name in ctx.dbs:
        st = ctx.studies[name]; e = ctx.exp["dbs"][name]; bad = []
        for t in complete(st):
            ua = t.user_attrs; fs = ctx.fstats(name, t)
            if fs is None:
                continue
            for k in ("n_papers", "n_cells", "n_doc_files", "missing_docs", "source_folder"):
                if k not in ua:
                    bad.append(f"#{t.number}: missing {k}"); break
            else:
                if ua["n_papers"] != fs["n_papers"] or ua["n_cells"] != fs["n_cells"] or sorted(ua["missing_docs"]) != fs["missing_docs"]:
                    bad.append(f"#{t.number}: attrs disagree with folder")
                partial = bool(ua.get("partial_completion"))
                if partial != (fs["n_papers"] < n_tune):
                    bad.append(f"#{t.number}: partial_completion={partial} but n_papers={fs['n_papers']}")
                origin, num, _ = resolve_source(name, t, ctx.studies)
                if origin == name:
                    expct = e["partial"].get(str(t.number))
                    if isinstance(expct, list) and sorted(expct) != fs["missing_docs"]:
                        bad.append(f"#{t.number}: expected missing {expct} got {fs['missing_docs']}")
                    if expct is None and fs["n_papers"] < n_tune:
                        bad.append(f"#{t.number}: partial but not in expected.partial ({fs['missing_docs']})")
        ctx.add("I5 completeness_attrs", not bad, f"{name}: bad={bad[:5]} ({len(bad)})")


def inv_I6(ctx):
    for name in ctx.dbs:
        st = ctx.studies[name]; bad = []
        for t in complete(st):
            ua = t.user_attrs
            if "unified_rescore" in ua and int(ua.get("unified_rescore_cells", 0)) <= 0:
                origin, num, _ = resolve_source(name, t, ctx.studies)
                if ctx.pre_aug and origin == "phase1" and num in AUG_BATCH:
                    continue
                bad.append(t.number)
        ctx.add("I6 unified_rescore_cells_positive", not bad, f"{name}: bad={bad}")


def inv_I7(ctx):
    for name in ctx.dbs:
        st = ctx.studies[name]; bad = []
        for t in complete(st):
            ua = t.user_attrs; origin, num, _ = resolve_source(name, t, ctx.studies)
            is_aug_copy = origin == "phase1" and num in AUG_BATCH
            want = JUDGE_GPTOSS if (ctx.pre_aug and is_aug_copy) else JUDGE_PHI3
            if ua.get("judge") != want:
                bad.append(f"#{t.number}: judge={ua.get('judge')} want={want}")
            elif is_aug_copy and not ctx.pre_aug and not all(k in ua for k in ("judge_before", "rescore_note", "rescore_date")):
                bad.append(f"#{t.number}: missing rescore provenance attrs")
        ctx.add("I7 judge_attr", not bad, f"{name}: bad={bad[:5]} ({len(bad)})")


def inv_I8(ctx):
    for name in ctx.dbs:
        st = ctx.studies[name]; e = ctx.exp["dbs"][name]["patience"]
        cb = PatienceCallback(patience=e["patience"], min_trials=0, from_number=e["from_number"])
        streak, n_complete, best_val = cb.streak(st)
        done = sorted(complete(st), key=lambda t: (t.datetime_complete or t.datetime_start))
        best_i = next(i for i, t in enumerate(done) if abs(t.value - best_val) <= 1e-12)
        after = [t for t in done[best_i + 1:] if t.number >= e["from_number"]]
        overshoot = [t.number for t in after[e["patience"]:]]
        ok = streak == e["expected_streak"] and overshoot == e["overshoot"]
        gate = e.get("gate_bug_attr_trial")
        if gate is not None:
            g = trial_by_number(st, gate)
            ok = ok and g is not None and "post_certification" in g.user_attrs
        ctx.add("I8 stopping_rule", ok, f"{name}: streak={streak} (exp {e['expected_streak']}) overshoot={overshoot} (exp {e['overshoot']}) best={best_val:.4f}")


def inv_I9(ctx):
    for name in ctx.dbs:
        st = ctx.studies[name]; e = ctx.exp["dbs"][name]["best"]
        b = max(complete(st), key=lambda t: t.value)
        fs = ctx.fstats(name, b)
        ok = b.number == e["number"] and abs(b.value - e["value"]) <= 1e-12 and fs is not None \
            and fs["config_md5"] == e["md5"] and fs["folder"].startswith(e["hash10"]) \
            and b.user_attrs.get("config_md5") == e["md5"]
        if "seeded_from_trial" in e:
            ok = ok and int(b.user_attrs.get("seeded_from_trial", -1)) == e["seeded_from_trial"] \
                and abs(trial_by_number(ctx.studies["phase2"], e["seeded_from_trial"]).value - b.value) <= 1e-12
        ctx.add("I9 champions_and_hashes", ok, f"{name}: best #{b.number}={b.value:.6f} folder={fs['folder'] if fs else None}")


def inv_I10(ctx):
    for name in ctx.dbs:
        st = ctx.studies[name]; e = ctx.exp["dbs"][name]
        groups = defaultdict(list)
        for t in complete(st):
            groups[json.dumps(t.params, sort_keys=True)].append(t)
        found = sorted(sorted(t.number for t in g) for g in groups.values() if len(g) > 1)
        expct = sorted(sorted(g) for g in e["repeat_draws"])
        ident = {tuple(sorted(g)) for g in e["identical_outputs"]}
        bad = []
        for g in groups.values():
            if len(g) > 1:
                vals = {round(t.value, 12) for t in g}
                key = tuple(sorted(t.number for t in g))
                if key in ident and len(vals) != 1: bad.append(f"{key}: expected identical values")
                if key not in ident and len(vals) == 1: bad.append(f"{key}: unexpectedly identical")
        ok = found == expct and not bad
        ctx.add("I10 repeat_draws", ok, f"{name}: groups={found if found != expct else 'as expected'} bad={bad}")


def inv_I11(ctx):
    e = ctx.exp["aug_batch"]; p = ROOT / e["sidecar"]; bad = []
    if not p.exists():
        ctx.add("I11 aug_sidecar", False, "sidecar missing"); return
    df = pd.read_csv(p)
    col = "p1_trial" if "p1_trial" in df.columns else "trial"
    if sorted(df[col].astype(int)) != e["p1_trials"]:
        bad.append(f"rows={sorted(df[col].astype(int))}")
    p1 = ctx.studies["phase1"]
    for _, r in df.iterrows():
        n = int(r[col]); d = ctx.resc(n)
        if d is None: bad.append(f"{n}: no parquet"); continue
        if abs(float(r["phi3_unified"]) - d.accuracy_unified.mean()) > 1e-9: bad.append(f"{n}: phi3_unified != parquet")
        if "p1_db_value" in df.columns and abs(float(r["p1_db_value"]) - trial_by_number(p1, n).value) > 1e-9:
            bad.append(f"{n}: p1_db_value != DB")
    if not ctx.pre_aug:
        for name, m in e["copies"].items():
            if name not in ctx.dbs: continue
            for num, p1n in m.items():
                t = trial_by_number(ctx.studies[name], int(num)); d = ctx.resc(int(p1n))
                if abs(t.value - d.accuracy_unified.mean()) > 1e-9: bad.append(f"{name}#{num}: value != phi3 of p1#{p1n}")
    ctx.add("I11 aug_sidecar", not bad, f"bad={bad[:6]} ({len(bad)})")


def inv_I12(ctx):
    for name in ctx.dbs:
        st = ctx.studies[name]; e = ctx.exp["dbs"][name]
        found = sorted(t.number for t in complete(st)
                       if t.params.get("disable_reasoning") is True and int(t.user_attrs.get("reasoning_tokens", 0) or 0) > 0)
        bad = [n for n in found if trial_by_number(st, n).user_attrs.get("effective_reasoning_observed") != "on"]
        ok = found == sorted(e["reasoning_toggle_ignored"]) and not bad
        ctx.add("I12 reasoning_toggle", ok, f"{name}: found={found} expected={e['reasoning_toggle_ignored']} bad_attrs={bad}")


def inv_I13(ctx):
    if ctx.skip_batch:
        ctx.add("I13 batch_api_replicate_pair", True, "skipped"); return
    d = ROOT / "archive/batch_api"
    outs = sorted(d.glob("batch_warmup_*_out.jsonl"))
    if not outs:
        ctx.add("I13 batch_api_replicate_pair", True, "no batched warm-up output file found (skipped)"); return
    by = {}
    for f in outs:
        for line in open(f):
            if not line.strip(): continue
            o = json.loads(line); cid = str(o.get("custom_id", ""))
            if "|" not in cid: continue
            cfg, paper = cid.split("|", 1)
            body = (o.get("response") or {}).get("body") or {}
            content = ((body.get("choices") or [{}])[0].get("message") or {}).get("content")
            by[(cfg, paper)] = (body.get("id"), content, body.get("created"))
    papers = sorted({p for c, p in by if c == "cfg1"} & {p for c, p in by if c == "cfg2"})
    ident = sum(1 for p in papers if by[("cfg1", p)][1] == by[("cfg2", p)][1] and by[("cfg1", p)][0] != by[("cfg2", p)][0])
    ok = len(papers) >= 35 and ident == len(papers)
    ctx.add("I13 batch_api_replicate_pair", ok, f"cfg1/cfg2 (P1 #383/#384) paper pairs={len(papers)} identical-content & distinct-id={ident}")


def inv_I14(ctx):
    for name in ctx.dbs:
        e = ctx.exp["dbs"][name].get("top_set_common_papers")
        if not e: continue
        st = ctx.studies[name]
        ref = trial_by_number(st, e["reference"]); rm = ctx.per_paper_means(name, ref)
        common = set(rm.index); rows = [f"ref #{ref.number}={rm.loc[sorted(common)].mean():.4f} on {len(common)} papers"]
        ok = True
        for n in e["candidates"]:
            t = trial_by_number(st, n)
            if t is None: continue
            m = ctx.per_paper_means(name, t); c = sorted(common & set(m.index))
            v = m.loc[c].mean(); rows.append(f"#{n}={v:.4f}")
            if name == "open_tpe" and v > rm.loc[sorted(common)].mean() + 1e-9: ok = False
        ctx.add("I14 top_set_common_papers", ok, f"{name}: " + "; ".join(rows))


def inv_I15(ctx, md5_start):
    e = ctx.exp["phase1_db"]
    ok = db_md5(DB_PATHS["phase1"]) == e["md5"] == md5_start["phase1"]
    ctx.add("I15 phase1_untouched", ok, f"md5={db_md5(DB_PATHS['phase1'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="all")
    ap.add_argument("--expected", default=str(ROOT / "scripts/study_expected.json"))
    ap.add_argument("--json")
    ap.add_argument("--pre-aug-write", action="store_true")
    ap.add_argument("--skip-batch-api", action="store_true")
    a = ap.parse_args()
    exp = json.load(open(a.expected))
    dbs = list(PHASE2_DBS) if a.db == "all" else [a.db]
    md5_start = {n: db_md5(DB_PATHS[n]) for n in ("phase1",) + tuple(dbs)}
    ctx = Ctx(exp, dbs, a.pre_aug_write, a.skip_batch_api)
    for fn in (inv_I1, inv_I2, inv_I3, inv_I4, inv_I5, inv_I6, inv_I7, inv_I8, inv_I9, inv_I10,
               inv_I11, inv_I12, inv_I13, inv_I14):
        try:
            fn(ctx)
        except Exception as ex:  # a crashed invariant is a FAIL, not a crash of the report
            ctx.add(fn.__name__.replace("inv_", "") + " (crashed)", False, f"{type(ex).__name__}: {ex}")
    inv_I15(ctx, md5_start)
    md5_end = {n: db_md5(DB_PATHS[n]) for n in md5_start}
    ctx.add("I0 self_read_only", md5_start == md5_end, "DB hashes unchanged by this run" if md5_start == md5_end else f"CHANGED: {md5_start} -> {md5_end}")
    n_fail = 0
    for r in ctx.results:
        print(f"{'PASS' if r['ok'] else 'FAIL'}  {r['inv']:36} {r['detail']}")
        n_fail += (not r["ok"])
    print(f"\n{len(ctx.results) - n_fail} PASS / {n_fail} FAIL")
    if a.json:
        json.dump({"results": ctx.results, "md5_start": md5_start, "md5_end": md5_end,
                   "pre_aug_write": a.pre_aug_write}, open(a.json, "w"), indent=1)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
