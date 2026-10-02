"""How much of the individual-vs-batched comparison could be prompt wording?

Trial-level analysis over every Phase-1 individual-mode trial and every Phase-2
prompt-enabled batched trial (plus Phase-1 batched_native for reference):
  (a) spread across prompt variants within the same model+parser+mode;
  (b) cells where the variant is held fixed across modes;
  (c) OLS of accuracy on mode with model, parser and prompt-variant fixed effects
      (cluster-robust by model), and a mode x variant-group interaction.
Individual values use the unified Phi-3 cells in analysis/cache/mode_by_type/
indiv_all/resc/ where present; trials still on the old basis get an `old_basis` dummy.
Writes analysis/tables/hpo/mode_prompt_confound_{trials,summary}.csv|json.
usage: pipeline_env/bin/python analysis/hpo_analysis/mode_prompt_confound.py
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd, optuna
import statsmodels.formula.api as smf
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from _study_record import load_studies  # noqa: E402
optuna.logging.set_verbosity(optuna.logging.WARNING)
OUT = ROOT / "analysis" / "tables" / "hpo"


def trials() -> pd.DataFrame:
    S = load_studies(("phase1", "phase2", "open_tpe", "consumer_tpe"))
    resc = {}
    for f in glob.glob(str(ROOT / "analysis/cache/mode_by_type/indiv_all/resc/[0-9]*.parquet")):
        x = pd.read_parquet(f); resc[int(x.p1_number.iloc[0])] = float(x.accuracy_unified.mean())
    rows = []
    for t in S["phase1"].trials:
        if t.state.name != "COMPLETE" or t.value is None: continue
        p = t.params
        if p.get("mode") != "full" or p.get("question_mode") == "batched": continue
        if p.get("question_mode") is None and t.number not in (32, 37): continue
        rows.append(dict(src="p1", n=t.number, model=p["llm_model"].split("/")[-1], parser=p["parser"].replace("Parser", ""),
                         mode="individual", variant=p.get("default_prompt_type"), binary=int(p.get("use_default_prompt") is False),
                         acc=resc.get(t.number, t.value), old_basis=int(t.number not in resc)))
    for db in ("phase2", "open_tpe", "consumer_tpe"):
        for t in S[db].trials:
            if t.state.name != "COMPLETE" or t.value is None: continue
            p = t.params
            if p.get("mode") != "full" or "batched_prompt_type" not in p or "seeded_from_trial" in t.user_attrs: continue
            v = p["batched_prompt_type"]
            rows.append(dict(src=db, n=t.number, model=p["llm_model"].split("/")[-1], parser=p["parser"].replace("Parser", ""),
                             mode="batched_native(P1)" if v == "batched_native" else "batched_variant(P2)", variant=v, binary=0,
                             acc=t.value, old_basis=0))
    return pd.DataFrame(rows)


def main():
    d = trials(); d.to_csv(OUT / "mode_prompt_confound_trials.csv", index=False)
    out = {"n_trials": d.groupby("mode").size().to_dict(), "individual_old_basis": int(d.old_basis.sum())}
    g = d.groupby(["model", "parser", "mode"]).agg(k=("variant", "nunique"), lo=("acc", "min"), hi=("acc", "max"))
    g = g[g.k >= 2]; g["range_pp"] = (g.hi - g.lo) * 100
    out["variant_range_pp_median"] = g.groupby("mode").range_pp.median().round(1).to_dict()
    out["variant_range_pp_max"] = g.groupby("mode").range_pp.max().round(1).to_dict()
    x = d[d["mode"].isin(["individual", "batched_variant(P2)"])].copy(); x["batched"] = (x["mode"] != "individual").astype(int)
    k = x.groupby(["model", "parser", "mode"]).acc.transform("size")
    x["res"] = x.acc - x.groupby(["model", "parser", "mode"]).acc.transform("mean")
    out["within_cell_sd_pp"] = (x[k >= 2].groupby("mode").res.std() * 100).round(1).to_dict()
    piv = x.pivot_table(index=["model", "variant"], columns="mode", values="acc", aggfunc="max").dropna()
    piv["bat_minus_ind_pp"] = ((piv["batched_variant(P2)"] - piv["individual"]) * 100).round(1)
    out["variant_matched_cells"] = {f"{m}|{v}": float(r) for (m, v), r in piv.bat_minus_ind_pp.items()}
    fits = {}
    for label, f in (("raw", "acc ~ batched"), ("model_fe", "acc ~ batched + C(model)"),
                     ("model_parser_fe", "acc ~ batched + C(model) + C(parser)"),
                     ("model_parser_variant_fe", "acc ~ batched + C(model) + C(parser) + C(variant) + binary + old_basis")):
        r = smf.ols(f, data=x).fit(cov_type="cluster", cov_kwds={"groups": x.model})
        fits[label] = {"batched_pp": round(r.params["batched"] * 100, 1), "se_pp": round(r.bse["batched"] * 100, 1), "n": int(r.nobs)}
        if label == "model_parser_variant_fe":
            out["variant_effects_pp_vs_base"] = {k.split("T.")[1].rstrip("]"): round(v * 100, 1) for k, v in r.params.items() if k.startswith("C(variant)")}
    out["mode_effect"] = fits
    x["vg"] = x.variant.replace({"cot_prompt": "cot_reasoning", "reasoning_prompt": "cot_reasoning", "true_few_shot_prompt": "few_shot",
                                 "synthetic_few_shot_prompt": "few_shot", "rewritten_prompt": "rewritten", "base_prompt": "base"})
    ri = smf.ols("acc ~ batched*C(vg) + C(model) + C(parser) + old_basis", data=x).fit(cov_type="cluster", cov_kwds={"groups": x.model})
    out["variant_group_by_mode_pp"] = {gname: {"individual": round(ri.params.get(f"C(vg)[T.{gname}]", np.nan) * 100, 1),
                                               "batched": round((ri.params.get(f"C(vg)[T.{gname}]", 0) + ri.params.get(f"batched:C(vg)[T.{gname}]", 0)) * 100, 1)}
                                       for gname in ("cot_reasoning", "few_shot", "rewritten")}
    json.dump(out, open(OUT / "mode_prompt_confound_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
