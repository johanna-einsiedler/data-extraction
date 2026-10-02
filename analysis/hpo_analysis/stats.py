"""Estimation: what actually moves accuracy, and how much data we need.

  Spec A  pooled LPM, paper + item fixed effects, SEs clustered by trial.
          A0 common levers | A1 RAG subsample | A2 full-doc subsample.
  Spec B  the ONLY cleanly identified lever: trial fixed effects absorb every
          trial-level confounder, and the binary-decomposition switch is identified
          off its interaction with fixed, pre-determined item eligibility.
  Spec D  variance components and the SE-vs-n_papers curves behind the
          "how big must the evaluation set be" decision.

The honest caveat, which belongs on the slide: 116k rows do NOT buy degrees of
freedom. Trial-level levers vary only across the 79 trials, so the effective n for
those coefficients is 79. What the cell-level model buys is (a) removal of the very
large paper and item variance from the residual, (b) automatic conditioning on the
unequal paper sets, and (c) the within-trial identification in Spec B.

Run:  pipeline_env/bin/python -m analysis.hpo_analysis.stats
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import design
from .paths import CACHE, TABLES

RNG = np.random.default_rng(42)


class RankDeficient(RuntimeError):
    """Raised when a spec is not identified, instead of returning 1e10 nonsense."""


# ------------------------------------------------------------------ FE OLS
def _design_matrix(df: pd.DataFrame, fe: list[str], x: list[str]):
    parts, names = [], []
    for c in fe + x:
        s = df[c]
        if s.dtype == bool:
            parts.append(s.astype(float).to_numpy()[:, None]); names.append(c)
            continue
        # already-binary indicators stay numeric; dummying them renames the term
        if (c not in fe and pd.api.types.is_numeric_dtype(s)
                and set(pd.unique(s.dropna())) <= {0.0, 1.0}):
            parts.append(s.astype(float).fillna(0.0).to_numpy()[:, None])
            names.append(c)
            continue
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 12 and c not in fe:
            parts.append(s.astype(float).fillna(s.astype(float).mean()).to_numpy()[:, None])
            names.append(c); continue
        d = pd.get_dummies(s.astype(str), prefix=c, drop_first=True, dtype=float)
        parts.append(d.to_numpy()); names.extend(d.columns.tolist())
    X = np.hstack(parts) if parts else np.empty((len(df), 0))
    X = np.hstack([np.ones((len(df), 1)), X])
    return X, ["const"] + names


def trial_support(df: pd.DataFrame, col: str) -> pd.Series:
    """Trials (not cells) per level — the only n that matters for a trial-level lever."""
    return df.groupby(col.replace("__", ""))["trial_number"].nunique()


def drop_sparse(df: pd.DataFrame, cols: list[str], min_trials: int = 5) -> list[str]:
    """Keep only levers whose every realised level has >= min_trials trials.
    A lever seen in 1-3 trials carries no information about its effect."""
    keep = []
    for c in cols:
        if c not in df.columns:
            continue
        n = df.groupby(df[c].astype(str))["trial_number"].nunique()
        if len(n) > 1 and n.min() >= min_trials:
            keep.append(c)
    return keep


def fe_ols(df: pd.DataFrame, fe: list[str], x: list[str], cluster: str = "trial_number",
           check_rank: bool = True):
    X, names = _design_matrix(df, fe, x)
    if check_rank:
        r = np.linalg.matrix_rank(X)
        n_trials = df[cluster].nunique()
        if r < X.shape[1]:
            raise RankDeficient(
                f"design matrix is rank-deficient: rank {r} < {X.shape[1]} columns "
                f"({n_trials} trials). Coefficients are not identified.")
    y = df["y"].to_numpy(float)
    ok = ~np.isnan(y)
    res = sm.OLS(y[ok], X[ok]).fit(
        cov_type="cluster", cov_kwds={"groups": df[cluster].to_numpy()[ok]})
    out = pd.DataFrame({"term": names, "coef": res.params, "se": res.bse,
                        "t": res.tvalues, "p": res.pvalues})
    out = out[~out.term.str.startswith(tuple(f"{c}_" for c in fe))]
    return out.reset_index(drop=True), res


def wild_cluster_bootstrap(df, fe, x, term, cluster="trial_number", B=999):
    """Webb 6-point wild cluster bootstrap p-value, null imposed. 79 clusters with
    ~40 regressors is exactly where CR1 under-covers."""
    X, names = _design_matrix(df, fe, x)
    y = df["y"].to_numpy(float)
    g = pd.factorize(df[cluster])[0]
    if term not in names:
        return np.nan, np.nan
    j = names.index(term)
    full = sm.OLS(y, X).fit()
    t_obs = full.params[j] / np.sqrt(np.diag(_cluster_vcov(X, full.resid, g))[j])

    keep = [i for i in range(X.shape[1]) if i != j]
    r0 = sm.OLS(y, X[:, keep]).fit()          # restricted fit: beta_j = 0
    yhat0, u0 = r0.fittedvalues, r0.resid
    webb = np.array([-1.5, -1, -0.5, 0.5, 1, 1.5]) / np.sqrt(1.5)
    ng = g.max() + 1
    ts = np.empty(B)
    for b in range(B):
        w = RNG.choice(webb, size=ng)[g]
        yb = yhat0 + u0 * w
        fb = sm.OLS(yb, X).fit()
        ts[b] = fb.params[j] / np.sqrt(np.diag(_cluster_vcov(X, fb.resid, g))[j])
    return float(t_obs), float(np.mean(np.abs(ts) >= abs(t_obs)))


def _cluster_vcov(X, u, g):
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for k in range(g.max() + 1):
        m = g == k
        Xg, ug = X[m], u[m]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    G = g.max() + 1
    c = G / max(G - 1, 1)
    return c * XtX_inv @ meat @ XtX_inv


# ------------------------------------------------------------------ specs
def spec_A(df: pd.DataFrame):
    fe = ["doc_name", "query_id"]
    common = ["llm_model", "mode", "parser_effective", "default_prompt_type",
              "use_default_prompt", "effective_reasoning"]
    a0, _ = fe_ols(df, fe, [c for c in common if df[c].nunique() > 1])

    # RAG branch: 36 trials cannot support 17 model dummies + 6 parsers + 6 prompts
    # + ~27 retrieval levers. Prune to levers with real trial-level support and drop
    # the model/parser controls, then still refuse to report if not identified.
    rag = df[df["mode"] == "rag"]
    a1 = pd.DataFrame()
    if len(rag):
        ragx = drop_sparse(rag, design.RAG_LEVERS, min_trials=5)
        try:
            a1, _ = fe_ols(rag, fe, ragx)
            a1["n_rag_trials"] = rag.trial_number.nunique()
        except (RankDeficient, np.linalg.LinAlgError) as e:
            a1 = pd.DataFrame([{"term": "NOT IDENTIFIED", "coef": np.nan,
                                "se": np.nan, "t": np.nan, "p": np.nan,
                                "note": str(e)}])

    full = df[df["mode"] == "full"]
    a2 = pd.DataFrame()
    if len(full):
        # is_vlm is redundant with parser_effective == "NA_vlm" (identical by
        # construction) and makes the design exactly rank-deficient.
        fullx = [c for c in ("question_mode",) if full[c].nunique() > 1]
        keep = [c for c in common if c != "mode" and full[c].nunique() > 1]
        try:
            a2, _ = fe_ols(full, fe, keep + fullx)
        except (RankDeficient, np.linalg.LinAlgError) as e:
            a2 = pd.DataFrame([{"term": "NOT IDENTIFIED", "coef": np.nan,
                                "se": np.nan, "t": np.nan, "p": np.nan,
                                "note": str(e)}])
    return a0, a1, a2


def spec_B(df: pd.DataFrame, B: int = 199):
    """Trial FE absorb every trial-level confounder; delta is the causal effect of
    binary decomposition on the 25 eligible items."""
    d = df.copy()
    d["binary_x_elig"] = (d.binary_eligible & ~d.udp).astype(float)
    tab, _ = fe_ols(d, ["trial_number", "doc_name", "query_id"], ["binary_x_elig"])
    row = tab[tab.term == "binary_x_elig"].iloc[0]

    t_obs, p_wild = wild_cluster_bootstrap(
        d, ["trial_number", "doc_name", "query_id"], ["binary_x_elig"],
        "binary_x_elig", B=B)

    items = d.query_id.unique()
    k = len(set(design.binary_eligible_items()) & set(items))

    # NOTE: a randomisation-inference p-value was computed here from a
    # covariance statistic on doubly-demeaned residuals. It is NOT the trial-FE
    # estimator the point estimate comes from, and it disagreed sharply with both
    # cluster-robust and wild-bootstrap inference (.002 vs .43/.45), so it was
    # measuring something else and has been removed rather than reported.
    itt_tab, _ = fe_ols(d, ["doc_name", "query_id"], ["use_default_prompt"])
    itt_row = itt_tab[itt_tab.term.str.startswith("use_default_prompt")]
    itt = -itt_row.coef.iloc[0] if len(itt_row) else np.nan
    return {"delta": float(row.coef), "se": float(row.se), "t_cluster": float(row.t),
            "p_cluster": float(row.p), "t_wild": t_obs, "p_wild": p_wild,
            "n_eligible_items": k,
            "itt_from_A0": float(itt),
            "itt_implied": float(row.coef) * k / len(items)}


def spec_D(df: pd.DataFrame, B: int = 2000):
    """Variance components and SE-vs-n_papers. The 43 items are the CODEBOOK —
    fixed, not sampled — so item variance does not enter the 'how many test
    papers' question; the decision-relevant object is a cluster bootstrap over
    papers."""
    full = df[df.n_docs_trial == 35]
    pt = full.pivot_table(index="doc_name", columns="trial_number", values="y", aggfunc="mean")
    it = full.pivot_table(index="query_id", columns="trial_number", values="y", aggfunc="mean")
    comp = {
        "sd_paper": float(pt.mean(axis=1).std()),
        "sd_item": float(it.mean(axis=1).std()),
        "sd_trial": float(pt.mean(axis=0).std()),
        "n_trials_balanced": int(pt.shape[1]),
    }
    paper_means = pt  # papers x trials
    rows = []
    for n in (5, 10, 15, 20, 27, 35, 50, 100, 163, 198):
        idx = paper_means.index.to_numpy()
        draws = RNG.choice(len(idx), size=(B, min(n, len(idx))), replace=True)
        mm = paper_means.to_numpy()
        abs_se = np.mean([mm[d].mean(axis=0).std() for d in draws[:200]])
        # single-config SE
        one = mm[:, 0]
        se_abs = one.std(ddof=1) / np.sqrt(n)
        # paired difference between two mid-ranked configs on the same papers
        if mm.shape[1] >= 2:
            dcol = mm[:, 0] - mm[:, 1]
            se_pair = dcol.std(ddof=1) / np.sqrt(n)
        else:
            se_pair = np.nan
        rows.append({"n_papers": n, "se_absolute": se_abs, "se_paired": se_pair,
                     "mde80_absolute": 2.8 * se_abs, "mde80_paired": 2.8 * se_pair})
    return comp, pd.DataFrame(rows)


def main() -> None:
    df = design.prepare()
    print(f"estimation sample: {len(df):,} cells, {df.trial_number.nunique()} trials, "
          f"{df.doc_name.nunique()} papers, {df.query_id.nunique()} items\n")

    a0, a1, a2 = spec_A(df)
    a0.to_csv(TABLES / "T3_specA0.csv", index=False)
    a1.to_csv(TABLES / "T3_specA1_rag.csv", index=False)
    a2.to_csv(TABLES / "T3_specA2_full.csv", index=False)
    print("--- Spec A0: common levers, paper+item FE, clustered by trial ---")
    top = a0[a0.term != "const"].reindex(a0[a0.term != "const"].coef.abs().sort_values(ascending=False).index)
    print(f"{'term':<48}{'coef':>8}{'se':>8}{'p':>8}")
    for _, r in top.head(14).iterrows():
        print(f"{r.term[:46]:<48}{r.coef:>+8.4f}{r.se:>8.4f}{r.p:>8.3f}")

    print("\n--- Spec B: binary decomposition, TRIAL fixed effects ---")
    b = spec_B(df)
    print(f"  delta (eligible items x binary on) = {b['delta']:+.4f}  (SE {b['se']:.4f})")
    print(f"  p  cluster-robust      : {b['p_cluster']:.4f}")
    print(f"  p  wild cluster boot   : {b['p_wild']:.4f}")
    print(f"  consistency: ITT implied {b['itt_implied']:+.4f} vs A0 ITT {b['itt_from_A0']:+.4f}")
    pd.DataFrame([b]).to_csv(TABLES / "T3_specB.csv", index=False)

    print("\n--- Spec D: variance components and evaluation-set size ---")
    comp, curve = spec_D(df)
    for k, v in comp.items():
        print(f"  {k:20s} {v:.4f}" if isinstance(v, float) else f"  {k:20s} {v}")
    curve.to_csv(TABLES / "T5_se_curve.csv", index=False)
    print(f"\n{'n_papers':>9}{'SE(abs)':>10}{'SE(paired)':>12}{'MDE80 abs':>12}{'MDE80 paired':>14}")
    for _, r in curve.iterrows():
        print(f"{int(r.n_papers):>9}{r.se_absolute:>10.4f}{r.se_paired:>12.4f}"
              f"{r.mde80_absolute:>12.4f}{r.mde80_paired:>14.4f}")
    print(f"\nwrote {TABLES}/T3_*.csv, T5_se_curve.csv")


if __name__ == "__main__":
    main()
