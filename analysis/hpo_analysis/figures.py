"""The six exhibits for the co-author call.

Palette is the validated reference instance (slots 1-3 clear the all-pairs gates
in light mode; the aqua slot's contrast WARN is discharged by direct labels).
One axis per chart, thin marks, recessive grid, legend whenever >= 2 series.

Run:  pipeline_env/bin/python -m analysis.hpo_analysis.figures
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import design
from .paths import CACHE, FIGURES, OUTPUT_BLACKOUT, TABLES

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8a85"
GRID = "#e3e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
NEG, POS = "#e34948", "#2a78d6"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 150,
})


def _finish(ax, title, sub=None, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left",
                 pad=24 if sub else 8)
    if sub:
        # matplotlib parses $...$ as mathtext; money must be escaped or the
        # subtitle silently renders as italic gibberish.
        ax.text(0, 1.015, sub.replace("$", r"\$"), transform=ax.transAxes,
                fontsize=8.5, color=INK2, va="bottom")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)


def _save(fig, name):
    p = FIGURES / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.name}")
    return p


# ---------------------------------------------------------------- F1 inventory
def fig1_inventory():
    tr = pd.read_parquet(CACHE / "trials.parquet")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    st = tr.state.value_counts()
    order = ["COMPLETE", "FAIL", "PRUNED", "WAITING", "RUNNING"]
    order = [s for s in order if s in st.index]
    vals = [st[s] for s in order]
    info = tr[(tr.state == "FAIL") & tr.informative].shape[0]
    cols = [S3, MUTED, S2, GRID, GRID][:len(order)]
    b = ax1.barh(order[::-1], vals[::-1], color=cols[::-1], height=0.6)
    for r, v in zip(b, vals[::-1]):
        ax1.text(v + 6, r.get_y() + r.get_height() / 2, f"{v}", va="center",
                 fontsize=8.5, color=INK2)
    ax1.text(vals[order.index("FAIL")] * 0.42,
             len(order) - 1 - order.index("FAIL"),
             f"only {info} informative\n(274 are one mass-fail event)",
             fontsize=7.8, color=SURFACE, va="center", ha="center", fontweight="bold")
    cells_all = pd.read_parquet(CACHE / "cells.parquet")
    n_keep = len(design.analysis_trials(cells_all, tr))
    n_rec = int(cells_all[cells_all.recovered].trial_number.nunique())
    _finish(ax1, f"509 trials, {vals[order.index('COMPLETE')]} complete",
            f"{n_keep} usable after recovering {n_rec} from the raw batch files",
            xlabel="trials")
    ax1.grid(axis="y", visible=False)

    # every COMPLETE trial, so the excluded partials stay visible
    comp = set(tr.loc[tr.state == "COMPLETE", "number"])
    d = cells_all[cells_all.trial_number.isin(comp)].copy()
    d["doc_name"] = d.doc_name.astype(str)
    nd = d.groupby("trial_number").doc_name.nunique().value_counts().sort_index()
    ax2.bar([str(i) for i in nd.index], nd.values, color=S1, width=0.6)
    for x, v in zip(range(len(nd)), nd.values):
        ax2.text(x, v + 1, str(v), ha="center", fontsize=8.5, color=INK2)
    n35 = int(nd.get(35, 0))
    _finish(ax2, "Trials did not run on the same papers",
            f"{n35} of {len(comp)} completed trials cover all 35 — "
            "raw means are not comparable",
            xlabel="papers completed", ylabel="trials")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, "F1_inventory")


# ---------------------------------------------------------------- F2 cost
def fig2_cost():
    tk = pd.read_parquet(CACHE / "tokens.parquet")
    cells = pd.read_parquet(CACHE / "cells.parquet")
    tr_all = pd.read_parquet(CACHE / "trials.parquet")
    keep = design.analysis_trials(cells, tr_all)
    d = tk[tk.trial_number.isin(keep) & tk.recoverable].copy()
    d["route"] = np.where(d["mode"] == "rag", "RAG",
                          "full / " + d.question_mode.astype(str))
    d = d[d.route.isin(["RAG", "full / batched", "full / individual"])]
    d = d[d.usd_per_paper_cal > 0]

    fig, ax = plt.subplots(figsize=(8.4, 5))
    colors = {"full / batched": S1, "full / individual": S2, "RAG": S3}
    for r, g in d.groupby("route"):
        ax.scatter(g.usd_per_paper_cal, g.accuracy, s=54, color=colors[r],
                   label=r, edgecolor=SURFACE, linewidth=1.6, zorder=3)
    # Pareto frontier: cheapest config at or above each accuracy level
    p = d.sort_values("usd_per_paper_cal")
    front, best = [], -1
    for _, r in p.iterrows():
        if r.accuracy > best:
            front.append(r); best = r.accuracy
    fr = pd.DataFrame(front)
    ax.step(fr.usd_per_paper_cal, fr.accuracy, where="post", color=MUTED,
            lw=1.4, ls="--", zorder=2, label="Pareto frontier")
    for i, (_, r) in enumerate(fr.iterrows()):
        ax.annotate(f"#{int(r.trial_number)}", (r.usd_per_paper_cal, r.accuracy),
                    textcoords="offset points",
                    xytext=(7, 6) if i % 2 == 0 else (7, -12),
                    fontsize=7.6, color=INK2)
    ax.set_xscale("log")
    med = d.groupby("route").usd_per_paper_cal.median()
    acc = d.groupby("route").accuracy.median()
    mb, mi = med.get("full / batched"), med.get("full / individual")
    ab, ai = acc.get("full / batched"), acc.get("full / individual")
    ratio = (mi / mb) if (mb and mi) else float("nan")
    _finish(ax, f"Batched full-document inference is {ratio:.0f}× cheaper, and no less accurate",
            f"median ${mb:.3f}/paper batched vs ${mi:.3f} individual · "
            f"accuracy {ab:.3f} vs {ai:.3f}",
            xlabel="cost per paper (USD, log scale)", ylabel="accuracy")
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    return _save(fig, "F2_cost_accuracy")


# ---------------------------------------------------------------- F3 effects
def fig3_effects():
    a0 = pd.read_csv(TABLES / "T3_specA0.csv")
    a0 = a0[a0.term != "const"].copy()
    pretty = {
        "effective_reasoning_on": "Reasoning ON (vs not offered)",
        "effective_reasoning_off": "Reasoning OFF (vs not offered)",
        "mode_rag": "RAG (vs full document)",
        "use_default_prompt": "Standard prompt (vs binary)",
    }
    a0["label"] = (a0.term
                   .str.replace("default_prompt_type_", "prompt: ", regex=False)
                   .str.replace("parser_effective_", "parser: ", regex=False)
                   .str.replace("llm_model_", "model: ", regex=False)
                   .replace(pretty))
    a0 = a0[~a0.label.str.startswith("model: ")]
    # Trial-level support per level. A lever varies only across trials, so a level
    # seen in <=3 trials carries no information however many cells it spans —
    # mark it rather than plotting a point that invites reading.
    d = design.prepare()
    support = {}
    for col in ("parser_effective", "default_prompt_type", "mode",
                "effective_reasoning"):
        for lvl, n in d.groupby(col, observed=True).trial_number.nunique().items():
            support[f"{col}_{lvl}"] = int(n)
    a0["n_trials"] = a0.term.map(support)
    a0["thin"] = a0.n_trials.notna() & (a0.n_trials <= 3)
    a0["label"] = np.where(a0.thin,
                           a0.label + a0.n_trials.fillna(0).map(
                               lambda v: f"  (n={int(v)}, no estimate)"),
                           a0.label)
    a0 = a0.reindex(a0.coef.sort_values().index)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    y = np.arange(len(a0))
    cols = [MUTED if thin else (NEG if c < 0 else POS)
            for c, thin in zip(a0.coef, a0.thin)]
    ax.errorbar(a0.coef, y, xerr=1.96 * a0.se, fmt="none", ecolor=GRID,
                elinewidth=1.4, capsize=0, zorder=2)
    ax.scatter(a0.coef, y, s=46, color=cols, zorder=3,
               edgecolor=SURFACE, linewidth=1.4,
               marker="o")
    # hollow the unsupported ones so they read as "not estimated"
    thin_idx = np.where(a0.thin.to_numpy())[0]
    if len(thin_idx):
        ax.scatter(a0.coef.to_numpy()[thin_idx], thin_idx, s=46,
                   facecolor=SURFACE, edgecolor=MUTED, linewidth=1.4, zorder=4)
    ax.axvline(0, color=INK2, lw=1, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels(a0.label, fontsize=8.4)
    for yy, (c, se_, p, thin) in enumerate(zip(a0.coef, a0.se, a0.p, a0.thin)):
        if p < 0.05 and not thin:
            end = c + 1.96 * se_ if c > 0 else c - 1.96 * se_
            ax.text(end + (0.006 if c > 0 else -0.006), yy, f"{c:+.3f}",
                    va="center", ha="left" if c > 0 else "right",
                    fontsize=7.8, color=INK2)
    ax.set_xlim(a0.coef.min() - 1.96 * a0.se.max() - 0.05,
                a0.coef.max() + 1.96 * a0.se.max() + 0.05)
    _finish(ax, "What actually moves accuracy",
            f"paper + item fixed effects, 95% CI, SEs clustered by trial "
            f"({d.trial_number.nunique()} trials) · model dummies fitted, not shown · "
            "hollow = too few trials to estimate",
            xlabel="change in probability of a correct cell")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, "F3_design_effects")


# ---------------------------------------------------------------- F4 humans
def fig4_humans():
    ha = pd.read_csv(TABLES / "T4_human_agreement.csv")
    ha = ha[(ha.n_raters >= 2) & (ha.bucket != "unknown")]
    hum = ha.groupby("query_id").agree.mean().rename("human_agree")

    d = design.prepare()
    mod = d.groupby("query_id").y.mean().rename("model_acc")
    j = pd.concat([hum, mod], axis=1).dropna()
    j["human_disagree"] = 1 - j.human_agree

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1.1]})
    ax1.scatter(j.human_disagree, j.model_acc, s=52, color=S1,
                edgecolor=SURFACE, linewidth=1.5, zorder=3)
    r = np.corrcoef(j.human_disagree, j.model_acc)[0, 1]
    b = np.polyfit(j.human_disagree, j.model_acc, 1)
    xs = np.linspace(0, j.human_disagree.max(), 20)
    ax1.plot(xs, np.polyval(b, xs), color=MUTED, lw=1.4, ls="--", zorder=2)
    for q in j.human_disagree.nlargest(4).index:
        ax1.annotate(q, (j.human_disagree[q], j.model_acc[q]),
                     textcoords="offset points", xytext=(6, 4),
                     fontsize=7.6, color=INK2)
    _finish(ax1, "Hard for humans, hard for the model",
            f"43 items · r = {r:.2f}",
            xlabel="two-coder disagreement", ylabel="model accuracy")

    types = ["single_choice", "multiple_choice", "numeric", "list", "open_ended"]
    hb = ha.groupby("type").agree.mean()
    mb = d.groupby("type", observed=True).y.mean()
    idx = [t for t in types if t in hb.index and t in mb.index]
    x = np.arange(len(idx)); w = 0.38
    ax2.bar(x - w / 2, [mb[t] for t in idx], w, color=S1, label="model accuracy",
            edgecolor=SURFACE, linewidth=1.2)
    ax2.bar(x + w / 2, [hb[t] for t in idx], w, color=S2, label="human agreement",
            edgecolor=SURFACE, linewidth=1.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([t.replace("_", "\n") for t in idx], fontsize=8)
    ax2.set_ylim(0, 1)
    for xi, t in zip(x, idx):
        ax2.text(xi - w / 2, mb[t] + .015, f"{mb[t]:.2f}", ha="center", fontsize=7.6, color=INK2)
        ax2.text(xi + w / 2, hb[t] + .015, f"{hb[t]:.2f}", ha="center", fontsize=7.6, color=INK2)
    _finish(ax2, "Measured against a ~74% human ceiling",
            "agreement under the scorer's own normalization")
    ax2.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0, -0.12),
               ncol=2, columnspacing=1.4)
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, "F4_humans")


# ---------------------------------------------------------------- F5 eval size
def fig5_evalsize():
    c = pd.read_csv(TABLES / "T5_se_curve.csv")
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(c.n_papers, c.mde80_absolute, color=S1, lw=2, marker="o", ms=5,
            label="absolute (one config)")
    ax.plot(c.n_papers, c.mde80_paired, color=S2, lw=2, marker="o", ms=5,
            label="paired (two configs, same papers)")
    for n, lab in ((35, "tuning\n35"), (163, "test\n163")):
        ax.axvline(n, color=GRID, lw=1.2, zorder=1)
        ax.text(n, ax.get_ylim()[1] * 0.96, lab, fontsize=7.6, color=MUTED,
                ha="center", va="top")
    top = c[c.n_papers == 35].mde80_absolute.iloc[0]
    ax.axhline(0.02, color=MUTED, ls=":", lw=1.2)
    ax.text(6, 0.022, "2pp — the gap between the top configs", fontsize=7.8, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 35, 50, 100, 198])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    _finish(ax, "At 35 papers we can only detect a 4pp difference",
            f"minimum detectable effect, 80% power · MDE at 35 papers = {top:.3f}",
            xlabel="papers in the evaluation set", ylabel="minimum detectable Δ accuracy")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    return _save(fig, "F5_eval_size")


# ---------------------------------------------------------------- F6 spec curve
def fig6_speccurve():
    d = design.prepare()
    g = (d.groupby("trial_number")
           .agg(acc=("y", "mean"), n_docs=("doc_name", "nunique"),
                model=("llm_model", "first"), mode=("mode", "first"),
                qmode=("question_mode", "first")))
    g = g[g.n_docs == 35].sort_values("acc").reset_index()
    se = pd.read_csv(TABLES / "T5_se_curve.csv")
    s35 = se[se.n_papers == 35].se_absolute.iloc[0]

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    x = np.arange(len(g))
    ax.errorbar(x, g.acc, yerr=1.96 * s35, fmt="none", ecolor=GRID, elinewidth=3, zorder=1)
    ax.scatter(x, g.acc, s=26, color=S1, zorder=3, edgecolor=SURFACE, linewidth=1)
    top = g.acc.max()
    band = top - 1.96 * s35 * np.sqrt(2)
    ax.axhspan(band, top + 0.005, color=S1, alpha=0.08, zorder=0)
    n_in = int((g.acc >= band).sum())
    ax.axhline(band, color=S1, lw=1.2, ls="--")
    ax.text(0.5, band + 0.004, f"{n_in} configs are statistically indistinguishable from the best",
            fontsize=8.2, color=INK2)
    _finish(ax, "There is no single best configuration — there is a top set",
            f"{len(g)} full-grid trials, 95% CI from paper-clustered SE ({s35:.3f})",
            xlabel="configuration (ranked)", ylabel="accuracy")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, "F6_spec_curve")


def main():
    print("building figures ->", FIGURES)
    for fn in (fig1_inventory, fig2_cost, fig3_effects, fig4_humans,
               fig5_evalsize, fig6_speccurve):
        try:
            fn()
        except Exception as e:
            print(f"  FAILED {fn.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
