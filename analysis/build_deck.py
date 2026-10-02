#!/usr/bin/env python
"""Assemble the co-author call deck, embedding the figures as data URIs.

  pipeline_env/bin/python analysis/build_deck.py
"""
import base64
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "analysis" / "figures" / "hpo"
TAB = ROOT / "analysis" / "tables" / "hpo"
OUT = ROOT / "analysis" / "deck" / "call_deck.html"
OUT.parent.mkdir(parents=True, exist_ok=True)


def img(name: str) -> str:
    b = (FIG / f"{name}.png").read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()


def figure(name: str, caption: str, note: str = "") -> str:
    n = f'<p class="note">{note}</p>' if note else ""
    return (f'<figure><img src="{img(name)}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption>{n}</figure>')


def decision(n: int, title: str, body: str, ask: str) -> str:
    return f"""<section class="pt">
  <h3><span class="num">{n}</span>{title}</h3>
  {body}
  <p class="ask"><strong>Decision needed.</strong> {ask}</p>
</section>"""


stab = json.loads((TAB / "T0_4_ranking_stability.json").read_text())
cal = json.loads((TAB / "T2_calibration.json").read_text())
se = pd.read_csv(TAB / "T5_se_curve.csv")
mde35 = se[se.n_papers == 35].mde80_absolute.iloc[0]

HTML = f"""<title>Phase 2 Decision Brief</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  color-scheme: light;
  --ground:#fbfbfa; --panel:#f2f3f1; --panel-2:#e9eae6;
  --ink:#14161a; --ink-2:#4a4e55; --ink-3:#7c8189;
  --rule:#dfe0dc; --rule-2:#c9cbc5;
  --accent:#2a78d6; --accent-soft:#eaf1fb;
  --flag:#b3401f; --flag-soft:#fbeee9;
  --good:#177f5c;
  --serif:"Source Serif 4",Georgia,"Times New Roman",serif;
  --sans:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,"SF Mono",Menlo,monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --ground:#15171a; --panel:#1d2024; --panel-2:#262a2f;
    --ink:#f2f3f1; --ink-2:#b9bec5; --ink-3:#878d95;
    --rule:#2d3238; --rule-2:#3d434a;
    --accent:#5c9ce8; --accent-soft:#1b2a3d;
    --flag:#e08063; --flag-soft:#33211b;
    --good:#3fb98d;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --ground:#15171a; --panel:#1d2024; --panel-2:#262a2f;
  --ink:#f2f3f1; --ink-2:#b9bec5; --ink-3:#878d95;
  --rule:#2d3238; --rule-2:#3d434a;
  --accent:#5c9ce8; --accent-soft:#1b2a3d;
  --flag:#e08063; --flag-soft:#33211b;
  --good:#3fb98d;
}}
*{{box-sizing:border-box}}
body{{background:var(--ground);color:var(--ink);font-family:var(--sans);
  font-size:16px;line-height:1.62;margin:0;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:920px;margin:0 auto;padding:0 24px 96px}}
p,li{{max-width:68ch}}
h1,h2,h3{{font-family:var(--serif);text-wrap:balance;margin:0}}
h1{{font-size:clamp(2rem,4.4vw,2.9rem);font-weight:700;letter-spacing:-.015em;line-height:1.12}}
h2{{font-size:1.55rem;font-weight:600;letter-spacing:-.01em}}
h3{{font-size:1.16rem;font-weight:600}}
a{{color:var(--accent)}}
.masthead{{border-bottom:2px solid var(--ink);padding:56px 0 22px;margin-bottom:34px}}
.eyebrow{{font-family:var(--mono);font-size:.7rem;letter-spacing:.13em;
  text-transform:uppercase;color:var(--ink-3);margin:0 0 14px}}
.standfirst{{font-family:var(--serif);font-size:1.16rem;color:var(--ink-2);
  margin:16px 0 0;max-width:64ch}}
.meta{{font-family:var(--mono);font-size:.76rem;color:var(--ink-3);
  margin-top:20px;display:flex;flex-wrap:wrap;gap:20px}}
.keys{{display:grid;grid-template-columns:repeat(auto-fit,minmax(178px,1fr));
  gap:2px;background:var(--rule);border:1px solid var(--rule);margin:0 0 46px}}
.key{{background:var(--panel);padding:18px 18px 16px}}
.key .v{{font-family:var(--mono);font-size:1.72rem;font-weight:500;
  letter-spacing:-.02em;display:block;line-height:1.1;font-variant-numeric:tabular-nums}}
.key .l{{font-size:.82rem;color:var(--ink-2);margin-top:7px;display:block;line-height:1.4}}
section{{margin:0 0 46px}}
.sec-head{{display:flex;align-items:baseline;gap:14px;border-top:1px solid var(--rule-2);
  padding-top:12px;margin:64px 0 8px}}
.sec-head .tag{{font-family:var(--mono);font-size:.7rem;letter-spacing:.13em;
  text-transform:uppercase;color:var(--ink-3);white-space:nowrap}}
figure{{margin:26px 0 30px}}
figure img{{width:100%;height:auto;display:block;border:1px solid var(--rule);
  background:#fcfcfb}}
figcaption{{font-size:.86rem;color:var(--ink-2);margin-top:10px;max-width:70ch}}
.note{{font-family:var(--mono);font-size:.75rem;color:var(--ink-3);
  margin:6px 0 0;max-width:70ch}}
.finding{{background:var(--panel);border-left:3px solid var(--accent);
  padding:16px 20px;margin:22px 0}}
.finding p{{margin:0}}
.finding + .finding{{margin-top:10px}}
.correction{{background:var(--flag-soft);border-left:3px solid var(--flag);
  padding:16px 20px;margin:22px 0}}
.correction p{{margin:0}}
.correction strong{{color:var(--flag)}}
.ask{{background:var(--accent-soft);border-left:3px solid var(--accent);
  padding:13px 18px;margin:14px 0 0;font-size:.94rem}}
.pt{{border-top:1px solid var(--rule);padding-top:22px;margin:0 0 32px}}
.pt h3{{display:flex;gap:12px;align-items:baseline;margin-bottom:10px}}
.num{{font-family:var(--mono);font-size:.85rem;color:var(--accent);
  font-weight:500;flex:none;padding-top:2px}}
table{{border-collapse:collapse;width:100%;font-size:.88rem;margin:18px 0}}
.scroll{{overflow-x:auto}}
th,td{{text-align:left;padding:9px 14px 9px 0;border-bottom:1px solid var(--rule)}}
th{{font-family:var(--mono);font-size:.7rem;letter-spacing:.09em;
  text-transform:uppercase;color:var(--ink-3);font-weight:500}}
td.n,th.n{{text-align:right;font-family:var(--mono);
  font-variant-numeric:tabular-nums;padding-right:20px}}
code{{font-family:var(--mono);font-size:.87em;background:var(--panel-2);
  padding:1px 5px;border-radius:2px}}
.hi{{color:var(--accent);font-weight:600}}
.dim{{color:var(--ink-3)}}
ul{{padding-left:20px}} li{{margin:5px 0}}
footer{{border-top:1px solid var(--rule-2);margin-top:64px;padding-top:20px;
  font-family:var(--mono);font-size:.76rem;color:var(--ink-3)}}
@media (max-width:640px){{.meta{{gap:10px}}.key .v{{font-size:1.45rem}}}}
</style>

<div class="wrap">
<header class="masthead">
  <p class="eyebrow">Automated article coding · internal brief</p>
  <h1>Phase 2 Decision Brief</h1>
  <p class="standfirst">What the 96 completed tuning trials actually tell us, and the
  nine choices we need to settle before the Phase&nbsp;2 preregistration is locked and the
  test set is opened.</p>
  <div class="meta">
    <span>Prepared for the co-author call</span>
    <span>Study <code>full_item_accuracy_tuning</code></span>
    <span>OSF quxdb</span>
  </div>
</header>

<div class="keys">
  <div class="key"><span class="v">87</span><span class="l">comparable trials — every one covering all 35 papers</span></div>
  <div class="key"><span class="v">31×</span><span class="l">cheaper per paper in batched mode, no accuracy cost</span></div>
  <div class="key"><span class="v">25.7%</span><span class="l">of cells the two human coders disagree on</span></div>
  <div class="key"><span class="v">{mde35*100:.1f}pp</span><span class="l">smallest difference 35 papers can detect</span></div>
</div>

<p>Three things changed while preparing this. The tuning run is <em>not</em> a converged
optimisation — TPE never engaged, so the 96 trials are an unbalanced randomised experiment,
which is better for causal claims about design and worse for finding an optimum. A scoring
fix we were about to apply would have destroyed 17 trials. And the single largest practical
result has nothing to do with model choice.</p>

<div class="sec-head"><span class="tag">Part 1</span><h2>What the runs say</h2></div>

{figure("F1_inventory",
  "Left: 509 trials, of which 96 completed. Right: completed trials by number of papers finished.",
  "274 of the 405 failures are a single mass-fail event on 2026-07-24 and carry no information; the informative failure set is 131.")}

<div class="correction"><p><strong>Seventeen trials were unusable, and we got them back.</strong>
Trials 382–398 carried no model outputs: they were ingested from the OpenAI Batch API, and both
ingest scripts score each answer and then discard it, writing only
<code>{{query_id, type, doc_name, accuracy}}</code>. But the raw batch files were still on disk,
and they hold the answers, the log-probabilities and the per-call token counts. Recovering them
lifts the analysis set from 70 to <strong>87</strong> — every completed trial that covers all 35
papers — raises log-probability coverage by a third (45,314 → 60,359 cells), and triples the
ground truth available for validating cost.</p></div>

<div class="correction"><p><strong>The scoring fix we were about to apply would have destroyed those 17.</strong>
<code>rescore_concat_fix.py</code> re-scores from stored outputs, so with <code>--write</code> it
would have marked every one of their answers wrong — in the raw dry run that looked like a
legitimate 9–14 point accuracy correction. Applied only where it is valid, the fix is
<em>monotone</em>: 2,351 cells corrected 0→1 and <strong>zero</strong> the other way. It now runs
as a read-only sidecar, and the unsafe script has been retired.</p></div>

<p>The fix moves the argmax from trial #31 to #336, but leaves the ranking intact
(Spearman ρ = {stab['spearman_rho']:.3f}, top-10 overlap {stab['top10_overlap']}/10). That
distinction matters more than it sounds, because of the next figure.</p>

{figure("F6_spec_curve",
  "Every full-grid configuration, ranked, with 95% confidence intervals from paper-clustered standard errors.",
  "Trials also differ in which papers they completed; on the common 27-paper set, means shift by 0.012 on average (max 0.038) — comparable to the entire scoring fix.")}

<div class="finding"><p><strong>There is no best configuration, only a top set.</strong>
The gap between the top configs is smaller than the noise. Reporting a single winner would
be reporting sampling noise, and the Phase&nbsp;2 search should target a stable
high-performing <em>region</em> instead.</p></div>

{figure("F2_cost_accuracy",
  "Cost per paper against accuracy for every usable configuration, with the Pareto frontier.",
  "Costs reconstructed offline from the parsed corpus and stored contexts; validated against the 18 trials that recorded true token counts (batched 3.0% MAPE, RAG 9.3%).")}

<div class="finding"><p><strong>This is the biggest practical result in the study.</strong>
Batched full-document inference costs <span class="hi">$0.022/paper</span> against
<span class="hi">$0.661</span> for one call per item — <strong>31× cheaper</strong>, on 25 and 21
trials respectively. It is 35 model calls per paper instead of 1,505, and it is not worse:
batched is 0.008 <em>above</em> individual on median accuracy, a difference we cannot distinguish
from zero. A full 198-paper test run costs about <strong>$4</strong> instead of
<strong>$131</strong>.</p></div>

<div class="correction"><p><strong>This number moved, and it is worth knowing why.</strong>
On the pre-recovery sample the gap looked like 41× and we called it a conservative floor, because
individual mode appeared to be under-estimated. That rested on <em>two</em> trials with recorded
token counts. The recovered batch files supply twelve more, and against fourteen the estimator is
essentially unbiased (median ratio 0.96). The honest figure is 31×, not 41× — and it is a point
estimate, not a floor.</p></div>

{figure("F3_design_effects",
  "Effect of each design choice on the probability that a cell is coded correctly.",
  "Linear probability model on 130,935 cells with paper and item fixed effects. Model dummies are fitted but omitted here; the model spread is much larger than any of these.")}

<p>Two results overturn what the descriptive tables suggested. <strong>Reasoning hurts</strong>
(−0.102, p&nbsp;&lt;&nbsp;.001; −0.130 among full-document trials) — and unlike the raw
marginal, this is adjusted for which model was used. And <strong>retrieval buys nothing</strong>:
RAG versus full-document inference is −0.011 (p&nbsp;=&nbsp;.47), a null.</p>

<div class="correction"><p><strong>What we cannot estimate, and will not plot.</strong>
The RAG branch has 36 trials against roughly 50 parameters. An unguarded fit returned
coefficients of 10<sup>10</sup> with standard errors of 10<sup>16</sup> — it would have looked
like a filled-in table. After pruning to levers with at least five trials per level, only
chunker, embedder and query-embedding survive. <strong>Chunk size, chunk overlap, retriever,
k, reranking and token budget have no support at all</strong> and are reported as
"not identified".</p></div>

{figure("F4_humans",
  "Left: per-item human disagreement against model accuracy. Right: model accuracy and two-coder agreement by question type.")}

<div class="finding"><p><strong>The model is scored against a moving target.</strong>
Under the same normalisation the pipeline is scored with, the two human coders disagree on
25.7% of cells — 51% on open-ended items. So ~79% against the adjudicated gold sits against
a ~74% human ceiling, and the items the model fails are the items humans argue about
(r&nbsp;=&nbsp;−0.26). Raw per-item accuracy will mislead readers unless we also report it
normalised by the ceiling.</p></div>

{figure("F5_eval_size",
  "Smallest accuracy difference detectable at 80% power, by evaluation-set size.")}

<p>At 35 tuning papers we can only resolve about {mde35*100:.1f} points — larger than the
spread among our best configurations. Detecting 2 points needs roughly 100 papers. Pairing
two configurations on the same papers buys only ~16%, because papers are not uniformly easy;
so racing and early stopping help by killing bad configurations quickly, not by resolving
close calls.</p>

<div class="sec-head"><span class="tag">Part 2</span><h2>Nine decisions</h2></div>
<p class="dim">In the order they were raised. Each ends with what we need to settle on the call.</p>

{decision(1, "Cost and whether to cap it", '''
<p>Cost is now reconstructed for all 93 recoverable trials at zero API spend. Worth saying
plainly: two-coder human extraction runs roughly $40–80 per paper, so even our most expensive
configuration is 20–40× cheaper than people. A cap is a budget constraint, not a scientific
one. Proposal: a hard eligibility cap of <strong>$0.50/paper</strong>, which excludes almost
nothing currently competitive, and report accuracy against cost as a second dimension rather
than optimising a blend. Batched mode should be promoted to a first-class arm.</p>''',
"Do we accept $0.50/paper as the eligibility cap, and what is the real remaining API budget?")}

{decision(2, "How the search should work", '''
<p>TPE never engaged — 96 completed trials against <code>n_startup_trials=100</code> — so every
trial came from the random prior. It also cannot exclude anything, only down-weight, and the
objective is a mean over 35 noisy instances. Proposal: a preregistered three-stage
<em>screen → race → confirm</em>. Screen levels on the 116k cells we already have (free);
race configurations in a new study with <code>WilcoxonPruner</code> and a grouped multivariate
sampler; confirm the top five on the full tuning set with a written tie-break ladder. Exclusion
rule: drop a level only when the upper bound of its 95% interval is below −0.02, with a floor
that no level with fewer than five trials is ever excluded on statistical evidence.</p>''',
"Accept “stable region, not argmax” as the goal, the exclusion rule, and ~$450–850 for the remaining trials?")}

{decision(3, "Reporting the design effects", '''
<p>Spec A above is the confirmatory answer to RQ1.2, with one honesty clause: the trials are an
unbalanced sample with cost-correlated missingness, so these are adjusted associations over the
region we sampled, not clean causal contrasts. The one-factor-at-a-time ladder around the
frozen winner is what converts the highest-stakes comparisons into real experiments.</p>''',
"Bayesian hierarchical model (handles the sparse levels gracefully) or frequentist? We recommend Bayesian.")}

{decision(4, "Question type and rater disagreement", '''
<p>Both are now computable and the normalisation is settled. Two asks. First, report
<strong>ceiling-normalised accuracy</strong> alongside raw, or items like 5.2 and 1.3 will read
as model failures when they are codebook failures. Second, a structural problem: the
<code>diff_human</code> and <code>diff_llm</code> ratings are per study × rater, <em>not</em>
per item, so RQ1.7 is only answerable at study level as designed. A 20-minute item-level
difficulty ranking from two to four coders would upgrade it to a genuine preregistered
prediction — but it has to happen before the test set is scored.</p>''',
"Do we run the item-level difficulty elicitation, and accept ceiling-normalised accuracy as co-primary?")}

{decision(5, "The agentic arm", '''
<p>Not in the Phase&nbsp;1 preregistration, so it would enter as exploratory-confirmatory:
conditions and hypotheses fixed in advance, all six run on the tuning set only, and only the
winner touches the test set. Six conditions isolate agency, the codebook, and examples — the
interesting one being whether a model that <em>writes its own codebook</em> from ten labelled
examples matches an agent given ours. Roughly $85–150 and a week of wall-clock. It is the
cheapest genuinely novel thing on this list.</p>''',
"Approve the arm and its framing, with gpt-oss-120b fixed as the single driver model?")}

{decision(6, "Calibration", '''
<p>With 15 calibration papers we have 645 cells — enough for a marginal conformal guarantee,
not enough per item, and the list type fails outright. The near-free fix: move 35 papers from
test to calibration using the same seed-42 permutation, which gives 50 calibration papers and
<em>restores the test set to the preregistered 163</em> rather than shrinking it. Everything so
far ran at temperature 0, so self-consistency needs new runs; it is also the only uncertainty
signal available for the five models that expose no log-probabilities.</p>''',
"Approve the re-split (calibration 50 / test 163), and a learned combined score or plain log-probabilities?")}

{decision(7, "Conformal prediction and propagating uncertainty", '''
<p>For multiple-choice, do not conformalise the power set — conformalise each option
separately, which maps onto the binary machinery we already have. Set the operating point at
the smallest flag rate whose auto-accepted accuracy beats the two-coder agreement rate on those
same cells, so the success criterion is “at least as good as a second human coder”. Downstream,
prediction-powered inference over all 500 papers, with the flip indicator against the review's
own thresholds. If a substantive conclusion flips across our top five configurations, that is
the most important finding in the paper.</p>''',
"Which eight adherence estimands? Please bring the review's claim list to the call.")}

{decision(8, "Evaluation-set size", '''
<p>Reduce the <em>search</em> set, never the evaluation set — conflating the two is the trap.
The test set carries RQ1.1 and every subgroup analysis, and it is the one place lost power
costs a preregistered question. For screening, 20 papers and 25 items is about a third of the
per-trial cost and still resolves the ~4.6 point differences that matter at that stage, with
the full set restored at confirmation.</p>''',
"Approve 20 papers × 25 items for screening only, with the stated hard-item selection bias?")}

{decision(9, "The model roster", '''
<p>Adding a model invalidates the existing study — a new categorical level makes every recorded
trial incompatible. But the search redesign forces a new study anyway, so the roster refresh is
free. Re-run the entry exam against the live provider catalogues rather than any published
list, freeze it to a dated snapshot, verify log-probability support empirically rather than
trusting metadata, and cut from 17 models to about 12 so each level gets more trials. Then run a
model-swap-only sweep at the frozen winning configuration — that paired curve is a far cleaner
answer to RQ1.5 than regressing over unbalanced trials.</p>''',
"Fresh study, 12 models with at least 8 exposing log-probabilities, and the paired swap as the primary RQ1.5 analysis?")}

<div class="sec-head"><span class="tag">Reference</span><h2>Where the numbers come from</h2></div>
<div class="scroll">
<table>
<thead><tr><th>Artifact</th><th>What it holds</th><th class="n">Scale</th></tr></thead>
<tbody>
<tr><td><code>cells.parquet</code></td><td>one row per trial × paper × item (87 analysed)</td><td class="n">144,742</td></tr>
<tr><td><code>rescored_cells.parquet</code></td><td>concat-fix sidecar, raw artifacts untouched</td><td class="n">70,685</td></tr>
<tr><td><code>tokens.parquet</code></td><td>reconstructed tokens and cost per trial</td><td class="n">100</td></tr>
<tr><td><code>human_agreement.parquet</code></td><td>two-coder agreement per paper × item</td><td class="n">1,505</td></tr>
<tr><td><code>recovered_cells.parquet</code></td><td>outputs + logprobs rebuilt from the raw batch files</td><td class="n">26,723</td></tr>
<tr><td><code>check_integrity.py</code></td><td>the gate that caught the output-less trials</td><td class="n">—</td></tr>
<tr><td><code>stats.py</code></td><td>Spec A / B / D, with a rank guard</td><td class="n">—</td></tr>
</tbody>
</table>
</div>
<p class="note">Token reconstruction validated per route against the 18 trials that recorded
true counts: batched {cal.get('full_batched',{}).get('mape',0)*100:.1f}% (n={cal.get('full_batched',{}).get('n',0)}) ·
RAG {cal.get('rag_context',{}).get('mape',0)*100:.1f}% ·
individual {cal.get('full_individual',{}).get('mape',0)*100:.1f}% (n={cal.get('full_individual',{}).get('n',0)}).</p>

<footer>
Every figure regenerates from the cached parquet files with no network access.
Raw artifacts — <code>optuna_study.db</code> and <code>trial_logs/</code> — are read-only
throughout; the scoring fix is a sidecar, not a rewrite.
</footer>
</div>
"""

OUT.write_text(HTML)
print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
