import pandas as pd, numpy as np, sys
S=sys.argv[1]; OUT=sys.argv[2]
df=pd.read_csv(f"{S}/mode_table_rows.csv"); po=pd.read_csv(f"{S}/mode_pooled.csv")
T=["overall","MC","SC","Num","OE","List"]
f3=lambda x:"---" if pd.isna(x) else f"{x:.3f}"; fd=lambda x:"---" if pd.isna(x) else f"{x:+.3f}"
fk=lambda x:"---" if pd.isna(x) else f"{x/1000:,.1f}"; fc=lambda x:"---" if pd.isna(x) else f"{x:.1f}"; fu=lambda x:"---" if pd.isna(x) else f"{x:.3f}"
t1=[]
for c,lab in (("batched − individual","batched $-$ individual"),("RAG − individual","RAG $-$ individual")):
    g=po[po.contrast==c]; cr=g.cost_ratio.dropna()
    t1.append(f"{lab} & {len(g)} & "+" & ".join(fd(g[t].mean()) for t in T)+f" & {cr.median():.0f}$\\times$ ({len(cr)}) \\\\")
    t1.append("\\quad median & & "+" & ".join(fd(g[t].median()) for t in T)+" & \\\\")
tab1="\n".join(t1)
rows=[]; prev=None
for _,r in df.iterrows():
    key=(r.model,r.parser)
    if prev is not None and key!=prev: rows.append("\\midrule")
    prev=key; est="" if r.rec else "$^{\\mathrm{e}}$"
    rows.append(f"{r.model} & {r.parser} & {r['mode']} & {r.era} & {int(r.trial)} & {int(r.n)} & "+" & ".join(f3(r[t]) for t in T)+f" & {fk(r.pt)}{est} & {fk(r.ct)}{est} & {fc(r.calls)} & {fu(r.usd)}{est} \\\\")
tab2="\n".join(rows)
g=lambda c:po[po.contrast==c]; B=g("batched − individual"); R=g("RAG − individual")
def m(x,t): return f"{x[t].mean()*100:+.1f}"
tex=r"""% ======================================================================
% Appendix: extraction mode x question type on matched configurations.
% Built by analysis/hpo_analysis/mode_by_type.py (pipeline repo); rows in
% analysis/tables/hpo/mode_by_type_{rows,pairs,pooled}.csv. Accuracies on the
% unified Phase-2 basis; open-ended verdicts from the judge the study actually
% executed (local Phi-3-mini, prompt v2-supersetrelaxed) for every member.
% \input{phase2_additions/appendix_mode_comparison_by_type}
% ======================================================================
\subsection{Extraction mode by question type: matched configurations}
\label{app:p2a-modes-type}

This appendix extends Appendix~\ref{app:p2a-modes} from overall accuracy to
accuracy by question type, token consumption and cost, for every
configuration in which the \emph{same model and parser} were run under at
least two of the three extraction modes: one call per item
(``individual''), one call per paper carrying all 43 items (``batched''),
and retrieval-augmented per-item calls (``RAG''). Twelve such
(model, parser) groups exist; three contain all three modes. Phase-1 and
Phase-2 runs are pooled: every member is scored on the unified Phase-2
basis (\S\ref{sec:metrics}) and, for the open-ended items, with the one
judge the study actually executed (local \texttt{Phi-3-mini}, prompt
\texttt{v2-supersetrelaxed}; see the deviations addendum for the judge
identity audit), so no contrast compares two judges. Within a group and
mode the member with the highest unified accuracy is used for the paired
contrasts (best-of-mode, hence observational), and all accuracies are
computed on the cells common to the group's selected members so that
differing paper coverage (two gemini runs skipped papers) cannot drive a
difference. The full table also lists the second batched member where a
group has one from each code era (P1: Phase-1 run on the inert
\texttt{batched\_native} prompt; P2: Phase-2 prompt-enabled run with
provider-recorded usage). The five individual-mode trials in the Phase-1
output blackout (\#389, \#390, \#396, \#397, \#398) were re-scored from the
recovered-outputs cache rather than dropped.

\begin{table}[h]
\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccccc}
\toprule
Contrast & $n$ & Overall & MC & SC & Num. & Open & List & cost ratio ($n$) \\
\midrule
TAB1
\bottomrule
\end{tabular}
\caption{Within-group difference in unified accuracy by question type
(MC: 21 items; SC: 10; numeric: 9; open-ended: 2; list: 1), one paired
difference per (model, parser) group between its best member of the named
mode and its best individual-mode member, pooled over Phase-1 and Phase-2
runs; mean and median shown. ``cost ratio'' is the median ratio of the
individual member's \$/paper to the partner's over the groups where both
are available.}
\label{tab:p2a-modes-type-pooled}
\end{table}

\begin{sidewaystable}
\centering\footnotesize
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{llllrrcccccc rrrr}
\toprule
Model & Parser & Mode & Era & Trial & cells & Overall & MC & SC & Num & Open & List
 & prompt k-tok/pp & compl.\ k-tok/pp & calls/pp & \$/pp \\
\midrule
TAB2
\bottomrule
\end{tabular}
\caption{Matched configurations: unified accuracy by question type on the
group's common cells, with per-paper prompt tokens, completion tokens,
LLM calls and cost (pp = per paper). Era P1 = Phase-1 run (batched members
on the inert prompt); P2 = Phase-2 prompt-enabled run. Token and cost
figures without a mark are provider-recorded; $^{\mathrm{e}}$ marks the
offline reconstruction of \S\ref{sec:p2a-search} (calibrated; validation
MAPE 3\% batched, 9\% RAG, 35\% individual, the last close to unbiased at
the median but noisy per trial); ``---'' marks members with neither
recorded usage nor a reconstructible prompt (VLM parser or missing
price). Phase-1 individual and RAG completion-token figures are lower
bounds. Calls/paper above 1 for a batched member reflect retried papers.}
\label{tab:p2a-modes-type-full}
\end{sidewaystable}

\paragraph{Reading.}
Pooled over the NB groups with a batched partner, batched extraction
differs from individual calls by B_OV~points overall (median B_MED),
with the type pattern more informative than the mean: multiple-choice
B_MC, single-choice B_SC, numeric B_NUM, open-ended B_OE, and the single
list item B_LIST. The list-item gap is a prompt-context effect rather
than a retrieval difference, and we checked the mechanism cell by cell: of
the individual members' 294 list errors, 122 contain a programming
language or environment (\texttt{R}, \texttt{Python}, \texttt{MATLAB},
\texttt{RStudio}, \texttt{SAS}) that the gold lists under the sibling
item 0.7.1, against 19 of 179 batched errors. Item 0.7.3 is defined as
``ML packages / libraries / modeling frameworks mentioned'' with no explicit
exclusion of languages; the batched prompt shows the model item 0.7.1 in
the same JSON and it routes \texttt{R} there, while the per-item prompt
presents 0.7.3 in isolation. A tolerant scorer does not close the gap
(Jaccard: batched 0.63 vs individual 0.39, against exact-set 0.53 vs
0.29), because adding two languages to a one-element gold set is a
substantive error under either rule. The contrast is therefore a fair
measurement of the modes as run, but not of model capability on that
item: a per-item prompt carrying the sibling-item definitions would
likely remove it (diagnostic in
\texttt{analysis/tables/hpo/mode\_by\_type\_list\_item\_diag.csv}). The pooled mean hides a sign change
between code eras: the inert-prompt Phase-1 batched members trail their
individual partners by P1_OV~points on average (P1_N groups; numeric
P1_NUM), while the prompt-enabled Phase-2 batched members lead them by
P2_OV (P2_N groups, P2_LEAD of which lead), so the batched deficit was a
property of the Phase-1 batched prompt rather than of the mode. RAG
(NR groups) differs by R_OV~points overall, concentrated in single-choice
(R_SC) and open-ended (R_OE) items whose answers are rarely co-located
with the retrieved chunks, while numeric items are nearly unaffected
(R_NUM); its cost advantage over individual calls is modest because it
still issues 43 calls per paper. The cost column makes the mode decision
of \S\ref{sec:p2a-search} concrete: within a group, individual mode costs
roughly an order of magnitude or more per paper than the batched member for
no systematic accuracy benefit once the prompt lever is live.

\paragraph{Caveats.}
Groups are observational and best-of-mode; they do not hold prompt,
reasoning or era fixed. Open-ended verdicts are those of the study's
executed judge (Phi-3-mini), which is more lenient than
\texttt{gpt-oss-120b} (deviations addendum, judge identity); this raises
open-ended levels in every row but applies identically to every member, so
the within-group contrasts are not affected by the choice of judge in
direction, only in size. Open-ended rests on 70 cells and list on 35 per
member, so those columns move by several points on a handful of cells.
Cost ratios mix recorded and reconstructed figures, and the
individual-mode reconstruction is the least precise of the three. Nothing
in this appendix is used for a confirmatory claim; its purpose is to
document, by question type, the evidence behind retiring the individual
tier and to give the per-paper token budget each mode implies for the test
run.
"""
pr=pd.read_csv(f"{S}/mode_pairs.csv"); e1=pr[pr.contrast=="batched (P1) − individual"]; e2=pr[pr.contrast=="batched (P2) − individual"]
rep={"P1_OV":f"{abs(e1.overall.mean())*100:.1f}","P1_N":str(len(e1)),"P1_NUM":f"{e1.Num.mean()*100:+.1f}",
     "P2_OV":f"{e2.overall.mean()*100:.1f}","P2_N":str(len(e2)),"P2_LEAD":str(int((e2.overall>0).sum()))}
rep.update({"TAB1":tab1,"TAB2":tab2,"NB":str(len(B)),"NR":str(len(R)),"B_OV":m(B,"overall"),"B_MED":f"{B.overall.median()*100:+.1f}",
     "B_MC":m(B,"MC"),"B_SC":m(B,"SC"),"B_NUM":m(B,"Num"),"B_OE":m(B,"OE"),"B_LIST":m(B,"List"),
     "R_OV":m(R,"overall"),"R_SC":m(R,"SC"),"R_OE":m(R,"OE"),"R_NUM":m(R,"Num")})
for k in sorted(rep,key=len,reverse=True): tex=tex.replace(k,rep[k])
open(OUT,"w").write(tex); print("tex written")
