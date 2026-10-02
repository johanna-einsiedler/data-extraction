"""Which judge model scored each trial's open-ended cells? The verdict cache is keyed on
sha1(JUDGE_MODEL_NAME, version, gold, pred). Run once per candidate model name with the
backends blocked: a cache hit means that model judged this (gold, pred); the returned
score is compared with the stored one. Output: per-trial counts.
usage: EQUIV_JUDGE_MODEL=<name> JUDGE_BACKEND=together python judge_attribution.py out.parquet"""
import os, sys, io, json, contextlib, glob, pandas as pd, optuna
sys.path.insert(0,"."); sys.path.insert(0,"src")
import evaluate as E
class Miss(Exception): pass
E._judge_via_together=lambda *a,**k:(_ for _ in ()).throw(Miss())
E._load_judge=lambda *a,**k:(_ for _ in ()).throw(Miss())
from analysis.hpo_analysis import unified_rescore as ur
ev,Q,TRUE,CUR=ur._evaluator(); optuna.logging.set_verbosity(optuna.logging.WARNING)
rec=pd.read_parquet("analysis/cache/recovered_cells.parquet"); rec["query_id"]=rec.query_id.astype(str)
targets=[]
p1=ur._study(ur.P1_DB)
for t in p1.trials:
    if t.state.name=="COMPLETE" and t.value is not None: targets.append(("phase1",t.number,dict(t.params),t.datetime_start))
for db in ("phase2","open_tpe","consumer_tpe"):
    st=optuna.load_study(study_name="full_item_accuracy_tuning",storage=f"sqlite:///stage2_analyses/phase2_study/{db}.db")
    for t in st.trials:
        if t.state.name=="COMPLETE" and t.value is not None and "phase1_trial" not in t.user_attrs: targets.append((db,t.number,dict(t.params),t.datetime_start))
rows=[]
for src,n,params,start in targets:
    f=ur.match_source_folder(n,params)
    if f is None: rows.append(dict(src=src,number=n,start=start,status="no_folder")); continue
    ts=json.load(open(f/"trial_summary.json")); cells=ts["per_query_doc_results"]
    id2=ur.doc_id_to_name(f) if any("doc_name" not in c for c in cells) else {}
    hit=miss=agree=nojudge=0
    for c in cells:
        qid=str(c["query_id"])
        if qid not in ("1.3","1.7"): continue
        out=c.get("raw_output") if c.get("raw_output") is not None else c.get("llm_output")
        doc=str(c.get("doc_name") or id2.get(c.get("doc_id")))
        if out is None and src=="phase1":
            r=rec[(rec.trial_number==n)&(rec.doc_name.astype(str)==doc)&(rec.query_id==qid)]
            out=r.raw_output.iloc[0] if len(r) else None
        if out is None: continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                e=ev(out,question_type=Q[qid]["type"],doc_name=doc,query_id=qid,TRUE_ANSWERS=TRUE,QUERIES=Q,other_callback=None,curated_mc_items=CUR)
            if e.get("reason") in ("normalized-exact","empty-answer"): nojudge+=1
            else:
                hit+=1; agree+=int(e["accuracy"]==c.get("accuracy"))
        except Miss: miss+=1
    rows.append(dict(src=src,number=n,start=start,status="ok",hit=hit,miss=miss,agree=agree,nojudge=nojudge))
pd.DataFrame(rows).to_parquet(sys.argv[1],index=False); print("done",len(rows))
