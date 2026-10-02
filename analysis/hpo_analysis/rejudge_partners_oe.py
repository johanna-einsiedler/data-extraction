"""Re-score only the open-ended cells (1.3, 1.7) of the batched/RAG partner trials
with the Together judge so every member of a group shares one judge."""
import sys, io, os, json, contextlib, optuna, pandas as pd
from dotenv import load_dotenv; load_dotenv(".env"); os.environ["JUDGE_BACKEND"]="together"
sys.path.insert(0,".")
from analysis.hpo_analysis import unified_rescore as ur
S=sys.argv[1]; optuna.logging.set_verbosity(optuna.logging.WARNING)
ev,Q,TRUE,CUR=ur._evaluator()
p1=ur._study(ur.P1_DB); P1={t.number:dict(t.params) for t in p1.trials if t.value is not None}
targets=[("p1",n,P1[n]) for n in (368,411,10,353,436,473,480,14,26,346,481,387,340)]
for db,n in (("phase2",92),("phase2",108),("phase2",63),("phase2",132),("open_tpe",94),("consumer_tpe",64)):
    st=optuna.load_study(study_name="full_item_accuracy_tuning",storage=f"sqlite:///stage2_analyses/phase2_study/{db}.db")
    t=[x for x in st.trials if x.number==n][0]; targets.append((db,n,dict(t.params)))
rows=[]
for src,n,params in targets:
    f=ur.match_source_folder(n,params); ts=json.load(open(f/"trial_summary.json"))
    cells=ts["per_query_doc_results"]; id2=ur.doc_id_to_name(f) if any("doc_name" not in c for c in cells) else {}
    k=0
    with contextlib.redirect_stdout(io.StringIO()):
        for c in cells:
            qid=str(c["query_id"])
            if qid not in ("1.3","1.7"): continue
            out=c.get("raw_output") if c.get("raw_output") is not None else c.get("llm_output")
            doc=str(c.get("doc_name") or id2.get(c.get("doc_id")))
            if out is None: continue
            e=ev(out,question_type=Q[qid]["type"],doc_name=doc,query_id=qid,TRUE_ANSWERS=TRUE,QUERIES=Q,other_callback=None,curated_mc_items=CUR)
            rows.append({"src":src,"number":n,"doc_name":doc,"query_id":qid,"oe_unified":e["accuracy"]}); k+=1
    print(f"{src}#{n}: {k} OE cells re-judged",flush=True)
pd.DataFrame(rows).to_parquet(f"{S}/resc/partner_oe.parquet",index=False); print("done")
