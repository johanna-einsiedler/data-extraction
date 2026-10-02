import sys, io, contextlib, os, pandas as pd
from dotenv import load_dotenv; load_dotenv(".env"); os.environ["JUDGE_BACKEND"]="local"; os.environ["EQUIV_JUDGE_MODEL"]="microsoft/Phi-3-mini-4k-instruct"; os.environ.setdefault("JUDGE_DTYPE","bfloat16")
sys.path.insert(0,".")
from analysis.hpo_analysis import unified_rescore as ur
S=sys.argv[1]; nums=[int(x) for x in sys.argv[2:]]
ev,Q,TRUE,CUR=ur._evaluator()
p1=ur._study(ur.P1_DB); params={t.number:dict(t.params) for t in p1.trials if t.value is not None}
rec=pd.read_parquet("analysis/cache/recovered_cells.parquet"); blackout=set(range(382,399))
for n in nums:
    if os.path.exists(f"{S}/resc/{n}.parquet"): continue
    f=ur.match_source_folder(n,params[n])
    with contextlib.redirect_stdout(io.StringIO()):
        r=pd.DataFrame(ur.rescore_folder(f,ev,Q,TRUE,CUR)); r["p1_number"]=n
        if n in blackout:
            st=r.set_index(["doc_name","query_id"]).accuracy_stored; rr=[]
            for _,c in rec[rec.trial_number==n].iterrows():
                qid=str(c.query_id)
                if qid not in Q: continue
                e=ev(c.raw_output,question_type=Q[qid]["type"],doc_name=str(c.doc_name),query_id=qid,TRUE_ANSWERS=TRUE,QUERIES=Q,other_callback=None,curated_mc_items=CUR)
                rr.append({"doc_name":str(c.doc_name),"query_id":qid,"accuracy_stored":st.get((str(c.doc_name),qid)),"accuracy_unified":e["accuracy"],"rescored":True,"p1_number":n})
            r=pd.DataFrame(rr)
    r.to_parquet(f"{S}/resc/{n}.parquet",index=False)
    print(f"#{n} cells={len(r)} rescored={int(r.rescored.sum())} stored={r.accuracy_stored.mean():.4f} unified={r.accuracy_unified.mean():.4f}",flush=True)
