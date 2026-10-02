"""#387 (output blackout) has no raw outputs in its folder: re-judge its OE cells
from the recovered-outputs cache and append to partner_oe.parquet."""
import sys, io, os, contextlib, pandas as pd
from dotenv import load_dotenv; load_dotenv(".env"); os.environ["JUDGE_BACKEND"]="local"; os.environ["EQUIV_JUDGE_MODEL"]="microsoft/Phi-3-mini-4k-instruct"; os.environ.setdefault("JUDGE_DTYPE","bfloat16")
sys.path.insert(0,".")
from analysis.hpo_analysis import unified_rescore as ur
S=sys.argv[1]; ev,Q,TRUE,CUR=ur._evaluator()
rec=pd.read_parquet("analysis/cache/recovered_cells.parquet"); rec=rec[(rec.trial_number==387)&(rec.query_id.astype(str).isin(["1.3","1.7"]))]
rows=[]
with contextlib.redirect_stdout(io.StringIO()):
    for _,c in rec.iterrows():
        qid=str(c.query_id); e=ev(c.raw_output,question_type=Q[qid]["type"],doc_name=str(c.doc_name),query_id=qid,TRUE_ANSWERS=TRUE,QUERIES=Q,other_callback=None,curated_mc_items=CUR)
        rows.append({"src":"p1","number":387,"doc_name":str(c.doc_name),"query_id":qid,"oe_unified":e["accuracy"]})
oe=pd.DataFrame(rows)
oe.to_parquet(f"{S}/resc/partner_oe.parquet",index=False); print(f"#387: {len(rows)} OE cells re-judged from recovered outputs")
