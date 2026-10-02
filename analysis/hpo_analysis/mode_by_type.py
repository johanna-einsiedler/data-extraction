"""Matched-configuration comparison of extraction modes: accuracy by question
type, tokens and cost. Groups = (model, parser) with >=2 modes among
individual / batched / RAG; best member per mode by unified accuracy; all
accuracies on the common (doc, item) cells of the group's selected members."""
import sys, json, optuna, pandas as pd, numpy as np
S="analysis/cache/mode_by_type"   # resc/ = Phi-3-judged cells (executed judge); gptoss_sensitivity/ = gpt-oss-120b re-judge
R="/Users/einsie0004/Documents/research/23_paper_extraction/pipeline_june_26"
optuna.logging.set_verbosity(optuna.logging.WARNING)
typ=pd.read_parquet(f"{R}/analysis/cache/cells.parquet",columns=["query_id","type"]).drop_duplicates("query_id").set_index("query_id")["type"]
TY={"multiple_choice":"MC","single_choice":"SC","numeric":"Num","open_ended":"OE","list":"List"}
import glob, os
# IND_DIR: per-trial unified cells of the individual-mode members (Phi-3 judge).
# USE_OE_OVERLAY=1 + OE_FILE: replace partner open-ended verdicts (used for #387,
# whose stored OE scores came from gpt-oss-120b in the Aug-4/5 batch).
USE_OE_OVERLAY=os.getenv("USE_OE_OVERLAY","1")=="1"
IND_DIR=os.getenv("IND_DIR",f"{S}/resc")
ind=pd.concat([pd.read_parquet(f) for f in glob.glob(f"{IND_DIR}/*.parquet") if not f.endswith("partner_oe.parquet")])
uni=pd.read_parquet(f"{R}/analysis/cache/unified_rescore_cells.parquet")
p2s=pd.read_parquet(f"{S}/p2_partner_cells_stored.parquet")
tok=pd.read_parquet(f"{R}/analysis/cache/tokens.parquet").set_index("trial_number")
OE=pd.read_parquet(os.getenv("OE_FILE",f"{S}/partner_oe_phi3.parquet")) if USE_OE_OVERLAY else pd.DataFrame(columns=["src","number","doc_name","query_id","oe_unified"]); OE["doc_name"]=OE.doc_name.astype(str)
def cells(src,n):
    if src=="ind": d=ind[ind.p1_number==n]
    elif src=="p1": d=uni[uni.p1_number==n]
    elif src=="p2fix": d=uni[(uni.p2_number==n)&(uni.p1_number==n)]
    else:
        db,num=src.split(":"); d=p2s[(p2s.db==db)&(p2s.number==int(num))].rename(columns={"accuracy_stored":"accuracy_unified"})
    d=d[["doc_name","query_id","accuracy_unified"]].copy(); d["doc_name"]=d.doc_name.astype(str); d["query_id"]=d.query_id.astype(str)
    d=d.drop_duplicates(["doc_name","query_id"])
    if src!="ind":   # overlay open-ended cells re-judged with the Together judge
        key=("p1" if src in("p1","p2fix") else src.split(":")[0], n) if src!="p2fix" else ("phase2",n)
        o=OE[(OE.src==key[0])&(OE.number==key[1])]
        if len(o):
            d=d.merge(o[["doc_name","query_id","oe_unified"]],how="left",on=["doc_name","query_id"])
            d["accuracy_unified"]=d.oe_unified.fillna(d.accuracy_unified); d=d.drop(columns="oe_unified")
    return d
def usage(src,n):
    """per-paper prompt tokens, completion tokens, calls, usd; flag recorded/estimated"""
    if src in("ind","p1"):
        r=tok.loc[n]; nd=r.n_docs
        rec=pd.notna(r.prompt_tokens_true)
        pt=r.prompt_tokens_true if rec else r.prompt_tokens_cal
        ct=r.completion_tokens_true if rec else r.completion_tokens_lo
        ca=r.llm_calls_true if rec else r.n_calls_est
        return dict(pt=pt/nd if pd.notna(pt) else np.nan,ct=ct/nd if pd.notna(ct) else np.nan,calls=ca/nd if pd.notna(ca) else np.nan,usd=r.usd_per_paper_cal,rec=bool(rec))
    db,num=(src.split(":") if ":" in src else ("phase2",n))
    st=optuna.load_study(study_name="full_item_accuracy_tuning",storage=f"sqlite:///{R}/stage2_analyses/phase2_study/{db}.db")
    t=[x for x in st.trials if x.number==int(num)][0]; u=t.user_attrs
    nd=max(1,round(len(cells(src,n))/43))   # papers actually completed (34 or 35)
    return dict(pt=u["prompt_tokens"]/nd,ct=u["completion_tokens"]/nd,calls=u["llm_calls"]/nd,usd=u["cost_usd"]/nd,rec=True)
# group -> list of (label, src, number, mode, era)
G=[
 ("GLM-5.2","PyMuPDF+Tess.",[("ind",348,"individual","P1"),("ind",397,"individual","P1"),("p1",368,"batched","P1")]),
 ("Kimi-K2.6","PyMuPDF+Tess.",[("ind",404,"individual","P1"),("p1",411,"batched","P1"),("p2fix",92,"batched","P2"),("p2fix",108,"RAG","P2")]),
 ("Kimi-K2.7-Code","PyMuPDF",[("ind",354,"individual","P1"),("p1",10,"RAG","P1"),("open_tpe:94",94,"batched","P2")]),
 ("Qwen3.5-9B","MinerU",[("ind",398,"individual","P1"),("p1",353,"batched","P1"),("p1",436,"RAG","P1")]),
 ("cogito-v2.1-671B","Gemini",[("ind",375,"individual","P1"),("p1",473,"RAG","P1")]),
 ("cogito-v2.1-671B","MinerU",[("ind",389,"individual","P1"),("ind",416,"individual","P1"),("p1",480,"RAG","P1")]),
 ("gemini-3.1-pro","Gemini",[("ind",485,"individual","P1"),("ind",488,"batched","P1"),("phase2:132",132,"batched","P2")]),
 ("gemini-3.1-pro","PyMuPDF",[("ind",336,"individual","P1"),("ind",487,"individual","P1"),("p1",14,"RAG","P1")]),
 ("gpt-4.1","PyMuPDF",[("ind",329,"individual","P1"),("p1",26,"RAG","P1")]),
 ("gpt-5.2","MinerU",[("ind",503,"individual","P1"),("p1",346,"batched","P1"),("p1",481,"RAG","P1")]),
 ("gpt-oss-120b","PyMuPDF+Tess.",[("ind",390,"individual","P1"),("ind",32,"individual","P1"),("p1",387,"batched","P1")]),
 ("ministral-8b","PyMuPDF",[("ind",341,"individual","P1"),("ind",361,"individual","P1"),("p1",340,"batched","P1"),("consumer_tpe:64",64,"batched","P2")]),
]
rows=[]; pairs=[]; pooled=[]
for model,parser,members in G:
    # best per (mode, era)
    best={}
    for src,n,mode,era in members:
        d=cells(src,n); acc=d.accuracy_unified.mean()
        k=(mode,era)
        if k not in best or acc>best[k][2]: best[k]=(src,n,acc,d)
    common=None
    for k,(src,n,acc,d) in best.items():
        s=set(zip(d.doc_name,d.query_id)); common=s if common is None else common&s
    for (mode,era),(src,n,acc,d) in sorted(best.items(),key=lambda kv:["individual","batched","RAG"].index(kv[0][0])*2+(kv[0][1]=="P2")):
        d=d[[ (a,b) in common for a,b in zip(d.doc_name,d.query_id)]].copy(); d["ty"]=d.query_id.map(typ).map(TY)
        bt=d.groupby("ty").accuracy_unified.mean()
        u=usage(src,n)
        rows.append(dict(model=model,parser=parser,mode=mode,era=era,trial=n,src=src,n=len(d),overall=d.accuracy_unified.mean(),
                         **{t:bt.get(t,np.nan) for t in ["MC","SC","Num","OE","List"]},**u))
    # pooled paired deltas: best member per mode ACROSS eras vs best individual
    gi=[r for r in rows if r["model"]==model and r["parser"]==parser]
    bi=max([r for r in gi if r["mode"]=="individual"],key=lambda r:r["overall"],default=None)
    if bi is not None:
        for mode in ("batched","RAG"):
            cand=[r for r in gi if r["mode"]==mode]
            if cand:
                r=max(cand,key=lambda r:r["overall"])
                pooled.append(dict(model=model,parser=parser,contrast=f"{mode} − individual",era=r["era"],
                    **{t:r[t]-bi[t] for t in ["overall","MC","SC","Num","OE","List"]},
                    cost_ratio=(bi["usd"]/r["usd"]) if (pd.notna(r["usd"]) and pd.notna(bi["usd"]) and r["usd"]>0) else np.nan))
    # paired deltas vs individual (P1), on common cells, per type
    if ("individual","P1") in best:
        bi=rows[[i for i,r in enumerate(rows) if r["model"]==model and r["parser"]==parser and r["mode"]=="individual"][0]]
        for r in rows:
            if r["model"]==model and r["parser"]==parser and r["mode"]!="individual":
                pairs.append(dict(model=model,parser=parser,contrast=f"{r['mode']} ({r['era']}) − individual",
                                  **{t:r[t]-bi[t] for t in ["overall","MC","SC","Num","OE","List"]},
                                  cost_ratio=(bi["usd"]/r["usd"]) if (pd.notna(r["usd"]) and pd.notna(bi["usd"]) and r["usd"]>0) else np.nan))
df=pd.DataFrame(rows); pr=pd.DataFrame(pairs); po=pd.DataFrame(pooled)
po.to_csv(f"{S}/mode_pooled.csv",index=False)
df.to_csv(f"{S}/mode_table_rows.csv",index=False); pr.to_csv(f"{S}/mode_pairs.csv",index=False)
pd.set_option("display.width",260)
print(df.round(3).to_string(index=False)); print(); print(pr.round(3).to_string(index=False))
print("\npooled by contrast:"); print(pr.groupby("contrast")[["overall","MC","SC","Num","OE","List","cost_ratio"]].agg(["mean","count"]).round(3).to_string())
print("\npooled across eras (best member per mode):"); print(po.groupby("contrast")[["overall","MC","SC","Num","OE","List","cost_ratio"]].agg(["mean","median","count"]).round(3).to_string())
