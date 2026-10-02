"""Write the Phi-3 unified-basis values of the Aug-4/5 Phase-1 batch (#382-#398) into the
study record: the Phase-1 study value and every seed copy in phase2/open_tpe/consumer_tpe
(trials carrying user_attr phase1_trial). Dry run unless --write. Backs up each DB first and
stamps provenance attrs (unified_rescore spec id, judge, date)."""
import sys, glob, shutil, datetime, pandas as pd, optuna
sys.path.insert(0,"."); from analysis.hpo_analysis import unified_rescore as ur
optuna.logging.set_verbosity(optuna.logging.WARNING)
WRITE="--write" in sys.argv; TODAY=datetime.date.today().isoformat()
A="analysis/cache/mode_by_type/aug_phi3/resc"
new={}
for f in glob.glob(f"{A}/*.parquet"):
    d=pd.read_parquet(f); new[int(d.p1_number.iloc[0])]=float(d.accuracy_unified.mean())
print(f"{'dry run' if not WRITE else 'WRITING'}: {len(new)} rescored trials: {sorted(new)}")
def stamp(storage,trial,val):
    storage.set_trial_user_attr(trial._trial_id,"unified_rescore",ur.SPEC_ID)
    storage.set_trial_user_attr(trial._trial_id,"judge","microsoft/Phi-3-mini-4k-instruct")
    storage.set_trial_user_attr(trial._trial_id,"rescore_note",f"Aug-4/5 batch: scored from recovered outputs with Phi-3 on {TODAY}; prior value {trial.value:.4f} (gpt-oss-judged, pre-semicolon-fix)")
    storage.set_trial_state_values(trial._trial_id,state=trial.state,values=[val])
plans=[]
targets=[("phase1",str(ur.P1_DB))]+[(db,f"stage2_analyses/phase2_study/{db}.db") for db in ("phase2","open_tpe","consumer_tpe")]
for name,path in targets:
    st=optuna.load_study(study_name="full_item_accuracy_tuning",storage=f"sqlite:///{path}")
    for t in st.trials:
        src=t.number if name=="phase1" else (int(t.user_attrs["phase1_trial"]) if "phase1_trial" in t.user_attrs else None)
        if src in new and t.value is not None and abs(t.value-new[src])>1e-9:
            plans.append((name,t.number,src,t.value,new[src]))
            if WRITE: stamp(st._storage,t,new[src])
    if WRITE and any(p[0]==name for p in plans):
        shutil.copy(path,path+f".bak_before_aug_phi3_{TODAY}")
for p in plans: print(f"  {p[0]:12} #{p[1]:<4} <- p1#{p[2]:<4} {p[3]:.4f} -> {p[4]:.4f}")
print(f"{len(plans)} values {'written' if WRITE else 'would change'}")
