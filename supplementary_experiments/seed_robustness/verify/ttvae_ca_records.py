import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[3])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import numpy as np, pandas as pd
S=_PA.R("/supplementary_experiments/seed_robustness/final/ttvae_CA")
P42=_PA.S("/hpo_v3/final/ttvae_CA/ttvae_CA_final_s42")
tp=np.load(f"{S}/ttvae_CA_final_s142/train_processed.npy",allow_pickle=True); print("train_processed",tp.shape)
tr=pd.read_csv(_PA.R("/data/pums_states/CA/traindata_pers.csv"),low_memory=False)
te=pd.read_csv(_PA.R("/data/pums_states/CA/testdata_pers.csv"),low_memory=False)
for rid in (0,17):
    G={s:pd.read_csv((P42 if s==42 else f"{S}/ttvae_CA_final_s{s}")+f"/generated_ensembles/final_generated_samples_{rid}.csv") for s in (42,142,242)}
    cols=list(G[42].columns)
    key=lambda d: d[cols].astype(str).agg("|".join,axis=1)
    ktr=set(key(tr[cols])); kte=set(key(te[cols]))
    for s,g in G.items():
        k=key(g); print(rid,s,"in train(1M)",round(k.isin(ktr).mean(),4),"in test",round(k.isin(kte).mean(),4))
    for a,b in ((42,142),(42,242),(142,242)):
        print(rid,a,b,"rowwise identical",round((key(G[a])==key(G[b])).mean(),4))
