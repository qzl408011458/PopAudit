import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[2])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import sys
from pathlib import Path
import pandas as pd, numpy as np
S=Path(_PA.R("/supplementary_experiments/sigma_sensitivity"))
sys.path.insert(0,str(S/"code")); sys.path.insert(0,str(S))
import decision_certificate_xstate_v3 as DC
from sigma_sens import cases, vdir, SRC
cert=pd.read_csv(Path(SRC)/"PAPER_xstate_v3/xstate_certificate_all.csv")
rows=[]
for v in ["base","beta0","floor0025","floor00525","floor01","hfree","lam05","lam145","lam2","mu05","mu13","mu2","win1013"]:
    DC.BASE=vdir(v)
    for m,st,y in cases():
        l2=DC.load_layer2(st,m,y); t=DC.layer2_test(l2,DC.ALPHA[1])
        rows.append(dict(variant=v,model=m,state=st,year=int(y),chi=l2["chi_H"],cut=t["cutoff"],L2=t["accept"]))
R=pd.DataFrame(rows)
b=R[R.variant=="base"].merge(cert,on=["model","state","year"])
print("n",len(b),"max|chi|",(b.chi-b.L2_chi_H).abs().max(),"max|cut|",(b.cut-b.L2_cutoff).abs().max(),
      "L2 flags match",(b.L2==b.L2_accept).sum(),"L2 pass",b.L2.sum(), "VALID prod",b.VALID.sum(), "L2&L3prod", (b.L2&b.L3_accept).sum())
# also compare to production layer2 dir chi (unrounded)
DC.BASE=Path(SRC)
mx=0
for r in b.itertuples():
    l2=DC.load_layer2(r.state,r.model,str(r.year)); t=DC.layer2_test(l2,DC.ALPHA[1]); mx=max(mx,abs(l2["chi_H"]-r.chi),abs(t["cutoff"]-r.cut))
print("max diff vs production L2 dirs (unrounded):",mx)
s=R.groupby(["variant","year"]).L2.sum().unstack(); s["tot"]=s.sum(1); print(s)
print(R[R.year==2021].groupby("variant").apply(lambda g: g.loc[g.chi.idxmin(),["model","state","chi","cut"]]))
R.to_csv(_PA.R("/supplementary_experiments/sigma_sensitivity/results/l2_all.csv"),index=False)
