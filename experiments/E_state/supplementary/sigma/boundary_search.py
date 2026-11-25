import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[4])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import sys
sys.argv=["x"]
from pathlib import Path
S=Path(_PA.R("/experiments/E_state/supplementary/sigma")); sys.path.insert(0,str(S))
import importlib.util, io, contextlib
# reuse ladder's data/admit without running its main loop: exec top part only
src=(S/"ladder.py").read_text().split("base = dict(")[0]
g={"__file__":str(S/"ladder.py")}; exec(compile(src,"ladder_head","exec"),g)
admit=g["admit"]
base=dict(floor=0.005,mu=1.0,lam=1.0,beta=1.0,H=None)
def n21(par,x):
    p=dict(base); p[par]=x; a=g["admit"](p); return a[(a.year==2021)&a.adm]
for par,lo,hi in (("floor",0.00525,0.0055),("mu",1.40,1.45),("lam",1.55,1.60)):
    for _ in range(18):
        mid=(lo+hi)/2
        if len(n21(par,mid)): hi=mid
        else: lo=mid
    a=n21(par,hi); print(par,"critical ~",round(lo,6),round(hi,6),list(zip(a.model,a.state)))
# base decomposition TTVAE OH 2021 and nearest cases
import pandas as pd, numpy as np
for m,st in (("ttvae","OH"),("ttvae","TX"),("semapop_vae","TX"),("bn","TX"),("semapop_vae","OH")):
    d=Path(_PA.R("/experiments/E_state/audits_v3"))/f"layer2_{m}_{st}_2021"
    w=pd.read_csv(d/"layer2_per_realization_weights.csv"); sig=pd.read_csv(d/"layer2_historical_sigma_summary.csv")
    att={c[len("attribute_loss__"):]:w[c].mean() for c in w.columns if c.startswith("attribute_loss__")}
    cm=sig[sig.attr=="Commute_Mode"]; nC=len(cm)
    wf=cm[cm.target=="Commute_Mode==Worked_at_home"].iloc[0]
    zw=0.5*(((w[wf.target]-wf.target_m)/wf.sigma)**2).mean()/nC
    print(m,st,"chi",round(w.scenario_loss.mean(),3),"commute",round(att["Commute_Mode"],3),"wfh part",round(zw,3),
          "sigma_wfh",wf.sigma,"target",round(wf.target_m,4),"prior mean wfh",round(w[wf.target].mean(),4))
