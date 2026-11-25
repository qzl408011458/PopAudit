"""[v3 port 2026-09-02: PAPER_xstate_v3; exemplar for panel (c) stays CA/ttvae (still L1-pass); docstring numbers below are from the v2 run -- see make_L2_commute_figure.log]
Concrete demonstration of the Layer-2 warning: when Layer 2 flags commute-mode
incompatibility (2021 COVID), force-projection CANNOT restore the flagged structure.
Three panels:
 (a) WFH share by horizon: realized surges (2.4->7.9->6.3%) while the projected and
     prior populations stay anchored near the 2013 level (~2%) -- the projection closes
     only ~3.5% of the gap at 2021.
 (b) post-projection commute-marginal TVD vs chi_H (r=+0.58): chi_H predicts how badly
     the projected population misses the realized commute distribution.
 (c) who works from home (2021): the projected WFH age composition equals the 2013 PRIOR,
     not the realized 2021 -- projection inherits, it does not restore.
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[4])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import ablation_lib as A
V2 = HERE.parent
WFH = "Worked_at_home"
AGES = ["0-17", "18-34", "35-49", "50-64", "65+"]

def add_ag(df):
    if "Age_Group" not in df.columns:
        df["Age_Group"] = pd.cut(pd.to_numeric(df.Age, errors="coerce"), [0,18,35,50,65,200],
            labels=AGES, right=False, include_lowest=True).astype(str)
    return df

c = pd.read_csv(V2/"xstate_certificate_all.csv"); c = c[c.L1_accept==True].copy(); c["year"]=c.year.astype(int)
rows=[]; ageC={}
for _,r in c.iterrows():
    fs=A._pw_files(r.model,r.state,str(r.year))
    if not fs: continue
    ens=add_ag(pd.concat([pd.read_csv(f) for f in fs],ignore_index=True))
    ref=add_ag(pd.read_csv(f"{_PA.DATA}/pums_states/{r.state}/reference_{r.year}_{r.state}.csv"))
    rw=pd.to_numeric(ref.PWGTP,errors="coerce").fillna(0).to_numpy(float); rw/=rw.sum()
    q=ens.projected_weight.to_numpy(float); q/=q.sum(); b=ens.base_weight.to_numpy(float); b/=b.sum()
    mq=(ens.Commute_Mode.astype(str)==WFH).values; mr=(ref.Commute_Mode.astype(str)==WFH).values
    cats=sorted(set(ref.Commute_Mode.astype(str))|set(ens.Commute_Mode.astype(str)))
    pr=np.array([rw[(ref.Commute_Mode.astype(str)==x).values].sum() for x in cats])
    pq=np.array([q[(ens.Commute_Mode.astype(str)==x).values].sum() for x in cats])
    rows.append(dict(state=r.state,model=r.model,year=r.year,L2pass=bool(r.L2_accept),chi=r.L2_chi_H,
        wfh_real=rw[mr].sum(),wfh_proj=q[mq].sum(),wfh_prior=b[mq].sum(),ctvd=0.5*np.abs(pq-pr).sum()))
    if r.state=="CA" and r.model=="ttvae":  # representative cell for panel (c)
        def agedist(df,w,m):
            s=pd.Series(w[m]).groupby(df.Age_Group.astype(str).values[m]).sum(); s=s.reindex(AGES).fillna(0); return (s/s.sum()).values
        ageC[r.year]=dict(real=agedist(ref,rw,mr),proj=agedist(ens,q,mq),prior=agedist(ens,b,mq))
D=pd.DataFrame(rows)

fig,ax=plt.subplots(1,3,figsize=(15,4.4)); fig.subplots_adjust(wspace=0.30,bottom=0.16)
YEARS=[2018,2021,2024]
# (a) WFH share by horizon
g=D.groupby("year").agg(real=("wfh_real","mean"),proj=("wfh_proj","mean"),prior=("wfh_prior","mean"))
xr=range(3)
ax[0].fill_between(xr,g.proj.values*100,g.real.values*100,color="#d62728",alpha=0.12,label="realized minus projected gap")
ax[0].plot(xr,g.real.values*100,"-o",color="#d62728",lw=2,label="realized (PUMS)")
ax[0].plot(xr,g.proj.values*100,"-o",color="#1f77b4",lw=2,label="projected $Q^\\star$")
ax[0].plot(xr,g.prior.values*100,"--o",color="#7f7f7f",lw=1.4,ms=4,label="2013 prior")
ax[0].set_xticks(list(xr)); ax[0].set_xticklabels(YEARS); ax[0].set_ylabel("work-from-home share (%)")
ax[0].set_title("(a) Force-projection cannot restore\nthe WFH surge",fontsize=10)
ax[0].legend(fontsize=8,loc="upper left")
# (b) within-2024 commute recovery by L2 verdict: horizon fixed -> no PASS/FAIL separation
d24=D[D.year==2024]
grp=[("L2 PASS","#1f77b4",d24[d24.L2pass].ctvd.values),("L2 FAIL","#d62728",d24[~d24.L2pass].ctvd.values)]
for i,(nm,col,vals) in enumerate(grp):
    xj=i+(np.arange(len(vals))-(len(vals)-1)/2)*0.05
    ax[1].scatter(xj,vals,c=col,s=54,alpha=0.85,edgecolor="w",lw=0.5)
    ax[1].hlines(vals.mean(),i-0.22,i+0.22,color="k",lw=2,zorder=5)
ax[1].set_xlim(-0.6,1.6); ax[1].set_xticks([0,1])
ax[1].set_xticklabels([f"L2 PASS\n(mean {grp[0][2].mean():.3f})", f"L2 FAIL\n(mean {grp[1][2].mean():.3f})"])
ax[1].set_ylabel("post-projection commute discrepancy (proj vs realized)")
ax[1].set_title("(b) Commute recovery does not\nseparate PASS from FAIL",fontsize=10)
# (c) who works from home, 2021
yv=ageC.get(2021); x=np.arange(len(AGES)); w=0.27
ax[2].bar(x-w,yv["real"]*100,w,color="#d62728",label="realized 2021")
ax[2].bar(x,   yv["proj"]*100,w,color="#1f77b4",label="projected $Q^\\star$")
ax[2].bar(x+w, yv["prior"]*100,w,color="#7f7f7f",label="2013 prior")
ax[2].set_xticks(x); ax[2].set_xticklabels(AGES,fontsize=8); ax[2].set_ylabel("share of WFH population (%)")
ax[2].set_title("(c) Who works from home:\nprojection inherits the 2013 prior",fontsize=10); ax[2].legend(fontsize=8)
for a in ax:
    for sp in ["top","right"]: a.spines[sp].set_visible(False)
for dd in [V2/"figures/png",V2/"figures/pdf"]: dd.mkdir(parents=True,exist_ok=True)
fig.savefig(V2/"figures/png/fig_L2_commute_unprojectable.png",dpi=150,bbox_inches="tight")
fig.savefig(V2/"figures/pdf/fig_L2_commute_unprojectable.pdf",bbox_inches="tight")
# v3: no copy to repo root (old manuscript figure must not be overwritten)
print("wrote fig_L2_commute_unprojectable (png+pdf)")
print("WFH by year (real/proj/prior %):"); print((g*100).round(2).to_string())
print("2024 commute discrepancy — PASS mean %.3f | FAIL mean %.3f" % (grp[0][2].mean(), grp[1][2].mean()))

# ---- standalone single-panel version: panel (a) only, no title ----
fig2,ax2=plt.subplots(figsize=(6.2,4.4)); fig2.subplots_adjust(bottom=0.12)
ax2.fill_between(xr,g.proj.values*100,g.real.values*100,color="#d62728",alpha=0.12,label="realized minus projected gap")
ax2.plot(xr,g.real.values*100,"-o",color="#d62728",lw=2,label="realized (PUMS)")
ax2.plot(xr,g.proj.values*100,"-o",color="#1f77b4",lw=2,label="projected $Q^\\star$")
ax2.plot(xr,g.prior.values*100,"--o",color="#7f7f7f",lw=1.4,ms=4,label="2013 prior")
ax2.set_xticks(list(xr)); ax2.set_xticklabels(YEARS); ax2.set_ylabel("work-from-home share (%)")
ax2.legend(fontsize=8,loc="upper left")
for sp in ["top","right"]: ax2.spines[sp].set_visible(False)
fig2.savefig(V2/"figures/png/fig_L2_wfh_unprojectable.png",dpi=150,bbox_inches="tight")
fig2.savefig(V2/"figures/pdf/fig_L2_wfh_unprojectable.pdf",bbox_inches="tight")
# v3: no copy to repo root
print("wrote fig_L2_wfh_unprojectable (single panel a, no title; png+pdf)")
