"""
v3 port (v3 protocol, 2026-09-02) of
  experiments/E_state/audits/PAPER_xstate_v2/figures/make_transfer_figures.py
Edits vs. the v2 original are limited to: BASE/V2/FIG paths -> audits_v3 / PAPER_xstate_v3,
DISP map -> the 9 new model tags, hard-coded "/12" cell counts -> len(J), output sub-dirs
created if missing, and an additional (purely additive) statistics dump
(tables/transfer_stats.json) with the numbers quoted in the manuscript text.
Algorithms, thresholds and formulas are byte-identical to the v2 script.

Two transfer-check figures for the manuscript:
  (Cor. 1, "Implications for audited marginal functionals")  -> fig_cor1_marginal_transfer
  (App. B, "Task-aligned joint-summary extension")           -> fig_appB_joint_transfer

For each of the valid cells (16 in v3) we take the Layer-3 projected distribution Q* (saved
cell weights) and the PWGTP-weighted scenario reference, and compute:
  - marginal event-share deviations |E_{Q*}[l] - m*| for WFH / car-commute / zero-car HH /
    3+ children, bounded (Cor. 1) by the aggregate marginal discrepancy d_A(Q*);
  - the JOINT employed-senior share (Age_Group in {65+} AND employed), which is NOT a
    single-attribute event and is bounded (App. B / Cor. 2) by the joint
    Age x Employment aggregate discrepancy d_{g_AE}(Q*), while the univariate d_A does not
    control it.
Honest bounds (no composite C_MA).

FIX 3 (2026-11-24): d_A(Q*) now comes from the shared helper ../dA_union.py (the same union-category d_A
that make_figE_postproj.py uses for D; zero target for categories absent from the reference).  This script
already summed over the union, so its numbers are unchanged.  Optional: --out_dir DIR redirects all outputs.
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[5])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import json, sys, argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(_PA.R(""))
BASE = ROOT / "experiments/E_state/audits_v3"
V2 = BASE / "PAPER_xstate_v3"; FIG = V2/"figures"
sys.path.insert(0, str(V2)); from dA_union import dA_from_frames   # FIX 3: shared union-category d_A
_ap = argparse.ArgumentParser(); _ap.add_argument("--out_dir", default=None); _args = _ap.parse_args()
if _args.out_dir: FIG = Path(_args.out_dir)   # check runs only; xstate_valid_only.csv is still read from V2
for _sub in ("tables", "png", "pdf"):
    (FIG/_sub).mkdir(parents=True, exist_ok=True)
DISP = {"semapop_vae":"SemaPop-VAE+","ttvae":"TTVAE","tabddpm":"TabDDPM","bn":"BN",
        "wgan_zcr":"WGAN-GP-ZCR","ctgan":"CTGAN","flowsyn":"FlowSyn","borysov":"Borysov-VAE","tvae":"TVAE"}
ATTRS = ["Age_Group","Household_Size","Number_of_children","Number_of_cars_of_household",
         "Employment_status","Household_Income_Class","Commute_Mode"]
EMP = {"Civilian_Employed_At_Work","Civilian_Employed_Not_At_Work","Armed_Forces_At_Work","Armed_Forces_Not_At_Work"}

def add_ag(df):
    if "Age_Group" not in df.columns:
        df["Age_Group"]=pd.cut(pd.to_numeric(df.Age,errors="coerce"),[0,18,35,50,65,200],
            labels=["0-17","18-34","35-49","50-64","65+"],right=False,include_lowest=True).astype(str)
    return df
def wshare(df,mask,w):   # weighted share
    return float((mask.to_numpy(float)*w).sum())
def qpath(s,m,y):
    d=BASE/f"layer3_paperW_{m}_{s}_{y}"/"projected_weights"
    c=list(d.glob("*.csv")) if d.exists() else []
    return c[0] if c else None

FUN=[("WFH share", lambda d:d.Commute_Mode=="Worked_at_home","#1f77b4"),
     ("car-commute share", lambda d:d.Commute_Mode=="Car_truck_or_van","#ff7f0e"),
     ("zero-car HH share", lambda d:pd.to_numeric(d.Number_of_cars_of_household,errors="coerce")==0,"#2ca02c"),
     ("3+ children HH", lambda d:pd.to_numeric(d.Number_of_children,errors="coerce")>=3,"#9467bd")]

v=pd.read_csv(V2/"xstate_valid_only.csv"); v["year"]=v.year.astype(int).astype(str)
marg=[]; joint=[]
for _,r in v.iterrows():
    wp=qpath(r.state,r.model,r.year)
    if wp is None: continue
    Q=add_ag(pd.read_csv(wp)); q=Q.projected_weight.to_numpy(float); q=q/q.sum()
    ref=add_ag(pd.read_csv(ROOT/f"data/pums_states/{r.state}/reference_{r.year}_{r.state}.csv"))
    pw=pd.to_numeric(ref.PWGTP,errors="coerce").fillna(0).to_numpy(); pw=pw/pw.sum()
    # d_A(Q*) aggregate marginal discrepancy on the union of Q* and reference categories (shared helper)
    dA=dA_from_frames(Q,q,ref,pw,ATTRS)["d_A"]
    # marginal functionals
    for name,fn,col in FUN:
        eq=wshare(Q,fn(Q),q); ep=wshare(ref,fn(ref),pw)
        marg.append({"cell":f"{r.state}·{DISP[r.model]}·{r.year}","functional":name,"color":col,
                     "delta":abs(eq-ep),"d_A":dA})
    # joint employed-senior: two definitions of "older" (65+ and 50+). The joint bound
    # d_{g_AE} is the FULL Age x Employment table aggregate -> SAME for both subsets;
    # only the actual event-share deviation differs.
    def js_of(sen): return lambda d:(d.Age_Group.astype(str).isin(sen))&(d.Employment_status.astype(str).isin(EMP))
    d65=abs(wshare(Q,js_of({"65+"})(Q),q)-wshare(ref,js_of({"65+"})(ref),pw))
    d50=abs(wshare(Q,js_of({"50-64","65+"})(Q),q)-wshare(ref,js_of({"50-64","65+"})(ref),pw))
    # d_{g_AE}(Q*) joint aggregate discrepancy over Age_Group x Employment
    Q["_g"]=Q.Age_Group.astype(str)+"|"+Q.Employment_status.astype(str)
    ref["_g"]=ref.Age_Group.astype(str)+"|"+ref.Employment_status.astype(str)
    cats=set(Q["_g"])|set(ref["_g"])
    mq=Q.assign(w=q).groupby("_g")["w"].sum(); mr=ref.assign(w=pw).groupby("_g")["w"].sum()
    dG=0.5*sum(abs(float(mq.get(c,0.0))-float(mr.get(c,0.0))) for c in cats)
    joint.append({"cell":f"{r.state}·{DISP[r.model]}·{r.year}","delta65":d65,"delta50":d50,"d_A":dA,"d_gAE":dG})

M=pd.DataFrame(marg); J=pd.DataFrame(joint)
M.to_csv(FIG/"tables"/"tableG_marginal_transfer.csv",index=False)
J.to_csv(FIG/"tables"/"tableH_joint_transfer.csv",index=False)

def save(fig,n): fig.savefig(FIG/"png"/f"{n}.png",dpi=150,bbox_inches="tight"); fig.savefig(FIG/"pdf"/f"{n}.pdf",bbox_inches="tight"); plt.close(fig); print("wrote",n)

# ---------- FIG Cor 1: marginal transfer ----------
# 2026-11-23 revision: legend names follow the manuscript wording (internal keys unchanged,
# so tableG / transfer_stats.json keys are unaffected); axis labels say "deviation"/"bound".
FUN_LABEL={"WFH share":"work-from-home share","car-commute share":"car-commute share",
           "zero-car HH share":"zero-car household share","3+ children HH":"households with at least three children"}
fig,ax=plt.subplots(figsize=(6.2,5.6))
mx=max(M.d_A.max(),M.delta.max())*1.08
ax.plot([0,mx],[0,mx],"k--",lw=1,label="certificate boundary: deviation $= d_{\\mathcal{A}}$")
for name,_,col in FUN:
    s=M[M.functional==name]; ax.scatter(s.delta,s.d_A,c=col,s=45,alpha=0.85,edgecolor="w",lw=0.5,label=FUN_LABEL[name])
ax.fill_between([0,mx],[0,mx],mx,color="green",alpha=0.05)
ax.set_xlabel("realized reference–projection deviation")
ax.set_ylabel("bound  $d_{\\mathcal{A}}(Q_\\theta^\\star)$")
ax.set_xlim(0,mx); ax.set_ylim(0,mx)
npass=int((M.delta<=M.d_A).sum())
ax.legend(fontsize=8,loc="lower right"); ax.set_aspect("equal")
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
save(fig,"fig_cor1_marginal_transfer")

# ---------- FIG App B: joint transfer + why marginals don't control it ----------
def draw_joint(dcol, senlabel, outname):
    D=J[dcol]
    fig,(ax,ax2)=plt.subplots(1,2,figsize=(12,5.2)); fig.subplots_adjust(wspace=0.32)
    mx=max(J.d_gAE.max(),D.max())*1.1
    ax.plot([0,mx],[0,mx],"k--",lw=1,label="unit-slope boundary: $e_{g_{\\mathrm{AE}}} = d_{g_{\\mathrm{AE}}}$")
    ax.scatter(D,J.d_gAE,c="#8c564b",s=55,edgecolor="w",lw=0.6)
    ax.fill_between([0,mx],[0,mx],mx,color="green",alpha=0.05)
    npass=int((D<=J.d_gAE).sum())
    ax.set_xlabel(f"{senlabel} share deviation  $e_{{g_{{\\mathrm{{AE}}}}}}$")
    ax.set_ylabel("table-aligned bound  $d_{g_{\\mathrm{AE}}}(Q_\\theta^\\star)$")
    ax.set_xlim(0,mx); ax.set_ylim(0,mx); ax.set_aspect("equal"); ax.legend(fontsize=8,loc="lower right")
    ax.set_title("(a) Table-aligned bound")
    xi=np.arange(len(J)); w=0.55
    ax2.bar(xi,D,w,color="#8c564b",label=f"{senlabel} share deviation $e_{{g_{{\\mathrm{{AE}}}}}}$")
    # r22: markers only (no lines joining categorical cases); short case codes at 11 pt (>=7 pt at print scale)
    ax2.plot(xi,J.d_gAE,"o",color="#2ca02c",ms=6,label="table-aligned bound $d_{g_{\\mathrm{AE}}}$")
    ax2.plot(xi,J.d_A,"x",color="#d62728",ms=7,mew=1.5,label="univariate marginal aggregate $d_{\\mathcal{A}}$")
    CODE={"SemaPop-VAE+":"SP","TTVAE":"TT","TabDDPM":"DD","BN":"BN","WGAN-GP-ZCR":"WG"}
    short=[f"{c.split('·')[0]}{c.split('·')[2][2:]}-{CODE[c.split('·')[1]]}" for c in J.cell]   # e.g. CA18-SP
    ax2.set_xticks(xi); ax2.set_xticklabels(short,rotation=90,fontsize=11)
    ax2.set_xlim(-0.7,len(J)-0.3)
    ax2.set_ylabel("deviation or bound"); ax2.set_ylim(0,0.53); ax2.legend(fontsize=7.5,loc="upper left",ncol=3)
    sharp=(J.d_A/J.d_gAE).mean()
    ax2.set_title("(b) Deviation and bounds per valid case")
    for a in (ax,ax2):
        for sp in ["top","right"]: a.spines[sp].set_visible(False)
    save(fig,outname)
    return npass,(D/J.d_gAE).mean(),D.mean()

p65,t65,m65 = draw_joint("delta65","employed-senior (65+)","fig_appB_joint_transfer")
p50,t50,m50 = draw_joint("delta50","employed-older (50+)","fig_appB_joint_transfer_50plus")

NC=len(J)
print("\n[marginal] pass %d/%d  mean tightness %.3f"%(int((M.delta<=M.d_A).sum()),len(M),(M.delta/M.d_A).mean()))
print("[joint 65+] pass %d/%d  mean|Δ|=%.4f  tightness|Δ|/d_gAE=%.3f  (d_A ≈%.1fx d_gAE)"%(p65,NC,m65,t65,(J.d_A/J.d_gAE).mean()))
print("[joint 50+] pass %d/%d  mean|Δ|=%.4f  tightness|Δ|/d_gAE=%.3f"%(p50,NC,m50,t50))
print(J[["cell","delta65","delta50","d_A","d_gAE"]].round(4).to_string(index=False))

# ---------- additive: statistics quoted in the manuscript text ----------
M["ratio"]=M.delta/M.d_A
per_fun=M.groupby("functional")["ratio"].agg(["mean","max","count"])
stats={
  "n_valid_cells":NC,
  "marginal":{"n_points":int(len(M)),"n_below_bound":int((M.delta<=M.d_A).sum()),
              "mean_tightness_ratio":float(M.ratio.mean()),"median_tightness_ratio":float(M.ratio.median()),
              "max_tightness_ratio":float(M.ratio.max()),
              "max_ratio_cell":str(M.loc[M.ratio.idxmax(),"cell"]),"max_ratio_functional":str(M.loc[M.ratio.idxmax(),"functional"]),
              "per_functional_mean_ratio":{k:float(v) for k,v in per_fun["mean"].items()},
              "per_functional_max_ratio":{k:float(v) for k,v in per_fun["max"].items()},
              "d_A_min":float(M.d_A.min()),"d_A_max":float(M.d_A.max()),"d_A_mean":float(J.d_A.mean())},
  "joint_65plus":{"n_points":NC,"n_below_bound":int(p65),"mean_abs_delta":float(m65),"max_abs_delta":float(J.delta65.max()),
                  "mean_tightness_ratio":float(t65),"max_tightness_ratio":float((J.delta65/J.d_gAE).max()),
                  "d_gAE_mean":float(J.d_gAE.mean()),"d_gAE_median":float(J.d_gAE.median()),
                  "d_gAE_min":float(J.d_gAE.min()),"d_gAE_max":float(J.d_gAE.max()),
                  "dA_over_dG_mean":float((J.d_A/J.d_gAE).mean()),"dA_over_dG_median":float((J.d_A/J.d_gAE).median()),
                  "dA_over_dG_min":float((J.d_A/J.d_gAE).min()),"dA_over_dG_max":float((J.d_A/J.d_gAE).max())},
  "joint_50plus":{"n_points":NC,"n_below_bound":int(p50),"mean_abs_delta":float(m50),"max_abs_delta":float(J.delta50.max()),
                  "mean_tightness_ratio":float(t50),"max_tightness_ratio":float((J.delta50/J.d_gAE).max())},
}
(FIG/"tables"/"transfer_stats.json").write_text(json.dumps(stats,indent=2))
print("\n[stats]",json.dumps(stats,indent=1))
