"""
§A–§F figures + tables for the paper-faithful audit (PAPER_xstate_v3, v3 protocol: 9 models x 3 states x 3 horizons = 81 cells, 16 valid).
Port of audits/PAPER_xstate_v2/figures/make_figures_v2.py (2026-09-02).
Only edits vs v2: BASE/V3 paths -> audits_v3/PAPER_xstate_v3, DISP/MODELS -> the 9 new tags, uniform l2dir
(no legacy layer2_pwgtp_ split), and every hard-coded count/annotation (63, 27, ylim 68/66, nESS band 0.85,
'TabDDPM (support-poor) nESS~0.04', 'realized 2024 (6%)') re-derived from the data.  Algorithms, thresholds
and formulas are byte-identical to v2.
Outputs -> figures/png, figures/pdf, figures/tables.
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[4])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import t as student_t

BASE = Path(_PA.R("/main_experiments/audits_v3"))
V3 = BASE / "PAPER_xstate_v3"
FIG = V3 / "figures"; (FIG/"png").mkdir(parents=True, exist_ok=True); (FIG/"pdf").mkdir(exist_ok=True); (FIG/"tables").mkdir(exist_ok=True)
DISP = {"semapop_vae":"SemaPop-VAE+","ttvae":"TTVAE","tabddpm":"TabDDPM","bn":"BN","wgan_zcr":"WGAN-GP-ZCR",
        "ctgan":"CTGAN","flowsyn":"FlowSyn","borysov":"Borysov-VAE","tvae":"TVAE"}
MODELS = ["semapop_vae","ttvae","tabddpm","bn","wgan_zcr","ctgan","flowsyn","borysov","tvae"]   # strong -> weak
YEARS = ["2018","2021","2024"]; STATES = ["CA","OH","TX"]
CGREEN,CORANGE,CRED,CGREY = "#55a868","#dd8452","#c44e52","#bdbdbd"   # 2026-10-24: muted gate palette (seaborn deep), was tab10 #2ca02c/#ff7f0e/#d62728
CPURPLE = "#8172b3"   # L3 reject (was #9467bd)
plt.rcParams.update({"font.size":10,"axes.titlesize":11,"axes.spilines" if False else "axes.grid":False})

d = pd.read_csv(V3/"xstate_certificate_all.csv")
d = d[d.L1_loaded & d.year.notna()].copy(); d["year"]=d.year.astype(int).astype(str)
for _c in ["L1_accept","L2_accept","L3_accept","VALID"]: d[_c]=d[_c].astype(bool)
N_CELLS = len(d)   # 81 in v3 (was 63)
def gate(r):
    if not r.L1_accept: return "L1"
    if not r.L2_accept: return "L2"
    if not r.L3_accept: return "L3"
    return "VALID"
d["g"]=d.apply(gate,axis=1)
def l2dir(s,m,y): return BASE/f"layer2_{m}_{s}_{y}"   # uniform v3 naming (v2 had a layer2_pwgtp_ split for CA ttvae/tvae_llm+)
def save(fig,name): fig.savefig(FIG/"png"/f"{name}.png",dpi=150,bbox_inches="tight"); fig.savefig(FIG/"pdf"/f"{name}.pdf",bbox_inches="tight"); plt.close(fig); print("  wrote",name)

# ============================================================ A: admission
def figA():
    out=np.full((len(MODELS),9),np.nan); labels=[]
    cols=[(s,y) for y in YEARS for s in STATES]
    code={"VALID":3,"L3":2,"L2":1,"L1":0}
    M=np.zeros((len(MODELS),9))
    for i,m in enumerate(MODELS):
        for j,(s,y) in enumerate(cols):
            r=d[(d.model==m)&(d.state==s)&(d.year==y)]
            M[i,j]=code[r.iloc[0].g] if len(r) else np.nan
    fig,(ax,ax2)=plt.subplots(1,2,figsize=(13,4.8),gridspec_kw={"width_ratios":[3,1.1]})
    cmap=ListedColormap([CRED,CORANGE,CPURPLE,CGREEN])
    ax.imshow(M,cmap=cmap,vmin=0,vmax=3,aspect="auto",alpha=1.0)
    ax.set_xticks(range(9)); ax.set_xticklabels([f"{s}\n{y}" for (s,y) in cols],fontsize=8)
    ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([DISP[m] for m in MODELS])
    for j in [3,6]: ax.axvline(j-0.5,color="w",lw=2)
    for i in range(len(MODELS)):
        for j in range(9):
            v=M[i,j]; ax.text(j,i,{0:"L1",1:"L2",3:"✓"}.get(v,""),ha="center",va="center",
                              color="white",fontsize=8,fontweight="bold")
    ax.set_title(f"(a) Layer-wise admission across the {len(d)} prior–state–horizon cases")   # r22: wording aligned with the text
    ax.set_xlabel("state × deployment horizon (H=5/8/11)")
    # waterfall
    n0=len(d); nL1=int(d.L1_accept.sum()); nL2=int((d.L1_accept&d.L2_accept).sum()); nV=int(d.VALID.sum())
    stages=["All\ncases","pass\nL1","pass\nL2","valid\n(pass L3)"]; vals=[n0,nL1,nL2,nV]
    ax2.bar(range(4),vals,color=[CGREY,"#8c8c8c","#636363",CGREEN],width=0.65)
    for k,v in enumerate(vals): ax2.text(k,v+1,str(v),ha="center",fontweight="bold")
    ax2.set_xticks(range(4)); ax2.set_xticklabels(stages,fontsize=8); ax2.set_ylim(0,n0+5)   # v2: fixed 68 for 63 cases
    ax2.set_title("(b) Admission waterfall"); ax2.set_ylabel("Number of cases")
    for sp in ["top","right"]: ax2.spines[sp].set_visible(False)
    save(fig,"figA_admission")

# ============================================================ B: failure mode
def figB():
    rows=[]
    for m in MODELS:
        sub=d[d.model==m]
        rows.append([int((sub.g=="L1").sum()),int((sub.g=="L2").sum()),int(sub.VALID.sum())])
    R=np.array(rows)
    fig,(ax,ax2)=plt.subplots(1,2,figsize=(12,4.4),gridspec_kw={"width_ratios":[1.3,1]})
    y=np.arange(len(MODELS))
    ax.barh(y,R[:,2],color=CGREEN,label="VALID",alpha=0.85)
    ax.barh(y,R[:,1],left=R[:,2],color=CORANGE,label="L2 reject (scenario)",alpha=0.85)
    ax.barh(y,R[:,0],left=R[:,1]+R[:,2],color=CRED,label="L1 reject (structural)",alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels([DISP[m] for m in MODELS]); ax.invert_yaxis()
    ax.set_xlabel("# of 9 state×horizon cells"); ax.set_xlim(0,9)
    ax.legend(fontsize=8,loc="lower right"); ax.set_title("(a) First-failed gate by model")
    # L1 binding diagnostic
    l1=d[d.g=="L1"].copy()
    l1["binding"]=l1.apply(lambda r: ["marg","dep","supp","inv"][int(np.argmax([r.L1_marg/0.05,r.L1_dep/0.10,(1-r.L1_support)/0.30,r.L1_invalid/0.005]))],axis=1)
    vc=l1.binding.value_counts()
    nm={"marg":"marginal discrepancy","dep":"dependency discrepancy","inv":"invalid mass","supp":"support cov."}
    ax2.bar([nm[k] for k in vc.index],vc.values,color="#7f7f7f")
    for i,v in enumerate(vc.values): ax2.text(i,v+0.2,str(v),ha="center",fontweight="bold")
    ax2.set_title("(b) Layer-1 rejections by binding diagnostic"); ax2.set_ylabel(f"# cases (of {len(l1)})")   # v2: fixed 'of 27' (= L1-rejected cells)
    ax2.tick_params(axis="x",labelsize=8); ax2.set_ylim(0,max(vc.values)+2)
    for sp in ["top","right"]: ax2.spines[sp].set_visible(False)
    save(fig,"figB_failuremode")
    return l1[["state","model","year","binding"]]

# ============================================================ C: chi_H decomposition
def chi_contrib(sub):  # returns dict attr-> chi_{H,a}
    # 2026-11-24: realization average of chi_{H,a}^{(r)} (Eq. chiH-ensemble), as in make_paper_tables_v3
    pr=pd.read_csv(l2dir(sub.state,sub.model,sub.year)/"layer2_per_realization_weights.csv")
    return pd.Series({k[len("attribute_loss__"):]:float(pr[k].mean()) for k in pr.columns if k.startswith("attribute_loss__")})
def figC():
    ATTR=["Commute_Mode","Household_Income_Class","Number_of_cars_of_household","Age_Group",
          "Number_of_children","Household_Size","Employment_status"]
    allc=[]; drift=[]
    for _,r in d.iterrows():
        try:g=chi_contrib(r)
        except Exception:continue
        tot=g.sum()
        if tot<=0:continue
        sh=(g/tot).reindex(ATTR).fillna(0.0)
        allc.append(sh)
        if r.L1_accept and (not r.L2_accept): drift.append(sh)
    A=pd.DataFrame(allc).mean()*100; D=pd.DataFrame(drift).mean()*100
    tab=pd.DataFrame({"all_cells_%":A.round(1),"drift_cells_%":D.round(1)}).reindex(ATTR)
    tab.to_csv(FIG/"tables"/"tableC_chiH_decomposition.csv")
    print(f"  figC: n_all={len(allc)}, n_drift={len(drift)}")
    fig,ax=plt.subplots(figsize=(8,4.2))
    y=np.arange(len(ATTR)); h=0.38
    disp=[a.replace("_"," ").replace("Household","HH") for a in ATTR]
    b1=ax.barh(y+h/2,A.values,h,color="#9ecae1",label=f"all {len(allc)} cells")   # v2: 'all 63 cells'
    b2=ax.barh(y-h/2,D.values,h,color=CORANGE,label=f"scenario-drift cells (pass L1, fail L2; n={len(drift)})")
    ax.set_yticks(y); ax.set_yticklabels(disp); ax.invert_yaxis()
    ax.set_xlabel("share of $\\overline{\\chi}_H$ (%)"); ax.legend(fontsize=8)
    for i,(a,dd) in enumerate(zip(A.values,D.values)):
        ax.text(a+0.5,i+h/2,f"{a:.0f}",va="center",fontsize=7); ax.text(dd+0.5,i-h/2,f"{dd:.0f}",va="center",fontsize=7,color=CORANGE)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    save(fig,"figC_chiH_decomposition")
    return tab

# ============================================================ D: L3 stress
def figD():
    # Two stress cells, TX 2024 SemaPop-VAE+ and TTVAE (two of the three priors, with BN, admitted
    # in both 2024 scenarios; see l3_stress_case.py).  Each curve is read from its own
    # l3_stress/<cell>/ and l3_counterfactual/<cell>/ directory.
    CELLS=[("semapop_vae","-"),("ttvae","--")]
    MK={"semapop_vae":dict(ms=6),"ttvae":dict(ms=9,mfc="none",mew=1.3)}   # hollow, larger: the two curves overlap
    TL={m:pd.read_csv(V3/"l3_stress"/f"{m}_TX_2024"/"l3_stress_curve.csv") for m,_ in CELLS}
    CF={m:pd.read_csv(V3/"l3_counterfactual"/f"{m}_TX_2024"/"l3_counterfactual_curve.csv") for m,_ in CELLS}
    # survivor L3 table
    v=pd.read_csv(V3/"xstate_valid_only.csv")
    st=v.groupby("model").agg(n_valid=("VALID","size"),med_R3=("R_3","median"),med_nESS=("L3_nESS","median")).reset_index()
    st["model"]=st.model.map(DISP); st.to_csv(FIG/"tables"/"tableD_L3_survivors.csv",index=False)
    ness_floor=float(np.floor(v.L3_nESS.min()*100)/100)             # lower edge of the survivor band = floor(min nESS over VALID cells, 2 dp)
    worst=d.loc[d.L3_nESS.idxmin()]                                   # weakest projection over all audited cells
    worst_lbl=f"weakest projection ({DISP[worst.model]}, {worst.state} {worst.year}) nESS≈{worst.L3_nESS:.2f}"
    cf0=CF["semapop_vae"]; wfh0=float(cf0.WFH_target.iloc[0]*100)
    ness0=float(np.mean([CF[m].nESS.iloc[0] for m,_ in CELLS]))
    for m,_ in CELLS:
        tl,cf=TL[m],CF[m]
        print(f"  figD[{m}]: tau-ladder feasible at every tau: {bool(tl.feasible.all())}; nESS {tl.nESS.iloc[0]:.4f}->{tl.nESS.iloc[-1]:.4f}, R3 {tl.R_3.iloc[0]:.4f}->{tl.R_3.iloc[-1]:.4f}; "
              f"counterfactual feasible at every w: {bool(cf.G3_feasible.all())}; nESS -> {cf.nESS.iloc[-1]:.4f} at WFH {cf.WFH_target.iloc[-1]*100:.0f}%")
    print(f"  figD: survivor band nESS>={ness_floor:.2f} (min valid {v.L3_nESS.min():.4f}); {worst_lbl}; realized WFH {wfh0:.1f}% nESS {ness0:.3f}")
    fig,(ax,ax2)=plt.subplots(1,2,figsize=(13,4.3)); fig.subplots_adjust(wspace=0.42)
    axb=ax.twinx()
    for m,ls in CELLS:
        tl=TL[m]
        ax.plot(tl.tau,tl.nESS,"o"+ls,color=CGREEN,label=f"nESS, {DISP[m]}",**MK[m])
        axb.plot(tl.tau,tl.R_3,"s"+ls,color="#8c564b",label=f"$R_3$, {DISP[m]}",**MK[m])
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel(r"projection tolerance $\tau$ (log, tighter →)"); ax.set_ylabel("nESS",color=CGREEN)
    ax.set_ylim(0,1.02); axb.set_ylabel("$R_3$",color="#8c564b")
    h1,l1=ax.get_legend_handles_labels(); h2,l2=axb.get_legend_handles_labels()
    ax.legend(h1+h2,l1+l2,fontsize=7,loc="center left",bbox_to_anchor=(0.0,0.62))
    allfeas=all(TL[m].feasible.all() for m,_ in CELLS)
    ax.set_title("(a) Tolerance ladder, Texas 2024")
    for m,ls in CELLS:
        cf=CF[m]; ax2.plot(cf.WFH_target*100,cf.nESS,"o"+ls,color="#1f77b4",label=f"nESS, {DISP[m]}",**MK[m])
    ax2.axhspan(ness_floor,1.0,color=CGREEN,alpha=0.12); ax2.text(22,ness_floor-0.01,f"valid cases, nESS≥{ness_floor:.2f}",fontsize=7,color=CGREEN,va="top")
    ax2.axhline(worst.L3_nESS,color=CRED,ls=":"); ax2.text(20,worst.L3_nESS-0.06,worst_lbl,fontsize=7,color=CRED)
    ax2.set_xlabel("counterfactual work-from-home target (%)"); ax2.set_ylabel("nESS"); ax2.set_ylim(0,1.02)
    ax2.legend(fontsize=7,loc="lower left",bbox_to_anchor=(0,0.12))
    allcf=all(CF[m].G3_feasible.all() for m,_ in CELLS)
    ax2.set_title("(b) Counterfactual work-from-home surge, Texas 2024")
    ax2.annotate(f"realized\n2024 ({wfh0:.0f}%)",(wfh0,ness0),(10.5,0.55),fontsize=7,arrowprops=dict(arrowstyle="->"))
    for a in (ax,ax2):
        for sp in ["top"]: a.spines[sp].set_visible(False)
    save(fig,"figD_L3_stress")
    return st

# ============================================================ E: decision d_A vs B0
def figE():
    v=pd.read_csv(V3/"xstate_valid_only.csv"); v["year"]=v.year.astype(int).astype(str)
    rows=[]
    for _,r in v.iterrows():
        t=pd.read_csv(l2dir(r.state,r.model,r.year)/"layer2_target_summary.csv")
        dA=0.5*t["abs_err_prior"].sum()
        rows.append({"cell":f"{r.state}·{DISP[r.model]}·{r.year}","d_A":dA,"R_2_MA":r.R_2_MA})
    e=pd.DataFrame(rows).sort_values("d_A").reset_index(drop=True)
    e.to_csv(FIG/"tables"/"tableE_decision_dA.csv",index=False)
    B0=0.30   # unchanged: decision_certificate_xstate_v3.py B0 = 0.30
    # selected-prior view: one group per audited state-horizon scenario;
    # within a group each admissible candidate's d_A is a point, the selected
    # theta^dagger (min d_A) is a filled star; scenarios with no admissible
    # candidate are annotated "none".
    e["state"]=e.cell.str.split("·").str[0]
    e["yr"]=e.cell.str.split("·").str[-1]
    scen=[(s,y) for y in ["2018","2021","2024"] for s in STATES]  # 9 audited scenarios
    fig,ax=plt.subplots(figsize=(10,4.6))
    for i,(st,yr) in enumerate(scen):
        g=e[(e.state==st)&(e.yr==yr)].sort_values("d_A")
        if len(g)==0:
            ax.text(i,0.015,"none\nadmissible",ha="center",va="bottom",fontsize=6.5,color="#999")
            continue
        xs=np.full(len(g),i)+(np.linspace(-0.13,0.13,len(g)) if len(g)>1 else np.array([0.0]))
        ax.scatter(xs,g.d_A.values,facecolors="none",edgecolors="#888",s=48,lw=1.1,zorder=2)
        sel_col=CGREEN if g.d_A.values[0]<=B0 else CRED
        ax.scatter([xs[0]],[g.d_A.values[0]],marker="*",s=260,color=sel_col,edgecolors="k",lw=0.6,zorder=4)
    for j in [2.5,5.5]: ax.axvline(j,color="0.85",lw=1)
    ax.axhline(B0,color="k",ls="--",lw=1.3); ax.text(-1.02,B0+0.006,f"$B_0={B0}$",fontsize=9,va="bottom")
    ax.set_xticks(range(9)); ax.set_xticklabels([f"{s}\n{y}" for (s,y) in scen],fontsize=8)
    ax.set_ylabel(r"post-projection marginal discrepancy $d_{\mathcal{A}}$")
    ax.set_xlim(-1.1,8.6); ax.set_ylim(0,max(0.46,e.d_A.max()*1.08))
    ax.scatter([],[],facecolors="none",edgecolors="#888",s=48,lw=1.1,label="admissible candidate prior")
    ax.scatter([],[],marker="*",s=170,color=CGREEN,edgecolors="k",lw=0.6,label=r"selected $\theta^\dagger$: deploy")
    ax.scatter([],[],marker="*",s=170,color=CRED,edgecolors="k",lw=0.6,label=r"selected $\theta^\dagger$: over budget")
    ax.legend(fontsize=8,loc="upper left",framealpha=0.9)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    save(fig,"figE_decision")
    return e

# ============================================================ F: comparison + sensitivity
def figF():
    # baselines on the 81 cases (audit vs marginal-only vs structural-only)
    # SRMSE-style marginal scenario fit
    def srmse(r):
        t=pd.read_csv(l2dir(r.state,r.model,r.year)/"layer2_target_summary.csv")
        p,q=t.prior_mean.values,t.target_m.values
        return float(np.sqrt(np.mean((p-q)**2))/ (q.mean()+1e-12))
    d2=d.copy(); d2["srmse"]=d2.apply(srmse,axis=1)
    audit=int(d.VALID.sum())
    srmse_acc=int((d2.srmse<=0.30).sum())
    struct=int(d.L1_accept.sum())   # structural-only scorecard (L1 pass, no scenario/projection gate)
    ACC=f"accepted /{N_CELLS}"   # v2 column name 'accepted /63'
    comp=pd.DataFrame({"protocol":["Proposed 3-layer audit","SRMSE marginal rule (≤0.30)","Structural-only scorecard (L1)"],
                       ACC:[audit,srmse_acc,struct]})
    comp.to_csv(FIG/"tables"/"tableF_comparison.csv",index=False)
    # sensitivity: valid vs kappa_2 (others baseline)
    d["eta2"]=d.L2_cutoff-5.0; L3=d.L3_accept.fillna(False).astype(bool)
    L1=(d.L1_marg<=0.05)&(d.L1_dep<=0.10)&((1-d.L1_support)<=0.30)&(d.L1_invalid<=0.005)
    # 2026-11-23 revision: panel (b) plots every kappa_2 value in the sensitivity sweep's curve
    # (sensitivity_kappa2_curve.csv, incl. 2.0 and 20.0) on a log x-axis; the inline recount is a cross-check.
    sk=V3/"sensitivity"/"sensitivity_kappa2_curve.csv"
    if sk.exists():
        s=pd.read_csv(sk).sort_values("k2")
        k2s=[float(k) for k in s.k2]; vv=[int(n) for n in s.n_valid_total]
    else:
        k2s=[3.5,4,5,6,7,10]; vv=None
    vin=[int((L1&(d.L2_chi_H<=(k2+d.eta2))&L3).sum()) for k2 in k2s]
    if vv is None: vv=vin
    print(f"  figF: kappa_2 curve sensitivity_sweep {dict(zip(k2s,vv))} vs inline {dict(zip(k2s,vin))} -> {'MATCH' if vv==vin else 'MISMATCH'}")
    fig,(ax,ax2)=plt.subplots(1,2,figsize=(12,4),gridspec_kw={"width_ratios":[1,1.1]})
    # r22: display labels follow the caption/Table wording (CSV protocol names unchanged); the two reduced
    # protocols use neutral greys instead of the L1/L2 gate colours of Fig. A
    PLBL=["Proposed\nthree-layer audit","SRMSE-style marginal\nrule (≤0.30)","Structural-only\nLayer 1 scorecard"]
    ax.bar(PLBL,comp[ACC],color=[CGREEN,"#7f8c99","#a6a6a6"])
    for i,v in enumerate(comp[ACC]): ax.text(i,v+0.6,str(v),ha="center",fontweight="bold")
    ax.set_ylabel(f"Accepted cases (of {N_CELLS})"); ax.set_title("(a) Admission under reduced protocols")
    ax.tick_params(axis="x",labelsize=7.5); ax.set_ylim(0,N_CELLS+3)   # v2: fixed 66 for 63 cases
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    ax2.plot(k2s,vv,"o-",color="#1f77b4"); ax2.axvline(5.0,color="k",ls="--")
    ax2.set_xscale("log"); ax2.xaxis.set_major_locator(FixedLocator(k2s)); ax2.xaxis.set_minor_locator(NullLocator())
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x:g}"))
    ib=k2s.index(5.0) if 5.0 in k2s else None
    if ib is not None: ax2.text(5.0*1.04,vv[ib]-6,"baseline\n$\\kappa_2$=5.0",fontsize=8)
    ax2.set_xlabel("$\\kappa_2$ (log scale)"); ax2.set_ylabel("Number of valid cases"); ax2.set_title("(b) Sensitivity to $\\kappa_2$")
    ax2.set_ylim(0,max(vv)+5)
    for i,(x,y) in enumerate(zip(k2s,vv)):
        dx,ha=(0.985,"right") if i==ib else (1.0,"center")  # baseline label just left of the dashed line
        ax2.text(x*dx,y+1.0,str(y),ha=ha,fontsize=8)
    for a in (ax,ax2):
        for sp in ["top","right"]: a.spines[sp].set_visible(False)
    save(fig,"figF_comparison_sensitivity")
    return comp

print("Building figures ->",FIG)
print(f"cells={N_CELLS}; gate counts: {d.g.value_counts().to_dict()}")
figA(); tB=figB(); tC=figC(); tD=figD(); tE=figE(); tF=figF()
print("\n[figB L1 binding diagnostic per rejected cell]\n",tB.groupby(["model","state"]).binding.first().to_string())
print("\n[tableC chi_H decomposition]\n",tC.to_string())
print("\n[tableD L3 survivors]\n",tD.to_string(index=False))
print("\n[tableE decision d_A]\n",tE.to_string(index=False))
print("\n[tableF comparison]\n",tF.to_string(index=False))
print("\nDONE. figures in",FIG/"png")
