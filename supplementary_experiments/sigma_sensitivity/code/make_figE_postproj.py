"""figE + tab:selected-prior-budget on the POST-projection d_A(Q*).
d_A(Q*) = 0.5 * sum_targets |m_{a,c}(Q*) - m*_{a,c}| from layer3_paperW_<tag>_<ST>_<year>/layer3_target_summary.csv
(projected_abs_err_mean; the same 66 targets Layer 2 uses).  Pre-projection value kept alongside for the record.
Outputs: figures/tables/tableE_decision_dA.csv (cell, d_A_pre, d_A, R_2_MA), exp_ii_scenario_budget_post.csv,
         figures/{pdf,png}/figE_decision.{pdf,png} (post) and figE_decision_preproj.* (old quantity)."""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[3])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
BASE=Path(_PA.R("/main_experiments/audits_v3")); V3=BASE/"PAPER_xstate_v3"; FIG=V3/"figures"
DISP={"semapop_vae":"SemaPop-VAE+","ttvae":"TTVAE","tabddpm":"TabDDPM","bn":"BN","wgan_zcr":"WGAN-GP-ZCR","ctgan":"CTGAN","flowsyn":"FlowSyn","borysov":"Borysov-VAE","tvae":"TVAE"}
B0=0.30; CGREEN="#55a868"; CRED="#c44e52"   # 2026-10-24: muted gate palette, was #2ca02c/#d62728
v=pd.read_csv(V3/"xstate_valid_only.csv"); v["year"]=v.year.astype(int)
rows=[]
for r in v.itertuples():
    t3=pd.read_csv(BASE/f"layer3_paperW_{r.model}_{r.state}_{r.year}"/"layer3_target_summary.csv")
    t2=pd.read_csv(BASE/f"layer2_{r.model}_{r.state}_{r.year}"/"layer2_target_summary.csv")
    rows.append({"state":r.state,"model":r.model,"year":r.year,"cell":f"{r.state}·{DISP[r.model]}·{r.year}",
                 "d_A_pre":0.5*t2["abs_err_prior"].sum(),           # Layer-2 ensemble-mean prior (old figE quantity)
                 "d_A_rstar_pre":0.5*t3["prior_abs_err_mean"].sum(),  # r* realization before projection
                 "d_A":0.5*t3["projected_abs_err_mean"].sum(),        # POST-projection d_A(Q*)
                 "R_2_MA":r.R_2_MA,"n_targets":len(t3)})
e=pd.DataFrame(rows).sort_values("d_A").reset_index(drop=True)
e.to_csv(FIG/"tables"/"tableE_decision_dA.csv",index=False)
print(e[["cell","d_A_pre","d_A_rstar_pre","d_A","R_2_MA"]].round(4).to_string(index=False))
# ---- scenario budget table (post-projection) ----
vH=pd.read_csv(V3/"exp_ii_vH.csv").set_index("year")["v_H"]
out=[]
for y in (2018,2021,2024):
    for st in ("CA","OH","TX"):
        g=e[(e.state==st)&(e.year==y)].sort_values("d_A")
        H=y-2013; vh=float(vH[y])
        if len(g)==0: out.append({"state":st,"year":y,"H":H,"n_adm":0,"v_H":vh}); continue
        D=float(g.d_A.iloc[0]); th=g.model.iloc[0]
        gmin=0.0 if D<=B0 else ((D/B0-1)/vh if vh>0 else np.inf)
        out.append({"state":st,"year":y,"H":H,"n_adm":len(g),"v_H":vh,"theta_dagger":th,"D_dagger":D,
                    "flat":"Deploy" if D<=B0 else "Defer","gamma_min":gmin,"max_dA":float(g.d_A.max()),
                    "theta_dagger_pre":e[(e.state==st)&(e.year==y)].sort_values("d_A_pre").model.iloc[0],
                    "D_dagger_pre":float(e[(e.state==st)&(e.year==y)].d_A_pre.min())})
sb=pd.DataFrame(out); sb.to_csv(V3/"exp_ii_scenario_budget_post.csv",index=False); print("\n",sb.round(4).to_string(index=False))
# ---- LaTeX table ----
cap=r'''\caption{
Scenario-level deployment decisions under the selected-prior budget rule.
For each scenario \(S_H\), \(\Theta_{\mathrm{adm}}(S_H)\) is the set of priors
passing all three audit layers. When this set is nonempty,
\(\theta^\dagger(S_H)\) is the admitted prior with the smallest
post-projection marginal-aggregate residual
\(D^\dagger(S_H)=D(\theta^\dagger(S_H),S_H)\). The flat decision deploys only
if \(D^\dagger(S_H)\le B_0=0.30\), and scenarios with
\(\Theta_{\mathrm{adm}}(S_H)=\emptyset\) are deferred by definition.
The last column reports the diagnostic value \(\gamma_{\min}\), the minimum
volatility-budget parameter required for the selected prior to satisfy
\(D^\dagger(S_H)\le B_0(1+\gamma v(H))\).
}'''
L=[r'\begin{table}[htbp]',r'\centering',cap,r'\label{tab:selected-prior-budget}',r'\begin{tabular}{llcccccc}',r'\toprule',
   r'State & Year & \(H\) &',r'\(|\Theta_{\mathrm{adm}}(S_H)|\) &',r'\(\theta^\dagger(S_H)\) &',r'\(D^\dagger(S_H)\) &',r'Flat decision &',r'\(\gamma_{\min}\) \\',r'\midrule']
for i,y in enumerate((2018,2021,2024)):
    if i: L.append(r'\midrule')
    for st in ("CA","OH","TX"):
        r=sb[(sb.state==st)&(sb.year==y)].iloc[0]
        if r.n_adm==0: L.append(f"{st} & {y} & {int(r.H):<2d} & 0 & N/A & N/A & Defer & N/A \\\\")
        else:
            gs='0' if r.gamma_min==0 else ('$\\infty$' if np.isinf(r.gamma_min) else f"{r.gamma_min:.2f}")
            L.append(f"{st} & {y} & {int(r.H):<2d} & {int(r.n_adm)} & \\texttt{{{DISP[r.theta_dagger]}}} & {r.D_dagger:.3f} & {r.flat} & {gs} \\\\")
L+=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
Path(_PA.R("/main_experiments/tables/tab_selected_prior_budget_v3.tex")).write_text("\n".join(L)+"\n")
# ---- figE (post) and figE_preproj ----
def draw(col, name, ylabel):
    ee=e.copy(); ee["yr"]=ee.year.astype(str)
    fig,ax=plt.subplots(figsize=(10,4.6)); x=0
    for yi,yr in enumerate(("2018","2021","2024")):
        for st in ("CA","OH","TX"):
            g=ee[(ee.state==st)&(ee.yr==yr)].sort_values(col)
            if len(g)==0: ax.text(x,0.015,"none\nadmissible",ha="center",va="bottom",fontsize=6.5,color="#999")
            else:
                xs=x+np.linspace(-0.18,0.18,len(g)) if len(g)>1 else [x]
                ax.scatter(xs,g[col].values,facecolors="none",edgecolors="#888",s=48,lw=1.1,zorder=2)
                sel=CGREEN if g[col].values[0]<=B0 else CRED
                ax.scatter([xs[0]],[g[col].values[0]],marker="*",s=260,color=sel,edgecolors="k",lw=0.6,zorder=4)
            ax.text(x,-0.03,f"{st}\n{yr}",ha="center",va="top",fontsize=9,transform=ax.get_xaxis_transform()); x+=1
        if yi<2: ax.axvline(x-0.5,color="#bbb",lw=1)
    ax.axhline(B0,color="k",ls="--",lw=1.3); ax.text(-1.02,B0+0.006,f"$B_0={B0}$",fontsize=9,va="bottom")
    ax.set_xlim(-1.1,8.6); ax.set_ylim(0,max(0.46,ee[col].max()*1.08)); ax.set_xticks([]); ax.set_ylabel(ylabel)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([],[],marker="o",ls="",mfc="none",mec="#888",label="admissible candidate prior"),
                       Line2D([],[],marker="*",ls="",ms=13,color=CGREEN,mec="k",label="selected $\\theta^\\dagger$: deploy"),
                       Line2D([],[],marker="*",ls="",ms=13,color=CRED,mec="k",label="selected $\\theta^\\dagger$: over budget")],loc="upper left",fontsize=8.5)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    fig.savefig(FIG/"png"/f"{name}.png",dpi=150,bbox_inches="tight"); fig.savefig(FIG/"pdf"/f"{name}.pdf",bbox_inches="tight"); plt.close(fig); print("wrote",name)
draw("d_A","figE_decision","post-projection marginal discrepancy $d_\\mathcal{A}(Q^\\ast)$")
draw("d_A_pre","figE_decision_preproj","pre-projection prior marginal discrepancy $d_\\mathcal{A}(P_\\theta)$")
