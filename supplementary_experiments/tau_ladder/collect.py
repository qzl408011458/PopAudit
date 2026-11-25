"""Collect D(tau), nESS, R_3, feasibility, CS bound and max chi_{H,a} for every run; crossing tau for B0=0.30."""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[2])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import json, numpy as np, pandas as pd
from pathlib import Path
H = Path(__file__).resolve().parent; AUD = Path(_PA.R("/main_experiments/audits_v3")); B0 = 0.30
# FIX 3 (2026-11-24): D = d_A(Q*,S_H) on the union of Q* and reference categories (shared helper
# PAPER_xstate_v3/dA_union.py, as make_figE_postproj.py).  D_ref keeps the Layer-3-target-only value.
import sys; sys.path.insert(0, str(AUD/"PAPER_xstate_v3")); from dA_union import dA_union_l3
DISP={"semapop_vae":"SemaPop-VAE+","ttvae":"TTVAE","tabddpm":"TabDDPM","bn":"BN"}
cert = pd.read_csv(AUD/"PAPER_xstate_v3/xstate_certificate_all.csv")
rows = []
for cd in sorted((H/"runs").iterdir()):
    m, st, yr = cd.name.rsplit("_", 2)
    sig = json.load(open(AUD/f"layer3_paperW_{cd.name}"/"sigma_H_targets.json"))
    prodT = pd.read_csv(AUD/f"layer3_paperW_{cd.name}"/"layer3_target_summary.csv")
    D_prod = dA_union_l3(AUD/f"layer3_paperW_{cd.name}", st, yr)["d_A"]
    for td in sorted(cd.glob("tau_*")):
        if not td.is_dir(): continue
        tau = float(td.name[4:]); js = td/"layer3_summary.json"
        if not js.exists(): rows.append(dict(cell=cd.name, model=m, state=st, tau=tau, feasible=False)); continue
        j = json.load(open(js)); feas = float(j.get("feasibility_rate", 0)) >= 0.999
        t = pd.read_csv(td/"layer3_target_summary.csv")
        u = dA_union_l3(td, st, yr) if feas else {}
        D = u.get("d_A", np.nan); D_ref = u.get("d_A_ref", np.nan); mu_out = u.get("mu_out", np.nan)
        # post-projection chi_{H,a} (Eq. attribute-wise-chi) with the same sigma used in the constraint
        s = t.sigma.values
        z2 = ((t.projected_mean - t.target_m)/s)**2
        chia = (z2.groupby(t.attr).sum() / (2*t.groupby("attr").size()))
        # CS bound (Prop cs-tightening): D <= 0.5*||sigma||_{2,w} sqrt(2 chi_H), chi_H <= |A| tau
        k = t.groupby("attr").size(); nw = np.sqrt(sum(k[a]*(t.sigma[t.attr==a]**2).sum() for a in k.index))
        KL = j.get("mean_distortion"); ness = j.get("mean_agent_nESS")
        rows.append(dict(cell=cd.name, model=m, state=st, tau=tau, feasible=feas, D=D, D_ref=D_ref, mu_out=mu_out, D_pre=0.5*t.prior_abs_err_mean.sum(),
            KL=KL, R_3=np.sqrt(KL/2) if (feas and KL is not None) else np.nan, nESS=ness,
            disp=np.sqrt(1/ness-1) if ness else np.nan, max_chi_a=chia.max(), chi_H=chia.sum(),
            CS_bound_tau=0.5*nw*np.sqrt(2*len(k)*tau), CS_bound_post=0.5*nw*np.sqrt(2*chia.sum()),
            n_targets=len(t), D_prod_tau1=D_prod))
df = pd.DataFrame(rows).sort_values(["cell","tau"], ascending=[True, False])
df.to_csv(H/"tau_ladder_all.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(df.round(4).to_string(index=False))
# tau=1 must reproduce production D (tableE_decision_dA.csv d_A, union) for every case
t1 = df[df.tau == 1.0]
bad = t1[(t1.D - t1.D_prod_tau1).abs() > 1e-6]
print("\ntau=1 vs production D: max |diff| =", (t1.D - t1.D_prod_tau1).abs().max())
if len(bad): raise SystemExit(f"tau=1 does not reproduce production D:\n{bad[['cell','D','D_prod_tau1']]}")
tE = pd.read_csv(AUD/"PAPER_xstate_v3/figures/tables/tableE_decision_dA.csv")
tE["cell"] = tE.model+"_"+tE.state+"_"+tE.year.astype(str)
chk = t1.merge(tE[["cell","d_A"]], on="cell", how="left")
print("tau=1 vs tableE_decision_dA.csv d_A: max |diff| =", (chk.D - chk.d_A).abs().max(), " n =", chk.d_A.notna().sum())
if chk.d_A.isna().any() or ((chk.D - chk.d_A).abs() > 1e-6).any(): raise SystemExit(f"tau=1 vs tableE mismatch:\n{chk}")

def crossing(tau, D, nESS, KL):
    """largest tau with D<=B0, linear interpolation between ladder points (tau sorted descending).
    Returns None when D>B0 everywhere, and tau_cross=NaN with already_le_B0=True when D(1)<=B0."""
    tau, D, nESS, KL = map(np.asarray, (tau, D, nESS, KL))
    i = np.where(D <= B0)[0]
    if len(i) == 0: return None
    i = i[0]
    if i == 0: return dict(already_le_B0=True, tau_ladder_first=tau[0], D_there=D[0], nESS_there=nESS[0],
                           tau_cross_interp=np.nan, nESS_cross_interp=np.nan, disp_cross_interp=np.nan, KL_cross_interp=np.nan)
    tc = tau[i] + (B0-D[i])*(tau[i-1]-tau[i])/(D[i-1]-D[i])
    nc = np.interp(tc, [tau[i], tau[i-1]], [nESS[i], nESS[i-1]]); kc = np.interp(tc, [tau[i], tau[i-1]], [KL[i], KL[i-1]])
    return dict(already_le_B0=False, tau_ladder_first=tau[i], D_there=D[i], nESS_there=nESS[i],
                tau_ladder_prev=tau[i-1], D_prev=D[i-1], tau_cross_interp=tc, nESS_cross_interp=nc,
                disp_cross_interp=np.sqrt(1/nc-1), KL_cross_interp=kc)
cr = []
for c, g in df.groupby("cell"):
    g = g.sort_values("tau", ascending=False).reset_index(drop=True)
    x = crossing(g.tau, g.D, g.nESS, g.KL)
    base = dict(cell=c, all_feasible=bool(g.feasible.all()), D_monotone=bool((np.diff(g.D.values) <= 1e-12).all()),
                nESS_monotone=bool((np.diff(g.nESS.values) <= 1e-12).all()), D_tau1=g[g.tau==1.0].D.iloc[0],
                nESS_tau1=g[g.tau==1.0].nESS.iloc[0], disp_tau1=g[g.tau==1.0].disp.iloc[0], KL_tau1=g[g.tau==1.0].KL.iloc[0],
                D_min_tau=g.D.iloc[-1], tau_min=g.tau.iloc[-1], nESS_min_tau=g.nESS.iloc[-1], D_prod=g.D_prod_tau1.iloc[0])
    cr.append({**base, **(x or dict(already_le_B0=False))})
cr = pd.DataFrame(cr); cr.to_csv(H/"tau_crossing_B0.csv", index=False); print(); print(cr.round(4).to_string(index=False))
# per scenario: selected prior (argmin D among the admitted priors) and scenario-minimum D along the ladder
rs = pd.read_csv(AUD/"PAPER_xstate_v3/figures/tables/tableE_decision_dA.csv")
sc = []
for (st,), g in df.groupby(["state"]):
    pv = g.pivot(index="tau", columns="model", values="D").sort_index(ascending=False)
    pn = g.pivot(index="tau", columns="model", values="nESS").sort_index(ascending=False)
    for tau, r in pv.iterrows():
        sel = r.idxmin(); srt = r.sort_values()
        sc.append(dict(state=st, tau=tau, n_priors=r.notna().sum(), selected=DISP[sel], D_sel=r[sel], nESS_sel=pn.loc[tau, sel],
                       runner_up=DISP[srt.index[1]], gap=srt.iloc[1]-srt.iloc[0], D_semapop=r.get("semapop_vae", np.nan),
                       decision="Deploy" if r[sel] <= B0 else "Defer",
                       **{f"D_{m}": r[m] for m in r.index}))
sc = pd.DataFrame(sc); sc.to_csv(H/"tau_scenario_selected.csv", index=False)
print(); print(sc.round(4).to_string(index=False))
mc = []
for st, g in sc.groupby("state"):
    g = g.sort_values("tau", ascending=False).reset_index(drop=True)
    x = crossing(g.tau, g.D_sel, g.nESS_sel, np.full(len(g), np.nan)) or {}
    mc.append(dict(state=st, selected_changes=g.selected.nunique() > 1, selected_set=";".join(dict.fromkeys(g.selected)),
                   decision_changes=g.decision.nunique() > 1, decisions=";".join(dict.fromkeys(g.decision)), **x))
mc = pd.DataFrame(mc); mc.to_csv(H/"tau_scenario_crossing.csv", index=False); print(); print(mc.round(4).to_string(index=False))
# pre-projection discrepancy of the selected realization r* (tau-independent), production tableE
print(); print(rs[rs.year == 2024][["state","model","d_A_pre","d_A_rstar_pre","d_A"]].round(4).to_string(index=False))
