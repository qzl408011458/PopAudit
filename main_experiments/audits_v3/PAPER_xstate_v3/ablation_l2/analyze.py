"""Analyze the Layer-2 predictive-validity ablation. Let the data decide the framing.

Central question: does Layer-2's compatibility statistic chi_H predict hold-out joint
fidelity BEYOND base-year quality (L1) and horizon (year)? And does that predictive
power GROW with joint order (univariate control -> full joint)?
"""
import os, sys, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)

d = pd.read_csv(f"{HERE}/ablation_results.csv")
d["prior"] = d.state + "/" + d.model
d["yr"] = d.year.astype(int)
METS = {"SRMSE_U_mean": "univariate SRMSE (CONTROL)", "SRMSE_B_mean": "5-way SRMSE",
        "SRMSE_M_mean": "6-way SRMSE", "SRMSE_J_mean": "full-joint SRMSE",
        "Recall_mean": "recall", "F1_mean": "F1"}
# for recall/F1 higher is better -> flip sign so "higher = worse fidelity" like SRMSE
FLIP = {"Recall_mean": -1, "F1_mean": -1}

def resid(y, X):
    Xc = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    return y - Xc @ beta

# control matrix: base-year quality (L1) + horizon (year dummies)
yd = pd.get_dummies(d.yr, prefix="y", drop_first=True).astype(float)
CTRL = np.column_stack([d.L1_marg, d.L1_dep, d.T1, yd.values])

print("="*90)
print("A. POOLED and PARTIAL association of chi_H with each metric")
print("   partial = residualize metric AND chi_H on [L1_marg,L1_dep,T1, year dummies], then correlate")
print("="*90)
print(f"{'metric':22s} {'pooled r':>9} {'pooledρ':>8} | {'partial r':>10} {'partial p':>10}")
chi = d.L2_chi_H.values
chi_r = resid(chi, CTRL)
for m, lab in METS.items():
    y = d[m].values * FLIP.get(m, 1)
    pr = stats.pearsonr(chi, y)[0]; sp = stats.spearmanr(chi, y)[0]
    yr_ = resid(y, CTRL)
    pcr, pcp = stats.pearsonr(chi_r, yr_)
    star = " ***" if pcp < .01 else (" **" if pcp < .05 else (" *" if pcp < .1 else ""))
    print(f"{lab:22s} {pr:9.3f} {sp:8.3f} | {pcr:10.3f} {pcp:10.4f}{star}")

print("\n" + "="*90)
print("B. Does chi_H predict full-joint SRMSE beyond L1 + horizon?  (cluster-robust by prior)")
print("="*90)
for m in ["SRMSE_J_mean", "Recall_mean"]:
    dd = d.copy(); dd["Y"] = dd[m]
    f = smf.ols("Y ~ L2_chi_H + L1_marg + L1_dep + T1 + C(yr)", data=dd)
    res = f.fit(cov_type="cluster", cov_kwds={"groups": dd.prior})
    b, t, p = res.params["L2_chi_H"], res.tvalues["L2_chi_H"], res.pvalues["L2_chi_H"]
    print(f"  {m:14s}: chi_H coef={b:+.5f}  t={t:+.2f}  p={p:.4f}  (R2={res.rsquared:.3f})")
# prior-FE spec (absorbs ALL base-year quality; chi_H identified within prior across horizons)
print("  --- prior fixed-effects (base-year quality fully absorbed) ---")
for m in ["SRMSE_J_mean", "Recall_mean"]:
    dd = d.copy(); dd["Y"] = dd[m]
    res = smf.ols("Y ~ L2_chi_H + C(prior)", data=dd).fit(cov_type="cluster", cov_kwds={"groups": dd.prior})
    b, t, p = res.params["L2_chi_H"], res.tvalues["L2_chi_H"], res.pvalues["L2_chi_H"]
    print(f"  {m:14s}: within-prior chi_H coef={b:+.5f}  t={t:+.2f}  p={p:.4f}")

print("\n" + "="*90)
print("C. WITHIN-PRIOR: rank-corr of chi_H with metric across the 3 horizons (base-year fixed)")
print("   (mean Spearman over the priors that have 3 horizons)")
print("="*90)
for m, lab in METS.items():
    rs = []
    for pr, g in d.groupby("prior"):
        if g.yr.nunique() == 3:
            y = g[m].values * FLIP.get(m, 1)
            rho = stats.spearmanr(g.L2_chi_H.values, y)[0]
            if not np.isnan(rho): rs.append(rho)
    print(f"  {lab:22s} mean within-prior ρ(chi_H, worse-fidelity) = {np.mean(rs):+.3f}  (n_priors={len(rs)})")

print("\n" + "="*90)
print("D. REPLACEABILITY: is chi_H redundant with base-year L1 or with Layer-3 nESS?")
print("="*90)
print(f"  corr(chi_H, L1_marg) = {stats.pearsonr(d.L2_chi_H, d.L1_marg)[0]:+.3f}   "
      f"corr(chi_H, L3_nESS) = {stats.pearsonr(d.L2_chi_H, d.L3_nESS)[0]:+.3f}")
# does an L1-only or nESS-only threshold reproduce the 16 L2-pass set?
from sklearn.metrics import roc_auc_score
y_pass = d.L2pass.astype(int).values
for surr, name, sign in [(d.L1_marg, "L1_marg", -1), (d.L3_nESS, "L3_nESS", +1), (d.T1, "T1(base)", -1)]:
    auc = roc_auc_score(y_pass, sign * surr.values)
    print(f"  AUC of {name:9s} predicting the L2-pass set = {auc:.3f}")

print("\n" + "="*90)
print("E. DISCRIMINATION vs JOINT ORDER (partial |r| after controls): should rise 1-way -> 7-way")
print("="*90)
for m in ["SRMSE_U_mean", "SRMSE_B_mean", "SRMSE_M_mean", "SRMSE_J_mean"]:
    yr_ = resid(d[m].values, CTRL)
    pcr, pcp = stats.pearsonr(chi_r, yr_)
    order = {"SRMSE_U_mean": "1-way (control)", "SRMSE_B_mean": "5-way", "SRMSE_M_mean": "6-way", "SRMSE_J_mean": "7-way"}[m]
    print(f"  {order:16s} partial r(chi_H) = {pcr:+.3f}  (p={pcp:.4f})")

print("\n" + "="*90)
print("F. Binary pass/fail group means (context; confounded by horizon+L1)")
print("="*90)
g = d.groupby("L2pass")[["SRMSE_J_mean", "Recall_mean", "F1_mean", "L1_marg", "L2_chi_H"]].mean()
print(g.round(4).to_string())
