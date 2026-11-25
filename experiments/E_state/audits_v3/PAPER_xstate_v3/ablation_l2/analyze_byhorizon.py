"""Disaggregate the ablation by horizon (2018/2021/2024). Within a fixed year the
horizon-drift confound is removed by construction, so a within-year pass-vs-fail
contrast (2018: 11 pass/5 fail; 2024: 5 pass/11 fail) is horizon-controlled. Look for
LOCAL evidence for the hypothesis that L2-pass cells project to more faithful pops."""
import os
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
d = pd.read_csv(f"{HERE}/ablation_results.csv")
d["prior"] = d.state + "/" + d.model

def presid(x, y, z):
    """partial corr of x,y controlling z (all 1D)"""
    def r(a, b):
        Z = np.column_stack([np.ones(len(z)), z])
        ra = a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
        rb = b - Z @ np.linalg.lstsq(Z, b, rcond=None)[0]
        return stats.pearsonr(ra, rb)
    return r(x, y)

for yr in [2018, 2021, 2024]:
    g = d[d.year == yr].copy().sort_values("L2_chi_H")
    npass, nfail = int(g.L2pass.sum()), int((~g.L2pass).sum())
    print("="*100)
    print(f"HORIZON {yr}   ({npass} L2-pass, {nfail} L2-fail)   [within-year => horizon controlled]")
    print("="*100)
    show = g[["state", "model", "L2pass", "L2_chi_H", "L1_marg",
              "SRMSE_J_mean", "SRMSE_U_mean", "Recall_mean", "F1_mean"]].round(4)
    print(show.to_string(index=False))
    # within-year cross-prior association of chi_H with fidelity (drift removed; L1 still varies)
    for m, flip, lab in [("SRMSE_J_mean", 1, "joint-SRMSE"), ("Recall_mean", -1, "recall(worse)"),
                          ("F1_mean", -1, "F1(worse)")]:
        y = g[m].values * flip
        rho = stats.spearmanr(g.L2_chi_H.values, y)[0]
        pr = stats.pearsonr(g.L2_chi_H.values, y)[0]
        # partial controlling base-year L1_marg
        try:
            pcr, pcp = presid(g.L2_chi_H.values, y, g.L1_marg.values)
        except Exception:
            pcr, pcp = np.nan, np.nan
        print(f"   corr(chi_H, {lab:14s}):  pearson={pr:+.2f}  spearman={rho:+.2f}  | partial|L1_marg={pcr:+.2f} (p={pcp:.3f})")
    if npass and nfail:
        for m in ["SRMSE_J_mean", "Recall_mean", "F1_mean", "L1_marg"]:
            mp, mf = g[g.L2pass][m].mean(), g[~g.L2pass][m].mean()
            print(f"   mean {m:14s}: PASS={mp:.4f}  FAIL={mf:.4f}  (diff {mp-mf:+.4f})")
    print()
