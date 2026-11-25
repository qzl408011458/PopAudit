"""Sensitivity of the certificate to K=50 and to the best-of-K choice r*.
Reads the main-experiment outputs (main_experiments/audits_v3) only. Gates copied from decision_certificate_xstate_v3.py:
  L1: invalid_mass_mean <= 0.005; marg <= 0.05 - DKW(a/3, K n, 10); dep <= 0.10 - DKW(...);
      1 - support_cov <= 0.30 - DKW(a/3, n, 1)
  L2: chibar_H = mean_r chi_H^(r) <= 5 + t_{0.95,K-1} * sd(chi_H^(r))/sqrt(K)
  L3: not recomputed for subsets (0 production rejections after L2; see a10_l3_median.py).
Subsets: K in {10, 25}, 200 random subsets each (rng seed 20261125), drawn independently per case.
Outputs (this dir): a10_spread_all81.csv, a10_subsample_case.csv, a10_subsample_validcounts.csv
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[2])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.argv = [sys.argv[0]]                      # keep the production module in PRIMARY mode
sys.path.insert(0, _PA.R("/main_experiments"))
import decision_certificate_xstate_v3 as DC
from scipy.stats import t as student_t

OUT = Path(__file__).resolve().parent
BASE = DC.BASE
NSUB = 200
KS = (10, 25)
rng = np.random.default_rng(20261125)
cert = pd.read_csv(BASE / "PAPER_xstate_v3/xstate_certificate_all.csv")
cert = cert[cert.year.notna()].copy(); cert["year"] = cert.year.astype(int).astype(str)

def l1_gate(pr, idx, n_agents):
    sub = pr.iloc[idx]; K = len(idx); a = DC.ALPHA[0] / 3
    inv = sub.invalid_mass.mean(); marg = sub.marginal_total_discrepancy.mean()
    dep = sub.dependency_total_discrepancy.mean()
    supp_lb = sub.support_coverage.max()      # union over the subset >= best single realization
    if inv > DC.INVALID_MASS_TOL: return False, "invalid_mass"
    sm = DC.dkw_critical(a, K * n_agents, 10); ss = DC.dkw_critical(a, n_agents, 1)
    ok = marg <= DC.KAPPA_1_MARG - sm and dep <= DC.KAPPA_1_DEP - sm and (1 - supp_lb) <= DC.KAPPA_1_SUPPORT - ss
    return bool(ok), ("" if ok else "marg/dep/support(lb)")

def l2_gate(loss):
    K = len(loss); chibar = loss.mean()
    cut = DC.KAPPA_2 + student_t.ppf(0.95, K - 1) * loss.std(ddof=1) / np.sqrt(K)
    return bool(chibar <= cut), chibar, cut

spread, case_rows, draws = [], [], {K: np.zeros((NSUB, len(cert)), bool) for K in KS}
for j, r in enumerate(cert.itertuples()):
    st, m, y = r.state, r.model, r.year
    n_agents = DC.STATE_N_AGENTS[st]
    w = pd.read_csv(BASE / f"layer2_{m}_{st}_{y}/layer2_per_realization_weights.csv")
    loss = w.scenario_loss.to_numpy(float)
    pr = pd.read_csv(BASE / f"layer1_paper_{m}_{st}/layer1_per_realization_metrics.csv")
    assert len(loss) == 50 and len(pr) == 50
    # full-K reproduction of the production certificate
    a2, chibar, cut = l2_gate(loss)
    assert abs(chibar - r.L2_chi_H) < 1e-4 and abs(cut - r.L2_cutoff) < 1e-4, (st, m, y)
    rstar = int(np.argmin(loss)); order = np.argsort(loss, kind="stable")
    rmed = int(order[24])                      # lower median (25th of 50)
    spread.append({"state": st, "model": m, "year": y, "VALID": bool(r.VALID),
                   "chi_min": loss.min(), "chi_q25": np.quantile(loss, .25), "chi_median": np.median(loss),
                   "chi_q75": np.quantile(loss, .75), "chi_max": loss.max(), "chi_mean": chibar,
                   "chi_sd": loss.std(ddof=1), "cutoff": cut, "rstar": rstar, "chi_rstar": loss[rstar],
                   "r_median": rmed, "chi_rmedian": loss[rmed],
                   "gap_median_minus_rstar": np.median(loss) - loss[rstar],
                   "rel_range": (loss.max() - loss.min()) / chibar,
                   "margin_to_cutoff": cut - chibar,
                   "n_real_below_5": int((loss <= 5.0).sum())})
    row = {"state": st, "model": m, "year": y, "VALID_K50": bool(r.VALID), "chibar_K50": chibar,
           "cutoff_K50": cut}
    for K in KS:
        acc = np.zeros(NSUB, bool); cuts = []
        for s in range(NSUB):
            idx = np.sort(rng.choice(50, K, replace=False))
            ok1, _ = l1_gate(pr, idx, n_agents)
            ok2, cb, c = l2_gate(loss[idx]); cuts.append(c)
            # L3 held at its production verdict (all True for L1&L2 passers)
            acc[s] = ok1 and ok2 and bool(r.L3_accept)
        draws[K][:, j] = acc
        row[f"admit_rate_K{K}"] = acc.mean()
        row[f"flip_rate_K{K}"] = (acc != bool(r.VALID)).mean()
        row[f"cutoff_mean_K{K}"] = float(np.mean(cuts))
    case_rows.append(row)

sp = pd.DataFrame(spread); sp.to_csv(OUT / "a10_spread_all81.csv", index=False)
cs = pd.DataFrame(case_rows); cs.to_csv(OUT / "a10_subsample_case.csv", index=False)
vc = []
for K in KS:
    n = draws[K].sum(1)
    vc.append({"K": K, "n_subsets": NSUB, "valid_min": n.min(), "valid_q05": np.quantile(n, .05),
               "valid_median": np.median(n), "valid_q95": np.quantile(n, .95), "valid_max": n.max(),
               "frac_equal_23": (n == 23).mean(),
               "frac_same_set": (draws[K] == cert.VALID.to_numpy(bool)).all(1).mean()})
    # count distribution
    print(f"K={K} VALID-count distribution:", dict(zip(*np.unique(n, return_counts=True))))
vc = pd.DataFrame(vc); vc.to_csv(OUT / "a10_subsample_validcounts.csv", index=False)
pd.set_option("display.width", 250)
print(vc.to_string(index=False))
print("\nCases whose verdict ever flips:")
print(cs[(cs.flip_rate_K10 > 0) | (cs.flip_rate_K25 > 0)].round(4).to_string(index=False))
v = sp[sp.VALID]
print("\nValid-case spread (23):")
print(v[["state","model","year","chi_min","chi_median","chi_max","chi_sd","chi_rstar","gap_median_minus_rstar","rel_range","margin_to_cutoff","n_real_below_5"]].round(4).to_string(index=False))
print("\nSummary over valid: rel_range", v.rel_range.describe().round(4).to_dict())
print("gap median-r*", v.gap_median_minus_rstar.describe().round(4).to_dict())
print("chi_max vs 5: n valid cases with chi_max>5:", int((v.chi_max > 5).sum()))
