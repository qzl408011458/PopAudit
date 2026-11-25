"""Supplementary check (key: srmse). Threshold sweep for the two reduced protocols of Sec 5.4 / S2.

Read-only on audits_v3/. Recomputes the marginal-only SRMSE (same formula as
PAPER_xstate_v3/baselines/baseline_reproduction.py::compute_srmse_marg and
figures/make_figures_v3.py::figF) from layer2_<model>_<state>_<year>/layer2_target_summary.csv
(prior_mean = ensemble-mean audited marginal over K realizations, target_m = m*(S_H)),
and takes VALID / L1 columns from PAPER_xstate_v3/xstate_certificate_all.csv.
Writes everything to this directory.
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[4])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(_PA.R("/experiments/E_state/audits_v3"))
PAPER = ROOT / "PAPER_xstate_v3"
OUT = Path(__file__).resolve().parent

fw = pd.read_csv(PAPER / "xstate_certificate_all.csv")
fw["year"] = fw.year.astype(int)


def srmse(state, model, year):
    t = pd.read_csv(ROOT / f"layer2_{model}_{state}_{year}" / "layer2_target_summary.csv")
    m, q = t.prior_mean.to_numpy(float), t.target_m.to_numpy(float)
    return float(np.sqrt(np.mean((m - q) ** 2)) / q.mean()), len(t)


vals = [srmse(r.state, r.model, r.year) for r in fw.itertuples()]
fw["srmse"] = [v[0] for v in vals]
fw["n_cells"] = [v[1] for v in vals]
fw["VALID"] = fw.VALID.astype(bool)
fw["L1_accept"] = fw.L1_accept.astype(bool)

# cross-check against the stored baseline file
b = pd.read_csv(PAPER / "baselines" / "baseline_comparison.csv")
chk = fw.merge(b[["state", "model", "year", "srmse_marg", "Beckman_ACCEPT"]], on=["state", "model", "year"])
print("max |srmse - stored srmse_marg| =", float(np.abs(chk.srmse - chk.srmse_marg).max()))
print("accepted at 0.30 (recomputed / stored):", int((fw.srmse <= 0.30).sum()), int(b.Beckman_ACCEPT.sum()))
print("n_cells per case:", sorted(fw.n_cells.unique()))

V = fw.VALID.values
N = len(fw)
print(f"N={N}  VALID={V.sum()}  L1_accept={fw.L1_accept.sum()}")


def summarize(acc, label, thr):
    acc = np.asarray(acc, bool)
    row = dict(protocol=label, threshold=thr, accepted=int(acc.sum()))
    for y in (2018, 2021, 2024):
        row[f"acc_{y}"] = int(acc[fw.year.values == y].sum())
    row["valid_rejected"] = int((V & ~acc).sum())
    row["invalid_accepted"] = int((~V & acc).sum())
    row["agree"] = int((V == acc).sum())
    row["agree_pct"] = round(100 * (V == acc).mean(), 1)
    # among invalid cases accepted, how many are L1 failures vs L2 failures
    ff = fw.first_fail.fillna("")
    row["inv_acc_L1fail"] = int((acc & ~V & ff.str.startswith("L1")).sum())
    row["inv_acc_L2fail"] = int((acc & ~V & (ff == "L2")).sum())
    row["inv_acc_L3fail"] = int((acc & ~V & (ff == "L3")).sum())
    return row


rows = []
for thr in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    rows.append(summarize(fw.srmse.values <= thr, "marginal-only SRMSE", thr))

# ---------- structural-only scorecard: the Layer-1 gate with its allowances
K1 = dict(marg=0.05, dep=0.10, supp=0.30, inv=0.005)
l1_rule = ((fw.L1_marg <= K1["marg"]) & (fw.L1_dep <= K1["dep"]) &
           ((1 - fw.L1_support) <= K1["supp"]) & (fw.L1_invalid <= K1["inv"]))
print("L1 rule from columns reproduces L1_accept:", bool((l1_rule == fw.L1_accept).all()),
      " mismatches:", int((l1_rule != fw.L1_accept).sum()))
for s in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
    acc = ((fw.L1_marg <= s * K1["marg"]) & (fw.L1_dep <= s * K1["dep"]) &
           ((1 - fw.L1_support) <= s * K1["supp"]) & (fw.L1_invalid <= K1["inv"]))
    rows.append(summarize(acc.values, "structural-only (allowances x s)", s))
for inv in [0.001, 0.0025, 0.005, 0.01, 0.02]:
    acc = ((fw.L1_marg <= K1["marg"]) & (fw.L1_dep <= K1["dep"]) &
           ((1 - fw.L1_support) <= K1["supp"]) & (fw.L1_invalid <= inv))
    rows.append(summarize(acc.values, "structural-only (invalid tol)", inv))

tab = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(tab.to_string(index=False))
tab.to_csv(OUT / "threshold_sweep.csv", index=False)

# ---------- best achievable agreement for the marginal-only rule over a continuous threshold
grid = np.unique(np.concatenate([[0.0], np.sort(fw.srmse.values), [10.0]]))
best = max(((int(((fw.srmse.values <= t) == V).sum()), t) for t in grid))
print(f"\nmarginal-only: best agreement over all thresholds = {best[0]}/81 at t={best[1]:.4f}")
print("AUC of -SRMSE for VALID:", round(roc_auc_score(V, -fw.srmse.values), 3))
print("AUC of -SRMSE for VALID among L1-pass cases:",
      round(roc_auc_score(V[fw.L1_accept], -fw.srmse.values[fw.L1_accept]), 3))
# smallest threshold that admits all 20 valid cases
print("max SRMSE among VALID:", round(fw.srmse[V].max(), 4), " min SRMSE among VALID:", round(fw.srmse[V].min(), 4))
print("min SRMSE among 2021 cases:", round(fw.srmse[fw.year == 2021].min(), 4))
t_all = fw.srmse[V].max()
print("at t = max valid SRMSE: accepted", int((fw.srmse <= t_all).sum()),
      "invalid accepted", int(((fw.srmse <= t_all) & ~V).sum()))
# structural-only best agreement: horizon invariant
print("structural-only: decision constant across horizons per prior-state;"
      " max achievable agreement if it admitted exactly the 2018-valid priors:")
ps = fw.groupby(["state", "model"]).VALID.apply(list)
ub = sum(max(sum(v), 3 - sum(v)) for v in ps)
print("  upper bound on agreement of ANY horizon-invariant rule:", ub, "/81")

# distribution of SRMSE by year and validity
print("\nSRMSE by year x VALID:")
print(fw.groupby(["year", "VALID"]).srmse.describe()[["count", "min", "50%", "max"]].round(3))
print("\nSRMSE by year x first_fail:")
print(fw.assign(ff=fw.first_fail.fillna("valid")).groupby(["year", "ff"]).srmse.describe()[["count", "min", "50%", "max"]].round(3))
# cases near 0.30
near = fw[(fw.srmse > 0.25) & (fw.srmse < 0.35)].sort_values("srmse")
print("\ncases with 0.25 < SRMSE < 0.35:")
print(near[["state", "model", "year", "srmse", "VALID", "first_fail", "L2_chi_H", "L2_cutoff"]].to_string(index=False))
fw[["state", "model", "year", "srmse", "VALID", "L1_accept", "first_fail", "L2_chi_H"]].to_csv(OUT / "per_case_srmse.csv", index=False)

# ---------- threshold-free ordering facts
v24min = fw.srmse[V & (fw.year == 2024)].min()
inv_below = fw[~V & (fw.srmse < v24min)]
print(f"\ninvalid cases with SRMSE below the smallest valid-2024 SRMSE ({v24min:.3f}): {len(inv_below)}"
      f"  by year {inv_below.year.value_counts().sort_index().to_dict()}")
t = fw.srmse[V & (fw.year == 2018)].max()
a = fw.srmse <= t
print(f"best-agreement threshold {t:.4f}: accepted {a.sum()}, valid rejected {(V & ~a).sum()}, invalid accepted {(~V & a).sum()}"
      f" -> {fw[~V & a][['state','model','year','srmse','first_fail']].values.tolist()}")
# rank concordance: fraction of (valid, invalid) pairs where valid has lower SRMSE
pv, pi = fw.srmse[V].values, fw.srmse[~V].values
print("P(SRMSE_valid < SRMSE_invalid) =", round((pv[:, None] < pi[None, :]).mean(), 3))
