"""Recompute the commute-mode sigma-widening curve from raw Layer-2 audit data.
[v3 port, v3 protocol 2026-09-02: reads audits_v3 (uniform layer2_<tag>_<ST>_<year>
 naming) and PAPER_xstate_v3; formulas unchanged.]

sigma_j(m): non-commute targets unchanged; commute-mode targets widened by
multiplier m as
    sigma_c(m) = max( floor_c * m,
                      m*H*|mean_delta_c| + m*sqrt(H)*sd_delta_c + sampling_se_c )
(scale drift + volatility + floor by m; leave the reference-sampling SE term).

chi_H(m)     = sum_a 0.5 * mean_{c in a} z_c(m)^2,  z_c(m)=|prior_mean_c - target_c|/sigma_c(m)
sigma_norm(m)= ||sigma(m)||_2 over all targets
R2A(m)       = 0.5 * sigma_norm(m) * sqrt(2*chi_H(m))
L2 gate      : PASS iff chi_H(m) <= kappa_2 + z_{1-alpha} sqrt(J/(2K))
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[4])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
from pathlib import Path
import numpy as np, pandas as pd, json
from scipy.stats import norm

AUD  = Path(_PA.R("/experiments/E_state/audits_v3"))
DATA = AUD / "PAPER_xstate_v3"
KAPPA_2, ALPHA = 5.0, 0.05
MS = [1, 2, 4, 8]

def l2dir(s, m, y):
    return AUD / f"layer2_{m}_{s}_{y}"

def sigma_commute(row, m):
    H = row["historical_projection_horizon"]
    floor = row["historical_sigma_floor"]
    raw = (m * H * abs(row["historical_mean_delta"])
           + m * np.sqrt(H) * row["historical_sd_delta"]
           + row["historical_sampling_se"])
    return max(floor * m, raw)

def cell_curve(s, model, y):
    t = pd.read_csv(l2dir(s, model, y) / "layer2_target_summary.csv")
    j = json.loads((l2dir(s, model, y) / "layer2_summary.json").read_text())
    K = int(j["n_realizations"]); J = int(j["n_targets"])
    z_cut = norm.ppf(1 - ALPHA)
    cutoff = KAPPA_2 + z_cut * np.sqrt(J / (2.0 * K))
    is_cm = t["attr"] == "Commute_Mode"
    sig_nc = t.loc[~is_cm, "sigma"].to_numpy()
    out = []
    for m in MS:
        sig = t["sigma"].to_numpy().copy()
        sig_cm = np.array([sigma_commute(r, m) for _, r in t.loc[is_cm].iterrows()])
        sig[is_cm.to_numpy()] = sig_cm
        # chi_H from prior_mean prediction
        z = np.abs(t["prior_mean"].to_numpy() - t["target_m"].to_numpy()) / sig
        chi = 0.0
        for a, idx in t.groupby("attr").groups.items():
            zi = z[[t.index.get_loc(ii) for ii in idx]]
            chi += 0.5 * np.mean(zi ** 2)
        snorm = float(np.sqrt(np.sum(sig ** 2)))
        r2a = 0.5 * snorm * np.sqrt(2.0 * chi)
        out.append(dict(model=model, state=s, year=int(y), m=m,
                        chi_H=chi, sigma_norm=snorm, R2A=r2a,
                        L2_gate="PASS" if chi <= cutoff else "FAIL"))
    return out

if __name__ == "__main__":
    # Verify against the logged exp_i_sigma_curve.csv (written by rebuild_lever_data.py)
    ex = pd.read_csv(DATA / "exp_i_sigma_curve.csv")
    cells = ex[ex.m == 1][["model", "state", "year"]].drop_duplicates()
    rows = []
    for _, c in cells.iterrows():
        rows += cell_curve(c.state, c.model, str(c.year))
    rc = pd.DataFrame(rows)
    cmp = ex.merge(rc, on=["model", "state", "year", "m"], suffixes=("_log", "_re"))
    cmp["d_chi"] = (cmp.chi_H_log - cmp.chi_H_re).abs()
    cmp["d_sig"] = (cmp.sigma_norm_log - cmp.sigma_norm_re).abs()
    cmp["gate_match"] = cmp.L2_gate_log == cmp.L2_gate_re
    print(f"=== recompute vs logged ({len(cells)} cells x {len(MS)} m) ===")
    print(f"max |dchi_H|   : {cmp.d_chi.max():.4f}   mean: {cmp.d_chi.mean():.4f}")
    print(f"max |dsigma|   : {cmp.d_sig.max():.5f}   mean: {cmp.d_sig.mean():.5f}")
    print(f"L2-gate match  : {cmp.gate_match.sum()}/{len(cmp)}")
    print()
    print("worst 5 chi_H rows:")
    print(cmp.nlargest(5, "d_chi")[["model","state","year","m","chi_H_log","chi_H_re","L2_gate_log","L2_gate_re"]].to_string(index=False))
