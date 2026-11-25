"""NEW helper (v3 protocol 2026-09-02, not a port): per-scenario inputs for the
selected-prior budget table (tab:selected-prior-budget).

For each audited scenario S_H = (state, year):
  |Theta_adm|      : number of VALID priors (all three gates)
  theta_dagger     : admitted prior with the smallest post-projection residual
                     D = d_A = 0.5 * sum_j |prior_mean_j - target_j|   (make_figures_v2.figE)
  D_dagger         : that residual;  flat decision Deploy iff D_dagger <= B0 = 0.30
  gamma_min        : (D_dagger/B0 - 1)/v(H) if D_dagger > B0 else 0   (v(H) from exp_ii_vH.csv)
plus the same quantities for the Cauchy-Schwarz bound R_2^MA (certificate column),
which is what exp_ii_budget_curve.csv sweeps.
"""
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AUD = HERE.parent
B0 = 0.30
H = {2018: 5, 2021: 8, 2024: 11}
cert = pd.read_csv(HERE / "xstate_certificate_all.csv")
vH = pd.read_csv(HERE / "exp_ii_vH.csv").set_index("year").v_H.to_dict()

def d_A(state, model, year):
    t = pd.read_csv(AUD / f"layer2_{model}_{state}_{year}" / "layer2_target_summary.csv")
    return 0.5 * float(t["abs_err_prior"].sum())

valid = cert[cert.VALID == True].copy()
valid["d_A"] = [d_A(r.state, r.model, int(r.year)) for r in valid.itertuples()]
valid.to_csv(HERE / "exp_ii_valid_dA.csv", index=False,
             columns=["state", "model", "year", "R_2_MA", "sigma_norm_w", "L2_chi_H", "d_A"])

def gmin(D, y):
    if D <= B0:
        return 0.0
    return (D / B0 - 1.0) / vH[y] if vH[y] > 0 else np.inf

rows = []
for y in (2018, 2021, 2024):
    for st in ("CA", "OH", "TX"):
        g = valid[(valid.state == st) & (valid.year == y)]
        row = {"state": st, "year": y, "H": H[y], "n_adm": len(g), "v_H": vH[y]}
        if len(g) == 0:
            row.update(theta_dagger_dA=None, D_dagger_dA=None, flat_dA="Defer", gamma_min_dA=None,
                       max_dA=None, theta_dagger_R2MA=None, D_dagger_R2MA=None, flat_R2MA="Defer",
                       gamma_min_R2MA=None, max_R2MA=None, B0_admit_dA=None, B0_admit_R2MA=None)
        else:
            a = g.loc[g.d_A.idxmin()]; b = g.loc[g.R_2_MA.idxmin()]
            row.update(theta_dagger_dA=a.model, D_dagger_dA=round(a.d_A, 4),
                       flat_dA="Deploy" if a.d_A <= B0 else "Defer",
                       gamma_min_dA=round(gmin(a.d_A, y), 3), max_dA=round(g.d_A.max(), 4),
                       theta_dagger_R2MA=b.model, D_dagger_R2MA=round(b.R_2_MA, 4),
                       flat_R2MA="Deploy" if b.R_2_MA <= B0 else "Defer",
                       gamma_min_R2MA=round(gmin(b.R_2_MA, y), 3), max_R2MA=round(g.R_2_MA.max(), 4),
                       B0_admit_dA=round(a.d_A, 4), B0_admit_R2MA=round(b.R_2_MA, 4))
        rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(HERE / "exp_ii_scenario_budget.csv", index=False)
pd.set_option("display.width", 250)
print(out.to_string(index=False))
print("\nper-valid-cell d_A vs R_2_MA:")
print(valid.sort_values(["year", "state", "d_A"])[["state", "model", "year", "d_A", "R_2_MA"]].to_string(index=False))
