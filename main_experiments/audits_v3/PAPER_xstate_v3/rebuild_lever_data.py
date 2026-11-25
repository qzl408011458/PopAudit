"""Rebuild the lever-experiment inputs from the regenerated audit + certificate.
[v3 port, v3 protocol 2026-09-02: reads PAPER_xstate_v3/xstate_certificate_all.csv
 (81 cells, 9 uniform tags).  The v3 certificate carries the paper's |C_a|-weighted
 Cauchy-Schwarz bound R_2_MA instead of the legacy R_2_optionA column, so the budget
 sweep tests R_2^MA <= B(H); formulas/thresholds otherwise unchanged.]

  exp_i_sigma_curve.csv   : Commute_Mode sigma-widening curve over the L1-survivor
                            cells (cell set = L1_accept) x m in {1,2,4,8}.
  exp_ii_vH.csv           : per-horizon mean sigma-norm (model-independent L2 sigma)
                            and volatility v(H) = sigma(H)/sigma(2018) - 1.
  exp_ii_budget_curve.csv : volatility-aware budget sweep over gamma; for each gamma,
                            B(H)=0.30(1+gamma*v(H)); deployable VALID case = R_2^MA <= B(H);
                            global_ADOPT = all VALID cases deployable.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from recompute_sigma_curve import cell_curve, MS  # noqa: E402

DATA = HERE
GAMMAS = [0.0, 0.3, 0.5, 1.0]
FLAT = 0.30

cert = pd.read_csv(DATA / "xstate_certificate_all.csv")

# ── exp_i: sigma-recovery curve over L1-survivor cells ───────────────────────
surv = cert[cert.L1_accept == True][["model", "state", "year"]].drop_duplicates()
rows = []
for _, c in surv.iterrows():
    rows += cell_curve(c.state, c.model, str(int(c.year)))
exp_i = pd.DataFrame(rows)
exp_i.to_csv(DATA / "exp_i_sigma_curve.csv", index=False)
print(f"exp_i_sigma_curve.csv: {surv.shape[0]} L1-survivor cells x {len(MS)} m  ->  {len(exp_i)} rows")

# ── exp_ii_vH: per-horizon mean sigma-norm + volatility ──────────────────────
base = exp_i[exp_i.m == 1]
msn = base.groupby("year").sigma_norm.mean()
s2018 = float(msn.loc[2018])
vH = pd.DataFrame({"year": msn.index,
                   "mean_sigma_norm": msn.values.round(4),
                   "v_H": (msn.values / s2018 - 1.0).round(4)})
vH.to_csv(DATA / "exp_ii_vH.csv", index=False)
print("exp_ii_vH.csv:\n" + vH.to_string(index=False))
v = {int(r.year): float(r.v_H) for _, r in vH.iterrows()}

# ── exp_ii_budget_curve: gamma sweep ─────────────────────────────────────────
valid = cert[cert.VALID == True].copy()
n_valid = len(valid)
rows = []
for g in GAMMAS:
    B = {y: FLAT * (1.0 + g * v[y]) for y in v}
    slack = valid.apply(lambda r: B[int(r.year)] - float(r.R_2_MA), axis=1)
    n_dep = int((slack >= 0).sum())
    rows.append({"gamma": g, "n_deployable": n_dep, "n_valid": n_valid,
                 "global_ADOPT": bool(n_dep == n_valid),
                 "B(2018)": round(B[2018], 3), "B(2021)": round(B[2021], 3),
                 "B(2024)": round(B[2024], 3), "min_slack": round(float(slack.min()), 3)})
bc = pd.DataFrame(rows)
bc.to_csv(DATA / "exp_ii_budget_curve.csv", index=False)
print("exp_ii_budget_curve.csv:\n" + bc.to_string(index=False))
