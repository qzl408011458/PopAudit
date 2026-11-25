"""Summary of a10_l3_median.py: D at the median-chi realization vs r*; re-select theta-dagger per scenario; B0=0.30 decisions."""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[2])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import pandas as pd, numpy as np
from pathlib import Path
OUT = Path(__file__).resolve().parent
V3 = Path(_PA.R("/main_experiments/audits_v3/PAPER_xstate_v3"))
prod = pd.read_csv(V3 / "figures/tables/tableE_decision_dA.csv")[["state", "model", "year", "d_A"]]
l3 = pd.read_csv(OUT / "a10_l3_med_rstar.csv")
w = l3.pivot_table(index=["state", "model", "year"], columns="tag", values=["D", "chi_H_r", "realization", "L3_accept"], aggfunc="first")
w.columns = [f"{a}_{b}" for a, b in w.columns]; w = w.reset_index().merge(prod, on=["state", "model", "year"])
w["repro_err"] = (w.D_rstar - w.d_A).abs()
w["dD"] = w.D_med - w.D_rstar
w.to_csv(OUT / "a10_D_med_vs_rstar.csv", index=False)
pd.set_option("display.width", 250)
print(w.round(5).to_string(index=False))
print("max |D_rstar(rerun) - D(production)| =", w.repro_err.max())
print("dD = D_med - D_rstar:", w.dD.describe().round(5).to_dict(), " n(dD>0) =", int((w.dD > 0).sum()), "/", len(w))
print("L3 accepts at median realization:", int(w.L3_accept_med.sum()), "/", len(w))
B0 = 0.30
vH = pd.read_csv(V3 / "exp_ii_vH.csv").set_index("year")["v_H"]
rows = []
for (st, y), g in w.groupby(["state", "year"]):
    r = {"state": st, "year": y}
    for col in ("D_rstar", "D_med"):
        gg = g.sort_values(col); D = float(gg[col].iloc[0])
        r[f"theta_{col}"] = gg.model.iloc[0]; r[f"Ddag_{col}"] = D
        r[f"gap_{col}"] = float(gg[col].iloc[1] - D) if len(gg) > 1 else np.nan
        r[f"dec_{col}"] = "Deploy" if D <= B0 else "Defer"
        r[f"gmin_{col}"] = 0.0 if D <= B0 else (D / B0 - 1) / float(vH[int(y)])
        r[f"maxD_{col}"] = float(g[col].max())
    rows.append(r)
s = pd.DataFrame(rows); s.to_csv(OUT / "a10_scenario_decision_med_vs_rstar.csv", index=False)
print(s.round(4).to_string(index=False))
