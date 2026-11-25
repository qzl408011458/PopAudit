"""Contribution of the Taxicab component to chi-bar_H in the production (current) Layer-2 outputs:
per case, mean over the K realizations of z_taxi^2 / (2 |C_commute|), with |C_commute| the number of
commute-mode targets Layer 2 standardized.  Read-only on experiments/E_state/audits_v3."""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[4])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import numpy as np, pandas as pd
from pathlib import Path
PROD = Path(_PA.R("/experiments/E_state/audits_v3"))
OUT = Path(__file__).resolve().parent / "results"
cert = pd.read_csv(PROD / "PAPER_xstate_v3/xstate_certificate_all.csv")
rows = []
for r in cert.itertuples():
    d = PROD / f"layer2_{r.model}_{r.state}_{int(r.year)}"
    sig = pd.read_csv(d / "layer2_historical_sigma_summary.csv")
    w = pd.read_csv(d / "layer2_per_realization_weights.csv")
    cm = sig[sig.attr == "Commute_Mode"]
    t = cm[cm.value == "Taxicab"].iloc[0]
    z2 = ((w[t.target].to_numpy(float) - float(t.target_m)) / float(t.sigma)) ** 2
    rows.append(dict(model=r.model, state=r.state, year=int(r.year), L1=bool(r.L1_accept), VALID=bool(r.VALID),
                     n_commute=len(cm), taxi_target=float(t.target_m), taxi_sigma=float(t.sigma),
                     taxi_gen_mean=float(w[t.target].mean()), contrib=float(np.mean(z2) / (2 * len(cm))),
                     chi_bar=float(r.L2_chi_H)))
T = pd.DataFrame(rows); T.to_csv(OUT / "taxi_contrib.csv", index=False)
for y in (2018, 2021, 2024):
    g = T[T.year == y]
    print(y, "max contrib all", round(g.contrib.max(), 5), "| L1-admissible", round(g[g.L1].contrib.max(), 5),
          "| valid", round(g[g.VALID].contrib.max(), 5) if g.VALID.any() else None,
          "| argmax", g.loc[g.contrib.idxmax(), ["model", "state"]].tolist(), "| n_commute", sorted(g.n_commute.unique()))
print(T[T.year == 2024].groupby("state")[["taxi_target", "taxi_sigma", "taxi_gen_mean"]].mean())
