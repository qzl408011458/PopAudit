"""Null reference for the Layer-2 gate kappa_2 (Section 6.4, Supplementary S3.6).

Under the null every standardized residual z_{a,c} of Eq. (attribute-wise-chi) is an
independent standard normal, so chi_H = sum_a (1/(2|C_a|)) sum_c z_{a,c}^2 has mean |A|/2.
The category sets C_a are those of each scenario reference (as in Layer 2, read from
layer2_historical_sigma_summary.csv of the bn run, which does not depend on the prior).
This is the distribution of chi_H for a single realization, not of the ensemble mean.
"""
import numpy as np, pandas as pd
from pathlib import Path
A = Path(__file__).resolve().parents[2]          # audits_v3
rng = np.random.default_rng(20261124)
N = 400_000
rows = []
for s in ["CA", "TX", "OH"]:
    for y in [2018, 2021, 2024]:
        d = pd.read_csv(A / f"layer2_bn_pgmpy_{s}_{y}" / "layer2_historical_sigma_summary.csv") \
            if (A / f"layer2_bn_pgmpy_{s}_{y}").exists() else \
            pd.read_csv(A / f"layer2_bn_{s}_{y}" / "layer2_historical_sigma_summary.csv")
        sizes = d.groupby("attr").size().to_dict()
        chi = np.zeros(N)
        for a, k in sizes.items():
            chi += (rng.standard_normal((N, k)) ** 2).sum(1) / (2 * k)
        rows.append(dict(state=s, year=y, components=sum(sizes.values()), n_attr=len(sizes),
                         mean=chi.mean(), sd=chi.std(), P95=np.quantile(chi, .95),
                         P98=np.quantile(chi, .98), P99=np.quantile(chi, .99),
                         pct_of_5=(chi <= 5.0).mean() * 100))
out = pd.DataFrame(rows)
out.to_csv(Path(__file__).with_name("kappa_null_sim.csv"), index=False)
print(out.round(3).to_string(index=False))
print("range pct_of_5: %.1f-%.1f, mean sd %.2f" % (out.pct_of_5.min(), out.pct_of_5.max(), out.sd.mean()))
