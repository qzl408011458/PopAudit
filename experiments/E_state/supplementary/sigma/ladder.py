"""Fine one-at-a-time ladders of floor / mu / lambda (Layer-2 admission only, exact fast path):
the smallest parameter value at which a 2021 case first passes Layer 2, and admission counts by year.
Layer 3 never rejected a Layer-2-admitted case in any of the ten full variants, so Layer-2 admission
equals VALID there.  Output: results/sigma_ladder.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import t as student_t
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sigma_sens import SRC, cases
from l2_fast import variant_sigma

data = []
for m, st, y in cases():
    src = Path(SRC) / f"layer2_{m}_{st}_{y}"
    data.append((m, st, int(y), pd.read_csv(src / "layer2_historical_sigma_summary.csv"),
                 pd.read_csv(src / "layer2_per_realization_weights.csv")))


def admit(p):
    out = []
    for m, st, y, sig, w in data:
        s, _ = variant_sigma(sig, p)
        loss = np.zeros(len(w))
        for a, idx in sig.groupby("attr", sort=False).groups.items():
            z2 = sum(((w[sig.target[i]].to_numpy(float) - sig.target_m[i]) / s[i]) ** 2 for i in idx)
            loss += 0.5 * z2 / len(idx)
        K = len(loss); eta2 = student_t.ppf(0.95, K - 1) * loss.std(ddof=1) / np.sqrt(K)
        out.append((m, st, y, loss.mean() <= 5.0 + eta2))
    return pd.DataFrame(out, columns=["model", "state", "year", "adm"])


base = dict(floor=0.005, mu=1.0, lam=1.0, beta=1.0, H=None)
rows = []
for par, grid in (("floor", np.round(np.arange(0.0025, 0.01001, 0.00025), 5)),
                  ("mu", np.round(np.arange(0.5, 2.001, 0.05), 3)),
                  ("lam", np.round(np.arange(0.5, 2.001, 0.05), 3))):
    for g in grid:
        p = dict(base); p[par] = float(g); a = admit(p)
        rows.append(dict(param=par, value=g, n_adm=int(a.adm.sum()),
                         **{f"n_{y}": int(a[a.year == y].adm.sum()) for y in (2018, 2021, 2024)},
                         adm_2021=";".join(f"{r.model}_{r.state}" for r in a[(a.year == 2021) & a.adm].itertuples())))
L = pd.DataFrame(rows); L.to_csv(HERE / "results/sigma_ladder.csv", index=False)
for par in ("floor", "mu", "lam"):
    l = L[L.param == par]; f = l[l.n_2021 > 0]
    print(par, "first 2021 admission at", None if f.empty else f.value.iloc[0], "->", None if f.empty else f.adm_2021.iloc[0])
pd.set_option("display.width", 200); print(L.drop(columns="adm_2021").to_string(index=False))
