"""Seed-robustness summary (rewritten 2026-11-24 for the g1fix rerun; old version in
a backup outside this release).
Reads results/seed_certificate_cases.csv + seed_scenario_decisions.csv (cert_seed.py) and the
production 81-case certificate (audits_v3/PAPER_xstate_v3/xstate_certificate_all.csv).
Writes results/seed_gate_ranges.csv, seed_valid81.csv, seed_s42_vs_production.csv and prints:
coverage, admission flips vs seed 42, VALID by seed (regenerated cases and 81-case counts with the
non-regenerated cases carried forward at their seed-42 outcome), near-threshold cases, decisions."""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[4])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import os
import pandas as pd, numpy as np
from pathlib import Path
H = Path(os.environ.get("SEED_OUT", Path(__file__).resolve().parent / "results"))
E = _PA.R("/experiments/E_state")
d = pd.read_csv(H / "seed_certificate_cases.csv")
prod = pd.read_csv(f"{E}/audits_v3/PAPER_xstate_v3/xstate_certificate_all.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
k = ["state", "model", "year"]

# --- seed 42 rows must equal the production certificate -----------------------------------
s42 = d[d.seed == 42].merge(prod[k + ["VALID", "L2_chi_H", "L2_cutoff", "L1_invalid", "first_fail"]],
                            on=k, suffixes=("", "_prod"))
s42["chi_diff"] = (s42.chi_H - s42.L2_chi_H).abs()
s42["valid_same"] = s42.VALID == s42.VALID_prod
s42.to_csv(H / "seed_s42_vs_production.csv", index=False)
print(f"seed-42 vs production certificate: {len(s42)} cases, VALID agree {int(s42.valid_same.sum())}, "
      f"max |chi_H diff| {s42.chi_diff.max():.2e}, max |L1_invalid diff| {(s42.L1_invalid - s42.L1_invalid_prod).abs().max():.1e}")

w = d.pivot_table(index=k, columns="seed", values="VALID", aggfunc="first")
have = w.dropna()
print(f"\ncases with all three seeds: {len(have)} of {len(w)}  ->  {2 * len(have)} seed comparisons")
missing = w[w.isna().any(axis=1)]
if len(missing):
    print("cases missing a seed:"); print(missing)
flips = have[(have[142] != have[42]) | (have[242] != have[42])]
print("\nADMISSION FLIPS vs seed 42:"); print(flips if len(flips) else "  none")
nflip = int((have[142] != have[42]).sum() + (have[242] != have[42]).sum())
print(f"flipped seed comparisons: {nflip} of {2 * len(have)}")
for s in (42, 142, 242):
    sub = have[s].astype(bool)
    print(f"seed {s}: VALID {int(sub.sum())}/{len(sub)} regenerated | by year",
          {y: int(sub.xs(y, level='year').sum()) for y in (2018, 2021, 2024)})

# --- 81-case counts: production seed-42 outcome, replaced by the seed's outcome where regenerated
rows = []
for s in (42, 142, 242):
    v = prod.set_index(k)["VALID"].astype(bool).copy()
    ds = d[d.seed == s].set_index(k)["VALID"].astype(bool)
    v.loc[ds.index] = ds
    r = dict(seed=s, n_regenerated=len(ds), VALID81=int(v.sum()))
    r.update({f"VALID_{y}": int(v.xs(y, level="year").sum()) for y in (2018, 2021, 2024)})
    r.update({f"VALID_{st}": int(v.xs(st, level="state").sum()) for st in ("CA", "OH", "TX")})
    r["valid_2024_set"] = ";".join(f"{m}_{st}" for (st, m, y), x in v.items() if x and y == 2024)
    rows.append(r)
v81 = pd.DataFrame(rows); v81.to_csv(H / "seed_valid81.csv", index=False)
print("\n81-case VALID by seed (non-regenerated cases carried at seed 42):"); print(v81.to_string(index=False))

g = d.groupby(k).agg(n=("seed", "size"), chi_mean=("chi_H", "mean"), chi_min=("chi_H", "min"), chi_max=("chi_H", "max"),
                     cut_min=("L2_cutoff", "min"), cut_max=("L2_cutoff", "max"),
                     inv_min=("L1_invalid", "min"), inv_max=("L1_invalid", "max"), D_min=("D", "min"), D_max=("D", "max"),
                     n_valid=("VALID", "sum")).reset_index()
g.to_csv(H / "seed_gate_ranges.csv", index=False)
# near-threshold: L1-passing cases with chi_H within 1.0 of the cutoff at some seed, or invalid mass > 0.004
dd = d.assign(chi_gap=d.chi_H - d.L2_cutoff)
near = dd[(dd.chi_gap.abs() < 1.0) | (dd.L1_invalid > 0.004)]
nt = near[k].drop_duplicates()
tab = dd.merge(nt, on=k).pivot_table(index=k, columns="seed", values=["chi_H", "L2_cutoff", "L1_invalid", "VALID"], aggfunc="first")
print("\nNear-threshold cases (|chi_H - cutoff| < 1 at some seed, or invalid mass > 0.004):"); print(tab.round(4).to_string())
print("\nPer-case ranges over seeds:"); print(g.round(4).to_string(index=False))
sc = pd.read_csv(H / "seed_scenario_decisions.csv")
print("\nScenario decisions by seed:"); print(sc.round(4).to_string(index=False))
adm = sc[sc.n_adm > 0].copy()
adm["gap_B0"] = (adm.D_dagger - 0.30).abs()
print("\nsmallest |D_dagger - B0|:"); print(adm.sort_values("gap_B0").head(5)[["seed", "state", "year", "theta_dagger", "D_dagger", "gap_B0"]].round(4).to_string(index=False))
adm["gap_runner"] = adm.D_runner - adm.D_dagger
print("\nrunner-up gaps:"); print(adm[["seed", "state", "year", "theta_dagger", "runner_up", "gap_runner"]].round(4).to_string(index=False))
dec = sc.pivot_table(index=["state", "year"], columns="seed", values="decision", aggfunc="first")
print("\ndecision by seed:"); print(dec)
print("all 9 decisions identical across seeds:", bool((dec.nunique(axis=1) == 1).all()))
