"""BN backend comparison (diagnostic, 2026-11-23).

Question: does the SemaPop repository's latest BN (train_bn_pgmpy.py @ e2b27f1, pgmpy HillClimbSearch,
BDeu ess 66.6383, max in-degree 4, MLE CPTs, empirical decoding) change the audit results
relative to the audited BN row (train_bn_v2.py, numpy hill climbing, BIC, <=3 parents,
Laplace CPTs, empirical decoding)?

Variants (finalize dirs under hpo_v3/final/, audit dirs under audits_v3/):
  bn            the audited row (unchanged artifacts)
  bn_pgmpy      option B: the SemaPop repository's reported configuration
  bn_pgmpy_bic  decomposition: the same pgmpy trainer at its literature defaults (BIC, 3, MLE)

NOTE (release): since 2026-11-23 run_audit_v3.py maps the audit tag ``bn`` to the study
bn_pgmpy_<ST> (pgmpy, BDeu ess 66.6383, max in-degree 4), so the ``bn`` and ``bn_pgmpy``
audit folders and certificate rows are identical; the train_bn_v2.py row described above is
no longer audited (only the ``bn`` rows of fidelity.csv still read its finalize records,
which are not shipped).  This is a diagnostic; the paper does not report it.

Nothing in PAPER_xstate_v3/ is modified.  The gate logic is decision_certificate_xstate_v3
imported unchanged; only its model list and output directory are redirected here.
Outputs (this directory):
  xstate_certificate_all.csv / xstate_valid_only.csv / xstate_model_by_gate.csv  (3 BN rows)
  fidelity.csv            reference SRMSE-1/2 and block F1 per variant x state (3 seeds)
  cell_compare.csv        per (state, year): gate stats of bn vs each variant side by side
  swap_headline.csv       VALID totals when bn is replaced by each variant in the audited nine
  swap_budget.csv         Table-4 scenario decisions (theta_dagger, D_dagger, flat) after the swap
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[4])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(_PA.R(""))
E = ROOT / "experiments/E_state"
BASE = E / "audits_v3"
V3 = BASE / "PAPER_xstate_v3"
OUT = BASE / "bn_pgmpy_compare"
FINAL = Path(_PA.S("/hpo_v3/final"))
VARIANTS = ["bn", "bn_pgmpy", "bn_pgmpy_bic"]
STATES = ["CA", "TX", "OH"]
YEARS = [2018, 2021, 2024]
B0 = 0.30

sys.argv = [sys.argv[0]]                      # make sure the module does not see --extras
sys.path.insert(0, str(E))
import decision_certificate_xstate_v3 as DC   # noqa: E402

# ---- 0. completeness: every variant x state has L1, and every (state, year) has L2+L3 ----
missing = []
for v in VARIANTS:
    for st in STATES:
        if not (BASE / f"layer1_paper_{v}_{st}" / "layer1_ensemble_summary.json").exists():
            missing.append(f"L1 {v} {st}")
        for y in YEARS:
            if not (BASE / f"layer2_{v}_{st}_{y}" / "layer2_summary.json").exists():
                missing.append(f"L2 {v} {st} {y}")
            if not (BASE / f"layer3_paperW_{v}_{st}_{y}" / "layer3_summary.json").exists():
                missing.append(f"L3 {v} {st} {y}")
assert not missing, f"incomplete audit outputs, refusing to compare: {missing}"

# ---- 1. certificate rows for the three BN variants (paper gate logic, unchanged) ----
DC.STATE_MODELS = {st: list(VARIANTS) for st in STATES}
DC.MODELS = list(VARIANTS)
DC.OUT_DIR = OUT
DC.main()
cert = pd.read_csv(OUT / "xstate_certificate_all.csv")
cert = cert[cert["year"].notna()].copy()
cert["year"] = cert["year"].astype(int)
assert all((cert.model == v).sum() == 9 for v in VARIANTS), cert.model.value_counts()

# sanity: the bn rows recomputed here must equal the paper's certificate rows
paper = pd.read_csv(V3 / "xstate_certificate_all.csv")
paper = paper[paper["year"].notna()].copy(); paper["year"] = paper["year"].astype(int)
cols = ["L1_marg", "L1_dep", "L1_support", "L1_invalid", "L2_chi_H", "L3_KL", "L3_nESS", "VALID", "first_fail"]
a = cert[cert.model == "bn"].set_index(["state", "year"])[cols].sort_index()
b = paper[paper.model == "bn"].set_index(["state", "year"])[cols].sort_index()
same = a.fillna("NA").astype(str).equals(b.fillna("NA").astype(str))
print(f"\n[check] recomputed bn rows identical to PAPER_xstate_v3: {same}")
if not same:
    print(pd.concat({"here": a, "paper": b}, axis=1).to_string())
assert same, "recomputed bn rows differ from the paper certificate"

# ---- 2. fidelity (finalize aggregate, 3 seeds) ----
fid = []
for v in VARIANTS:
    for st in STATES:
        p = FINAL / f"{v}_{st}" / "aggregate.json"
        if not p.exists():
            fid.append({"variant": v, "state": st}); continue
        agg = json.load(open(p))["aggregate"]
        ref = agg.get("reference", {})
        row = {"variant": v, "state": st}
        for k in ("srmse_1", "srmse_2", "block_f1"):
            if k in ref:
                row[f"ref_{k}_mean"] = ref[k].get("mean"); row[f"ref_{k}_sd"] = ref[k].get("sd")
        m = json.load(open(FINAL / f"{v}_{st}" / f"{v}_{st}_final_s42" / "metrics.json"))
        row["n_edges"] = (m.get("structure", {}) or {}).get("n_edges", m.get("n_edges"))
        fid.append(row)
fid = pd.DataFrame(fid)
fid.to_csv(OUT / "fidelity.csv", index=False)

# ---- 3. post-projection residual D = d_A(Q*) for VALID cells (make_figE_postproj rule) ----
# FIX 3 (2026-11-24): D on the union of Q* and reference categories, shared helper PAPER_xstate_v3/dA_union.py
sys.path.insert(0, str(V3))
from dA_union import D_union  # noqa: E402
def d_post(model, st, y):
    return float(D_union(model, st, y, BASE))

cert["D_post"] = [d_post(r.model, r.state, r.year) if r.VALID else np.nan
                  for r in cert.itertuples()]
cert.to_csv(OUT / "xstate_certificate_all_with_D.csv", index=False)

# ---- 4. side-by-side per cell ----
keep = ["L1_marg", "L1_dep", "L1_support", "L1_invalid", "L2_chi_H", "L3_KL", "L3_nESS",
        "first_fail", "VALID", "D_post"]
wide = cert.pivot_table(index=["state", "year"], columns="model", values=keep, aggfunc="first",
                        dropna=False)
wide = wide.reindex(columns=VARIANTS, level=1)
wide.columns = [f"{c}__{m}" for c, m in wide.columns]
wide.reset_index().to_csv(OUT / "cell_compare.csv", index=False)

# ---- 5. swap bn -> variant inside the audited nine ----
paper_valid = paper[paper.VALID == True].copy()
paper_valid["D_post"] = [d_post(r.model, r.state, r.year) for r in paper_valid.itertuples()]
heads, budgets = [], []
for v in VARIANTS:
    vrows = cert[cert.model == v]
    others = paper[paper.model != "bn"]
    allrows = pd.concat([others, vrows.drop(columns=["D_post"])], ignore_index=True)
    n_valid = int(allrows["VALID"].astype(bool).sum())
    by_year = {y: int(allrows[(allrows.year == y)]["VALID"].astype(bool).sum()) for y in YEARS}
    ff = allrows[~allrows["VALID"].astype(bool)]["first_fail"].fillna("").map(lambda s: s.split(":")[0])
    heads.append({"bn_variant": v, "VALID_total": n_valid, **{f"VALID_{y}": by_year[y] for y in YEARS},
                  "L1_reject": int((ff == "L1").sum()), "L2_reject": int((ff == "L2").sum()),
                  "L3_reject": int((ff == "L3").sum()),
                  "bn_valid_cells": ";".join(f"{r.state}{r.year}" for r in vrows[vrows.VALID == True].itertuples())})
    val = pd.concat([paper_valid[paper_valid.model != "bn"][["state", "model", "year", "D_post"]],
                     vrows[vrows.VALID == True][["state", "model", "year", "D_post"]]], ignore_index=True)
    for y in YEARS:
        for st in ("CA", "OH", "TX"):
            g = val[(val.state == st) & (val.year == y)].sort_values("D_post")
            if len(g) == 0:
                budgets.append({"bn_variant": v, "state": st, "year": y, "n_adm": 0,
                                "theta_dagger": "", "D_dagger": np.nan, "flat": "Defer"})
            else:
                D = float(g.D_post.iloc[0])
                budgets.append({"bn_variant": v, "state": st, "year": y, "n_adm": len(g),
                                "theta_dagger": g.model.iloc[0], "D_dagger": round(D, 4),
                                "flat": "Deploy" if D <= B0 else "Defer"})
heads = pd.DataFrame(heads); budgets = pd.DataFrame(budgets)
heads.to_csv(OUT / "swap_headline.csv", index=False)
budgets.to_csv(OUT / "swap_budget.csv", index=False)

# the bn-variant row of the swap must reproduce the paper's Table 4
orig = pd.read_csv(V3 / "exp_ii_scenario_budget_post.csv")
chk = budgets[budgets.bn_variant == "bn"].merge(orig, on=["state", "year"], suffixes=("", "_paper"))
ok = all((r.n_adm == r.n_adm_paper) and ((r.n_adm == 0) or (r.theta_dagger == r.theta_dagger_paper
         and abs(r.D_dagger - r.D_dagger_paper) < 1e-3)) for r in chk.itertuples())
print(f"[check] bn swap row reproduces exp_ii_scenario_budget_post.csv: {ok}")
assert len(chk) == 9 and ok, "bn swap row does not reproduce the paper's Table 4"

pd.set_option("display.width", 250)
print("\nFIDELITY (reference, 3 seeds)\n", fid.round(4).to_string(index=False))
print("\nHEADLINE after swapping bn\n", heads.to_string(index=False))
piv = budgets.pivot_table(index=["year", "state"], columns="bn_variant",
                          values=["n_adm", "theta_dagger", "D_dagger", "flat"], aggfunc="first")
print("\nTABLE-4 decisions after swapping bn\n", piv.to_string())
