"""Supplementary item (tau): D(tau), nESS(tau), R_3(tau), feasibility on the 2024 valid cases.
Shadow copy of PAPER_xstate_v3/l3_stress_case.py with (i) parametrized state/model/taus,
(ii) outputs under supplementary/tau/runs/<cell>/tau_<t>, (iii) D is computed in collect.py with the shared
union helper PAPER_xstate_v3/dA_union.py (FIX 3, 2026-11-24), exactly as Table 4; before FIX 3 it was
(make_figE_postproj.py: D = 0.5 * sum projected_abs_err_mean of layer3_target_summary.csv).
Layer-3 call identical to rerun_audit_new.do_l3 except --uncertainty_tau.
usage: python tau_ladder.py --model M --state ST --taus 1.0 0.5 ...
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[4])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import subprocess, json, argparse
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(_PA.R(""))
PY = _PA.PYTHON
AUD = ROOT / "experiments/E_state/audits_v3"
OUTROOT = Path(_PA.R("/experiments/E_state/supplementary/tau/runs"))
L3ATTRS = ["Age","Household_Size","Number_of_children","Number_of_cars_of_household",
           "Employment_status","Household_Income_Class","Commute_Mode"]
ap = argparse.ArgumentParser(); ap.add_argument("--model"); ap.add_argument("--state"); ap.add_argument("--year", default="2024")
ap.add_argument("--taus", nargs="+", type=float); a = ap.parse_args()
CELL = f"{a.model}_{a.state}_{a.year}"
c7 = list((AUD/"layer3_rstar_inputs_cells7_v2"/CELL).glob("*_cells7.csv")); assert len(c7) == 1, c7
SIG = AUD/f"layer3_paperW_{CELL}"/"sigma_H_targets.json"
REF = ROOT/f"data/pums_states/{a.state}/reference_{a.year}_{a.state}.csv"
OUT = OUTROOT/CELL; OUT.mkdir(parents=True, exist_ok=True)
for tau in a.taus:
    od = OUT/f"tau_{tau}"
    if (od/"layer3_summary.json").exists(): print("skip", od); continue
    rc = subprocess.run([PY, "-u", str(ROOT/"layer3_projection_uncertainty_branches.py"),
        "--realizations_csv", str(c7[0]), "--scenario_reference_csv", str(REF),
        "--target_attrs", *L3ATTRS, "--base_weight_col", "base_count", "--reference_weight_col", "PWGTP",
        "--output_dir", str(od), "--projection_mode", "uncertainty_set", "--uncertainty_aggregation", "attribute_constraints",
        "--uncertainty_tau", str(tau), "--solver", "SCALABLE", "--save_projected_weights",
        "--sigma_mode", "scenario_uncertainty", "--scenario_target_sigma_json", str(SIG), "--sigma_floor", "0.005"],
        stdout=open(str(od)+".log","w"), stderr=subprocess.STDOUT).returncode
    print(CELL, tau, "rc", rc, flush=True)
