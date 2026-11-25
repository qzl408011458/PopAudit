"""
Prespecified Layer-3 stress case (paper section D).  [v3 port, v3 protocol 2026-09-02:
paths -> audits_v3 / PAPER_xstate_v3, tag tvae_llm+ -> semapop_vae, r*=32; algorithm unchanged]
Cells = TX x SemaPop-VAE+ x 2024 and TX x TTVAE x 2024, two of the three priors (with BN) valid in all three 2024 scenarios; after the 2026-11-24 g1fix rerun their baseline R_3 is 0.046 and 0.046 (0.0458, 0.0462; the smallest among the admitted TX-2024 priors, BN 0.047; TabDDPM, 0.056, is no longer admitted in TX 2024).
[pre-g1fix comment: valid in both 2024 scenarios; R_3 0.044 and 0.046, BN 0.047, TabDDPM 0.056.]
Hold the model + scenario FIXED (so L1, L2 verdicts are unchanged, both PASS), and
tighten the projection tolerance uniformly along a prespecified ladder tau in
{1.0, 0.75, 0.5, 0.3, 0.2, 0.1}. Report feasibility (G3), R_3=sqrt(KL/2), nESS, and
the first attribute to become infeasible. Demonstrates G3's independent boundary.
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[4])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import subprocess, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(_PA.R(""))
PY = _PA.PYTHON
AUD = ROOT / "experiments/E_state/audits_v3"
# 2026-11-23 r* fix: the cells7 file is resolved from the cell (one file per cell, written by
# rerun_audit_new.build_cells7 for the TRUE r*), never hard-coded.  Two cells are run: SemaPop-VAE+
# and TTVAE in TX 2024 (see the module docstring for why).  usage: python <this> --model {semapop_vae,ttvae}
import argparse, glob, shutil
_ap = argparse.ArgumentParser(); _ap.add_argument("--model", default="semapop_vae")
MODEL = _ap.parse_args().model
CELL = f"{MODEL}_TX_2024"
_c = glob.glob(str(AUD / "layer3_rstar_inputs_cells7_v2" / CELL / "*_cells7.csv"))
assert len(_c) == 1, f"expected exactly one cells7 file for {CELL}, found {_c}"
CELLS7 = Path(_c[0])
# 2026-11-23: the projection uses sigma_{a,c}(H) of the cell (written by rerun_audit_new.do_l3),
# as in the paper's feasible set; the counterfactual keeps the realized-scenario sigma(H)
# (only its small sampling-error term would move with the shifted targets).
SIGMA_JSON = AUD / f"layer3_paperW_{CELL}" / "sigma_H_targets.json"
REF = ROOT / "data/pums_states/TX/reference_2024_TX.csv"
L3ATTRS = ["Age","Household_Size","Number_of_children","Number_of_cars_of_household",
           "Employment_status","Household_Income_Class","Commute_Mode"]
TAUS = [1.0, 0.75, 0.5, 0.3, 0.2, 0.1]
OUT = AUD / "PAPER_xstate_v3/l3_stress" / CELL
shutil.rmtree(OUT, ignore_errors=True)   # never re-read a stale layer3_summary.json
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for tau in TAUS:
    od = OUT / f"tau_{tau}"
    rc = subprocess.run([PY, "-u", str(ROOT/"layer3_projection_uncertainty_branches.py"),
        "--realizations_csv", str(CELLS7), "--scenario_reference_csv", str(REF),
        "--target_attrs", *L3ATTRS, "--base_weight_col", "base_count",
        "--reference_weight_col", "PWGTP", "--output_dir", str(od),
        "--projection_mode", "uncertainty_set", "--uncertainty_aggregation", "attribute_constraints",
        "--uncertainty_tau", str(tau), "--solver", "SCALABLE", "--save_projected_weights",
        "--sigma_mode", "scenario_uncertainty", "--scenario_target_sigma_json", str(SIGMA_JSON), "--sigma_floor", "0.005"],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT).returncode
    j = json.loads((od/"layer3_summary.json").read_text()) if (od/"layer3_summary.json").exists() else {}
    feas = float(j.get("feasibility_rate", 0.0))
    KL = j.get("mean_distortion"); ness = j.get("mean_agent_nESS")
    # first infeasible / binding attribute from per-realization diagnostics
    binding = ""
    prf = od/"layer3_per_realization_diagnostics.csv"
    if prf.exists():
        pr = pd.read_csv(prf)
        ap = pr.iloc[0].get("attribute_projected_losses_json")
        if isinstance(ap, str) and ap not in ("","nan"):
            a = json.loads(ap)
            binding = max(a, key=a.get) + f"={a[max(a,key=a.get)]:.3f}"
    R3 = float(np.sqrt(KL/2)) if (KL is not None and feas>=0.999) else None
    rows.append({"tau": tau, "feasible": feas>=0.999, "feasibility_rate": round(feas,3),
                 "R_3": round(R3,4) if R3 is not None else None,
                 "KL": round(float(KL),5) if KL is not None else None,
                 "nESS": round(float(ness),4) if ness is not None else None,
                 "binding_attr": binding, "rc": rc})
    print(f"tau={tau}: feasible={feas>=0.999} feas_rate={feas:.3f} "
          f"R3={R3 if R3 else 'n/a'} nESS={ness} binding={binding}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT/"l3_stress_curve.csv", index=False)
print("\nSaved ->", OUT/"l3_stress_curve.csv")
print(df.to_string(index=False))
