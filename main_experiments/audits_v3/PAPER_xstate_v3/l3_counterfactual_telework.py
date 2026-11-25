"""
Prespecified COUNTERFACTUAL scenario-projection stress (section D, part 3).
[v3 port, v3 protocol 2026-09-02: paths -> audits_v3 / PAPER_xstate_v3, tag tvae_llm+ ->
semapop_vae, r*=32; algorithm unchanged]
Purpose: the retrospective realized scenarios happen to lie within the priors'
support (L3 always feasible). To exercise G3's GENERAL discriminative capacity
(forecast / policy counterfactuals), we impose a telework-surge scenario on a
prior that passes all three layers on the realized scenario, and sweep it.

Cell: TX x SemaPop-VAE+ x 2024 (a full L1/L2/L3 survivor). We take the realized
2024 TX scenario reference and move a prespecified fraction w of car-commuter
survey weight to Worked_at_home, w in {0, 0.15, 0.30, 0.45, 0.60}, keeping other
attributes fixed. For each w we re-run Layer 3 on the SAME r* realization and
report the prior mismatch (chi_H), G3 feasibility, R_3, nESS, and commute binding.
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[3])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import subprocess, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(_PA.R(""))
PY = _PA.PYTHON
AUD = Path(_PA.R("/main_experiments/audits_v3"))
# 2026-11-23 r* fix: the cells7 file is resolved from the cell (one file per cell, written by
# rerun_audit_new.build_cells7 for the TRUE r*), never hard-coded.  Two cells are run: the same two
# admitted TX-2024 priors as l3_stress_case.py. After the 2026-11-24 g1fix rerun the admitted TX-2024 priors are
# SemaPop-VAE+, TTVAE and BN with baseline R_3 0.0458, 0.0462 and 0.0472 (TabDDPM, 0.0558, is no longer admitted).
# [pre-g1fix comment: 'largest baseline R_3, tied to three decimals (TTVAE 0.1723, SemaPop-VAE+ 0.1721; TabDDPM 0.1715)'.]
# usage: python <this> --model {semapop_vae,ttvae}
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
REF0 = ROOT / "data/pums_states/TX/reference_2024_TX.csv"
L3ATTRS = ["Age","Household_Size","Number_of_children","Number_of_cars_of_household",
           "Employment_status","Household_Income_Class","Commute_Mode"]
WS = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
OUT = AUD / "PAPER_xstate_v3/l3_counterfactual" / CELL
shutil.rmtree(OUT, ignore_errors=True)   # never re-read a stale layer3_summary.json
OUT.mkdir(parents=True, exist_ok=True)
CAR, WFH = "Car_truck_or_van", "Worked_at_home"

ref0 = pd.read_csv(REF0)
w_col = "PWGTP"
car_mask = ref0["Commute_Mode"] == CAR
base_wfh = ref0.loc[ref0["Commute_Mode"]==WFH, w_col].sum() / ref0[w_col].sum()
print(f"baseline realized WFH share (2024 TX) = {base_wfh:.3f}; car share = {ref0.loc[car_mask,w_col].sum()/ref0[w_col].sum():.3f}", flush=True)

rows = []
for w in WS:
    ref = ref0.copy()
    if w > 0:
        car_idx = ref.index[car_mask]
        cw = ref.loc[car_idx, w_col].to_numpy(float)
        target_move = w * cw.sum()
        order = np.argsort(-cw)  # move heaviest-weight car rows first (deterministic)
        cum = np.cumsum(cw[order]); k = int(np.searchsorted(cum, target_move)) + 1
        move_idx = car_idx[order[:k]]
        ref.loc[move_idx, "Commute_Mode"] = WFH
    cf_ref = OUT / f"ref_w{int(w*100)}.csv"; ref.to_csv(cf_ref, index=False)
    wfh_share = ref.loc[ref["Commute_Mode"]==WFH, w_col].sum()/ref[w_col].sum()
    od = OUT / f"w{int(w*100)}"
    subprocess.run([PY, "-u", str(_PA.R("/audit/layer3_projection_uncertainty_branches.py")),
        "--realizations_csv", str(CELLS7), "--scenario_reference_csv", str(cf_ref),
        "--target_attrs", *L3ATTRS, "--base_weight_col", "base_count",
        "--reference_weight_col", w_col, "--output_dir", str(od),
        "--projection_mode", "uncertainty_set", "--uncertainty_aggregation", "attribute_constraints",
        "--uncertainty_tau", "1.0", "--solver", "SCALABLE", "--save_projected_weights", "--sigma_mode", "scenario_uncertainty", "--scenario_target_sigma_json", str(SIGMA_JSON), "--sigma_floor", "0.005"],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    j = json.loads((od/"layer3_summary.json").read_text()) if (od/"layer3_summary.json").exists() else {}
    feas=float(j.get("feasibility_rate",0.0)); KL=j.get("mean_distortion"); ness=j.get("mean_agent_nESS")
    chi_prior=j.get("mean_gaussian_loss_prior")
    commute=None
    prf=od/"layer3_per_realization_diagnostics.csv"
    if prf.exists():
        ap=pd.read_csv(prf).iloc[0].get("attribute_projected_losses_json")
        if isinstance(ap,str) and ap not in ("","nan"): commute=json.loads(ap).get("Commute_Mode")
    R3=float(np.sqrt(KL/2)) if (KL is not None and feas>=0.999) else None
    rows.append({"w_shift":w,"WFH_target":round(wfh_share,3),"chiH_rstar_sigmaH":round(float(chi_prior),3) if chi_prior is not None else None,  # chi_H of the r* realization under sigma(H) (lower bound on the ensemble mean)
                 "G3_feasible":feas>=0.999,"R_3":round(R3,4) if R3 else None,
                 "nESS":round(float(ness),4) if ness is not None else None,
                 "commute_chi_proj":round(commute,3) if commute is not None else None})
    print(f"w={w:.2f} WFH_target={wfh_share:.3f} L3_gauss_loss_prior={chi_prior} G3_feasible={feas>=0.999} "
          f"R3={R3 if R3 else 'INFEAS'} nESS={ness}", flush=True)

df=pd.DataFrame(rows); df.to_csv(OUT/"l3_counterfactual_curve.csv", index=False)
print("\nSaved ->", OUT/"l3_counterfactual_curve.csv"); print(df.to_string(index=False))
