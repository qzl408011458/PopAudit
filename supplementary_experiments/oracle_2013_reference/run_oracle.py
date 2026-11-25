"""Oracle reference: the OBSERVED 2013 base-year sample audited in place of a prior (Supplementary S3.7).

For each scenario (CA/OH/TX x 2018/2021/2024) the observed 2013 sample is fed to the production
Layer-2 and Layer-3 scripts as an ensemble of ONE realization, with exactly the production
invocation of rerun_audit_new.do_year / do_l3 (same reference CSV, PWGTP reference weights,
historical_dynamics sigma(H) from data/pums_states/<ST>/historical, attribute-normalized loss,
uncertainty_set projection, tau = 1, SCALABLE solver, sigma floor 0.005, sigma(H) JSON from the
Layer-2 output).  Reads the main-experiment inputs only; all outputs go under this folder.

Oracle samples (variant -> source):
  full : data/pums_states/<ST>/person_level_2013_<ST>.csv  (the PWGTP-weighted resample of the
         2013 PUMS that train/val/test were split from; rows are equally weighted)
  test : data/pums_states/<ST>/testdata_pers.csv           (the Layer-1 reference split)
Only the 12 attribute columns of a generated realization are written, so no weight column of
the source can leak into the realization statistics.
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[2])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import os, sys, json, shutil, subprocess, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

ROOT = Path(_PA.R(""))
PY = _PA.PYTHON
OUT = Path(_PA.R("/supplementary_experiments/oracle_2013_reference"))
sys.path.insert(0, _PA.R("/main_experiments"))
from rerun_audit_new import L2_ATTRS, L3_ATTRS, AGE_BINS, AGE_LABELS, CELL_KEYS  # noqa: E402

STATES = ["CA", "OH", "TX"]
YEARS = ["2018", "2021", "2024"]
GEN_COLS = ["Age", "Household_Size", "Number_of_children", "Number_of_cars_of_household", "Gender",
            "Marital_status", "Employment_status", "Studenthood_status", "Household_Type",
            "Commute_Mode", "Personal_Income_Class", "Household_Income_Class"]
SRC = {"full": "person_level_2013_{st}.csv", "test": "testdata_pers.csv"}


def run(cmd, logpath):
    with open(logpath, "w") as lg:
        return subprocess.run(cmd, cwd=ROOT, stdout=lg, stderr=subprocess.STDOUT).returncode


def make_ens(var, st):
    d = OUT / "ens" / f"{var}_{st}"
    f = d / "oracle_2013_0.csv"
    if not f.exists():
        d.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(ROOT / f"data/pums_states/{st}" / SRC[var].format(st=st), usecols=GEN_COLS)
        df[GEN_COLS].to_csv(f, index=False)
    return d


def do_case(var, st, y):
    state_dir = ROOT / f"data/pums_states/{st}"
    ref = state_dir / f"reference_{y}_{st}.csv"
    ens = OUT / "ens" / f"{var}_{st}"
    tag = f"{var}_{st}_{y}"
    l2 = OUT / "runs" / f"layer2_{tag}"
    l3 = OUT / "runs" / f"layer3_{tag}"
    shutil.rmtree(l2, ignore_errors=True); shutil.rmtree(l3, ignore_errors=True)
    rc2 = run([PY, "-u", _PA.R("/audit/layer2_historical_sigma.py"),
               "--realizations_dir", str(ens),
               "--scenario_reference_csv", str(ref),
               "--target_attrs", *L2_ATTRS,
               "--reference_weight_col", "PWGTP",
               "--sigma_mode", "historical_dynamics",
               "--historical_reference_dir", str(state_dir / "historical"),
               "--output_dir", str(l2)], f"{l2}.relog")
    if rc2 != 0:
        return f"{tag} L2 FAILED rc={rc2}"
    # cells7 of the single realization (same grouping as rerun_audit_new.build_cells7)
    df = pd.read_csv(ens / "oracle_2013_0.csv")
    df["Age_Group"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    g = df.groupby(CELL_KEYS, observed=True).size().reset_index(name="base_count")
    g.insert(0, "realization_id", 0)
    cdir = OUT / "runs" / f"cells7_{tag}"; cdir.mkdir(parents=True, exist_ok=True)
    cells7 = cdir / "oracle_2013_0_cells7.csv"
    g.to_csv(cells7, index=False)
    sig = pd.read_csv(l2 / "layer2_historical_sigma_summary.csv")
    l3.mkdir(parents=True, exist_ok=True)
    sig_json = l3 / "sigma_H_targets.json"
    json.dump({r.target: float(r.sigma) for r in sig.itertuples()}, open(sig_json, "w"))
    rc3 = run([PY, "-u", _PA.R("/audit/layer3_projection_uncertainty_branches.py"),
               "--realizations_csv", str(cells7),
               "--scenario_reference_csv", str(ref),
               "--target_attrs", *L3_ATTRS,
               "--base_weight_col", "base_count",
               "--reference_weight_col", "PWGTP",
               "--output_dir", str(l3),
               "--projection_mode", "uncertainty_set",
               "--uncertainty_aggregation", "attribute_constraints",
               "--uncertainty_tau", "1.0",
               "--solver", "SCALABLE",
               "--save_projected_weights",
               "--sigma_mode", "scenario_uncertainty", "--scenario_target_sigma_json", str(sig_json),
               "--sigma_floor", "0.005"], f"{l3}.relog")
    if rc3 != 0:
        return f"{tag} L3 FAILED rc={rc3}"
    lv = pd.read_csv(l3 / "layer3_scenario_sigma_summary.csv").sigma_level
    if (lv != "target").any():
        return f"{tag} L3 sigma fallback on {(lv != 'target').sum()} targets"
    return f"{tag} ok"


def main():
    for var in SRC:
        for st in STATES:
            make_ens(var, st)
    (OUT / "runs").mkdir(exist_ok=True)
    tasks = [(v, s, y) for v in SRC for s in STATES for y in YEARS]
    with ThreadPoolExecutor(max_workers=9) as ex:
        futs = {ex.submit(do_case, *t): t for t in tasks}
        for fu in as_completed(futs):
            try:
                print(f"[{time.strftime('%T')}] {fu.result()}", flush=True)
            except Exception as e:
                print(f"[{time.strftime('%T')}] {futs[fu]} EXCEPTION {e!r}", flush=True)


if __name__ == "__main__":
    main()
