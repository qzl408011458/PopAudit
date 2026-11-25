"""
P4 audit re-run driver (reconstructed).

For each NEW cell (from experiments/hpo/cell_selection.csv), re-run the full
3-layer audit on the NEW finalized K=50 ensemble:

  L1  : layer1.py vs base-year real test  (testdata_pers.csv)          -> layer1_<m>_<s>/
  per year in {2018,2021,2024}:
    L2     : layer2_historical_sigma.py vs reference_<y>_<s>.csv        -> layer2[_pwgtp]_<m>_<s>_<y>/
    cells7 : r*=argmin scenario_loss; group r* realization (8 attrs)    -> layer3_rstar_inputs_cells7_v2/<m>_<s>_<y>/
    L3     : layer3_projection_uncertainty_branches.py                  -> layer3_paperW_<m>_<s>_<y>/

OLD cells are left untouched (their canonical audit dirs already reflect the
published ensembles). Invocations mirror run_2024_audits.sh verbatim; the
cells7 builder + L1 reference were validated to reproduce surviving outputs
exactly (r*=23 wgan_zc_OH_2024; support_cov 0.779356 wgan_llm_OH).
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[2])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import os, sys, csv, glob, json, time, argparse, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

ROOT = _PA.R("")
PY = _PA.PYTHON
AUD = f"{ROOT}/experiments/E_state/audits"
YEARS = ["2018", "2021", "2024"]

# L2 uses the SAME 7-attr transport-visible set as L1/L3 (canonical incl. Commute_Mode
# + Number_of_children); run_2024_audits.sh's 5-attr ATTRS_L2 was stale/wrong.
L2_ATTRS = ["Age", "Household_Size", "Number_of_children",
            "Number_of_cars_of_household", "Employment_status",
            "Household_Income_Class", "Commute_Mode"]
L3_ATTRS = ["Age", "Household_Size", "Number_of_children",
            "Number_of_cars_of_household", "Employment_status",
            "Household_Income_Class", "Commute_Mode"]
AGE_BINS = [0, 18, 35, 50, 65, 200]
AGE_LABELS = ["0-17", "18-34", "35-49", "50-64", "65+"]
CELL_KEYS = ["Age", "Age_Group", "Household_Size", "Number_of_children",
             "Number_of_cars_of_household", "Employment_status",
             "Household_Income_Class", "Commute_Mode"]


def l2_dir(model, state, year):
    if state == "CA" and model in ("ttvae", "tvae_llm+"):
        return f"{AUD}/layer2_pwgtp_{model}_CA_{year}"
    return f"{AUD}/layer2_{model}_{state}_{year}"


def run(cmd, logpath):
    with open(logpath, "w") as lg:
        return subprocess.run(cmd, cwd=ROOT, stdout=lg, stderr=subprocess.STDOUT).returncode


# Marginal columns of layer2_per_realization_weights.csv used to confirm that the file
# resolved for r* is the realization Layer 2 actually scored.
_CHECK_PREFIX = ("Household_Size==", "Commute_Mode==", "Employment_status==",
                 "Number_of_cars_of_household==", "Household_Income_Class==")


def resolve_rstar_file(l2_out, ens_dir):
    """r* = argmin scenario_loss in the Layer-2 output -> the realization FILE Layer 2 used.

    FIX 2026-11-23.  layer2_historical_sigma.load_realizations_from_dir numbers the
    realizations by their POSITION in sorted(Path(dir).glob('*.csv')), i.e. lexicographic
    order (_0, _1, _10, _11, ..., _19, _2, ...).  The old code read
    final_generated_samples_{rstar}.csv, treating that position as the file-name suffix;
    the two agree only for positions 0 and 1, so Layer 3 projected a different realization
    in 77 of the 81 audited cells.  The file is now taken from the same sorted list Layer 2
    used, and its marginal shares are checked against the Layer-2 row for r*.
    """
    from pathlib import Path
    w = pd.read_csv(f"{l2_out}/layer2_per_realization_weights.csv")
    rstar = int(w.loc[w["scenario_loss"].idxmin(), "realization_id"])
    files = sorted(Path(ens_dir).glob("*.csv"))          # == layer2 loader (default glob)
    if not (len(files) == len(w) == int(w["realization_id"].max()) + 1):
        raise RuntimeError(f"{ens_dir}: {len(files)} files vs {len(w)} Layer-2 realizations")
    rcsv = files[rstar]
    row = w.loc[w["realization_id"] == rstar].iloc[0]
    cols = [c for c in w.columns if c.startswith(_CHECK_PREFIX)]
    if not cols:
        raise RuntimeError("no marginal columns to verify the r* file against")
    d = pd.read_csv(rcsv, usecols=sorted({c.split("==")[0] for c in cols}))
    err = 0.0
    for c in cols:
        a, v = c.split("==")
        try:
            m = pd.to_numeric(d[a], errors="coerce") == float(v)
        except ValueError:
            m = d[a].astype(str) == v
        err = max(err, abs(float(m.mean()) - float(row[c])))
    if err > 1e-9:
        raise RuntimeError(f"r*={rstar}: {rcsv.name} does not reproduce the Layer-2 row "
                           f"(max share error {err:.3g}); realization mapping is broken")
    return rstar, rcsv


def build_cells7(model, state, year, ens_dir):
    """r* = argmin scenario_loss from the L2 output; group r* realization to 7-attr cells."""
    outdir = f"{AUD}/layer3_rstar_inputs_cells7_v2/{model}_{state}_{year}"
    for stale in glob.glob(f"{outdir}/*_cells7.csv"):  # never leave a previous r* input behind
        os.remove(stale)
    rstar, rcsv = resolve_rstar_file(l2_dir(model, state, year), ens_dir)
    df = pd.read_csv(rcsv)
    df["Age_Group"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    g = (df.groupby(CELL_KEYS, observed=True).size()
           .reset_index(name="base_count"))
    g.insert(0, "realization_id", rstar)
    outdir = f"{AUD}/layer3_rstar_inputs_cells7_v2/{model}_{state}_{year}"
    os.makedirs(outdir, exist_ok=True)
    for stale in glob.glob(f"{outdir}/*_cells7.csv"):  # drop prior r* file(s)
        os.remove(stale)
    # the file name carries the realization FILE's suffix (= ENSEMBLE_SEEDS index)
    out = f"{outdir}/{rcsv.stem}_cells7.csv"
    g.to_csv(out, index=False)
    return out


def do_l1(model, state, ens_dir):
    out = f"{AUD}/layer1_{model}_{state}"
    rc = run([PY, "layer1.py",
              "--real_test_data", f"{ROOT}/data/pums_states/{state}/testdata_pers.csv",
              "--realizations_dir", f"{ens_dir}/generated_ensembles",
              "--output_dir", out], f"{out}.relog")
    return f"L1 {model} {state} rc={rc}"


def do_year(model, state, year, ens_dir):
    state_dir = f"{ROOT}/data/pums_states/{state}"
    ref = f"{state_dir}/reference_{year}_{state}.csv"
    # --- L2 ---
    l2 = l2_dir(model, state, year)
    rc2 = run([PY, "-u", "layer2_historical_sigma.py",
               "--realizations_dir", f"{ens_dir}/generated_ensembles",
               "--scenario_reference_csv", ref,
               "--target_attrs", *L2_ATTRS,
               "--reference_weight_col", "PWGTP",
               "--sigma_mode", "historical_dynamics",
               "--historical_reference_dir", f"{state_dir}/historical",
               "--output_dir", l2], f"{l2}.relog")
    if rc2 != 0:
        return f"L2 {model} {state} {year} FAILED rc={rc2}"
    return f"L2 {model} {state} {year} rc2={rc2}; " + do_l3(model, state, year, ens_dir)


def do_l3(model, state, year, ens_dir):
    """cells7 (true r*) + Layer 3 only; Layer 2 must already exist for this cell."""
    state_dir = f"{ROOT}/data/pums_states/{state}"
    ref = f"{state_dir}/reference_{year}_{state}.csv"
    # --- L3 output dir is cleared first, so a failed run can never leave a stale summary ---
    l3 = f"{AUD}/layer3_paperW_{model}_{state}_{year}"
    import shutil
    shutil.rmtree(l3, ignore_errors=True)
    # --- cells7 (raises on a resolve failure) ---
    cells7 = build_cells7(model, state, year, f"{ens_dir}/generated_ensembles")
    # --- sigma_{a,c}(H) from Layer 2 (FIX 2026-11-23: the paper's feasible set uses chi_{H,a},
    # i.e. the historical scale; the Layer-3 script's default is a fixed 0.02) ---
    import json as _json
    sig = pd.read_csv(f"{l2_dir(model, state, year)}/layer2_historical_sigma_summary.csv")
    os.makedirs(l3, exist_ok=True)
    sig_json = f"{l3}/sigma_H_targets.json"
    _json.dump({r.target: float(r.sigma) for r in sig.itertuples()}, open(sig_json, "w"))
    # --- L3 ---
    rc3 = run([PY, "-u", f"{ROOT}/layer3_projection_uncertainty_branches.py",
               "--realizations_csv", cells7,
               "--scenario_reference_csv", ref,
               "--target_attrs", *L3_ATTRS,
               "--base_weight_col", "base_count",
               "--reference_weight_col", "PWGTP",
               "--output_dir", l3,
               "--projection_mode", "uncertainty_set",
               "--uncertainty_aggregation", "attribute_constraints",
               "--uncertainty_tau", "1.0",
               "--solver", "SCALABLE",
               "--save_projected_weights",
               "--sigma_mode", "scenario_uncertainty", "--scenario_target_sigma_json", sig_json,
               "--sigma_floor", "0.005"], f"{l3}.relog")
    if rc3 != 0:
        raise RuntimeError(f"L3 {model} {state} {year} failed rc3={rc3} (see {l3}.relog)")
    _lv = pd.read_csv(f"{l3}/layer3_scenario_sigma_summary.csv").sigma_level
    if (_lv != "target").any():
        raise RuntimeError(f"L3 {model} {state} {year}: {int((_lv != 'target').sum())} targets fell back to the default sigma")
    return f"L3 {model} {state} {year} rc3={rc3} ({os.path.basename(cells7)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--only", default="", help="comma list of model_state to restrict (e.g. flowsyn_OH)")
    ap.add_argument("--skip-l1", action="store_true", help="skip L1 (unchanged by L2 attr fix)")
    args = ap.parse_args()

    cells = []
    with open(f"{ROOT}/experiments/hpo/cell_selection.csv") as f:
        for r in csv.DictReader(f):
            if r["use"] != "NEW":
                continue
            tag = f"{r['model']}_{r['state']}"
            if args.only and tag not in args.only.split(","):
                continue
            cells.append((r["model"], r["state"], f"{ROOT}/{r['new_ensemble_dir']}"))

    print(f"[{time.strftime('%F %T')}] P4 audit re-run: {len(cells)} NEW cells, jobs={args.jobs}")
    tasks = []
    for m, s, ens in cells:
        if not args.skip_l1:
            tasks.append(("L1", do_l1, (m, s, ens)))
        for y in YEARS:
            tasks.append((f"Y{y}", do_year, (m, s, y, ens)))

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(fn, *a): name for name, fn, a in tasks}
        for fu in as_completed(futs):
            print(f"[{time.strftime('%T')}] {fu.result()}", flush=True)
    print(f"[{time.strftime('%F %T')}] all audit tasks done.")


if __name__ == "__main__":
    main()
