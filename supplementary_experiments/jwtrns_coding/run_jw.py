"""Layer 2 + Layer 3 for the JWTRNS light-rail alternatives on the CURRENT production setup.
Invocations are exactly those of production (main_experiments/rerun_audit_new.do_year / do_l3,
driven by run_audit_v3.py --l23-only): the production scripts at the repo root, the production
ensembles (run_audit_v3.PRIMARY / ens_dir), r* from rerun_audit_new.resolve_rstar_file, sigma(H)
from the shadow Layer-2 run.  Only --scenario_reference_csv and the output root change.
Cases: the 2021/2024 cases that pass Layer 1 in the production certificate (36; the 18 Layer-1-
admissible 2018 cases use JWTR references that no mapping affects).
usage: run_jw.py --variant lr2rail lr2subway lr2rail_k13 lr2subway_k13 [--repro] --jobs 12
  --repro : also re-run 4 cases with the production references (variant 'census') to check that this
            driver reproduces production outputs.
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[2])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import os, sys, json, time, shutil, argparse, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

ROOT = _PA.R("")
PY = _PA.PYTHON
HERE = Path(__file__).resolve().parent
sys.path.insert(0, _PA.R("/main_experiments")); sys.path.insert(0, ROOT)
import rerun_audit_new as R
from run_audit_v3 import PRIMARY, ens_dir

PROD = Path(_PA.R("/main_experiments/audits_v3"))
REPRO = [("semapop_vae", "CA", "2024"), ("ttvae", "OH", "2021"), ("semapop_vae", "TX", "2024"), ("bn", "OH", "2024")]


def cases():
    cert = pd.read_csv(PROD / "PAPER_xstate_v3/xstate_certificate_all.csv")
    c = cert[(cert.L1_accept == True) & cert.year.isin([2021, 2024])]
    return [(r.model, r.state, str(int(r.year))) for r in c.itertuples()]


def vdir(v): return HERE / "runs" / v


def ref(v, st, y):
    if v == "census":
        return f"{ROOT}/data/pums_states/{st}/reference_{y}_{st}.csv"
    return str(HERE / "refs" / v / st / f"reference_{y}_{st}.csv")


def run(cmd, log):
    with open(log, "w") as lg:
        return subprocess.run(cmd, cwd=ROOT, stdout=lg, stderr=subprocess.STDOUT).returncode


def do_l2(v, m, st, y):
    sd = f"{ROOT}/data/pums_states/{st}"
    out = vdir(v) / f"layer2_{m}_{st}_{y}"
    shutil.rmtree(out, ignore_errors=True)
    ens = ens_dir(PRIMARY[m].format(st=st))
    cmd = [PY, "-u", "layer2_historical_sigma.py",
           "--realizations_dir", f"{ens}/generated_ensembles",
           "--scenario_reference_csv", ref(v, st, y),
           "--target_attrs", *R.L2_ATTRS,
           "--reference_weight_col", "PWGTP",
           "--sigma_mode", "historical_dynamics",
           "--historical_reference_dir", f"{sd}/historical",
           "--output_dir", str(out)]
    if run(cmd, f"{out}.relog") != 0:
        raise RuntimeError(f"L2 {v} {m} {st} {y} failed")
    return f"L2 {v} {m} {st} {y} ok"


def do_l3(v, m, st, y):
    l2 = vdir(v) / f"layer2_{m}_{st}_{y}"
    l3 = vdir(v) / f"layer3_paperW_{m}_{st}_{y}"
    shutil.rmtree(l3, ignore_errors=True); os.makedirs(l3)
    ens = ens_dir(PRIMARY[m].format(st=st))
    rstar, rcsv = R.resolve_rstar_file(str(l2), f"{ens}/generated_ensembles")
    df = pd.read_csv(rcsv)
    df["Age_Group"] = pd.cut(df["Age"], bins=R.AGE_BINS, labels=R.AGE_LABELS, right=False)
    g = df.groupby(R.CELL_KEYS, observed=True).size().reset_index(name="base_count")
    g.insert(0, "realization_id", rstar)
    c7 = l3 / f"{rcsv.stem}_cells7.csv"
    g.to_csv(c7, index=False)
    sig = pd.read_csv(l2 / "layer2_historical_sigma_summary.csv")
    sj = l3 / "sigma_H_targets.json"
    json.dump({r.target: float(r.sigma) for r in sig.itertuples()}, open(sj, "w"))
    rc = run([PY, "-u", f"{ROOT}/layer3_projection_uncertainty_branches.py",
              "--realizations_csv", str(c7),
              "--scenario_reference_csv", ref(v, st, y),
              "--target_attrs", *R.L3_ATTRS,
              "--base_weight_col", "base_count",
              "--reference_weight_col", "PWGTP",
              "--output_dir", str(l3),
              "--projection_mode", "uncertainty_set",
              "--uncertainty_aggregation", "attribute_constraints",
              "--uncertainty_tau", "1.0",
              "--solver", "SCALABLE",
              "--save_projected_weights",
              "--sigma_mode", "scenario_uncertainty", "--scenario_target_sigma_json", str(sj),
              "--sigma_floor", "0.005"], f"{l3}.relog")
    c7.unlink()
    if rc != 0:
        return f"L3 {v} {m} {st} {y} FAILED rc={rc}"
    if (pd.read_csv(l3 / "layer3_scenario_sigma_summary.csv").sigma_level != "target").any():
        raise RuntimeError(f"L3 {v} {m} {st} {y}: default-sigma fallback")
    return f"L3 {v} {m} {st} {y} ok r*={rstar}"


def do_case(v, m, st, y):
    a = do_l2(v, m, st, y)
    return a + "; " + do_l3(v, m, st, y)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", nargs="+", default=[])
    ap.add_argument("--repro", action="store_true")
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    tasks = [(v, *c) for v in a.variant for c in cases()]
    if a.repro:
        tasks += [("census", *c) for c in REPRO]
    for v in set(t[0] for t in tasks):
        vdir(v).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[{time.strftime('%T')}] {len(tasks)} tasks, jobs={a.jobs}", flush=True)
    with ThreadPoolExecutor(max_workers=min(a.jobs, 12)) as ex:
        futs = {ex.submit(do_case, *t): t for t in tasks}
        for fu in as_completed(futs):
            try:
                print(f"[{time.strftime('%T')}] {fu.result()}", flush=True)
            except Exception as e:
                print(f"[{time.strftime('%T')}] {futs[fu]} EXCEPTION {e!r}", flush=True)
    print(f"[{time.strftime('%T')}] done in {time.time() - t0:.0f}s", flush=True)
