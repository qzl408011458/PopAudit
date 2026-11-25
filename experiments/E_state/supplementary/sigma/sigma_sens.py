"""Supplementary shadow analysis (2026-11-24): sensitivity of the audit to the specification of
sigma_{a,c}(H) (Eq. sigma-scale).  For each variant, re-runs Layer 2 (copied
layer2_historical_sigma.py, only the sigma CLI parameters change) and Layer 3
(copied layer3_projection_uncertainty_branches.py, variant sigma in the
constraints, tau=1, r* re-selected under the variant) for the 54
Layer-1-passing cases.  Layer 1 is reused (sigma-independent).
Invocations mirror rerun_audit_new.do_year / do_l3 verbatim except for the
sigma flags and output root.  Nothing under audits_v3/ is written.

usage: sigma_sens.py --variant base --stage l2|l3|all --jobs 12
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[4])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import os, sys, glob, json, time, shutil, argparse, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

ROOT = _PA.R("")
PY = _PA.PYTHON
HERE = Path(__file__).resolve().parent
CODE = HERE / "code"
sys.path.insert(0, str(CODE))
import rerun_audit_new as R   # shadow copy; only pure helpers (resolve_rstar_file, constants) are used

SRC = f"{ROOT}/experiments/E_state/audits_v3"
FINAL = _PA.S("/hpo_v3/final")
PRIMARY = {"semapop_vae": "semapop_vae_{st}_train", "ttvae": "ttvae_{st}", "bn": "bn_pgmpy_{st}",
           "wgan_zcr": "wgan_zcr_{st}", "ctgan": "ctgan_{st}", "flowsyn": "flowsyn_{st}",
           "borysov": "borysov_{st}", "tabddpm": "tabddpm_{st}_valsrmse", "tvae": "tvae_ref_{st}"}

# floor, mu (drift), lambda (volatility), beta (sampling), fixed horizon (None = scenario year - 2013)
VARIANTS = {
    "base":       dict(floor=0.005,  mu=1.0, lam=1.0, beta=1.0, H=None),
    "floor0025":  dict(floor=0.0025, mu=1.0, lam=1.0, beta=1.0, H=None),
    "floor01":    dict(floor=0.01,   mu=1.0, lam=1.0, beta=1.0, H=None),
    "mu05":       dict(floor=0.005,  mu=0.5, lam=1.0, beta=1.0, H=None),
    "mu2":        dict(floor=0.005,  mu=2.0, lam=1.0, beta=1.0, H=None),
    "lam05":      dict(floor=0.005,  mu=1.0, lam=0.5, beta=1.0, H=None),
    "lam2":       dict(floor=0.005,  mu=1.0, lam=2.0, beta=1.0, H=None),
    "beta0":      dict(floor=0.005,  mu=1.0, lam=1.0, beta=0.0, H=None),
    "hfree":      dict(floor=0.005,  mu=1.0, lam=1.0, beta=1.0, H=5.0),
    # historical window 2010-2013 (three annual changes) instead of 2009-2013
    "win1013":    dict(floor=0.005,  mu=1.0, lam=1.0, beta=1.0, H=None, y0=2010),
    # boundary probes from ladder.py: first values at which a 2021 case passes Layer 2
    "floor00525": dict(floor=0.00525, mu=1.0, lam=1.0, beta=1.0, H=None),
    "mu13":       dict(floor=0.005,  mu=1.3, lam=1.0, beta=1.0, H=None),
    "lam145":     dict(floor=0.005,  mu=1.0, lam=1.45, beta=1.0, H=None),
    # g1fix (2026-11-24) boundary probes: first grid values at which a 2021 case passes Layer 2
    # on the regenerated historical files / references (ladder.py, logs_g1fix/ladder.log)
    "floor0055":  dict(floor=0.0055, mu=1.0, lam=1.0, beta=1.0, H=None),
    "mu145":      dict(floor=0.005,  mu=1.45, lam=1.0, beta=1.0, H=None),
    "lam16":      dict(floor=0.005,  mu=1.0, lam=1.6, beta=1.0, H=None),
}


def ens_dir(study):
    d = f"{FINAL}/{study}/{study}_final_s42"
    if os.path.isdir(d):
        return d
    c = glob.glob(f"{FINAL}/{study}/*_final_s42")
    return c[0] if len(c) == 1 else d


def cases():
    cert = pd.read_csv(f"{SRC}/PAPER_xstate_v3/xstate_certificate_all.csv")
    c = cert[cert.L1_accept == True]
    return [(r.model, r.state, str(int(r.year))) for r in c.itertuples()]


def vdir(v):
    return HERE / os.environ.get("SIGMA_RUNS", "runs") / v


def run(cmd, log):
    with open(log, "w") as lg:
        return subprocess.run(cmd, cwd=ROOT, stdout=lg, stderr=subprocess.STDOUT).returncode


def do_l2(v, m, st, y):
    p = VARIANTS[v]
    sd = f"{ROOT}/data/pums_states/{st}"
    out = vdir(v) / f"layer2_{m}_{st}_{y}"
    cmd = [PY, "-u", str(CODE / "layer2_historical_sigma.py"),
           "--realizations_dir", f"{ens_dir(PRIMARY[m].format(st=st))}/generated_ensembles",
           "--scenario_reference_csv", f"{sd}/reference_{y}_{st}.csv",
           "--target_attrs", *R.L2_ATTRS,
           "--reference_weight_col", "PWGTP",
           "--sigma_mode", "historical_dynamics",
           "--historical_reference_dir", f"{sd}/historical",
           "--output_dir", str(out),
           "--sigma_floor", str(p["floor"]),
           "--historical_drift_scale", str(p["mu"]),
           "--historical_vol_scale", str(p["lam"]),
           "--historical_sampling_scale", str(p["beta"])]
    if p["H"] is not None:
        cmd += ["--projection_horizon", str(p["H"])]
    if p.get("y0") == 2010:
        cmd += ["--historical_reference_glob", "person_level_*_201[0-3].csv"]
    rc = run(cmd, f"{out}.relog")
    if rc != 0:
        raise RuntimeError(f"L2 {v} {m} {st} {y} rc={rc}")
    return f"L2 {v} {m} {st} {y} ok"


def do_l3(v, m, st, y):
    p = VARIANTS[v]
    sd = f"{ROOT}/data/pums_states/{st}"
    ref = f"{sd}/reference_{y}_{st}.csv"
    l2 = vdir(v) / f"layer2_{m}_{st}_{y}"
    l3 = vdir(v) / f"layer3_paperW_{m}_{st}_{y}"
    shutil.rmtree(l3, ignore_errors=True); os.makedirs(l3)
    # cells7 of the variant's r* (same code as rerun_audit_new.build_cells7)
    rstar, rcsv = R.resolve_rstar_file(str(l2), f"{ens_dir(PRIMARY[m].format(st=st))}/generated_ensembles")
    df = pd.read_csv(rcsv)
    df["Age_Group"] = pd.cut(df["Age"], bins=R.AGE_BINS, labels=R.AGE_LABELS, right=False)
    g = df.groupby(R.CELL_KEYS, observed=True).size().reset_index(name="base_count")
    g.insert(0, "realization_id", rstar)
    c7 = l3 / f"{rcsv.stem}_cells7.csv"
    g.to_csv(c7, index=False)
    sig = pd.read_csv(l2 / "layer2_historical_sigma_summary.csv")
    sj = l3 / "sigma_H_targets.json"
    json.dump({r.target: float(r.sigma) for r in sig.itertuples()}, open(sj, "w"))
    rc = run([PY, "-u", str(CODE / "layer3_projection_uncertainty_branches.py"),
              "--realizations_csv", str(c7), "--scenario_reference_csv", ref,
              "--target_attrs", *R.L3_ATTRS, "--base_weight_col", "base_count",
              "--reference_weight_col", "PWGTP", "--output_dir", str(l3),
              "--projection_mode", "uncertainty_set", "--uncertainty_aggregation", "attribute_constraints",
              "--uncertainty_tau", "1.0", "--solver", "SCALABLE", "--save_projected_weights",
              "--sigma_mode", "scenario_uncertainty", "--scenario_target_sigma_json", str(sj),
              "--sigma_floor", str(p["floor"])], f"{l3}.relog")
    # g1fix (2026-11-24): projected_weights are KEPT (~1 MB/case); analyze.py computes
    # D = d_A(Q*, S_H) on the category union from them via PAPER_xstate_v3/dA_union.py.
    if rc != 0:
        return f"L3 {v} {m} {st} {y} FAILED rc={rc}"
    lv = pd.read_csv(l3 / "layer3_scenario_sigma_summary.csv").sigma_level
    if (lv != "target").any():
        raise RuntimeError(f"L3 {v} {m} {st} {y}: default-sigma fallback")
    return f"L3 {v} {m} {st} {y} ok r*={rstar}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", nargs="+", required=True)
    ap.add_argument("--stage", choices=["l2", "l3", "all"], default="all")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--only", default="", help="comma list model_state_year")
    ap.add_argument("--admitted-only", action="store_true",
                    help="Layer 3 only for cases passing the variant's Layer-2 gate, skipping finished ones")
    a = ap.parse_args()
    cs = cases()
    if a.only:
        cs = [c for c in cs if "_".join(c) in a.only.split(",")]
    for v in a.variant:
        vdir(v).mkdir(parents=True, exist_ok=True)
        for stage, fn in (("l2", do_l2), ("l3", do_l3)):
            if a.stage not in (stage, "all"):
                continue
            todo = cs
            if stage == "l3" and a.admitted_only:
                import decision_certificate_xstate_v3 as DC
                DC.BASE = vdir(v)
                todo = [c for c in cs
                        if DC.layer2_test(DC.load_layer2(c[1], c[0], c[2]), 0.05)["accept"]
                        and not (vdir(v) / f"layer3_paperW_{c[0]}_{c[1]}_{c[2]}" / "layer3_summary.json").exists()]
            t0 = time.time()
            print(f"[{time.strftime('%T')}] {v} {stage}: {len(todo)} cases", flush=True)
            with ThreadPoolExecutor(max_workers=a.jobs) as ex:
                futs = {ex.submit(fn, v, *c): c for c in todo}
                for fu in as_completed(futs):
                    try:
                        print(f"[{time.strftime('%T')}] {fu.result()}", flush=True)
                    except Exception as e:
                        print(f"[{time.strftime('%T')}] {futs[fu]} EXCEPTION {e!r}", flush=True)
            print(f"[{time.strftime('%T')}] {v} {stage} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
