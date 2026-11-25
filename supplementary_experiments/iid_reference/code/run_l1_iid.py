"""Shadow Layer-1 runs with a FRESH independent iid test reference (no retraining).
Same production chain as audits_v3/layer1_paper_<tag>_<ST>/ (run_audit_v3.do_l1 -> rerun_l1_paper.run_cell
+ l1_invalid_fix_shadow FIXED invalid-mass patch), reusing the household_disjoint shadow metric driver
supplementary_experiments/household_disjoint/code/rerun_l1_paper_disjoint.py (= rerun_l1_paper.py with ONLY the output
root AUD and reference path REF settable; see household_disjoint/code/driver.diff).  This file is
household_disjoint/code/run_l1_disjoint.py with only the variant table and the output root (iid_reference/runs) changed.
usage: run_l1_iid.py --variant orig|fresh [--only semapop_vae_CA,...] [--jobs 9]
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[3])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import os, sys, json, glob, time, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
ROOT = _PA.R(""); E = _PA.R("/main_experiments")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))           # .../iid_reference
L1D = _PA.R("/supplementary_experiments/household_disjoint/code")
sys.path.insert(0, L1D); sys.path.insert(0, E); sys.path.insert(0, ROOT)
import rerun_l1_paper_disjoint as L1P          # shadow metric driver (reused, read-only)
import l1_invalid_fix_shadow as FIX            # production FIXED rule set (read-only use)
import run_audit_v3 as V3                      # production cell table + ensemble resolver (read-only use)

REFS = {"orig": f"{ROOT}/data/pums_states/{{state}}/testdata_pers.csv",
        "fresh": f"{HERE}/refs/{{state}}/testdata_fresh.csv"}
def ref_for(v): return REFS[v]

def setup(variant):
    out = f"{HERE}/runs/{variant}"; os.makedirs(out, exist_ok=True)
    L1P.AUD = Path(out); L1P.REF = ref_for(variant)
    return out

def do_l1(tag, st, variant):
    out = setup(variant); study = V3.PRIMARY[tag].format(st=st); ens = V3.ens_dir(study)
    r = L1P.run_cell(tag, st, ens)
    if r.get("status") != "OK": return f"L1 {tag} {st} {variant} status={r.get('status')}"
    files = sorted(glob.glob(f"{ens}/generated_ensembles/final_generated_samples_*.csv"))
    f = [FIX.inv(pd.read_csv(fp), FIX.FIXED)[0] for fp in files]
    p = f"{out}/layer1_paper_{tag}_{st}/layer1_ensemble_summary.json"
    j = json.load(open(p)); j["invalid_mass_mean_orig_rules"] = j.get("invalid_mass_mean")
    j.update(invalid_mass_mean=float(np.mean(f)), invalid_mass_std=float(np.std(f, ddof=1)),
             invalid_mass_min=float(np.min(f)), invalid_mass_max=float(np.max(f)),
             invalid_rules_note="7 fixed R_hard rules (l1_invalid_fix_shadow.FIXED)",
             reference_csv=L1P.REF.format(state=st), ensemble_dir=ens)
    json.dump(j, open(p, "w"), indent=1)
    return (f"L1 {tag} {st} {variant} marg={r['paper_marg']} dep={r['paper_dep']} supp={r['paper_support']} "
            f"inv7={np.mean(f):.6f} ({r['secs']}s)")

def _w(a):
    try: return do_l1(*a)
    except Exception as e:
        import traceback; return f"L1 {a} EXCEPTION {e!r} {traceback.format_exc()[-600:]}"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", required=True)
    ap.add_argument("--only", default=""); ap.add_argument("--jobs", type=int, default=9)
    ap.add_argument("--redo", action="store_true"); a = ap.parse_args()
    out = setup(a.variant)
    cells = [(t, st) for t in V3.PRIMARY for st in V3.STATES if not a.only or f"{t}_{st}" in a.only.split(",")]
    cells = [c for c in cells if a.redo or not os.path.exists(f"{out}/layer1_paper_{c[0]}_{c[1]}/layer1_ensemble_summary.json")]
    cells.sort(key=lambda c: {"CA": 0, "TX": 1, "OH": 2}[c[1]])
    print(f"[{time.strftime('%F %T')}] variant={a.variant} ref={L1P.REF} -> {out}: {len(cells)} cells", flush=True)
    with ProcessPoolExecutor(max_workers=a.jobs) as px:
        futs = [px.submit(_w, (t, st, a.variant)) for t, st in cells]
        for fu in as_completed(futs): print(f"[{time.strftime('%T')}] {fu.result()}", flush=True)
    print(f"[{time.strftime('%F %T')}] done {a.variant}", flush=True)

if __name__ == "__main__":
    main()
