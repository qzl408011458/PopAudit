"""Shadow Layer-1 runs for the household-disjoint reference test (no retraining).
Exactly the production chain that produced audits_v3/layer1_paper_<tag>_<ST>/:
  run_audit_v3.do_l1 -> rerun_l1_paper.run_cell(tag, st, ens_dir(study))      (2026-09-02)
  + l1_invalid_fix_shadow FIXED invalid-mass patch of layer1_ensemble_summary.json (2026-11-23)
using code/rerun_l1_paper_disjoint.py (= rerun_l1_paper.py with ONLY the output root AUD and the
reference path REF made settable; see driver.diff).  Ensembles = run_audit_v3.PRIMARY/ens_dir (seed 42).
usage: run_l1_disjoint.py --variant orig|hhdisjoint_trainval|hhdisjoint_train|hhdisjoint_trainval_ipw|placebo_s1|...
                          [--only ctgan_TX,...] [--jobs 9]
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[5])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import os, sys, json, glob, time, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
ROOT = _PA.R(""); E = f"{ROOT}/experiments/E_state"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "code")); sys.path.insert(0, E); sys.path.insert(0, ROOT)
import rerun_l1_paper_disjoint as L1P          # shadow copy
import l1_invalid_fix_shadow as FIX            # production FIXED rule set (read-only use)
import run_audit_v3 as V3                      # production cell table + ensemble resolver (read-only use)

REFS = {
    "orig": f"{ROOT}/data/pums_states/{{state}}/testdata_pers.csv",
    "hhdisjoint_trainval": f"{HERE}/refs/{{state}}/testdata_hhdisjoint_trainval.csv",
    "hhdisjoint_train": f"{HERE}/refs/{{state}}/testdata_hhdisjoint_train.csv",
    "hhdisjoint_trainval_ipw": f"{HERE}/refs/{{state}}/testdata_hhdisjoint_trainval_ipw.csv",
}
def ref_for(v):
    if v in REFS: return REFS[v]
    if v.startswith("placebo_s"): return f"{HERE}/refs/{{state}}/testdata_{v}.csv"
    raise KeyError(v)

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
    # CA first (longest)
    cells.sort(key=lambda c: {"CA": 0, "TX": 1, "OH": 2}[c[1]])
    print(f"[{time.strftime('%F %T')}] variant={a.variant} ref={L1P.REF} -> {out}: {len(cells)} cells", flush=True)
    with ProcessPoolExecutor(max_workers=a.jobs) as px:
        futs = [px.submit(_w, (t, st, a.variant)) for t, st in cells]
        for fu in as_completed(futs): print(f"[{time.strftime('%T')}] {fu.result()}", flush=True)
    print(f"[{time.strftime('%F %T')}] done {a.variant}", flush=True)

if __name__ == "__main__":
    main()
