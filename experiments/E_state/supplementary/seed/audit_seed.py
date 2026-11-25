"""Shadow 3-layer audit of the seed-2/3 K=50 ensembles with the exact current pipeline.
L1  : rerun_l1_paper.run_cell (Age_Group joint-TV) + invalid mass recomputed with the 7 fixed
      R_hard rules (l1_invalid_fix_shadow.FIXED), exactly as audits_v3 layer1 summaries were patched.
L2/L3: rerun_audit_new.do_year verbatim (L2 historical sigma; cells7 on true r*; L3 SCALABLE,
      tau=1, sigma(H) from Layer 2, sigma floor 0.005).  Only the output root changes.
usage: audit_seed.py --seed 142 [--only tabddpm_OH,...] [--jobs 12]
       --seed 42 re-audits the ORIGINAL ensembles into audits_s42/ (pipeline reproduction check).
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[4])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import os, sys, json, glob, time, argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd
ROOT = _PA.R(""); E = f"{ROOT}/experiments/E_state"
sys.path.insert(0, E); sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
import rerun_audit_new as R
import rerun_l1_paper as L1P
import l1_invalid_fix_shadow as FIX
FINAL42 = _PA.S("/hpo_v3/final")
STUDY = {"semapop_vae": "semapop_vae_{st}_train", "ttvae": "ttvae_{st}", "bn": "bn_pgmpy_{st}",
         "wgan_zcr": "wgan_zcr_{st}", "tabddpm": "tabddpm_{st}_valsrmse"}
STATES = ["CA", "TX", "OH"]; YEARS = R.YEARS

def ens_dir(tag, st, seed):
    s = STUDY[tag].format(st=st)
    base = FINAL42 if seed == 42 else f"{HERE}/final"
    return f"{base}/{s}/{s}_final_s{seed}"

def ready(ens):
    fs = glob.glob(f"{ens}/generated_ensembles/*.csv")
    return len(fs) == 50 and time.time() - max(os.path.getmtime(f) for f in fs) > 90

def setup(seed):
    aud = f"{HERE}/audits_s{seed}"; os.makedirs(aud, exist_ok=True)
    R.AUD = aud; L1P.AUD = Path(aud)
    R.l2_dir = lambda m, s, y: f"{aud}/layer2_{m}_{s}_{y}"
    return aud

def do_l1(tag, st, seed):
    aud = setup(seed); ens = ens_dir(tag, st, seed)
    r = L1P.run_cell(tag, st, ens)
    files = sorted(glob.glob(f"{ens}/generated_ensembles/final_generated_samples_*.csv"))
    f = [FIX.inv(pd.read_csv(fp), FIX.FIXED)[0] for fp in files]
    p = f"{aud}/layer1_paper_{tag}_{st}/layer1_ensemble_summary.json"
    j = json.load(open(p)); j["invalid_mass_mean_orig_rules"] = j.get("invalid_mass_mean")
    j.update(invalid_mass_mean=float(np.mean(f)), invalid_mass_std=float(np.std(f, ddof=1)),
             invalid_mass_min=float(np.min(f)), invalid_mass_max=float(np.max(f)),
             invalid_rules_note="7 fixed R_hard rules (l1_invalid_fix_shadow.FIXED)")
    json.dump(j, open(p, "w"), indent=1)
    return f"L1 {tag} {st} s{seed} marg={r.get('paper_marg')} dep={r.get('paper_dep')} supp={r.get('paper_support')} inv7={np.mean(f):.6f} ({r.get('secs')}s)"

def do_year(tag, st, y, seed):
    setup(seed)
    return f"s{seed} " + R.do_year(tag, st, y, ens_dir(tag, st, seed))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--only", default=""); ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--l1-jobs", type=int, default=6); ap.add_argument("--redo", action="store_true")
    a = ap.parse_args(); aud = setup(a.seed)
    cells = [(t, st) for t in STUDY for st in STATES
             if (not a.only or f"{t}_{st}" in a.only.split(","))
             and ready(ens_dir(t, st, a.seed))]
    print(f"[{time.strftime('%T')}] seed {a.seed}: {len(cells)} ready cells -> {aud}: {cells}", flush=True)
    l1 = [c for c in cells if a.redo or not os.path.exists(f"{aud}/layer1_paper_{c[0]}_{c[1]}/layer1_ensemble_summary.json")]
    yr = [(t, st, y) for t, st in cells for y in YEARS
          if a.redo or not os.path.exists(f"{aud}/layer3_paperW_{t}_{st}_{y}/layer3_summary.json")]
    with ProcessPoolExecutor(max_workers=a.l1_jobs) as px, ThreadPoolExecutor(max_workers=a.jobs) as tx:
        futs = {px.submit(do_l1, t, st, a.seed): f"L1 {t} {st}" for t, st in l1}
        futs.update({tx.submit(do_year, t, st, y, a.seed): f"Y {t} {st} {y}" for t, st, y in yr})
        for fu in as_completed(futs):
            try: print(f"[{time.strftime('%T')}] {fu.result()}", flush=True)
            except Exception as e: print(f"[{time.strftime('%T')}] {futs[fu]} EXCEPTION {e!r}", flush=True)
    print(f"[{time.strftime('%T')}] done seed {a.seed}", flush=True)

if __name__ == "__main__":
    main()
