"""Supplementary diagnostic (filter): full 3-layer audit of the truncated priors.

Uses the shadow COPIES of the current pipeline (rerun_l1_paper.run_cell, rerun_audit_new.do_year =
layer2_historical_sigma.py -> resolve_rstar_file/cells7 -> layer3 with sigma(H) from Layer 2,
tau=1, SCALABLE), with only the output root redirected to supplementary/filter/audit/.

Tags:  <model>_filt  (filtered ensembles under ../ens/<model>_filt_<ST>/)
       ctgan_ctrl    (control: the UNFILTERED CTGAN-Ohio ensemble; must reproduce audits_v3)
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[5])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import os, sys, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
ROOT = _PA.R("")
S = _PA.R("/experiments/E_state/supplementary/filter")
sys.path.insert(0, f"{S}/scripts"); sys.path.insert(0, ROOT)
import rerun_audit_new as R          # shadow copy: R.AUD = S/audit
import rerun_l1_paper as L1P         # shadow copy: L1P.AUD = S/audit
assert R.AUD == f"{S}/audit" and str(L1P.AUD) == f"{S}/audit", (R.AUD, L1P.AUD)
R.l2_dir = lambda m, s, y: f"{R.AUD}/layer2_{m}_{s}_{y}"   # uniform naming (as run_audit_v3)
FINAL = _PA.S("/hpo_v3/final")


def cells():
    out = []
    for m in ("ctgan", "tvae", "borysov"):
        for st in ("CA", "TX", "OH"):
            out.append((f"{m}_filt", st, f"{S}/ens/{m}_filt_{st}"))
    out.append(("ctgan_ctrl", "OH", f"{FINAL}/ctgan_OH/ctgan_OH_final_s42"))
    return out


def do_l1(tag, st, ens):
    r = L1P.run_cell(tag, st, ens)     # ROOT / absolute path -> absolute path
    return f"L1 {tag} {st} {r['status']} marg={r.get('paper_marg')} dep={r.get('paper_dep')} supp={r.get('paper_support')} inv={r.get('invalid_mass')} ({r.get('secs')}s)"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only", default=""); a = ap.parse_args()
    tasks = []
    for tag, st, ens in cells():
        if a.only and f"{tag}_{st}" not in a.only.split(","):
            continue
        if not os.path.exists(f"{R.AUD}/layer1_paper_{tag}_{st}/layer1_ensemble_summary.json"):
            tasks.append((f"L1 {tag} {st}", do_l1, (tag, st, ens)))
        for y in R.YEARS:
            if not os.path.exists(f"{R.AUD}/layer3_paperW_{tag}_{st}_{y}/layer3_summary.json"):
                tasks.append((f"Y{y} {tag} {st}", R.do_year, (tag, st, y, ens)))
    # control first, then the small states first so failures surface early
    order = {"ctgan_ctrl": 0}; sto = {"OH": 0, "TX": 1, "CA": 2}
    tasks.sort(key=lambda t: (order.get(t[2][0], 1), sto[t[2][1]]))
    os.makedirs(R.AUD, exist_ok=True)
    print(f"[{time.strftime('%F %T')}] {len(tasks)} tasks, jobs={a.jobs}", flush=True)
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(fn, *args): name for name, fn, args in tasks}
        for fu in as_completed(futs):
            try:
                print(f"[{time.strftime('%T')}] {fu.result()}", flush=True)
            except Exception as e:
                print(f"[{time.strftime('%T')}] {futs[fu]} EXCEPTION {e!r}", flush=True)
    print(f"[{time.strftime('%F %T')}] done.", flush=True)


if __name__ == "__main__":
    main()
