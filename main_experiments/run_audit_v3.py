"""
3-layer audit (v3 protocol, 2026-09-02) of the hpo_v3 finalized K=50 ensembles.

9 primary models x 3 states x 3 horizons = 81 cases, plus diagnostic extras
(as-shipped rows, TabDDPM under the authors' ml_eff criterion, SemaPop-VAE+ M-U
prior arm, TTVAE zero-context "prior" arm).  Reuses the validated paper-faithful
L1 runner (rerun_l1_paper.run_cell) and the L2 -> cells7 -> L3 chain
(rerun_audit_new.do_year) verbatim; only the ensemble source and the output root
change:  outputs -> main_experiments/audits_v3/  (old audits/ untouched).

Ensemble = final/<study>/<study>_final_s42/generated_ensembles (seed 42 = the
certificate seed; 50 realizations, N = |reference|, audit_config.ENSEMBLE_SEEDS).
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[1])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import os, sys, glob, time, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
ROOT = _PA.R("")
sys.path.insert(0, _PA.R("/main_experiments")); sys.path.insert(0, ROOT)
import rerun_audit_new as R
import rerun_l1_paper as L1P

FINAL = _PA.S("/hpo_v3/final")
AUD_V3 = _PA.R("/main_experiments/audits_v3")
R.AUD = AUD_V3
L1P.AUD = Path(AUD_V3)
R.l2_dir = lambda m, s, y: f"{AUD_V3}/layer2_{m}_{s}_{y}"   # uniform naming, no legacy pwgtp split
STATES = ["CA", "TX", "OH"]
YEARS = R.YEARS

# audit tag -> study-name template ({st} = state).  Primary = the fidelity-table rows
# selected under the uniform val_srmse criterion (or the authors' own where no_hpo).
PRIMARY = {
    "semapop_vae": "semapop_vae_{st}_train",   # SemaPop-VAE+ (M-R main arm)
    "ttvae":       "ttvae_{st}",               # triangular (published sampler)
    "bn":          "bn_pgmpy_{st}",           # 2026-11-23: the SemaPop repository pgmpy BN (was train_bn_v2 bn_{st})
    "wgan_zcr":    "wgan_zcr_{st}",
    "ctgan":       "ctgan_{st}",               # tuned
    "flowsyn":     "flowsyn_{st}",
    "borysov":     "borysov_{st}",             # tuned
    "tabddpm":     "tabddpm_{st}_valsrmse",    # val_srmse reselection
    "tvae":        "tvae_ref_{st}",            # tuned
}
EXTRAS = {
    "ctgan_defaults":   "ctgan_defaults_{st}",
    "tvae_defaults":    "tvae_ref_defaults_{st}",
    "borysov_defaults": "borysov_defaults_{st}",
    "tabddpm_mleff":    "tabddpm_{st}",
    "semapop_vae_mu":   "semapop_vae_{st}_train_mu",
    "ttvae_prior":      "ttvae_{st}_prior",
    # 2026-11-23 diagnostic: the SemaPop repository's latest pgmpy BN (reported config) and the same
    # trainer at its BIC literature defaults, compared against the audited bn row
    "bn_pgmpy":         "bn_pgmpy_{st}",
    "bn_pgmpy_bic":     "bn_pgmpy_bic_{st}",
}


def ens_dir(study):
    d = f"{FINAL}/{study}/{study}_final_s42"
    if os.path.isdir(d):
        return d
    # renamed finalize dirs (ctgan_defaults_<ST>, tvae_ref_defaults_<ST>) keep the
    # original run name inside: exactly one *_final_s42 subdir
    cands = glob.glob(f"{FINAL}/{study}/*_final_s42")
    return cands[0] if len(cands) == 1 else d


def n_files(study):
    return len(glob.glob(f"{ens_dir(study)}/generated_ensembles/final_generated_samples_*.csv"))


def l3_done(tag, st, y):
    return os.path.exists(f"{AUD_V3}/layer3_paperW_{tag}_{st}_{y}/layer3_summary.json")


def l1_done(tag, st):
    return os.path.exists(f"{AUD_V3}/layer1_paper_{tag}_{st}/layer1_ensemble_summary.json")


def do_l1(tag, st, study):
    r = L1P.run_cell(tag, st, ens_dir(study))      # absolute path: ROOT / abs -> abs
    return (f"L1 {tag} {st} status={r['status']} marg={r.get('paper_marg')} "
            f"dep={r.get('paper_dep')} supp={r.get('paper_support')} inv={r.get('invalid_mass')} "
            f"({r.get('secs')}s)")


def do_l23(tag, st, y, ens):
    """Layer 2 + cells7 + Layer 3 from scratch: the Layer-2 dir is cleared first so no file of a
    previous reference can survive (do_year itself clears the Layer-3 dir and the cells7 input)."""
    import shutil
    shutil.rmtree(R.l2_dir(tag, st, y), ignore_errors=True)
    return R.do_year(tag, st, y, ens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--extras", action="store_true", help="run the diagnostic extras instead of the primary 9")
    ap.add_argument("--only", default="", help="comma list of tag_state, e.g. tabddpm_OH")
    ap.add_argument("--list", action="store_true", help="print the cell table and exit")
    ap.add_argument("--redo", action="store_true", help="re-run even if outputs exist")
    ap.add_argument("--l3-only", action="store_true",
                    help="2026-11-23 r* fix: rebuild cells7 from the EXISTING Layer-2 output "
                         "(rerun_audit_new.resolve_rstar_file) and re-run Layer 3 only")
    ap.add_argument("--l23-only", action="store_true",
                    help="2026-11-24 group-1 fix (JWTRNS crosswalk + constant-2013-dollar income): the "
                         "scenario/historical references changed, Layer 1 (2013 test data) did not.  Clears "
                         "and re-runs Layer 2 + cells7 + Layer 3 (rerun_audit_new.do_year) for every cell; "
                         "Layer-1 outputs are reused untouched")
    a = ap.parse_args()
    table = EXTRAS if a.extras else PRIMARY
    cells = []
    for tag, tpl in table.items():
        for st in STATES:
            if a.only and f"{tag}_{st}" not in a.only.split(","):
                continue
            study = tpl.format(st=st); k = n_files(study)
            cells.append((tag, st, study, k))
    print(f"[{time.strftime('%F %T')}] audit_v3 {'EXTRAS' if a.extras else 'PRIMARY'}: {len(cells)} cells -> {AUD_V3}")
    for tag, st, study, k in cells:
        print(f"  {tag:18s} {st}  {study:30s} K={k:2d}  {'READY' if k == 50 else 'SKIP (incomplete)'}")
    if a.list:
        return
    os.makedirs(f"{AUD_V3}/PAPER_xstate", exist_ok=True)
    tasks = []
    if a.l23_only:
        for tag, st, study, k in cells:
            if k == 50:
                for y in YEARS:
                    tasks.append((f"Y{y} {tag} {st}", do_l23, (tag, st, y, ens_dir(study))))
        cells = []
    if a.l3_only:
        for tag, st, study, k in cells:
            if k == 50:
                for y in YEARS:
                    tasks.append((f"L3 {tag} {st} {y}", R.do_l3, (tag, st, y, ens_dir(study))))
        cells = []
    for tag, st, study, k in cells:
        if k != 50:
            continue
        if a.redo or not l1_done(tag, st):
            tasks.append((f"L1 {tag} {st}", do_l1, (tag, st, study)))
        for y in YEARS:
            if a.redo or not l3_done(tag, st, y):
                tasks.append((f"Y{y} {tag} {st}", R.do_year, (tag, st, y, ens_dir(study))))
    print(f"[{time.strftime('%T')}] {len(tasks)} tasks, jobs={a.jobs}", flush=True)
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(fn, *args): name for name, fn, args in tasks}
        for fu in as_completed(futs):
            try:
                print(f"[{time.strftime('%T')}] {fu.result()}", flush=True)
            except Exception as e:
                print(f"[{time.strftime('%T')}] {futs[fu]} EXCEPTION {e!r}", flush=True)
    print(f"[{time.strftime('%F %T')}] audit_v3 done.", flush=True)


if __name__ == "__main__":
    main()
