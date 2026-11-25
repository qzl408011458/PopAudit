"""
Paper-faithful Layer-1 re-run (2026-07-02).

Fixes two code<->paper deviations found in the manuscript consistency audit:
  (bug 4) marginal/support used RAW integer Age; paper C_a for age = Age_Group.
  (bug 5) dependency used a TV / sliced-W / mixed(TV+conditional-W) blend;
          paper eq:ensemble-joint-dependency is 0.5-scaled JOINT-TV over binned/
          categorical cells for every audited 2-way subset.

This driver reuses layer1.py's tested metric functions but:
  - adds Age_Group (bins=[0,18,35,50,65,200], right=False) matching Layer 2/3,
  - audits the 7 transport attrs with Age_Group replacing raw Age,
  - computes ALL 13 dependency pairs as joint-TV (pair_type="categorical"),
  - keeps invalid-mass rules on the raw records (Age still present).

Outputs -> experiments/E_state/audits/layer1_paper_<model>_<state>/
Also validates ensemble provenance by reproducing each cell's CURRENT (raw-age)
ensemble support coverage against the stored layer1_<model>_<state> summary.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[2])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import sys, os, glob, json, csv, time, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

ROOT = Path(_PA.R(""))
sys.path.insert(0, str(ROOT))
import layer1 as L1

AUD = ROOT / "experiments/E_state/audits"
AGE_BINS = [0, 18, 35, 50, 65, 200]
AGE_LABELS = ["0-17", "18-34", "35-49", "50-64", "65+"]

# 7 transport-visible audit attrs, paper representation (Age -> Age_Group)
AUDIT_ATTRS = ["Age_Group", "Household_Size", "Number_of_children",
               "Number_of_cars_of_household", "Employment_status",
               "Household_Income_Class", "Commute_Mode"]
# raw representation used ONLY to reproduce/validate the current stored L1 support
RAW_ATTRS = ["Age", "Household_Size", "Number_of_children",
             "Number_of_cars_of_household", "Employment_status",
             "Household_Income_Class", "Commute_Mode"]

# same 13 two-way subsets as the original audit, but Age -> Age_Group and ALL joint-TV
PAIRS_ATTR = [
    ("Household_Size", "Number_of_children"),
    ("Household_Size", "Number_of_cars_of_household"),
    ("Number_of_children", "Number_of_cars_of_household"),
    ("Age_Group", "Employment_status"),
    ("Age_Group", "Household_Income_Class"),
    ("Age_Group", "Commute_Mode"),
    ("Household_Size", "Household_Income_Class"),
    ("Number_of_children", "Household_Income_Class"),
    ("Number_of_cars_of_household", "Household_Income_Class"),
    ("Number_of_cars_of_household", "Commute_Mode"),
    ("Employment_status", "Household_Income_Class"),
    ("Employment_status", "Commute_Mode"),
    ("Household_Income_Class", "Commute_Mode"),
]

# 2026-07-03 SWAP: keep the single LLM-conditioned model = SemaPop-VAE+ (tvae_llm+),
# drop SemaPop-GAN (wgan_llm). Under the corrected kappa_2=5.0 gate, June-HPO
# tvae_llm+ = 5/9 vs recovered wgan_llm = 4/9, and tvae_llm+'s June HPO did NOT
# regress (no re-HPO needed). Same one-LLM-model anti-favoritism posture.
MODELS = ["ttvae", "tabvae", "wgan_zc", "tvae_llm+", "ctgan", "tabddpm", "flowsyn"]
STATES = ["CA", "OH", "TX"]

# Ensembles for cells not in cell_selection.csv NEW list.
# 2026-07-02: user directive = use June HPO checkpoints everywhere available.
# wgan_llm x3 and tabddpm_CA -> switched to their June-29 HPO ensembles.
# wgan_zc_CA and ctgan_CA -> June HPO was KILLED (no ensemble); pending decision
#   (kept on May published for now).
# tvae_llm+ is NEW (June ensembles) -> comes from cell_selection.csv, no entry needed.
OLD_ENS = {
    ("tabddpm", "CA"):  "modelSave/tabddpm_CA_2026-06-29_07-31-32",
    ("wgan_zc", "CA"):  "modelSave/wgan_zc_CA_2026-07-03_10-28-05",   # retrained June HPO (2026-07-03)
    ("ctgan", "CA"):    "modelSave/ctgan_CA_2026-07-03_05-31-43",     # retrained June HPO (2026-07-03)
}


def _fast_patterns(df, cols, missing_token="__MISSING__"):
    """Vectorized replacement for L1._row_patterns (string key == tuple key for set membership)."""
    sub = df.loc[:, list(cols)].astype(object)
    sub = sub.where(sub.notna(), missing_token).astype(str)
    out = sub[cols[0]]
    for c in cols[1:]:
        out = out + "\x1f" + sub[c]
    return out


L1._row_patterns = _fast_patterns  # monkeypatch for speed; membership semantics preserved


def add_age_group(df):
    age = pd.to_numeric(df["Age"], errors="coerce")
    ag = pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS, right=False, include_lowest=True)
    df = df.copy()
    df["Age_Group"] = ag.astype(object)
    return df


def load_ensemble_map():
    m = {}
    with open(ROOT / "experiments/hpo/cell_selection.csv") as f:
        for r in csv.DictReader(f):
            if r["use"] == "NEW" and r["new_ensemble_dir"]:
                m[(r["model"], r["state"])] = r["new_ensemble_dir"]
    m.update(OLD_ENS)
    return m


def paper_config():
    cfg = L1.build_default_layer1_config()
    cfg.task_attrs = AUDIT_ATTRS
    cfg.dependency_pairs = [L1.DependencyPairSpec(a, b, "categorical") for a, b in PAIRS_ATTR]
    return cfg


def stored_raw_support(model, state):
    p = AUD / f"layer1_{model}_{state}" / "layer1_ensemble_summary.json"
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text())["ensemble_support_coverage"])
    except Exception:
        return None


def run_cell(model, state, ens_rel):
    t0 = time.time()
    ens = ROOT / ens_rel / "generated_ensembles"
    files = sorted(glob.glob(str(ens / "final_generated_samples_*.csv")))
    if not files:
        return {"cell": f"{model}_{state}", "status": "NO_ENSEMBLE", "ens": ens_rel}
    real = pd.read_csv(ROOT / f"data/pums_states/{state}/testdata_pers.csv")
    real = add_age_group(real)
    reals = [add_age_group(pd.read_csv(f)) for f in files]
    ids = [Path(f).stem for f in files]

    # paper-faithful L1
    cfg = paper_config()
    res = L1.run_layer1_ensemble_audit(real, reals, cfg, ids)
    outdir = AUD / f"layer1_paper_{model}_{state}"
    L1._save_outputs(res, outdir)
    s = res.ensemble_summary

    # provenance validation: reproduce CURRENT raw-age ensemble support coverage
    raw_cov = L1.compute_ensemble_support_coverage(real, reals, RAW_ATTRS).coverage
    stored = stored_raw_support(model, state)
    match = (stored is not None) and (abs(raw_cov - stored) < 1e-4)

    return {
        "cell": f"{model}_{state}", "status": "OK", "ens": ens_rel, "K": len(files),
        "paper_marg": round(float(s["marginal_total_discrepancy_mean"]), 6),
        "paper_dep":  round(float(s["dependency_total_discrepancy_mean"]), 6),
        "paper_support": round(float(s["ensemble_support_coverage"]), 6),
        "invalid_mass": round(float(s.get("invalid_mass_mean", 0.0)), 6),
        "raw_support_recomputed": round(float(raw_cov), 6),
        "raw_support_stored": None if stored is None else round(stored, 6),
        "provenance_match": bool(match),
        "secs": round(time.time() - t0, 1),
    }


def _worker(args):
    try:
        return run_cell(*args)
    except Exception as e:
        import traceback
        return {"cell": f"{args[0]}_{args[1]}", "status": f"ERROR: {e}",
                "trace": traceback.format_exc()[-800:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="", help="comma list model_state e.g. wgan_llm_OH")
    args = ap.parse_args()

    emap = load_ensemble_map()
    cells = []
    for m in MODELS:
        for s in STATES:
            tag = f"{m}_{s}"
            if args.only and tag not in args.only.split(","):
                continue
            if (m, s) not in emap:
                print(f"[skip] {tag}: no ensemble mapping", flush=True)
                continue
            cells.append((m, s, emap[(m, s)]))

    print(f"[{time.strftime('%F %T')}] paper L1 re-run: {len(cells)} cells, jobs={args.jobs}", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_worker, c): c for c in cells}
        for fu in as_completed(futs):
            r = fu.result()
            results.append(r)
            flag = "" if r.get("provenance_match") else "  <-- PROVENANCE MISMATCH" if r["status"] == "OK" else ""
            if r["status"] == "OK":
                print(f"[{time.strftime('%T')}] {r['cell']:20s} K={r['K']} "
                      f"marg={r['paper_marg']:.4f} dep={r['paper_dep']:.4f} supp={r['paper_support']:.4f} "
                      f"inv={r['invalid_mass']:.4f} | rawsupp {r['raw_support_recomputed']} vs stored {r['raw_support_stored']} "
                      f"({r['secs']}s){flag}", flush=True)
            else:
                print(f"[{time.strftime('%T')}] {r['cell']:20s} {r['status']}", flush=True)

    # persist a compact summary table
    df = pd.DataFrame([r for r in results if r["status"] == "OK"])
    if not df.empty:
        df = df.sort_values("cell")
        out = AUD / "PAPER_xstate" / "layer1_paper_recompute_summary.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved summary -> {out}", flush=True)
        bad = df[~df["provenance_match"]]
        if len(bad):
            print(f"WARNING: {len(bad)} cells failed provenance validation:", flush=True)
            print(bad[["cell", "raw_support_recomputed", "raw_support_stored"]].to_string(index=False), flush=True)
        else:
            print("All cells passed provenance validation (raw support matches stored).", flush=True)
    print(f"[{time.strftime('%F %T')}] done.", flush=True)


if __name__ == "__main__":
    main()
