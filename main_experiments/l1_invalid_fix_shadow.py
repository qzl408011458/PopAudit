"""SHADOW (2026-11-23): Layer-1 invalid mass with the corrected rule set.
Two of the eight coded rules never fired: rule_child_not_employed (its word-boundary regexes
cannot match underscore-coded labels such as 'Civilian_Employed_At_Work') and
rule_student_age_reasonable (same cause; it is also violated by the real 2013 splits, so it
is not a valid hard rule). Fixed set = the six working rules + rule_child_not_employed applied
to labels with '_' replaced by ' '. Only invalid mass changes. Writes audits_v3_l1fix/:
layer1_paper_<tag>_<ST>/layer1_ensemble_summary.json (copy with invalid_mass_* replaced) and
symlinks to the unchanged Layer-2/3 outputs. First reproduces the stored invalid mass with the
original rules as a check.
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[1])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import os, sys, json, glob, shutil
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
ROOT = _PA.R("")
sys.path.insert(0, ROOT); sys.path.insert(0, _PA.R("/main_experiments"))
import layer1 as L1
import run_audit_v3 as V3
SRC = _PA.R("/main_experiments/audits_v3"); DST = _PA.R("/main_experiments/audits_v3_l1fix")

def rule_child_not_employed_fixed(df):
    if "Employment_status" not in df: return L1._true(df)
    age = L1._ncol(df, "Age"); emp = L1._scol(df, "Employment_status").str.replace("_", " ", regex=False)
    employed = L1._has(emp, [r"\bemployed\b", r"\bat work\b", r"\bworked\b", r"\bworking\b"]) & (~L1._has(emp, [r"\bunemployed\b", r"\bnot employed\b", r"\bnot in labor\b", r"\bnot in labour\b", r"\binactive\b"]))
    return ~(age.notna() & (age < 14) & employed)

ORIG = [L1.rule_age_reasonable_pums, L1.rule_household_size_positive, L1.rule_children_nonnegative, L1.rule_cars_reasonable_pums,
        L1.rule_children_household_consistency, L1.rule_child_not_employed, L1.rule_child_not_married, L1.rule_student_age_reasonable]
FIXED = [L1.rule_age_reasonable_pums, L1.rule_household_size_positive, L1.rule_children_nonnegative, L1.rule_cars_reasonable_pums,
         L1.rule_children_household_consistency, rule_child_not_employed_fixed, L1.rule_child_not_married]

def inv(df, rules):
    ok = pd.Series(True, index=df.index)
    for r in rules: ok &= r(df).fillna(False).astype(bool)
    return float((~ok).mean()), float((~rule_child_not_employed_fixed(df)).mean())

def cell(args):
    tag, st, ens = args
    files = sorted(glob.glob(f"{ens}/generated_ensembles/final_generated_samples_*.csv"))
    o, f, e = [], [], []
    for fp in files:
        df = pd.read_csv(fp)
        a, _ = inv(df, ORIG); b, c = inv(df, FIXED); o.append(a); f.append(b); e.append(c)
    return tag, st, len(files), float(np.mean(o)), float(np.mean(f)), float(np.std(f, ddof=1)), float(np.min(f)), float(np.max(f)), float(np.mean(e))

def main():
    table = dict(V3.PRIMARY); table.update(V3.EXTRAS)
    jobs = [(t, st, V3.ens_dir(tpl.format(st=st))) for t, tpl in table.items() for st in V3.STATES
            if os.path.exists(f"{SRC}/layer1_paper_{t}_{st}/layer1_ensemble_summary.json")]
    os.makedirs(DST, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        for r in ex.map(cell, jobs):
            tag, st, k, orig, fixd, sd, mn, mx, emp = r
            stored = json.load(open(f"{SRC}/layer1_paper_{tag}_{st}/layer1_ensemble_summary.json"))
            rows.append(dict(tag=tag, state=st, K=k, stored=stored["invalid_mass_mean"], recomputed_orig=orig, fixed=fixd, employed_u14=emp))
            print(tag, st, f"stored={stored['invalid_mass_mean']:.6f} orig={orig:.6f} fixed={fixd:.6f} emp_u14={emp:.6f}", flush=True)
            d = f"{DST}/layer1_paper_{tag}_{st}"; os.makedirs(d, exist_ok=True)
            stored["rule_child_not_employed_mean"] = emp; stored.pop("rule_child_not_employed_std", None)
            stored.pop("rule_student_age_reasonable_mean", None); stored.pop("rule_student_age_reasonable_std", None)
            stored.update(invalid_mass_mean=fixd, invalid_mass_std=sd, invalid_mass_min=mn, invalid_mass_max=mx,
                          invalid_rules_note="2026-11-23 shadow: rule_child_not_employed fixed, rule_student_age_reasonable dropped")
            json.dump(stored, open(f"{d}/layer1_ensemble_summary.json", "w"), indent=1)
    df = pd.DataFrame(rows); df.to_csv(f"{DST}/invalid_mass_fix.csv", index=False)
    print("max |stored-orig| =", (df.stored - df.recomputed_orig).abs().max())
    for name in os.listdir(SRC):
        if name.startswith(("layer2_", "layer3_paperW_", "layer3_rstar")) and not os.path.lexists(f"{DST}/{name}"):
            os.symlink(f"{SRC}/{name}", f"{DST}/{name}")

if __name__ == "__main__":
    main()
