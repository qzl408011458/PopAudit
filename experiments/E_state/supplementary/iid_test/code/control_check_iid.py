"""Control: runs/orig (shadow driver with the ORIGINAL reference) vs production audits_v3, cell by cell:
every numeric field of layer1_ensemble_summary.json and every common numeric cell (excluding invalid_*/rule_*
CSV columns, which differ by design; see l1_disjoint/code/control_check.py) of the 6 Layer-1 CSVs.
= l1_disjoint/code/control_check.py with HERE -> iid_test.  Output ../results/control_reproduction.csv"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[5])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import os, json, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROD = _PA.R("/experiments/E_state/audits_v3")
rows = []
for d in sorted(glob.glob(f"{HERE}/runs/orig/layer1_paper_*")):
    name = os.path.basename(d); p = f"{PROD}/{name}"
    a = json.load(open(f"{d}/layer1_ensemble_summary.json")); b = json.load(open(f"{p}/layer1_ensemble_summary.json"))
    keys = [k for k in b if isinstance(b[k], (int, float)) and not isinstance(b[k], bool)]
    jd = max(abs(float(a[k]) - float(b[k])) for k in keys if k in a)
    missing = [k for k in keys if k not in a]
    cd = 0.0; ncells = 0
    for f in ["layer1_ensemble_summary.csv", "layer1_per_realization_metrics.csv", "layer1_marginal_attr_summary.csv",
              "layer1_marginal_attr_per_realization.csv", "layer1_dependency_pair_summary.csv", "layer1_dependency_pair_per_realization.csv"]:
        x = pd.read_csv(f"{d}/{f}"); y = pd.read_csv(f"{p}/{f}")
        assert len(x) == len(y), (name, f)
        num = [c for c in y.select_dtypes("number").columns if c in x.columns and not c.startswith(("invalid", "rule_"))]
        if num:
            cd = max(cd, float(np.nanmax(np.abs(x[num].to_numpy(float) - y[num].to_numpy(float))))); ncells += x[num].size
    rows.append(dict(cell=name, json_fields=len(keys), json_max_abs_diff=jd, csv_cells=ncells, csv_max_abs_diff=cd,
                     missing=";".join(missing), marg=a["marginal_total_discrepancy_mean"], dep=a["dependency_total_discrepancy_mean"],
                     cov=a["ensemble_support_coverage"], inv=a["invalid_mass_mean"]))
df = pd.DataFrame(rows); os.makedirs(f"{HERE}/results", exist_ok=True)
df.to_csv(f"{HERE}/results/control_reproduction.csv", index=False)
print(df.to_string(index=False)); print("MAX json", df.json_max_abs_diff.max(), "MAX csv", df.csv_max_abs_diff.max())
