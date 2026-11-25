"""Step 3: strict reproduction check of runs/orig against production audits_v3 (all 27 pairs):
every numeric field of layer1_ensemble_summary.json and every numeric cell of the 6 Layer-1 CSVs."""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[3])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import os, json, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROD = _PA.R("/main_experiments/audits_v3")
rows = []
for d in sorted(glob.glob(f"{HERE}/runs/orig/layer1_paper_*")):
    name = os.path.basename(d); p = f"{PROD}/{name}"
    a = json.load(open(f"{d}/layer1_ensemble_summary.json")); b = json.load(open(f"{p}/layer1_ensemble_summary.json"))
    keys = [k for k in b if isinstance(b[k], (int, float)) and not isinstance(b[k], bool)]
    jd = max(abs(float(a[k]) - float(b[k])) for k in keys if k in a)
    missing = [k for k in keys if k not in a]
    cd = 0.0; coldiff = set()
    for f in ["layer1_ensemble_summary.csv", "layer1_per_realization_metrics.csv", "layer1_marginal_attr_summary.csv",
              "layer1_marginal_attr_per_realization.csv", "layer1_dependency_pair_summary.csv", "layer1_dependency_pair_per_realization.csv"]:
        x = pd.read_csv(f"{d}/{f}"); y = pd.read_csv(f"{p}/{f}")
        assert len(x) == len(y), (name, f)
        # layer1.py's built-in rule list no longer contains rule_student_age_reasonable (dropped with the
        # 2026-11-23 invalid-rule fix), so rule columns / invalid_* in the CSVs differ by design; the paper's
        # invalid mass is the patched JSON value.  Compare every common numeric column except invalid/rule ones.
        num = [c for c in y.select_dtypes("number").columns if c in x.columns and not c.startswith(("invalid", "rule_"))]
        coldiff.update(set(x.columns) ^ set(y.columns))
        cd = max(cd, float(np.nanmax(np.abs(x[num].to_numpy(float) - y[num].to_numpy(float)))) if len(num) else 0.0)
    rows.append(dict(cell=name, json_max_abs_diff=jd, csv_max_abs_diff=cd, json_fields=len(keys), missing=";".join(missing), csv_col_symdiff=";".join(sorted(coldiff))))
df = pd.DataFrame(rows); df.to_csv(f"{HERE}/results/control_reproduction.csv", index=False)
print(df.to_string(index=False)); print("MAX json", df.json_max_abs_diff.max(), "MAX csv", df.csv_max_abs_diff.max())
