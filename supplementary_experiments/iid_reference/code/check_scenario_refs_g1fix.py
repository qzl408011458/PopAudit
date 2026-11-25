"""Check scenario references / historical files are raw (non-resampled) PUMS persons with weights."""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[3])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import json
from pathlib import Path
import pandas as pd
ROOT = Path(_PA.R(""))
OUT = Path(_PA.R("/supplementary_experiments/iid_reference/corrected"))
res = {}
for st in ["CA", "OH", "TX"]:
    sd = ROOT / f"data/pums_states/{st}"
    files = {f"ref_{y}": sd / f"reference_{y}_{st}.csv" for y in (2018, 2021, 2024)}
    files.update({f"hist_{y}": sd / "historical" / f"person_level_{st}_{y}.csv" for y in range(2009, 2014)})
    for k, f in files.items():
        if not f.exists():
            res[f"{st}_{k}"] = "missing"; continue
        d = pd.read_csv(f, usecols=lambda c: c in {"Person_ID", "Source_Person_ID", "Sampled_Person_ID", "PWGTP", "WGTP"}, dtype=str)
        idc = "Person_ID" if "Person_ID" in d.columns else "Source_Person_ID"
        res[f"{st}_{k}"] = {"rows": len(d), "id_col": idc, "id_unique": bool(d[idc].is_unique),
                            "has_Sampled_Person_ID": "Sampled_Person_ID" in d.columns,
                            "has_PWGTP": "PWGTP" in d.columns, "has_WGTP": "WGTP" in d.columns,
                            "pwgtp_sum": float(pd.to_numeric(d["PWGTP"]).sum()) if "PWGTP" in d.columns else None}
print(json.dumps(res, indent=1))
(OUT / "scenario_refs_check.json").write_text(json.dumps(res, indent=2))
