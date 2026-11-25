"""Design forensics step 1: try to reproduce data/pums/person_level_merged_compact_2013.csv
bit-for-bit from the national non-resampled 2013 table using _pums_proc.weighted_resample_by_pwgtp
logic (rng = default_rng(42); choice(arange(N), size=N, replace=True, p=PWGTP/sum)).

Read-only on data/. Writes a small JSON summary to ../out/.
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[5])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(_PA.R(""))
OUT = Path(_PA.R("/experiments/E_state/supplementary/iid_test/out"))

nat = pd.read_csv(ROOT / "data/pums_historical/person_level_merged_compact_pwgtp_2013.csv",
                  usecols=["Person_ID", "PWGTP", "State_Code"], dtype={"Person_ID": str})
res = pd.read_csv(ROOT / "data/pums/person_level_merged_compact_2013.csv",
                  usecols=["Sampled_Person_ID", "Source_Person_ID", "PWGTP", "State_Code"],
                  dtype={"Source_Person_ID": str})
summ = {}
summ["n_national_unweighted"] = int(len(nat))
summ["n_resampled"] = int(len(res))
summ["person_id_unique_national"] = bool(nat["Person_ID"].is_unique)
summ["n_pwgtp_le0"] = int((nat["PWGTP"] <= 0).sum())
summ["sampled_id_is_0_to_n"] = bool((res["Sampled_Person_ID"].to_numpy() == np.arange(len(res))).all())
summ["source_ids_subset_of_national"] = bool(res["Source_Person_ID"].isin(set(nat["Person_ID"])).all())

work = nat[nat["PWGTP"] > 0].reset_index(drop=True)
p = work["PWGTP"].to_numpy(float)
p = p / p.sum()
rng = np.random.default_rng(42)
pos = rng.choice(np.arange(len(work)), size=len(work), replace=True, p=p)
rep_ids = work["Person_ID"].to_numpy()[pos]
match = rep_ids == res["Source_Person_ID"].to_numpy() if len(rep_ids) == len(res) else None
summ["exact_reproduction_seed42_rowwise_match_frac"] = float(match.mean()) if match is not None else None
summ["exact_reproduction_all_rows"] = bool(match.all()) if match is not None else False
# PWGTP and state consistent too?
if match is not None and match.all():
    summ["pwgtp_match"] = bool((work["PWGTP"].to_numpy()[pos] == res["PWGTP"].to_numpy()).all())
    summ["state_match"] = bool((work["State_Code"].to_numpy()[pos] == res["State_Code"].to_numpy()).all())
print(json.dumps(summ, indent=2))
(OUT / "reproduce_resample.json").write_text(json.dumps(summ, indent=2))
