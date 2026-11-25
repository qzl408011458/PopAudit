"""Loosening check (tau_a > 1) for the 10 VALID 2024 cases (g1fix, 2026-11-24): D (union, dA_union.py), nESS, KL
from runs_loose/ (tau_loose.py / launch_loose.sh).  As tau grows the projection becomes the identity and D tends to
the unprojected discrepancy of the selected realization r*.  Writes tau_loose_all.csv."""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[2])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import sys, json, pandas as pd
from pathlib import Path
H = Path(__file__).resolve().parent; AUD = Path(_PA.R("/main_experiments/audits_v3"))
sys.path.insert(0, str(AUD/"PAPER_xstate_v3")); from dA_union import dA_union_l3
rows = []
for cd in sorted((H/"runs_loose").iterdir()):
    m, st, yr = cd.name.rsplit("_", 2)
    for td in sorted(cd.glob("tau_*")):
        if not td.is_dir(): continue
        j = json.load(open(td/"layer3_summary.json")); feas = float(j.get("feasibility_rate", 0)) >= 0.999
        u = dA_union_l3(td, st, yr)
        rows.append(dict(cell=cd.name, state=st, model=m, tau=float(td.name[4:]), feasible=feas, D=u["d_A"],
                         nESS=j.get("mean_agent_nESS"), KL=j.get("mean_distortion")))
df = pd.DataFrame(rows).sort_values(["cell","tau"]); df.to_csv(H/"tau_loose_all.csv", index=False)
print(df.round(4).to_string(index=False)); print(df.groupby("state").D.max())
print(df.loc[df.groupby(["state","tau"]).D.idxmin()][["state","tau","model","D"]].round(4).to_string(index=False))
