# Per-attribute residual D of the observed 2013 sample (full variant) and of SemaPop-VAE+, TTVAE and BN in
# each scenario -> a9_D_per_attribute.csv.  Run after run_oracle.py and collect.py.
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[2])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, _PA.R("/main_experiments/audits_v3/PAPER_xstate_v3"))
from dA_union import dA_union_l3, dA_union_case
OUT = Path(__file__).parent
ctx = pd.read_csv(OUT / "a9_production_context.csv", dtype={"year": str})
rows = []
for r in ctx.itertuples():
    o = dA_union_l3(OUT / f"runs/layer3_full_{r.state}_{r.year}", r.state, r.year)
    rows.append({"case": f"{r.state}{r.year}", "who": "oracle", **o["per_attr"], "D": o["d_A"]})
    for m in (["semapop_vae", "ttvae", "bn"] if True else []):
        p = dA_union_case(m, r.state, r.year)
        rows.append({"case": f"{r.state}{r.year}", "who": m, **p["per_attr"], "D": p["d_A"]})
df = pd.DataFrame(rows); df.to_csv(OUT / "a9_D_per_attribute.csv", index=False)
pd.set_option("display.width", 250)
print(df.round(4).to_string(index=False))
