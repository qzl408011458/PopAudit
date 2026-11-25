"""Compare l2_fast outputs (runs/) with full runs of the copied Layer-2/Layer-3 scripts (runs_script/)."""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[2])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import glob, os, json
from pathlib import Path
import sys
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
# g1fix (2026-11-24): also compare D on the category union (shared production helper)
sys.path.insert(0, _PA.R("/main_experiments/audits_v3/PAPER_xstate_v3"))
from dA_union import dA_union_l3
rows = []
for d in sorted(glob.glob(str(HERE / "runs_script/*/layer2_*"))):
    if d.endswith(".relog") or not os.path.exists(f"{d}/layer2_per_realization_weights.csv"):
        continue
    v = Path(d).parent.name; c = Path(d).name
    f = HERE / "runs" / v / c
    a = pd.read_csv(f"{d}/layer2_per_realization_weights.csv"); b = pd.read_csv(f / "layer2_per_realization_weights.csv")
    sa = pd.read_csv(f"{d}/layer2_historical_sigma_summary.csv"); sb = pd.read_csv(f / "layer2_historical_sigma_summary.csv")
    r = dict(variant=v, case=c[7:], max_rel_loss=float((a.scenario_loss - b.scenario_loss).abs().max() / a.scenario_loss.abs().max()),
             max_sigma=float((sa.sigma - sb.sigma).abs().max()),
             rstar_same=int(a.scenario_loss.idxmin() == b.scenario_loss.idxmin()))
    l3a = Path(str(d).replace("layer2_", "layer3_paperW_")); l3b = HERE / "runs" / v / l3a.name
    if (l3a / "layer3_target_summary.csv").exists() and (l3b / "layer3_target_summary.csv").exists():
        ta = pd.read_csv(l3a / "layer3_target_summary.csv"); tb = pd.read_csv(l3b / "layer3_target_summary.csv")
        r["D_script"] = 0.5 * ta.projected_abs_err_mean.sum(); r["D_fast"] = 0.5 * tb.projected_abs_err_mean.sum()
        st, y = c.split("_")[-2:]
        r["Dunion_script"] = dA_union_l3(l3a, st, y)["d_A"]; r["Dunion_fast"] = dA_union_l3(l3b, st, y)["d_A"]
    rows.append(r)
df = pd.DataFrame(rows); pd.set_option("display.width", 200)
print(df.to_string(index=False))
df.to_csv(HERE / "results/verify_fast_vs_script.csv", index=False)
