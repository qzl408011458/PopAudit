"""Validation of the single-realization path: feed the production r* realization of two real
cases through run_oracle.do_case and compare with the production Layer-2 row of r* and the
production Layer-3 / D of that case.  Also checks that sigma(H) of every oracle run equals the
production sigma(H) of the same scenario."""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[2])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import sys, json, shutil
from pathlib import Path
import pandas as pd, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, _PA.R("/main_experiments"))
sys.path.insert(0, _PA.R("/main_experiments/audits_v3/PAPER_xstate_v3"))
import run_oracle as R
from rerun_audit_new import resolve_rstar_file
from run_audit_v3 import PRIMARY, ens_dir
from dA_union import dA_union_l3
AUD = Path(_PA.R("/main_experiments/audits_v3"))
out = []
for model, st, y in [("semapop_vae", "CA", "2024"), ("ttvae", "OH", "2021"), ("bn", "TX", "2018")]:
    ed = Path(ens_dir(PRIMARY[model].format(st=st))) / "generated_ensembles"
    rstar, f = resolve_rstar_file(AUD / f"layer2_{model}_{st}_{y}", ed)
    var = f"chk{model}"
    d = R.OUT / "ens" / f"{var}_{st}"; d.mkdir(parents=True, exist_ok=True)
    shutil.copy(f, d / "oracle_2013_0.csv")
    print(R.do_case(var, st, y))
    w = pd.read_csv(AUD / f"layer2_{model}_{st}_{y}/layer2_per_realization_weights.csv")
    prod_loss = float(w.loc[w.realization_id == rstar, "scenario_loss"].iloc[0])
    mine = json.load(open(R.OUT / f"runs/layer2_{var}_{st}_{y}/layer2_summary.json"))["prior_mean_scenario_loss"]
    p3 = json.load(open(AUD / f"layer3_paperW_{model}_{st}_{y}/layer3_summary.json"))
    m3 = json.load(open(R.OUT / f"runs/layer3_{var}_{st}_{y}/layer3_summary.json"))
    Dp = dA_union_l3(AUD / f"layer3_paperW_{model}_{st}_{y}", st, y)["d_A"]
    Dm = dA_union_l3(R.OUT / f"runs/layer3_{var}_{st}_{y}", st, y)["d_A"]
    out.append(dict(case=f"{model}_{st}_{y}", rstar=rstar, chi_prod_rstar=prod_loss, chi_oracle_path=mine,
                    KL_prod=p3["mean_distortion"], KL_path=m3["mean_distortion"],
                    nESS_prod=p3.get("mean_agent_nESS"), nESS_path=m3.get("mean_agent_nESS"),
                    D_prod=Dp, D_path=Dm))
v = pd.DataFrame(out); print(v.to_string()); v.to_csv(R.OUT / "validation_single_realization_path.csv", index=False)
# sigma(H) identity
bad = 0
for var in R.SRC:
    for st in R.STATES:
        for y in R.YEARS:
            a = pd.read_csv(R.OUT / f"runs/layer2_{var}_{st}_{y}/layer2_historical_sigma_summary.csv")
            b = pd.read_csv(AUD / f"layer2_semapop_vae_{st}_{y}/layer2_historical_sigma_summary.csv")
            ok = a.target.tolist() == b.target.tolist() and np.allclose(a.sigma, b.sigma, rtol=0, atol=1e-15)
            bad += not ok
print("sigma(H) identical to production in", 18 - bad, "/ 18 oracle runs")
