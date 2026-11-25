"""Real-data yardsticks for each Layer-1 reference variant (same paper_config / Age_Group / 13 joint-TV
pairs as the shadow driver): score REAL 2013 splits as if they were a single realization (K=1):
  test_split : the production reference itself (testdata_pers.csv) -> what a perfect model of the
               population scores against the variant (pure composition/sampling gap)
  train_split: traindata_pers.csv (the 40k rows the priors were trained on) -> a pure memoriser
  val_split  : valdata_pers.csv (10k held-out rows)
Invalid mass is not computed (reference-independent; real data ~0)."""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[3])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import os, sys, glob
import pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{HERE}/code")
import rerun_l1_paper_disjoint as L1P
import run_l1_disjoint as W
L1 = L1P.L1
D = _PA.R("/data/pums_states")
variants = sorted(d for d in os.listdir(f"{HERE}/runs") if os.path.isdir(f"{HERE}/runs/{d}"))
rows = []
cfg = L1P.paper_config()
for st in ["CA", "TX", "OH"]:
    real_sets = {k: L1P.add_age_group(pd.read_csv(f"{D}/{st}/{f}")) for k, f in
                 [("test_split", "testdata_pers.csv"), ("train_split", "traindata_pers.csv"), ("val_split", "valdata_pers.csv")]}
    for v in variants:
        ref = L1P.add_age_group(pd.read_csv(W.ref_for(v).format(state=st)))
        for k, df in real_sets.items():
            res = L1.run_layer1_ensemble_audit(ref, [df[[c for c in df.columns if c in ref.columns]]], cfg, [k])
            s = res.ensemble_summary
            rows.append(dict(state=st, variant=v, real_set=k, n_real_set=len(df), n_ref=len(ref),
                             marg=float(s["marginal_total_discrepancy_mean"]), dep=float(s["dependency_total_discrepancy_mean"]),
                             support_cov=float(s["ensemble_support_coverage"])))
            print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"{HERE}/results/l1_real_yardsticks.csv", index=False)
