"""'Scenario-only (Layer 2 alone)' row of Table 5 (tab:baseline-counts) and Supplementary S2.
Read-only on production outputs. Admit a case iff L2_accept, ignoring Layer 1 (and Layer 3)."""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[2])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import pandas as pd
from pathlib import Path
ROOT = Path(_PA.R('/main_experiments/audits_v3'))
OUT = Path(__file__).resolve().parent
d = pd.read_csv(ROOT/'PAPER_xstate_v3/xstate_certificate_all.csv')
b = pd.read_csv(ROOT/'PAPER_xstate_v3/baselines/baseline_comparison.csv')
assert len(d) == 81
# cross-check chi_H against per-case layer2 summaries
for _, r in d.iterrows():
    s = pd.read_csv(ROOT/f'layer2_{r.model}_{r.state}_{r.year}/layer2_summary.csv')
    assert abs(s.prior_mean_scenario_loss.iloc[0] - r.L2_chi_H) < 1e-4, (r.state, r.model, r.year)
d = d.merge(b[['state','model','year','srmse_marg']], on=['state','model','year'])
protocols = {
    'full_audit': d.VALID,
    'srmse_0.30': d.srmse_marg <= 0.30,
    'structural_only_L1': d.L1_accept,
    'scenario_only_L2': d.L2_accept,
    'L1_and_L2_no_L3': d.L1_accept & d.L2_accept,
}
rows = []
for name, acc in protocols.items():
    rows.append(dict(protocol=name, accepted=int(acc.sum()),
        **{f'acc_{y}': int(acc[d.year == y].sum()) for y in (2018, 2021, 2024)},
        agree=int((acc == d.VALID).sum()),
        valid_rejected=int((d.VALID & ~acc).sum()),
        invalid_accepted=int((~d.VALID & acc).sum())))
tab = pd.DataFrame(rows)
tab.to_csv(OUT/'a27_protocol_summary.csv', index=False)
print(tab.to_string(index=False))
l1rej = d[~d.L1_accept].copy()
l1rej[['state','model','year','L1_invalid','L2_chi_H','L2_cutoff','L2_accept','L3_accept']].to_csv(OUT/'a27_l1_rejections_l2.csv', index=False)
print('\nL1 rejections passing L2:', int(l1rej.L2_accept.sum()), 'of', len(l1rej))
print(pd.crosstab(l1rej.model, l1rej.year, values=l1rej.L2_accept, aggfunc='sum', margins=True))
print(l1rej[l1rej.L2_accept][['state','model','year','L1_invalid','L2_chi_H','L2_cutoff']].to_string(index=False))
print('\nL1 rejections by model/year count:'); print(pd.crosstab(l1rej.model, l1rej.year))
print('\nL1-pass but L2-fail (31):'); print(pd.crosstab(d[d.L1_accept & ~d.L2_accept].model, d[d.L1_accept & ~d.L2_accept].year, margins=True))
