"""Projection-and-budget rule: the fourth reduced protocol of Table 4 (tab:baseline-counts) and Supplementary S2.

The rule skips the admission gates. In each state--horizon scenario it takes every candidate prior whose
Layer-3 projection is feasible, selects the one with the smallest post-projection residual
D = d_A(Q*, S_H) (over all categories, as in dA_union.py), and deploys it if D <= B0 = 0.30. Nothing is
re-solved: the main experiment projected all 81 cases, those rejected at Layer 1 or 2 as diagnostic stress
tests (Supplementary S3.3).

For every case the script also records the commute-mode work-from-home residual of the selected realization
before and after projection, in units of sigma_{a,c}(H) (the saturation of Proposition 4(b)), and the age
composition of the projected home workers against the realized one (the discrepancy of Supplementary
Table S5).

Writes projection_budget_cases.csv (81 rows), projection_budget_scenarios.csv (9 rows) and
projection_budget_summary.csv; run with
`python projection_budget_rule.py | tee projection_budget_rule.log`.
Reads the main-experiment outputs (main_experiments/audits_v3) and the processed scenario references only.
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[2])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AUD = Path(_PA.R('/main_experiments/audits_v3'))
V3 = AUD / 'PAPER_xstate_v3'
sys.path.insert(0, str(V3))
from dA_union import add_age_group, dA_union_l3, load_projected, reference_path  # noqa: E402

OUT = Path(__file__).resolve().parent
B0 = 0.30
TAU = 1.0
WFH = 'Worked_at_home'
AGES = ['0-17', '18-34', '35-49', '50-64', '65+']
STATES = ('CA', 'OH', 'TX')
YEARS = (2018, 2021, 2024)

cert = pd.read_csv(V3 / 'xstate_certificate_all.csv')
cert = cert[cert.year.notna()].copy()
cert['year'] = cert.year.astype(int)
assert len(cert) == 81, len(cert)
cert['first_fail'] = cert.first_fail.fillna('').astype(str)
cert['feasible'] = cert.L3_feasible.astype(float) == 1.0


def realized_wfh_age(st, y):
    ref = add_age_group(pd.read_csv(reference_path(st, y), usecols=lambda c: c in {'Age', 'Commute_Mode', 'PWGTP'},
                                    low_memory=False))
    w = pd.to_numeric(ref.PWGTP, errors='coerce').fillna(0).to_numpy(float)
    m = (ref.Commute_Mode.astype(str) == WFH).to_numpy()
    prof = pd.Series(w[m]).groupby(ref.Age_Group.astype(str).to_numpy()[m]).sum()
    return (prof / prof.sum()).reindex(AGES, fill_value=0.0)


REAL = {(st, y): realized_wfh_age(st, y) for st in STATES for y in YEARS}

rows = []
for r in cert.itertuples():
    l3 = AUD / f'layer3_paperW_{r.model}_{r.state}_{r.year}'
    dres = dA_union_l3(l3, r.state, r.year)          # D over all categories; cross-checked against Layer 3
    t = pd.read_csv(l3 / 'layer3_target_summary.csv')
    t['z_pre'] = (t.prior_mean - t.target_m) / t.sigma
    t['z_post'] = (t.projected_mean - t.target_m) / t.sigma
    # attribute-wise chi_{H,a} after projection, over the categories that Layers 2-3 standardize
    chi_post = t.groupby('attr').apply(lambda g: float((g.z_post ** 2).sum() / (2 * len(g))), include_groups=False)
    w = t[t.target == f'Commute_Mode=={WFH}'].iloc[0]
    Q, q = load_projected(l3)
    m = (Q.Commute_Mode.astype(str) == WFH).to_numpy()
    prof = pd.Series(q[m]).groupby(Q.Age_Group.astype(str).to_numpy()[m]).sum()
    prof = (prof / prof.sum()).reindex(AGES, fill_value=0.0)
    real = REAL[(r.state, r.year)]
    rows.append(dict(
        state=r.state, model=r.model, year=r.year, first_fail=r.first_fail or 'VALID', VALID=bool(r.VALID),
        feasible=bool(r.feasible), L1_invalid=r.L1_invalid, chi_bar=r.L2_chi_H, D=dres['d_A'],
        within_budget=bool(r.feasible and dres['d_A'] <= B0),
        n_commute_targets=int((t.attr == 'Commute_Mode').sum()), chi_commute_post=chi_post['Commute_Mode'],
        n_binding=int((chi_post >= TAU - 1e-3).sum()),
        wfh_target=w.target_m, wfh_sigma=w.sigma, wfh_pre=w.prior_mean, wfh_post=w.projected_mean,
        z_wfh_pre=w.z_pre, z_wfh_post=w.z_post,
        wfh_age_disc=0.5 * float(np.abs(prof - real).sum()),
        proj_wfh_18_34=prof['18-34'], real_wfh_18_34=real['18-34']))
cases = pd.DataFrame(rows)
cases.to_csv(OUT / 'projection_budget_cases.csv', index=False)

# ---- scenario decisions: audit (admission, then budget) and projection-and-budget rule ----
srows = []
for st in STATES:
    for y in YEARS:
        c = cases[(cases.state == st) & (cases.year == y)]
        adm = c[c.VALID]
        if len(adm):
            a = adm.loc[adm.D.idxmin()]
            audit = dict(audit_prior=a.model, audit_D=a.D, audit_decision='deploy' if a.D <= B0 else 'defer: over budget')
        else:
            audit = dict(audit_prior='', audit_D=np.nan, audit_decision='defer: no admissible prior')
        b = c[c.feasible].loc[c[c.feasible].D.idxmin()]
        srows.append(dict(state=st, year=y, **audit, rule_prior=b.model, rule_D=b.D, rule_first_fail=b.first_fail,
                          rule_decision='deploy' if b.D <= B0 else 'defer: over budget',
                          rule_wfh_post=b.wfh_post, wfh_realized=b.wfh_target, rule_wfh_age_disc=b.wfh_age_disc,
                          rule_proj_wfh_18_34=b.proj_wfh_18_34, real_wfh_18_34=b.real_wfh_18_34))
scen = pd.DataFrame(srows)
scen.to_csv(OUT / 'projection_budget_scenarios.csv', index=False)

# the audit rows must reproduce the main-text decisions (Table 4)
TAB4 = {('CA', 2018): ('semapop_vae', 0.234), ('OH', 2018): ('semapop_vae', 0.147), ('TX', 2018): ('ttvae', 0.142),
        ('CA', 2024): ('semapop_vae', 0.321), ('OH', 2024): ('semapop_vae', 0.213), ('TX', 2024): ('semapop_vae', 0.238)}
for (st, y), (mod, dd) in TAB4.items():
    s = scen[(scen.state == st) & (scen.year == y)].iloc[0]
    assert s.audit_prior == mod and abs(s.audit_D - dd) < 5e-4, (st, y, s.audit_prior, s.audit_D)
assert scen[scen.year == 2021].audit_decision.eq('defer: no admissible prior').all()

# ---- summary ----
acc = cases[cases.within_budget]
summ = (acc.assign(group=acc.first_fail.where(acc.first_fail.isin(['VALID', 'L2']), 'L1'))
        .groupby(['group', 'year']).size().unstack(fill_value=0))
summ['total'] = summ.sum(axis=1)
summ.loc['all'] = summ.sum()
summ.to_csv(OUT / 'projection_budget_summary.csv')

pd.set_option('display.width', 220)
print(f'feasible projections: {int(cases.feasible.sum())} of 81; infeasible:',
      cases[~cases.feasible][['state', 'model', 'year']].to_dict('records'))
print(f'\ncases with a feasible projection and D <= B0 = {B0}: {len(acc)} of 81')
print(summ.to_string())
print('L1 rejections within budget:', acc[acc.first_fail.str.startswith('L1')][['state', 'model', 'year', 'L1_invalid', 'D']]
      .round(4).to_string(index=False), sep='\n')
print('\nscenario decisions')
print(scen[['state', 'year', 'audit_prior', 'audit_D', 'audit_decision', 'rule_prior', 'rule_D', 'rule_first_fail',
            'rule_decision']].round(4).to_string(index=False))
print('scenarios deployed: audit', int((scen.audit_decision == 'deploy').sum()), 'of 9; rule',
      int((scen.rule_decision == 'deploy').sum()), 'of 9')
same = scen[scen.year != 2021]
print('same selected prior and decision in 2018/2024:',
      bool((same.audit_prior == same.rule_prior).all() and (same.audit_decision == same.rule_decision).all()))
dep = scen[scen.audit_decision == 'deploy']
print(f'2021 residuals under the rule: {scen[scen.year == 2021].rule_D.min():.3f} to {scen[scen.year == 2021].rule_D.max():.3f};'
      f' deployed 2018/2024 residuals: {dep.audit_D.min():.3f} to {dep.audit_D.max():.3f}')

print('\nwork-from-home saturation (Layer-1-passing feasible cases)')
l1p = cases[cases.feasible & ~cases.first_fail.str.startswith('L1')]
for y in YEARS:
    c = l1p[l1p.year == y]
    print(f'  {y}: n={len(c)}, commute binding in {int((c.chi_commute_post >= TAU - 1e-3).sum())}, '
          f'z_pre {c.z_wfh_pre.min():.2f} to {c.z_wfh_pre.max():.2f}, z_post {c.z_wfh_post.min():.2f} to {c.z_wfh_post.max():.2f}, '
          f'sigma {sorted(c.wfh_sigma.unique())}, commute targets {sorted(c.n_commute_targets.unique())}')
for (st, y), c in l1p[l1p.year == 2021].groupby(['state', 'year']):
    print(f'  {st} {y}: projected WFH share {c.wfh_post.min():.4f} to {c.wfh_post.max():.4f}, realized {c.wfh_target.iloc[0]:.4f}')
print(f'  single-category limit sqrt(2*13*tau) = {np.sqrt(2 * 13 * TAU):.3f}')

print('\nage composition of the projected home workers (discrepancy from the realized composition)')
print(scen[['state', 'year', 'rule_prior', 'rule_wfh_age_disc', 'rule_proj_wfh_18_34', 'real_wfh_18_34']]
      .round(3).to_string(index=False))

print('\nwrote', ', '.join(p.name for p in sorted(OUT.glob('projection_budget_*.csv'))))
