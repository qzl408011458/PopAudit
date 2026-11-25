"""[v3 port 2026-09-02: exemplars tvae_llm+/TX -> semapop_vae/TX, ttvae/CA unchanged]
Probe: does force-projection restore the STRUCTURE INSIDE commute mode (who works
from home), or only the commute marginal? Compare the projected population's
commute-conditional composition P(X | commute) to the realized one, for a strong
prior across 2018 (L2 pass) / 2021 (L2 fail, COVID) / 2024, and against the prior's
own commute composition. Signal we expect if L2's warning is real: at 2021 the WFH
share is matched but P(X|WFH) stays near the 2013 prior and far from realized."""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[4])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import sys, glob
import numpy as np, pandas as pd
HERE = _PA.R("/main_experiments/audits_v3/PAPER_xstate_v3/ablation_l2")
sys.path.insert(0, HERE)
import ablation_lib as A

AUD = A.AUD
COND = ["Employment_status", "Age_Group", "Household_Income_Class"]
WFH = "Worked_at_home"; CAR = "Car_truck_or_van"

def add_ag(df):
    if "Age_Group" not in df.columns:
        df["Age_Group"] = pd.cut(pd.to_numeric(df.Age, errors="coerce"), [0,18,35,50,65,200],
            labels=["0-17","18-34","35-49","50-64","65+"], right=False, include_lowest=True).astype(str)
    return df

def wdist(vals, w, cats):
    s = pd.Series(w).groupby(pd.Series(vals).values).sum()
    s = s.reindex(cats).fillna(0.0); tot = s.sum()
    return (s/tot).values if tot > 0 else s.values

def tvd(p, q): return 0.5*np.abs(p-q).sum()

def cond_on(df, w, mode, X, cats):
    m = (df.Commute_Mode.astype(str) == mode).values
    if m.sum() == 0 or w[m].sum() == 0: return None
    return wdist(df[X].astype(str).values[m], w[m], cats)

def ens_weights(model, state, year):
    fs = A._pw_files(model, state, str(year))
    parts = []
    for f in fs:
        t = add_ag(pd.read_csv(f))
        parts.append(t)
    ens = pd.concat(parts, ignore_index=True)
    return ens

for model, state in [("semapop_vae", "TX"), ("ttvae", "CA")]:
    print("="*88); print(f"{A.__dict__};" if False else f"MODEL={model}  STATE={state}"); print("="*88)
    for year, tag in [(2018,"L2-PASS"),(2021,"L2-FAIL COVID"),(2024,"")]:
        ens = ens_weights(model, state, year)
        ref = add_ag(pd.read_csv(f"{_PA.DATA}/pums_states/{state}/reference_{year}_{state}.csv"))
        pw = pd.to_numeric(ref.PWGTP, errors="coerce").fillna(0).to_numpy(float)
        wq = ens.projected_weight.to_numpy(float); wq/=wq.sum()   # projected
        wb = ens.base_weight.to_numpy(float); wb/=wb.sum()        # prior
        # WFH marginal (should be ~matched by projection)
        wfh_q = wq[(ens.Commute_Mode.astype(str)==WFH).values].sum()
        wfh_r = pw[(ref.Commute_Mode.astype(str)==WFH).values].sum()/pw.sum()
        wfh_b = wb[(ens.Commute_Mode.astype(str)==WFH).values].sum()
        # within-WFH nESS (how concentrated the projected WFH mass is)
        mq = wq[(ens.Commute_Mode.astype(str)==WFH).values]; mq=mq/mq.sum()
        ness_wfh = 1.0/np.sum(mq**2)/len(mq)
        print(f"\n  {year} [{tag}]  WFH share: real={wfh_r:.3f} proj={wfh_q:.3f} prior={wfh_b:.3f}  | within-WFH nESS(proj)={ness_wfh:.3f}")
        for X in COND:
            cats = sorted(set(ref[X].astype(str))|set(ens[X].astype(str)))
            pr = cond_on(ref, pw, WFH, X, cats)
            pq = cond_on(ens, wq, WFH, X, cats)
            pb = cond_on(ens, wb, WFH, X, cats)
            if pr is None or pq is None: continue
            print(f"    P({X:22s}|WFH):  TVD(proj,real)={tvd(pq,pr):.3f}   TVD(prior,real)={tvd(pb,pr):.3f}   TVD(proj,prior)={tvd(pq,pb):.3f}")
