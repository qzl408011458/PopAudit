"""Projection-EFFECT ablation (within-cell differenced; base-year quality cancels).

From each projected_weights file we MC-sample TWO populations over the 7 audit attrs:
  - PRIOR      ~ base_weight       (the L1-passing prior, un-projected)
  - PROJECTED  ~ projected_weight  (after Layer-3 force-projection onto the scenario)
Both compared to the PWGTP-resampled realized horizon population. We then study the
PROJECTION EFFECT deltas (projected - prior) as a function of Layer-2 (in)compatibility:
  dU = U_proj - U_prior   (univariate: projection should IMPROVE -> dU<0)
  dJ = J_proj - J_prior   (full joint: does projection DEGRADE it? dJ>0 = degraded)
  dRecall = R_proj - R_prior
Hypothesis (L1->L3 without L2 is deceptive): for L2-incompatible (high chi_H) cells the
projection buys marginal fit (dU<<0) by DISTORTING the joint (dJ>0), i.e. it produces a
population that looks calibrated on the margins but whose dependency structure is worse.
"""
import os, sys, io, glob, time, contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ablation_lib as A

_REF = {}
def _ref(s, y):
    if (s, y) not in _REF: _REF[(s, y)] = A.load_reference(s, y)
    return _REF[(s, y)]

def _sample_by(pw_df, col, n, seed):
    w = pd.to_numeric(pw_df[col], errors="coerce").fillna(0).to_numpy(float); w = w / w.sum()
    idx = np.random.default_rng(seed).choice(len(pw_df), size=n, replace=True, p=w)
    return pw_df.iloc[idx][A.ATTRS].reset_index(drop=True)

def do_cell(row):
    r = pd.Series(row)
    files = A._pw_files(r.model, r.state, str(r.year))[:A.MAX_REAL]
    if not files: return []
    ref = _ref(r.state, str(r.year)); recs = []
    for ri, f in enumerate(files):
        pw = pd.read_csv(f)
        for s in range(A.K_SEEDS):
            seed = 1000 * ri + s
            prior = _sample_by(pw, "base_weight", A.N_POP, seed)
            proj = _sample_by(pw, "projected_weight", A.N_POP, seed)
            with contextlib.redirect_stdout(io.StringIO()):
                mp = A.metrics_one(ref, prior); mq = A.metrics_one(ref, proj)
            recs.append(dict(model=r.model, state=r.state, year=int(r.year), realization=ri, seed=s,
                             U_prior=mp["SRMSE_U"], J_prior=mp["SRMSE_J"], R_prior=mp["Recall"], F_prior=mp["F1"],
                             U_proj=mq["SRMSE_U"], J_proj=mq["SRMSE_J"], R_proj=mq["Recall"], F_proj=mq["F1"]))
    return recs

def main():
    cells = A.list_l1pass_cells()
    print(f"[{time.strftime('%T')}] projection-effect over {len(cells)} cells (prior vs projected)")
    allrecs = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(do_cell, row) for row in cells.to_dict("records")]
        for i, fu in enumerate(as_completed(futs)):
            allrecs.extend(fu.result())
            if (i + 1) % 6 == 0: print(f"[{time.strftime('%T')}] {i+1}/{len(cells)} cells", flush=True)
    raw = pd.DataFrame(allrecs); raw.to_csv(f"{HERE}/projeffect_raw.csv", index=False)
    for c in ["U", "J", "R", "F"]:
        raw[f"d{c}"] = raw[f"{c}_proj"] - raw[f"{c}_prior"]
    agg = raw.groupby(["state", "model", "year"]).mean(numeric_only=True).reset_index()
    out = cells.merge(agg, on=["state", "model", "year"], how="left")
    out.to_csv(f"{HERE}/projeffect_results.csv", index=False)
    print(f"[{time.strftime('%T')}] wrote projeffect_results.csv ({len(out)} cells)")

if __name__ == "__main__":
    main()
