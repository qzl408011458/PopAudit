"""No-L2 performance table on the ENSEMBLE population (pool all projected-weight
realizations = the mixture the framework actually deploys), which removes the
which-single-draw arbitrariness that makes per-realization recall/precision unstable.

For each of the 48 L1-pass cells: pool all realization projected_weights into one
mixture (weight = projected_weight / n_realizations), MC-sample N agents for K seeds,
compare to the PWGTP-resampled realized test set. Report mean +/- seed-std."""
import os, sys, io, glob, time, contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ablation_lib as A

K = 5
_REF = {}
def _ref(s, y):
    if (s, y) not in _REF: _REF[(s, y)] = A.load_reference(s, y)
    return _REF[(s, y)]

def do_cell(row):
    r = pd.Series(row)
    files = A._pw_files(r.model, r.state, str(r.year))
    if not files: return None
    parts = []
    for f in files:
        t = pd.read_csv(f)[A.ATTRS + ["projected_weight"]].copy()
        t["projected_weight"] = t.projected_weight / t.projected_weight.sum() / len(files)
        parts.append(t)
    ens = pd.concat(parts, ignore_index=True)   # ensemble mixture (weights sum to 1)
    ref = _ref(r.state, str(r.year))
    accs = []
    for s in range(K):
        gen = A.sample_synth(ens, seed=s)
        with contextlib.redirect_stdout(io.StringIO()):
            accs.append(A.metrics_one(ref, gen))
    df = pd.DataFrame(accs)
    out = {f"{c}": df[c].mean() for c in df.columns}
    out.update({f"{c}_sd": df[c].std() for c in ["SRMSE_J", "Recall", "F1"]})
    # 2026-11-24: reported SRMSE_J is the exact S2 quantity on Q* and the PWGTP-weighted reference
    out["SRMSE_J_mc"], out["SRMSE_J_mc_sd"] = out["SRMSE_J"], out.pop("SRMSE_J_sd")
    out["SRMSE_J"] = A.srmse_j_exact(r.state, str(r.year), ens)
    out.update(dict(state=r.state, model=r.model, year=int(r.year)))
    return out

def main():
    cells = A.list_l1pass_cells()
    print(f"[{time.strftime('%T')}] ENSEMBLE table over {len(cells)} cells (K={K} seeds)")
    recs = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(do_cell, row) for row in cells.to_dict("records")]
        for i, fu in enumerate(as_completed(futs)):
            r = fu.result()
            if r: recs.append(r)
            if (i + 1) % 6 == 0: print(f"[{time.strftime('%T')}] {i+1}/{len(cells)}", flush=True)
    out = cells.merge(pd.DataFrame(recs), on=["state", "model", "year"], how="left")
    out.to_csv(f"{HERE}/ensemble_results.csv", index=False)
    print(f"[{time.strftime('%T')}] wrote ensemble_results.csv")

if __name__ == "__main__":
    main()
