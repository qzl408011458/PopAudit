"""Run the Layer-2 predictive-validity ablation over all 48 Layer-1-pass cells.

For each cell: load projected weights (<=MAX_REAL realizations), MC-sample K_SEEDS
synthetic populations of N_POP agents, compare to a fixed PWGTP-resampled realized
test set, compute joint-order fidelity metrics. Parallel over cells (CPU only).
Outputs ablation_raw.csv (per eval) and ablation_results.csv (per-cell mean/std).
"""
import os, sys, io, glob, time, contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ablation_lib as A

_REFCACHE = {}


def _ref(state, year):
    k = (state, year)
    if k not in _REFCACHE:
        _REFCACHE[k] = A.load_reference(state, year)
    return _REFCACHE[k]


def do_cell(row):
    r = pd.Series(row)
    files = A._pw_files(r.model, r.state, str(r.year))[:A.MAX_REAL]
    if not files:
        return r.model, r.state, int(r.year), []
    ref = _ref(r.state, str(r.year))
    recs = []
    for ri, f in enumerate(files):
        pw = pd.read_csv(f)
        j_exact = A.srmse_j_exact(r.state, str(r.year), pw)   # 2026-11-24: exact S2 SRMSE_J
        for s in range(A.K_SEEDS):
            gen = A.sample_synth(pw, seed=1000 * ri + s)
            with contextlib.redirect_stdout(io.StringIO()):
                mm = A.metrics_one(ref, gen)
            mm["SRMSE_J_mc"], mm["SRMSE_J"] = mm["SRMSE_J"], j_exact
            mm.update(dict(model=r.model, state=r.state, year=int(r.year),
                           realization=ri, seed=s))
            recs.append(mm)
    return r.model, r.state, int(r.year), recs


def main():
    cells = A.list_l1pass_cells()
    print(f"[{time.strftime('%T')}] ablation over {len(cells)} L1-pass cells "
          f"(N={A.N_POP}, K={A.K_SEEDS}, <= {A.MAX_REAL} realizations)")
    rows = cells.to_dict("records")
    allrecs = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(do_cell, row): row for row in rows}
        for fu in as_completed(futs):
            m, s, y, recs = fu.result()
            allrecs.extend(recs)
            print(f"[{time.strftime('%T')}] done {s} {m} {y}: {len(recs)} evals", flush=True)

    raw = pd.DataFrame(allrecs)
    raw.to_csv(f"{HERE}/ablation_raw.csv", index=False)

    metcols = ["SRMSE_U", "SRMSE_B", "SRMSE_M", "SRMSE_J", "Precision", "Recall", "F1"]
    agg = raw.groupby(["state", "model", "year"])[metcols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg = agg.reset_index()
    agg["n_evals"] = raw.groupby(["state", "model", "year"]).size().values

    out = cells.merge(agg, on=["state", "model", "year"], how="left")
    out.to_csv(f"{HERE}/ablation_results.csv", index=False)
    print(f"\n[{time.strftime('%T')}] wrote ablation_results.csv ({len(out)} cells)")
    show = out[["state", "model", "year", "L2pass", "L2_chi_H",
                "SRMSE_U_mean", "SRMSE_B_mean", "SRMSE_M_mean", "SRMSE_J_mean",
                "Recall_mean", "F1_mean"]].round(4)
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
