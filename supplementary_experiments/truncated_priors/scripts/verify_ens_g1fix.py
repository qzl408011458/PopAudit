"""g1fix check (2026-11-24) for the truncated-prior diagnostic.

(a) Re-scores the 2013 splits and the REGENERATED 2018/2021/2024 references (JWTRNS crosswalk,
    constant-2013-dollar income) against the 7 R_hard rules -> diag/reference_rule_violations.csv
    (same columns as filter_ensembles.py step (a)).
(b) Verifies, without rewriting anything, that every filtered realization under ens/ is exactly
    df[valid] of the CURRENT original realization (same rows, same values, same order), and that
    diag/removal_per_realization.csv still matches -> diag/ens_verification_g1fix.csv.
Filtering depends only on the unchanged 2013 rule set and the unchanged ensembles, so (b) must pass.
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[3])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import os, sys, glob
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

S = _PA.R("/supplementary_experiments/truncated_priors")
sys.path.insert(0, f"{S}/scripts")
import filter_ensembles as FE   # RULES, valid_mask, ens_dir, STUDY, STATES (no side effects on import)

ROOT = FE.ROOT


def check_file(args):
    tag, st, fp = args
    df = pd.read_csv(fp)
    ok, per = FE.valid_mask(df)
    exp = df[ok].reset_index(drop=True)
    ffp = f"{S}/ens/{tag}_filt_{st}/generated_ensembles/{os.path.basename(fp)}"
    got = pd.read_csv(ffp)
    same_cols = list(got.columns) == list(exp.columns)
    identical = bool(same_cols and len(got) == len(exp) and got.equals(exp.astype(got.dtypes.to_dict())))
    return dict(model=tag, state=st, file=os.path.basename(fp), n=len(df), n_kept_recomputed=int(ok.sum()),
                n_rows_filtered_file=len(got), removed_recomputed=float(1 - ok.mean()), identical=identical,
                residual_violations=int((~FE.valid_mask(got)[0]).sum()))


def main():
    # (a) reference rule check on the regenerated references
    rows = []
    for st in FE.STATES:
        base = f"{ROOT}/data/pums_states/{st}"
        for name in ["traindata_pers.csv", "valdata_pers.csv", "testdata_pers.csv",
                     f"reference_2018_{st}.csv", f"reference_2021_{st}.csv", f"reference_2024_{st}.csv"]:
            df = pd.read_csv(f"{base}/{name}")
            ok, per = FE.valid_mask(df)
            rows.append(dict(state=st, split=name, n=len(df), n_invalid=int((~ok).sum()), **per))
    ref = pd.DataFrame(rows); ref.to_csv(f"{S}/diag/reference_rule_violations.csv", index=False)
    print(ref[["state", "split", "n", "n_invalid"]].to_string(index=False), flush=True)
    print(f"total records {ref.n.sum()}, total violations {ref.n_invalid.sum()}", flush=True)

    # (b) ens/ verification
    jobs = []
    for tag, tpl in FE.STUDY.items():
        for st in FE.STATES:
            src = f"{FE.ens_dir(tpl.format(st=st))}/generated_ensembles"
            files = sorted(glob.glob(f"{src}/final_generated_samples_*.csv")); assert len(files) == 50
            assert len(glob.glob(f"{S}/ens/{tag}_filt_{st}/generated_ensembles/*.csv")) == 50
            jobs += [(tag, st, fp) for fp in files]
    with ProcessPoolExecutor(max_workers=12) as ex:
        res = pd.DataFrame(list(ex.map(check_file, jobs, chunksize=2)))
    old = pd.read_csv(f"{S}/diag/removal_per_realization.csv")
    m = res.merge(old[["model", "state", "file", "n_kept", "removed"]], on=["model", "state", "file"], how="left")
    m["matches_old_diag"] = (m.n_kept == m.n_kept_recomputed) & np.isclose(m.removed, m.removed_recomputed, atol=1e-12)
    m.to_csv(f"{S}/diag/ens_verification_g1fix.csv", index=False)
    print(m.groupby(["model", "state"]).agg(files=("file", "size"), identical=("identical", "sum"),
                                            diag_match=("matches_old_diag", "sum"),
                                            resid=("residual_violations", "sum"),
                                            removed_mean=("removed_recomputed", "mean")).to_string(), flush=True)
    allok = bool(m.identical.all() and m.matches_old_diag.all() and (m.residual_violations == 0).all())
    print("ALL ENS VERIFIED:", allok, flush=True)


if __name__ == "__main__":
    main()
