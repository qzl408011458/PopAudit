"""Supplementary diagnostic (filter): truncated priors P_theta(. | valid).

For the three Layer-1-rejected priors (CTGAN, TVAE, Borysov-VAE) in CA/TX/OH, drop every record
that violates any of the seven R_hard rules (layer1.py default rule set, identical to the paper's
Supplementary S4.2 list) from each of the K=50 realizations, and write the filtered realizations
under ../ens/<tag>_filt_<ST>/generated_ensembles/ with the ORIGINAL file names (so the Layer-2
loader's sorted-glob realization numbering and the r* resolution are unchanged).

Also records, per realization, the removed fraction (overall and by rule) and, per prior-state,
the shift of the audited marginal shares (7 audit attributes, Age as Age_Group) from the
unfiltered to the filtered ensemble, plus the same shares in the 2013 held-out reference.
Rule set check: the 2013 train/val/test splits and the nine scenario references are also scored.
Nothing outside supplementary_experiments/truncated_priors/ is written.
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[3])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import os, sys, glob, json
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

ROOT = _PA.R("")
sys.path.insert(0, ROOT)
import layer1 as L1

S = _PA.R("/supplementary_experiments/truncated_priors")
FINAL = _PA.S("/hpo_v3/final")
STUDY = {"ctgan": "ctgan_{st}", "tvae": "tvae_ref_{st}", "borysov": "borysov_{st}"}   # run_audit_v3.PRIMARY
STATES = ["CA", "TX", "OH"]
RULES = L1.build_default_layer1_config().validity_rules
assert len(RULES) == 7, [r.__name__ for r in RULES]
AGE_BINS = [0, 18, 35, 50, 65, 200]; AGE_LABELS = ["0-17", "18-34", "35-49", "50-64", "65+"]
ATTRS = ["Age_Group", "Household_Size", "Number_of_children", "Number_of_cars_of_household",
         "Employment_status", "Household_Income_Class", "Commute_Mode"]


def ens_dir(study):
    d = f"{FINAL}/{study}/{study}_final_s42"
    if os.path.isdir(d):
        return d
    c = glob.glob(f"{FINAL}/{study}/*_final_s42")
    return c[0] if len(c) == 1 else d


def valid_mask(df):
    ok = pd.Series(True, index=df.index); per = {}
    for r in RULES:
        m = r(df).fillna(False).astype(bool); per[r.__name__] = float((~m).mean()); ok &= m
    return ok, per


def shares(df):
    d = df.copy()
    d["Age_Group"] = pd.cut(pd.to_numeric(d["Age"], errors="coerce"), bins=AGE_BINS, labels=AGE_LABELS, right=False).astype(object)
    out = {}
    for a in ATTRS:
        vc = d[a].astype(str).value_counts(normalize=True)
        for k, v in vc.items():
            out[f"{a}=={k}"] = float(v)
    return out


def one_file(args):
    fp, outdir = args
    df = pd.read_csv(fp)
    ok, per = valid_mask(df)
    filt = df[ok]
    filt.to_csv(f"{outdir}/{os.path.basename(fp)}", index=False)
    return dict(file=os.path.basename(fp), n=len(df), n_kept=int(ok.sum()), removed=float(1 - ok.mean()),
                **{f"rate_{k}": v for k, v in per.items()}), shares(df), shares(filt)


def main():
    os.makedirs(f"{S}/diag", exist_ok=True)
    # ---- (a) reference splits: the rule set must be violation-free on real data ----
    ref_rows = []
    for st in STATES:
        base = f"{ROOT}/data/pums_states/{st}"
        for name in ["traindata_pers.csv", "valdata_pers.csv", "testdata_pers.csv",
                     f"reference_2018_{st}.csv", f"reference_2021_{st}.csv", f"reference_2024_{st}.csv"]:
            df = pd.read_csv(f"{base}/{name}")
            ok, per = valid_mask(df)
            ref_rows.append(dict(state=st, split=name, n=len(df), n_invalid=int((~ok).sum()), **per))
    pd.DataFrame(ref_rows).to_csv(f"{S}/diag/reference_rule_violations.csv", index=False)
    print(pd.DataFrame(ref_rows)[["state", "split", "n", "n_invalid"]].to_string(index=False), flush=True)

    # ---- (b) filter the 9 rejected prior-state ensembles ----
    jobs, meta = [], []
    for tag, tpl in STUDY.items():
        for st in STATES:
            src = f"{ens_dir(tpl.format(st=st))}/generated_ensembles"
            files = sorted(glob.glob(f"{src}/final_generated_samples_*.csv"))
            assert len(files) == 50, (tag, st, len(files))
            out = f"{S}/ens/{tag}_filt_{st}/generated_ensembles"; os.makedirs(out, exist_ok=True)
            for fp in files:
                jobs.append((fp, out)); meta.append((tag, st))
    rows, sh_u, sh_f = [], {}, {}
    with ProcessPoolExecutor(max_workers=12) as ex:
        for (tag, st), (r, su, sf) in zip(meta, ex.map(one_file, jobs, chunksize=2)):
            rows.append(dict(model=tag, state=st, **r))
            sh_u.setdefault((tag, st), []).append(su); sh_f.setdefault((tag, st), []).append(sf)
    rr = pd.DataFrame(rows); rr.to_csv(f"{S}/diag/removal_per_realization.csv", index=False)
    rate_cols = [c for c in rr.columns if c.startswith("rate_")]
    summ = rr.groupby(["model", "state"]).agg(removed_mean=("removed", "mean"), removed_min=("removed", "min"),
                                              removed_max=("removed", "max"), n_kept_min=("n_kept", "min"),
                                              **{c: (c, "mean") for c in rate_cols}).reset_index()
    summ.to_csv(f"{S}/diag/removal_summary.csv", index=False)
    print(summ.to_string(index=False), flush=True)

    # ---- (c) marginal distortion: ensemble-mean shares, unfiltered vs filtered vs 2013 reference ----
    srows = []
    for (tag, st) in sh_u:
        U = pd.DataFrame(sh_u[(tag, st)]).fillna(0).mean(); F = pd.DataFrame(sh_f[(tag, st)]).fillna(0).mean()
        R = pd.Series(shares(pd.read_csv(f"{ROOT}/data/pums_states/{st}/testdata_pers.csv")))
        keys = sorted(set(U.index) | set(F.index) | set(R.index))
        for k in keys:
            u, f, r = float(U.get(k, 0)), float(F.get(k, 0)), float(R.get(k, 0))
            srows.append(dict(model=tag, state=st, attr=k.split("==")[0], cat=k.split("==", 1)[1],
                              share_unfilt=u, share_filt=f, share_ref2013=r, shift=f - u,
                              abserr_unfilt=abs(u - r), abserr_filt=abs(f - r)))
    sd = pd.DataFrame(srows); sd.to_csv(f"{S}/diag/share_shift_long.csv", index=False)
    tv = sd.groupby(["model", "state", "attr"]).apply(
        lambda g: pd.Series(dict(TV_filt_vs_unfilt=0.5 * g["shift"].abs().sum(),
                                 TV_unfilt_vs_ref=0.5 * g.abserr_unfilt.sum(),
                                 TV_filt_vs_ref=0.5 * g.abserr_filt.sum(),
                                 max_abs_shift=g["shift"].abs().max()))).reset_index()
    tv.to_csv(f"{S}/diag/share_shift_tv_by_attr.csv", index=False)
    print(tv.groupby(["model", "state"])[["TV_filt_vs_unfilt", "max_abs_shift", "TV_unfilt_vs_ref", "TV_filt_vs_ref"]]
            .agg({"TV_filt_vs_unfilt": "max", "max_abs_shift": "max", "TV_unfilt_vs_ref": "mean", "TV_filt_vs_ref": "mean"})
            .to_string(), flush=True)


if __name__ == "__main__":
    main()
