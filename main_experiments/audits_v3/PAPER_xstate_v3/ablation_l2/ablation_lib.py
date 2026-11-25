"""
[v3 port 2026-09-02: audits_v3 / PAPER_xstate_v3 / 9 uniform tags; algorithms unchanged]
Layer-2 predictive-validity ablation — shared library.

Question: if we DROP Layer 2 and force-project every Layer-1-passing prior onto its
state-horizon scenario, does the resulting synthetic population's fidelity to the
REALIZED horizon data track Layer-2's (prospective) accept/reject verdict?

We take each cell's Layer-3 projected weights Q* (already computed for all 48
L1-pass cells), Monte-Carlo sample a full synthetic population of N agents, and
compare it to a PWGTP-resampled realized-population test set for the same
state-horizon using the SAME evals the paper's model-quality tables use
(utils.evals). Metrics span joint order:
  - SRMSE_U : 1-way (univariate) marginals            -> NEGATIVE CONTROL
              (Layer-3 constrains the 7 univariate marginals to a chi<=tau band,
               so this is pinned ~equal for BOTH L2-pass and L2-fail cells)
  - SRMSE_B : 5-of-7-way joints  (evaluate_bivar)      -> discriminator
  - SRMSE_M : 6-of-7-way joints  (evaluate_marg)       -> discriminator
  - SRMSE_J : full 7-way joint   (eval_travelattrs)    -> headline discriminator
  - P/R/F1  : joint support overlap (eval_zerocells_travel) -> support discriminators

Only computable because the realized future PUMS is held out -> exactly why the
prospective Layer-2 screen is needed when it is NOT.
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[4])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import os, sys, glob
import numpy as np
import pandas as pd

ROOT = _PA.R("")
sys.path.insert(0, ROOT)
from utils.evals import (transform_cat_columns, discretize,  # noqa: E402
                         evaluate_marg, evaluate_bivar, eval_travelattrs,
                         eval_zerocells_travel)

AUD = _PA.R("/main_experiments/audits_v3")
CERT = f"{AUD}/PAPER_xstate_v3/xstate_certificate_all.csv"

# 7 audit attributes = the Layer-3 projection target set (L3_ATTRS in rerun_l3_scalable.py)
NUM = ["Age", "Household_Size", "Number_of_children", "Number_of_cars_of_household"]
CAT = ["Employment_status", "Household_Income_Class", "Commute_Mode"]
ATTRS = NUM + CAT

N_POP = 50_000        # synthetic + reference population size per evaluation (comparable across all cells)
K_SEEDS = 8           # Monte-Carlo synthetic resample seeds
MAX_REAL = 3          # projected-weight realizations per cell to average over
REF_SEED = 20240703   # fixed reference-resample seed (test set held constant per state-year)


def list_l1pass_cells():
    """The 48 Layer-1-passing cells with the columns the ablation/analysis needs."""
    c = pd.read_csv(CERT)
    c["year"] = c.year.astype(int)
    l1 = c[c.L1_accept == True].copy()
    l1["L2pass"] = l1.L2_accept == True
    keep = ["state", "model", "year", "L2pass", "L1_marg", "L1_dep", "L1_support",
            "L1_invalid", "T1", "L2_chi_H", "L2_cutoff", "L3_nESS", "L3_KL", "R_2", "R_2_MA"]
    return l1[keep].sort_values(["state", "model", "year"]).reset_index(drop=True)


def _pw_files(model, state, year):
    d = f"{AUD}/layer3_paperW_{model}_{state}_{year}/projected_weights"
    return sorted(glob.glob(f"{d}/*.csv"))


def load_reference(state, year, n=N_POP, seed=REF_SEED):
    """PWGTP-resample the realized horizon PUMS to n agents (fixed seed) -> test set."""
    ref = pd.read_csv(f"{ROOT}/data/pums_states/{state}/reference_{year}_{state}.csv",
                      usecols=ATTRS + ["PWGTP"])
    w = pd.to_numeric(ref.PWGTP, errors="coerce").fillna(0).to_numpy(float)
    w = w / w.sum()
    idx = np.random.default_rng(seed).choice(len(ref), size=n, replace=True, p=w)
    return ref.iloc[idx][ATTRS].reset_index(drop=True)


def sample_synth(pw_df, n=N_POP, seed=0):
    """Monte-Carlo sample n agents from a cell-level projected_weights table."""
    q = pd.to_numeric(pw_df.projected_weight, errors="coerce").fillna(0).to_numpy(float)
    q = q / q.sum()
    idx = np.random.default_rng(seed).choice(len(pw_df), size=n, replace=True, p=q)
    return pw_df.iloc[idx][ATTRS].reset_index(drop=True)


def evaluate_univariate(real_df, model_df, cat_cols, num_cols):
    """1-way (univariate) SRMSE averaged over attributes, matching evaluate_marg's
    per-combo arithmetic (SRMSE_contrib = ||p_real - p_model||_2, averaged over combos).
    This is the NEGATIVE CONTROL: Layer-3 pins the univariate marginals for both groups."""
    real_df, model_df = transform_cat_columns(real_df, model_df, cat_cols, num_cols)
    dic = {k: v for k, v in {"Age": 5, "Number_of_cars_of_household": 2}.items() if k in num_cols}
    real_df = discretize(real_df, dic)
    model_df = discretize(model_df, dic)
    ori_r, ori_m = real_df.copy(deep=True), model_df.copy(deep=True)
    vars_list = real_df.columns.tolist()
    srmse = 0.0
    for v in vars_list:
        md = ori_m[[v]].copy(); md["count"] = 1
        md = md.groupby(v, observed=True).count(); md /= md["count"].sum()
        rd = ori_r[[v]].copy(); rd["count"] = 1
        rd = rd.groupby(v, observed=True).count(); rd /= rd["count"].sum()
        merged = pd.merge(rd, md, suffixes=["_real", "_sampled"], on=v, how="outer").fillna(0)
        diff = (merged["count_real"] - merged["count_sampled"]).to_numpy()
        N = len(ori_r[v].unique())
        srmse += np.sqrt((np.sum(diff ** 2) / N) * N)   # = sqrt(sum diff^2)
    return srmse / len(vars_list)


AGE_BINS = [0, 18, 35, 50, 65, 200]


def srmse_j_audit(real_df, gen_df):
    """Standardized SRMSE over the full joint of the seven audit attributes in the AUDIT coding
    (age in the five audit age groups, the other attributes at their coded categories), as
    defined in Supplementary Section S2: [n^-1 sum_j (p_hat_j - p_j)^2]^{1/2} / (n^-1 sum_j p_j),
    with n the number of joint cells occupied in either population (2026-11-23; the earlier
    SRMSE_J of eval_travelattrs was the unnormalized Euclidean distance on 5-year age bins)."""
    def coded(d):
        d = d.copy()
        d["Age"] = pd.cut(pd.to_numeric(d["Age"], errors="coerce"), bins=AGE_BINS, right=False, labels=False)
        return d[ATTRS].astype(str).agg("|".join, axis=1)
    pr = coded(real_df).value_counts(normalize=True)
    pg = coded(gen_df).value_counts(normalize=True)
    idx = pr.index.union(pg.index)
    d = pg.reindex(idx, fill_value=0.0).to_numpy() - pr.reindex(idx, fill_value=0.0).to_numpy()
    n = len(idx)
    return float(np.sqrt((d ** 2).sum() / n) / (1.0 / n))


_REF_SHARES = {}


def _audit_code(d):
    d = d.copy()
    d["Age"] = pd.cut(pd.to_numeric(d["Age"], errors="coerce"), bins=AGE_BINS, right=False, labels=False)
    return d[ATTRS].astype(str).agg("|".join, axis=1)


def srmse_j_exact(state, year, pw_df, weight_col="projected_weight"):
    """SRMSE_J exactly as defined in Supplementary Section S2 (2026-11-24): standardized SRMSE
    between the joint cell shares of the projected population Q* (record weights `weight_col`)
    and the PWGTP-weighted shares of the full realized reference, over the seven audit
    attributes in the audit coding, n = number of joint cells occupied in either population.
    Deterministic (no Monte-Carlo sampling); replaces the sample-based srmse_j_audit in the
    reported SRMSE_J."""
    k = (state, str(year))
    if k not in _REF_SHARES:
        ref = pd.read_csv(f"{ROOT}/data/pums_states/{state}/reference_{year}_{state}.csv",
                          usecols=ATTRS + ["PWGTP"])
        w = pd.to_numeric(ref.PWGTP, errors="coerce").fillna(0).to_numpy(float)
        _REF_SHARES[k] = pd.Series(w, index=_audit_code(ref)).groupby(level=0).sum() / w.sum()
    pr = _REF_SHARES[k]
    q = pd.to_numeric(pw_df[weight_col], errors="coerce").fillna(0).to_numpy(float)
    pg = pd.Series(q, index=_audit_code(pw_df)).groupby(level=0).sum() / q.sum()
    pg = pg[pg > 0]
    idx = pr[pr > 0].index.union(pg.index)
    d = pg.reindex(idx, fill_value=0.0).to_numpy() - pr.reindex(idx, fill_value=0.0).to_numpy()
    n = len(idx)
    return float(np.sqrt((d ** 2).sum() / n) / (1.0 / n))


def metrics_one(real_test, gen_df):
    """All fidelity metrics for one (reference, synthetic) pair over the 7 audit attrs."""
    u = float(evaluate_univariate(real_test.copy(), gen_df.copy(), CAT, NUM))
    b = float(evaluate_bivar(real_test.copy(), gen_df.copy(), CAT, NUM))
    m = float(evaluate_marg(real_test.copy(), gen_df.copy(), CAT, NUM))
    j_euclid = float(eval_travelattrs(real_test.copy(), gen_df.copy(), CAT, NUM))
    j = srmse_j_audit(real_test, gen_df)
    p, r, f1 = eval_zerocells_travel(real_test.copy(), gen_df.copy(), CAT, NUM)
    return dict(SRMSE_U=u, SRMSE_B=b, SRMSE_M=m, SRMSE_J=j, SRMSE_J_euclid=j_euclid, Precision=p, Recall=r, F1=f1)
