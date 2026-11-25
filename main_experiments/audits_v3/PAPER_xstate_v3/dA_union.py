"""Shared helper for the post-projection residual D = d_A(Q*, S_H)  (FIX 3, 2026-11-24).

Definition (paper Eq. marginal-aggregate, Cor. 1):
    d_A(Q*, S_H) = 0.5 * sum_{a in A} sum_{c in C_a} | m_{a,c}(Q*) - m*_{a,c}(S_H) |
where C_a is the UNION of the categories carried by the Layer-3 projected population Q*
and the categories of the PWGTP-weighted scenario reference.  A category that the reference
does not populate has target m* = 0, so its projected share enters D in full.

Layers 2 and 3 standardize only the reference-observed categories (m* > 0; the Layer-3
targets), so
    d_A_union = d_A_ref + 0.5 * mu_out,
    d_A_ref   = 0.5 * sum_targets projected_abs_err_mean   (the pre-FIX-3 value of D),
    mu_out    = Q* mass on categories with m* = 0.
Both parts are returned, and d_A_ref is cross-checked against layer3_target_summary.csv,
which guards the category matching between Q*, the reference and the Layer-3 targets.

Used by figures/make_figE_postproj.py (D, Table 4, figE, gamma_min, runner-up gaps),
figures/make_transfer_figures_v3.py (Cor.-1 check; unchanged result), bn_pgmpy_compare,
and the supplementary collectors (tau ladder, seeds, truncated priors).
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[3])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)

import functools
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(_PA.R(""))
BASE = Path(_PA.R("/main_experiments/audits_v3"))
ATTRS = ("Age_Group", "Household_Size", "Number_of_children", "Number_of_cars_of_household",
         "Employment_status", "Household_Income_Class", "Commute_Mode")
_AGE_BINS = [0, 18, 35, 50, 65, 200]
_AGE_LABELS = ["0-17", "18-34", "35-49", "50-64", "65+"]


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Age_Group as Layer 3 builds it (only if the frame does not carry it already)."""
    if "Age_Group" not in df.columns:
        df["Age_Group"] = pd.cut(pd.to_numeric(df["Age"], errors="coerce"), _AGE_BINS,
                                 labels=_AGE_LABELS, right=False, include_lowest=True).astype(str)
    return df


def norm_cat(s: pd.Series) -> pd.Series:
    """Category label as a string; integer-like columns -> '3' (never '3.0')."""
    x = pd.to_numeric(s, errors="coerce")
    if len(s) and x.notna().mean() > 0.95:
        return x.round().astype("Int64").astype(str)
    return s.astype(str)


def reference_path(state: str, year) -> Path:
    return ROOT / f"data/pums_states/{state}/reference_{int(year)}_{state}.csv"


def marginals(df: pd.DataFrame, w, attrs=ATTRS) -> dict:
    """{attr: Series(category -> weighted share)} with the weights normalized to sum 1."""
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    return {a: pd.Series(w).groupby(norm_cat(df[a]).values).sum() for a in attrs}


@functools.lru_cache(maxsize=32)
def _ref_marginals_cached(path: str, mtime: float, attrs: tuple, weight_col: str) -> dict:
    need = set(attrs) | {"Age", weight_col}
    ref = add_age_group(pd.read_csv(path, usecols=lambda c: c in need, low_memory=False))
    pw = pd.to_numeric(ref[weight_col], errors="coerce").fillna(0).to_numpy(float)
    return marginals(ref, pw, attrs)


def reference_marginals(state=None, year=None, ref_csv=None, attrs=ATTRS, weight_col="PWGTP") -> dict:
    p = Path(ref_csv) if ref_csv is not None else reference_path(state, year)
    return _ref_marginals_cached(str(p), os.path.getmtime(p), tuple(attrs), weight_col)


def dA_from_marginals(mq: dict, mr: dict, attrs=ATTRS) -> dict:
    """d_A on the union of categories (zero target outside the reference) and its parts."""
    dun = dref = mu = 0.0
    per_attr, outcats = {}, {}
    for a in attrs:
        q, r = mq[a], mr[a]
        rpos = r[r > 0]                                   # reference categories = Layer-3 targets
        cats = set(q.index) | set(r.index)
        s_un = sum(abs(float(q.get(c, 0.0)) - float(r.get(c, 0.0))) for c in cats)
        s_ref = sum(abs(float(q.get(c, 0.0)) - float(rpos[c])) for c in rpos.index)
        oc = {c: float(q[c]) for c in q.index if c not in rpos.index and q[c] != 0}
        dun += 0.5 * s_un; dref += 0.5 * s_ref; mu += sum(oc.values())
        per_attr[a] = 0.5 * s_un; outcats[a] = oc
    assert abs(dun - (dref + 0.5 * mu)) < 1e-10, (dun, dref, mu)
    return {"d_A": dun, "d_A_ref": dref, "mu_out": mu, "per_attr": per_attr, "out_categories": outcats}


def dA_from_frames(Q: pd.DataFrame, q, ref: pd.DataFrame, pw, attrs=ATTRS) -> dict:
    """Same as dA_from_marginals, from already loaded Q* / reference frames and weights."""
    return dA_from_marginals(marginals(add_age_group(Q), q, attrs), marginals(add_age_group(ref), pw, attrs), attrs)


def projected_weights_file(l3_dir) -> Path:
    files = sorted((Path(l3_dir) / "projected_weights").glob("*.csv"))
    if len(files) != 1:
        raise FileNotFoundError(f"expected exactly one projected_weights/*.csv in {l3_dir}, found {len(files)}")
    return files[0]


def load_projected(l3_dir):
    """(Q*, q): the saved cell-level projected population of a Layer-3 run and its normalized weights."""
    Q = add_age_group(pd.read_csv(projected_weights_file(l3_dir)))
    q = pd.to_numeric(Q["projected_weight"], errors="coerce").fillna(0).to_numpy(float)
    return Q, q / q.sum()


def dA_union_l3(l3_dir, state=None, year=None, ref_csv=None, attrs=ATTRS, check=True, tol=1e-9) -> dict:
    """D for one Layer-3 output directory.  The reference defaults to reference_<year>_<state>.csv.
    check=True: d_A_ref must equal 0.5*sum(projected_abs_err_mean) of layer3_target_summary.csv
    (and the number of reference categories the number of targets), else RuntimeError."""
    l3_dir = Path(l3_dir)
    Q, q = load_projected(l3_dir)
    res = dA_from_marginals(marginals(Q, q, attrs),
                            reference_marginals(state, year, ref_csv, attrs), attrs)
    ts = l3_dir / "layer3_target_summary.csv"
    if check and ts.exists():
        t3 = pd.read_csv(ts)
        res["d_A_ref_L3summary"] = 0.5 * float(t3["projected_abs_err_mean"].sum())
        mr = reference_marginals(state, year, ref_csv, attrs)
        n_ref = int(sum((mr[a] > 0).sum() for a in attrs))
        if abs(res["d_A_ref"] - res["d_A_ref_L3summary"]) > tol or n_ref != len(t3):
            raise RuntimeError(f"{l3_dir.name}: reference-category d_A {res['d_A_ref']:.12f} vs Layer-3 summary "
                               f"{res['d_A_ref_L3summary']:.12f}; {n_ref} reference categories vs {len(t3)} targets")
    return res


def dA_union_case(model: str, state: str, year, base=BASE, **kw) -> dict:
    """D for layer3_paperW_<model>_<state>_<year> under `base` (default: audits_v3)."""
    return dA_union_l3(Path(base) / f"layer3_paperW_{model}_{state}_{int(year)}", state, year, **kw)


def D_union(model: str, state: str, year, base=BASE, **kw) -> float:
    """Scalar D = d_A(Q*, S_H) on the category union."""
    return dA_union_case(model, state, year, base, **kw)["d_A"]
