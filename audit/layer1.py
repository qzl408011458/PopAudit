from __future__ import annotations

"""Ensemble-aware Layer 1 structural prior audit.

X is the full training attribute space. X_A is the task-relevant space used for
Layer 1-3. If multiple generated population realizations are supplied, this
script computes Layer-1 metrics for each realization, reports mean/std, and also
reports ensemble-level support coverage over the union of realization supports.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple
import argparse, json
import numpy as np
import pandas as pd

# -----------------------------
# Attribute spaces
# -----------------------------
FULL_NUM_COLS = ["Age", "Household_Size", "Number_of_children", "Number_of_cars_of_household"]
FULL_CAT_COLS = [
    "Gender", "Marital_status", "Employment_status",
    "Studenthood_status", "Household_Type", "Commute_Mode",
    "Personal_Income_Class", "Household_Income_Class",
]
# v3 (2026-08-29): Region_Category removed — constant within each state
# (CA=West / OH=Midwest / TX=South), excluded from the modelled columns in
# cols_config_v3.yaml, and absent from v3 model outputs. Audit gates are
# unaffected: TASK_ATTRS_XA, the dependency pairs and the structural rules
# never referenced it (check_full_attrs defaults to False).
FULL_ATTRS_X = FULL_NUM_COLS + FULL_CAT_COLS

TASK_NUM_COLS_XA = ["Age", "Household_Size", "Number_of_children", "Number_of_cars_of_household"]
TASK_CAT_COLS_XA = ["Employment_status", "Household_Income_Class", "Commute_Mode"]
TASK_ATTRS_XA = TASK_NUM_COLS_XA + TASK_CAT_COLS_XA

PairType = Literal["categorical", "metric", "mixed", "discrete", "continuous"]
ValidityRule = Callable[[pd.DataFrame], pd.Series]

@dataclass
class DependencyPairSpec:
    var1: str
    var2: str
    pair_type: PairType
    categories_var1: Optional[Sequence[Any]] = None
    categories_var2: Optional[Sequence[Any]] = None
    metric_var: Optional[str] = None
    categorical_var: Optional[str] = None
    weight: Optional[float] = None

@dataclass
class Layer1AuditConfig:
    full_attrs: Sequence[str]
    task_attrs: Sequence[str]
    dependency_pairs: Sequence[DependencyPairSpec]
    validity_rules: Sequence[ValidityRule]
    support_min_count: int = 1
    missing_token: str = "__MISSING__"
    min_count_per_category: int = 5
    n_sliced_projections: int = 64
    check_full_attrs: bool = False

@dataclass
class SupportCoverageResult:
    coverage: float
    n_real_rows: int
    n_real_supported: int
    n_unique_real_patterns: int
    n_unique_synth_patterns: int
    uncovered_fraction: float

@dataclass
class DependencyPairResult:
    var1: str
    var2: str
    pair_type: PairType
    discrepancy: float
    details: Optional[Dict[str, Any]] = None

@dataclass
class DependencyDiscrepancyResult:
    total_discrepancy: float
    pair_results: List[DependencyPairResult]

@dataclass
class InvalidMassResult:
    invalid_mass: float
    n_rows: int
    n_invalid_rows: int
    rule_violation_rates: Dict[str, float]

@dataclass
class MarginalAttrResult:
    attr: str
    discrepancy: float
    n_categories: int
    top_abs_diff: float

@dataclass
class MarginalDiscrepancyResult:
    total_discrepancy: float
    attr_results: List[MarginalAttrResult]

@dataclass
class Layer1AuditResult:
    support: SupportCoverageResult
    dependency: DependencyDiscrepancyResult
    invalidity: InvalidMassResult
    marginal: MarginalDiscrepancyResult

@dataclass
class Layer1EnsembleAuditResult:
    ensemble_summary: Dict[str, Any]
    per_realization_frame: pd.DataFrame
    dependency_pair_per_realization_frame: pd.DataFrame
    dependency_pair_summary_frame: pd.DataFrame
    marginal_attr_per_realization_frame: pd.DataFrame
    marginal_attr_summary_frame: pd.DataFrame
    per_realization_results: List[Layer1AuditResult]
    ensemble_support: SupportCoverageResult

# -----------------------------
# Helpers
# -----------------------------
def _check_columns(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name} is missing required columns: {miss}")

def _read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {p}")
    return pd.read_csv(p)

def _clip01(x: float) -> float:
    return np.nan if not np.isfinite(x) else float(np.clip(x, 0.0, 1.0))

def _disc(s: pd.Series, missing_token: str) -> pd.Series:
    return s.astype("object").where(~s.isna(), other=missing_token).astype(str)

def _num(s: pd.Series) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
    return x[np.isfinite(x)]

def _num_pair(df: pd.DataFrame, a: str, b: str) -> np.ndarray:
    arr = df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(float)
    return arr[np.isfinite(arr).all(axis=1)] if arr.size else np.empty((0, 2))

def _row_patterns(df: pd.DataFrame, cols: Sequence[str], missing_token: str) -> pd.Series:
    sub = df.loc[:, list(cols)].copy().fillna(missing_token)
    return sub.apply(lambda r: tuple(r.values.tolist()), axis=1)

def _json_safe(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer)): return x.item()
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (pd.Series, pd.DataFrame)): return x.to_dict()
    if isinstance(x, dict): return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_json_safe(v) for v in x]
    try:
        if pd.isna(x): return None
    except Exception:
        pass
    return x

# -----------------------------
# Normalized Wasserstein
# -----------------------------
def _w1_1d(u: np.ndarray, v: np.ndarray) -> float:
    u, v = np.asarray(u, float), np.asarray(v, float)
    u, v = np.sort(u[np.isfinite(u)]), np.sort(v[np.isfinite(v)])
    if len(u) == 0 or len(v) == 0: return np.nan
    z = np.sort(np.concatenate([u, v]))
    if len(z) <= 1: return 0.0
    dz = np.diff(z)
    cu = np.searchsorted(u, z[:-1], side="right") / len(u)
    cv = np.searchsorted(v, z[:-1], side="right") / len(v)
    return float(np.sum(np.abs(cu - cv) * dz))

def _minmax2(real: np.ndarray, synth: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    pooled = np.vstack([real, synth])
    lo, hi = np.nanmin(pooled, axis=0), np.nanmax(pooled, axis=0)
    rng = np.where((hi - lo) > 0, hi - lo, 1.0)
    return (real - lo) / rng, (synth - lo) / rng, {"min": lo.tolist(), "max": hi.tolist(), "range": rng.tolist()}

def _norm_sliced_w1_2d(real: np.ndarray, synth: np.ndarray, L: int) -> Tuple[float, Dict[str, Any]]:
    if len(real) == 0 or len(synth) == 0:
        return np.nan, {"method": "normalized_sliced_wasserstein_2d", "reason": "empty numeric pair"}
    r, s, info = _minmax2(real, synth)
    vals = []
    for ang in np.linspace(0, np.pi, max(int(L), 1), endpoint=False):
        d = np.array([np.cos(ang), np.sin(ang)])
        vals.append(_w1_1d(r @ d, s @ d))
    raw = float(np.nanmean(vals))
    out = _clip01(raw / np.sqrt(2.0))
    return out, {"method": "normalized_sliced_wasserstein_2d", "scale": "[0,1]", "raw_sliced_w1": raw, "n_projections": int(L), "minmax_info": info}

# -----------------------------
# Metric 1: Support Coverage
# -----------------------------
def compute_support_coverage(real_df, synth_df, task_attrs, support_min_count=1, missing_token="__MISSING__") -> SupportCoverageResult:
    _check_columns(real_df, task_attrs, "real_df"); _check_columns(synth_df, task_attrs, "synth_df")
    real_pat = _row_patterns(real_df, task_attrs, missing_token)
    synth_pat = _row_patterns(synth_df, task_attrs, missing_token)
    counts = synth_pat.value_counts()
    support = set(counts[counts >= support_min_count].index)
    covered = real_pat.isin(support)
    cov = float(covered.mean()) if len(covered) else np.nan
    return SupportCoverageResult(cov, int(len(real_pat)), int(covered.sum()), int(real_pat.nunique()), int(synth_pat.nunique()), float(1-cov) if np.isfinite(cov) else np.nan)

def compute_ensemble_support_coverage(real_df, synth_realizations, task_attrs, support_min_count=1, missing_token="__MISSING__") -> SupportCoverageResult:
    _check_columns(real_df, task_attrs, "real_df")
    real_pat = _row_patterns(real_df, task_attrs, missing_token)
    union_support: set = set()
    for i, df in enumerate(synth_realizations):
        _check_columns(df, task_attrs, f"synth_realization[{i}]")
        pat = _row_patterns(df, task_attrs, missing_token)
        counts = pat.value_counts()
        union_support.update(set(counts[counts >= support_min_count].index))
    covered = real_pat.isin(union_support)
    cov = float(covered.mean()) if len(covered) else np.nan
    return SupportCoverageResult(cov, int(len(real_pat)), int(covered.sum()), int(real_pat.nunique()), int(len(union_support)), float(1-cov) if np.isfinite(cov) else np.nan)

# -----------------------------
# Metric 2: Dependency discrepancy
# -----------------------------
def _pair_type(t: PairType) -> str:
    return "categorical" if t == "discrete" else "metric" if t == "continuous" else str(t)

def _joint_tv(real1, real2, synth1, synth2, missing_token, cats1=None, cats2=None):
    r1, r2, s1, s2 = _disc(real1, missing_token), _disc(real2, missing_token), _disc(synth1, missing_token), _disc(synth2, missing_token)
    c1 = sorted(set(r1.unique()).union(s1.unique()) if cats1 is None else set(map(str, cats1)).union(r1.unique()).union(s1.unique()))
    c2 = sorted(set(r2.unique()).union(s2.unique()) if cats2 is None else set(map(str, cats2)).union(r2.unique()).union(s2.unique()))
    p = pd.crosstab(r1, r2, normalize=True).reindex(index=c1, columns=c2, fill_value=0.0)
    q = pd.crosstab(s1, s2, normalize=True).reindex(index=c1, columns=c2, fill_value=0.0)
    l1 = float((p - q).abs().to_numpy().sum())
    tv = _clip01(0.5 * l1)
    return tv, {"method": "joint_total_variation", "scale": "[0,1]", "l1_discrepancy": l1, "normalized_discrepancy": tv}

def _cat_tv(real_cat, synth_cat, cats):
    p = real_cat.value_counts(normalize=True).reindex(cats, fill_value=0.0)
    q = synth_cat.value_counts(normalize=True).reindex(cats, fill_value=0.0)
    return _clip01(0.5 * float((p - q).abs().sum())), p.to_dict(), q.to_dict()

def _mixed_disc(real_df, synth_df, metric_var, cat_var, missing_token, cats=None, min_count=5):
    _check_columns(real_df, [metric_var, cat_var], "real_df"); _check_columns(synth_df, [metric_var, cat_var], "synth_df")
    rc, sc = _disc(real_df[cat_var], missing_token), _disc(synth_df[cat_var], missing_token)
    cats = sorted(set(rc.unique()).union(sc.unique()) if cats is None else set(map(str, cats)).union(rc.unique()).union(sc.unique()))
    rm_raw, sm_raw = pd.to_numeric(real_df[metric_var], errors="coerce"), pd.to_numeric(synth_df[metric_var], errors="coerce")
    pooled = pd.concat([rm_raw.dropna(), sm_raw.dropna()])
    if len(pooled) == 0: return np.nan, {"method": "normalized_mixed_tv_conditional_wasserstein", "reason": "empty metric"}
    lo, rng = float(pooled.min()), float(pooled.max() - pooled.min())
    if not np.isfinite(rng) or rng <= 0: rng = 1.0
    rm, sm = (rm_raw - lo) / rng, (sm_raw - lo) / rng
    tv, pcat, qcat = _cat_tv(rc, sc, cats)
    weights = rc.value_counts(normalize=True)
    cond_total, cat_details = 0.0, {}
    for c in cats:
        w = float(weights.get(c, 0.0))
        if w == 0: continue
        rv, sv = rm.loc[rc == c].dropna().to_numpy(float), sm.loc[sc == c].dropna().to_numpy(float)
        if len(rv) < min_count:
            cat_details[c] = {"weight": w, "n_real": len(rv), "n_synth": len(sv), "used": False, "conditional_w1": np.nan}
            continue
        cw1 = 1.0 if len(sv) == 0 else _clip01(_w1_1d(rv, sv))
        cond_total += w * cw1
        cat_details[c] = {"weight": w, "n_real": int(len(rv)), "n_synth": int(len(sv)), "used": True, "conditional_w1": float(cw1), "weighted_conditional_w1": float(w*cw1)}
    cond_total = _clip01(cond_total)
    disc = _clip01(0.5 * tv + 0.5 * cond_total)
    return disc, {"method": "normalized_mixed_tv_conditional_wasserstein", "scale": "[0,1]", "categorical_tv": tv, "conditional_wasserstein_total": cond_total, "real_category_probs": pcat, "synth_category_probs": qcat, "category_details": cat_details}

def compute_dependency_discrepancy(real_df, synth_df, dependency_pairs, missing_token="__MISSING__", min_count_per_category=5, n_sliced_projections=64) -> DependencyDiscrepancyResult:
    req = sorted(set([p.var1 for p in dependency_pairs] + [p.var2 for p in dependency_pairs]))
    _check_columns(real_df, req, "real_df"); _check_columns(synth_df, req, "synth_df")
    results, wsum, dsum = [], 0.0, 0.0
    for spec in dependency_pairs:
        typ = _pair_type(spec.pair_type)
        if typ == "categorical":
            disc, details = _joint_tv(real_df[spec.var1], real_df[spec.var2], synth_df[spec.var1], synth_df[spec.var2], missing_token, spec.categories_var1, spec.categories_var2)
            out_type: PairType = "categorical"
        elif typ == "metric":
            disc, details = _norm_sliced_w1_2d(_num_pair(real_df, spec.var1, spec.var2), _num_pair(synth_df, spec.var1, spec.var2), n_sliced_projections)
            out_type = "metric"
        elif typ == "mixed":
            mv, cv = spec.metric_var or spec.var1, spec.categorical_var or spec.var2
            if {mv, cv} != {spec.var1, spec.var2}: raise ValueError(f"Invalid mixed spec: {spec}")
            cats = spec.categories_var1 if cv == spec.var1 else spec.categories_var2
            disc, details = _mixed_disc(real_df, synth_df, mv, cv, missing_token, cats, min_count_per_category)
            out_type = "mixed"
        else:
            raise ValueError(f"Unsupported pair_type: {spec.pair_type}")
        disc = _clip01(float(disc)) if np.isfinite(disc) else np.nan
        w = 1.0 if spec.weight is None else float(spec.weight)
        if w < 0: raise ValueError("Pair weight must be non-negative.")
        if np.isfinite(disc) and w > 0:
            dsum += w * disc; wsum += w
        details = details or {}; details["weight"] = w; details["weighted_discrepancy"] = float(w*disc) if np.isfinite(disc) else np.nan
        results.append(DependencyPairResult(spec.var1, spec.var2, out_type, disc, details))
    return DependencyDiscrepancyResult(_clip01(dsum / wsum) if wsum > 0 else np.nan, results)

# -----------------------------
# Metric 2b: Univariate marginal discrepancy
# -----------------------------
def _univariate_tv(real_col: pd.Series, synth_col: pd.Series, missing_token: str) -> Tuple[float, int, float]:
    r = _disc(real_col, missing_token)
    s = _disc(synth_col, missing_token)
    cats = sorted(set(r.unique()) | set(s.unique()))
    p = r.value_counts(normalize=True).reindex(cats, fill_value=0.0)
    q = s.value_counts(normalize=True).reindex(cats, fill_value=0.0)
    diff = (p - q).abs()
    tv = _clip01(0.5 * float(diff.sum()))
    top = float(diff.max()) if len(diff) else 0.0
    return tv, len(cats), top

def compute_marginal_discrepancy(real_df, synth_df, task_attrs, missing_token="__MISSING__") -> MarginalDiscrepancyResult:
    _check_columns(real_df, task_attrs, "real_df"); _check_columns(synth_df, task_attrs, "synth_df")
    results: List[MarginalAttrResult] = []
    for a in task_attrs:
        tv, ncat, top = _univariate_tv(real_df[a], synth_df[a], missing_token)
        results.append(MarginalAttrResult(attr=a, discrepancy=tv, n_categories=ncat, top_abs_diff=top))
    finite = [r.discrepancy for r in results if np.isfinite(r.discrepancy)]
    total = float(np.mean(finite)) if finite else np.nan
    return MarginalDiscrepancyResult(_clip01(total), results)

# -----------------------------
# Metric 3: Invalid mass
# -----------------------------
def compute_invalid_mass(synth_df, validity_rules) -> InvalidMassResult:
    if len(synth_df) == 0: raise ValueError("synth_df is empty.")
    valid = pd.Series(True, index=synth_df.index)
    rates = {}
    for i, rule in enumerate(validity_rules):
        mask = rule(synth_df)
        if not isinstance(mask, pd.Series) or mask.dtype != bool or not mask.index.equals(synth_df.index):
            raise ValueError(f"Validity rule {i} must return a boolean Series with matching index.")
        name = getattr(rule, "__name__", f"rule_{i}")
        rates[name] = float((~mask).mean())
        valid &= mask
    n_invalid = int((~valid).sum())
    return InvalidMassResult(float(n_invalid / len(synth_df)), int(len(synth_df)), n_invalid, rates)

# -----------------------------
# Run single and ensemble audits
# -----------------------------
def run_layer1_audit(real_df, synth_df, config: Layer1AuditConfig) -> Layer1AuditResult:
    if config.check_full_attrs:
        _check_columns(real_df, config.full_attrs, "real_df"); _check_columns(synth_df, config.full_attrs, "synth_df")
    req = sorted(set(config.task_attrs) | {p.var1 for p in config.dependency_pairs} | {p.var2 for p in config.dependency_pairs})
    _check_columns(real_df, req, "real_df"); _check_columns(synth_df, req, "synth_df")
    return Layer1AuditResult(
        compute_support_coverage(real_df, synth_df, config.task_attrs, config.support_min_count, config.missing_token),
        compute_dependency_discrepancy(real_df, synth_df, config.dependency_pairs, config.missing_token, config.min_count_per_category, config.n_sliced_projections),
        compute_invalid_mass(synth_df, config.validity_rules),
        compute_marginal_discrepancy(real_df, synth_df, config.task_attrs, config.missing_token),
    )

def run_layer1_ensemble_audit(real_df, synth_realizations, config: Layer1AuditConfig, realization_ids=None) -> Layer1EnsembleAuditResult:
    if not synth_realizations: raise ValueError("synth_realizations is empty.")
    realization_ids = list(range(len(synth_realizations))) if realization_ids is None else list(realization_ids)
    if len(realization_ids) != len(synth_realizations): raise ValueError("realization_ids length mismatch.")
    results = [run_layer1_audit(real_df, df, config) for df in synth_realizations]
    per = layer1_ensemble_per_realization_to_frame(results, realization_ids)
    ens_sup = compute_ensemble_support_coverage(real_df, synth_realizations, config.task_attrs, config.support_min_count, config.missing_token)
    pair_per = dependency_pair_per_realization_from_results(results, realization_ids)
    pair_sum = dependency_pair_summary_from_frame(pair_per)
    marg_per = marginal_attr_per_realization_from_results(results, realization_ids)
    marg_sum = marginal_attr_summary_from_frame(marg_per)
    summary = layer1_ensemble_summary_dict(per, ens_sup, len(results))
    return Layer1EnsembleAuditResult(summary, per, pair_per, pair_sum, marg_per, marg_sum, results, ens_sup)

# -----------------------------
# Output frames
# -----------------------------
def layer1_result_to_summary_dict(r: Layer1AuditResult) -> Dict[str, Any]:
    return {"support_coverage": r.support.coverage, "support_n_real_rows": r.support.n_real_rows, "support_n_real_supported": r.support.n_real_supported, "support_unique_real_patterns": r.support.n_unique_real_patterns, "support_unique_synth_patterns": r.support.n_unique_synth_patterns, "support_uncovered_fraction": r.support.uncovered_fraction, "marginal_total_discrepancy": r.marginal.total_discrepancy, "marginal_total_scale": "[0,1] mean of per-attribute total-variation distances", "dependency_total_discrepancy": r.dependency.total_discrepancy, "dependency_total_scale": "[0,1] weighted average of normalized pair discrepancies", "invalid_mass": r.invalidity.invalid_mass, "invalid_n_rows": r.invalidity.n_rows, "invalid_n_invalid_rows": r.invalidity.n_invalid_rows, "rule_violation_rates": r.invalidity.rule_violation_rates}

def dependency_results_to_frame(dep: DependencyDiscrepancyResult) -> pd.DataFrame:
    rows=[]
    for r in dep.pair_results:
        rows.append({"var1":r.var1,"var2":r.var2,"pair_type":r.pair_type,"method":(r.details or {}).get("method"),"discrepancy_normalized_0_1":r.discrepancy,"weight":(r.details or {}).get("weight"),"weighted_discrepancy":(r.details or {}).get("weighted_discrepancy")})
    return pd.DataFrame(rows)

def layer1_ensemble_per_realization_to_frame(results, realization_ids) -> pd.DataFrame:
    rows=[]
    for rid, r in zip(realization_ids, results):
        d = layer1_result_to_summary_dict(r); rates = d.pop("rule_violation_rates")
        d["realization_id"] = rid
        for name, val in rates.items(): d[f"invalid_rule_rate__{name}"] = val
        rows.append(d)
    return pd.DataFrame(rows)[["realization_id"] + [c for c in pd.DataFrame(rows).columns if c != "realization_id"]]

def _summary(s):
    x = pd.to_numeric(s, errors="coerce").dropna().astype(float)
    return {"mean": float(x.mean()) if len(x) else np.nan, "std": float(x.std(ddof=1)) if len(x)>1 else 0.0, "min": float(x.min()) if len(x) else np.nan, "max": float(x.max()) if len(x) else np.nan}

def layer1_ensemble_summary_dict(per, ens_sup, n_realizations):
    sc, dep, im = _summary(per.support_coverage), _summary(per.dependency_total_discrepancy), _summary(per.invalid_mass)
    mg = _summary(per.marginal_total_discrepancy)
    out = {
        "audit_object": "learned_prior_distribution_P_theta",
        "estimator": "multiple_monte_carlo_population_realizations",
        "interpretation": (
            "Layer-1 metrics are prior-level structural quantities. "
            "They are estimated per generated realization and summarized by mean/std/min/max; "
            "ensemble_support_coverage uses the union of empirical supports across realizations."
        ),
        "n_realizations": int(n_realizations),
        "per_realization_support_coverage_mean": sc["mean"],
        "per_realization_support_coverage_std": sc["std"],
        "per_realization_support_coverage_min": sc["min"],
        "per_realization_support_coverage_max": sc["max"],
        "ensemble_support_coverage": ens_sup.coverage,
        "ensemble_support_uncovered_fraction": ens_sup.uncovered_fraction,
        "ensemble_support_n_real_supported": ens_sup.n_real_supported,
        "ensemble_support_unique_real_patterns": ens_sup.n_unique_real_patterns,
        "ensemble_support_unique_synth_patterns_union": ens_sup.n_unique_synth_patterns,
        "marginal_total_discrepancy_mean": mg["mean"],
        "marginal_total_discrepancy_std": mg["std"],
        "marginal_total_discrepancy_min": mg["min"],
        "marginal_total_discrepancy_max": mg["max"],
        "marginal_total_scale": "[0,1] mean of per-attribute total-variation distances",
        "dependency_total_discrepancy_mean": dep["mean"],
        "dependency_total_discrepancy_std": dep["std"],
        "dependency_total_discrepancy_min": dep["min"],
        "dependency_total_discrepancy_max": dep["max"],
        "dependency_total_scale": "[0,1] weighted average of normalized pair discrepancies",
        "invalid_mass_mean": im["mean"],
        "invalid_mass_std": im["std"],
        "invalid_mass_min": im["min"],
        "invalid_mass_max": im["max"],
    }
    for c in [c for c in per.columns if c.startswith("invalid_rule_rate__")]:
        st = _summary(per[c]); name = c.replace("invalid_rule_rate__", "")
        out[f"{name}_mean"] = st["mean"]; out[f"{name}_std"] = st["std"]
    return out

def marginal_attr_per_realization_from_results(results, realization_ids) -> pd.DataFrame:
    rows = []
    for rid, r in zip(realization_ids, results):
        for a in r.marginal.attr_results:
            rows.append({
                "realization_id": rid,
                "attr": a.attr,
                "discrepancy_normalized_0_1": a.discrepancy,
                "method": "univariate_total_variation",
                "n_categories": a.n_categories,
                "top_abs_diff": a.top_abs_diff,
            })
    if not rows:
        return pd.DataFrame(columns=["realization_id","attr","discrepancy_normalized_0_1","method","n_categories","top_abs_diff"])
    out = pd.DataFrame(rows)
    return out[["realization_id","attr","discrepancy_normalized_0_1","method","n_categories","top_abs_diff"]]

def marginal_attr_summary_from_frame(marg_frame: pd.DataFrame) -> pd.DataFrame:
    if marg_frame.empty:
        return pd.DataFrame()
    return (
        marg_frame
        .groupby(["attr", "method"], dropna=False)["discrepancy_normalized_0_1"]
        .agg(
            discrepancy_mean="mean",
            discrepancy_std="std",
            discrepancy_min="min",
            discrepancy_max="max",
            n_realizations_used="count",
        )
        .reset_index()
        .fillna({"discrepancy_std": 0.0})
        .sort_values("discrepancy_mean", ascending=False)
        .reset_index(drop=True)
    )

def dependency_pair_per_realization_from_results(results, realization_ids) -> pd.DataFrame:
    frames = []
    for rid, r in zip(realization_ids, results):
        df = dependency_results_to_frame(r.dependency)
        df["realization_id"] = rid
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    cols = ["realization_id"] + [c for c in out.columns if c != "realization_id"]
    return out[cols]

def dependency_pair_summary_from_frame(pair_frame: pd.DataFrame) -> pd.DataFrame:
    if pair_frame.empty:
        return pd.DataFrame()
    return (
        pair_frame
        .groupby(["var1", "var2", "pair_type", "method"], dropna=False)["discrepancy_normalized_0_1"]
        .agg(
            discrepancy_mean="mean",
            discrepancy_std="std",
            discrepancy_min="min",
            discrepancy_max="max",
            n_realizations_used="count",
        )
        .reset_index()
        .fillna({"discrepancy_std": 0.0})
    )

# Backward-compatible helper name.
def dependency_pair_summary_from_results(results):
    return dependency_pair_summary_from_frame(
        dependency_pair_per_realization_from_results(results, list(range(len(results))))
    )

# -----------------------------
# Conservative PUMS invalid-mass rules
# -----------------------------
def _true(df): return pd.Series(True, index=df.index)
def _ncol(df,col,default=np.nan): return pd.Series(default,index=df.index,dtype=float) if col not in df else pd.to_numeric(df[col], errors="coerce")
def _scol(df,col): return pd.Series("",index=df.index,dtype=object) if col not in df else df[col].astype(object).where(~df[col].isna(),"").astype(str).str.lower().str.strip()
def _has(s, pats): return s.str.contains("|".join(pats), regex=True, na=False) if pats else pd.Series(False,index=s.index)

def rule_age_reasonable_pums(df):
    age=_ncol(df,"Age"); return age.notna() & (age>=0) & (age<=110)
def rule_household_size_positive(df):
    x=_ncol(df,"Household_Size"); return x.notna() & (x>=1)
def rule_children_nonnegative(df):
    x=_ncol(df,"Number_of_children"); return x.notna() & (x>=0)
def rule_cars_reasonable_pums(df):
    x=_ncol(df,"Number_of_cars_of_household"); return x.notna() & (x>=0) & (x<=10)
def rule_children_household_consistency(df):
    ch,hh=_ncol(df,"Number_of_children"),_ncol(df,"Household_Size"); return ch.notna() & hh.notna() & (ch<=hh)
def rule_child_not_employed(df):
    if "Employment_status" not in df: return _true(df)
    # 2026-11-23 fix: labels are underscore-coded (e.g. 'Civilian_Employed_At_Work'), and '_'
    # is a word character, so the word-boundary patterns never matched; compare on spaces.
    age, emp = _ncol(df,"Age"), _scol(df,"Employment_status").str.replace("_"," ",regex=False)
    employed = _has(emp,[r"\bemployed\b",r"\bat work\b",r"\bworked\b",r"\bworking\b"]) & (~_has(emp,[r"\bunemployed\b",r"\bnot employed\b",r"\bnot in labor\b",r"\bnot in labour\b",r"\binactive\b"]))
    return ~(age.notna() & (age<14) & employed)
def rule_child_not_married(df):
    if "Marital_status" not in df: return _true(df)
    age, mar = _ncol(df,"Age"), _scol(df,"Marital_status")
    ever = _has(mar,[r"\bmarried\b",r"\bdivorced\b",r"\bseparated\b",r"\bwidowed\b"]) & (~_has(mar,[r"\bnever married\b",r"\bsingle\b",r"\bnot married\b"]))
    return ~(age.notna() & (age<14) & ever)
def rule_student_age_reasonable(df):
    # 2026-11-23: NOT in the default rule set (violated by the real 2013 splits, so not a hard rule).
    if "Studenthood_status" not in df: return _true(df)
    age, st = _ncol(df,"Age"), _scol(df,"Studenthood_status")
    is_st = _has(st,[r"\bstudent\b",r"\bschool\b",r"\benrolled\b"]) & (~_has(st,[r"\bnot student\b",r"\bnot a student\b",r"\bnot enrolled\b",r"\bnon[- ]?student\b"]))
    return ~(is_st & age.notna() & ((age<3)|(age>80)))

# aliases
rule_age_reasonable = rule_age_reasonable_pums
rule_age_nonnegative = lambda df: _ncol(df,"Age").fillna(-1) >= 0
rule_cars_nonnegative = lambda df: _ncol(df,"Number_of_cars_of_household").fillna(-1) >= 0
rule_car_household_consistency = lambda df: ~((_ncol(df,"Number_of_cars_of_household").fillna(0)>0) & (_ncol(df,"Household_Size").fillna(0)<1))

# -----------------------------
# Default config
# -----------------------------
def build_default_dependency_pairs() -> List[DependencyPairSpec]:
    return [
        DependencyPairSpec("Household_Size","Number_of_children","metric"),
        DependencyPairSpec("Household_Size","Number_of_cars_of_household","metric"),
        DependencyPairSpec("Number_of_children","Number_of_cars_of_household","metric"),
        DependencyPairSpec("Age","Employment_status","mixed",metric_var="Age",categorical_var="Employment_status"),
        DependencyPairSpec("Age","Household_Income_Class","mixed",metric_var="Age",categorical_var="Household_Income_Class"),
        DependencyPairSpec("Age","Commute_Mode","mixed",metric_var="Age",categorical_var="Commute_Mode"),
        DependencyPairSpec("Household_Size","Household_Income_Class","mixed",metric_var="Household_Size",categorical_var="Household_Income_Class"),
        DependencyPairSpec("Number_of_children","Household_Income_Class","mixed",metric_var="Number_of_children",categorical_var="Household_Income_Class"),
        DependencyPairSpec("Number_of_cars_of_household","Household_Income_Class","mixed",metric_var="Number_of_cars_of_household",categorical_var="Household_Income_Class"),
        DependencyPairSpec("Number_of_cars_of_household","Commute_Mode","mixed",metric_var="Number_of_cars_of_household",categorical_var="Commute_Mode"),
        DependencyPairSpec("Employment_status","Household_Income_Class","categorical"),
        DependencyPairSpec("Employment_status","Commute_Mode","categorical"),
        DependencyPairSpec("Household_Income_Class","Commute_Mode","categorical"),
    ]

def build_default_layer1_config(support_min_count=1, min_count_per_category=5, n_sliced_projections=64, check_full_attrs=False):
    return Layer1AuditConfig(FULL_ATTRS_X, TASK_ATTRS_XA, build_default_dependency_pairs(), [rule_age_reasonable_pums, rule_household_size_positive, rule_children_nonnegative, rule_cars_reasonable_pums, rule_children_household_consistency, rule_child_not_employed, rule_child_not_married], support_min_count, "__MISSING__", min_count_per_category, n_sliced_projections, check_full_attrs)

# -----------------------------
# File interfaces and CLI
# -----------------------------
def _load_combined(path, rid_col="realization_id"):
    df=_read_csv(path)
    if rid_col not in df: return [df], [0]
    outs, ids=[], []
    for rid, g in df.groupby(rid_col, sort=True):
        outs.append(g.drop(columns=[rid_col]).reset_index(drop=True)); ids.append(rid)
    return outs, ids

def _load_dir(path):
    p=Path(path)
    if not p.is_dir(): raise NotADirectoryError(f"Realizations directory not found: {p}")
    files=sorted(p.glob("*.csv"))
    if not files: raise FileNotFoundError(f"No CSV files found in {p}")
    return [_read_csv(f) for f in files], [f.stem for f in files]

def _save_outputs(res: Layer1EnsembleAuditResult, outdir):
    out=Path(outdir); out.mkdir(parents=True, exist_ok=True)
    with (out/"layer1_ensemble_summary.json").open("w",encoding="utf-8") as f: json.dump(_json_safe(res.ensemble_summary), f, indent=2, ensure_ascii=False)
    pd.DataFrame([res.ensemble_summary]).to_csv(out/"layer1_ensemble_summary.csv", index=False)
    res.per_realization_frame.to_csv(out/"layer1_per_realization_metrics.csv", index=False)
    res.dependency_pair_per_realization_frame.to_csv(out/"layer1_dependency_pair_per_realization.csv", index=False)
    res.dependency_pair_summary_frame.to_csv(out/"layer1_dependency_pair_summary.csv", index=False)
    res.marginal_attr_per_realization_frame.to_csv(out/"layer1_marginal_attr_per_realization.csv", index=False)
    res.marginal_attr_summary_frame.to_csv(out/"layer1_marginal_attr_summary.csv", index=False)

# backward-compatible single CSV interface
def run_layer1_audit_from_csv(real_test_data_path, df_final_path, output_dir=None, config=None):
    real, gen = _read_csv(real_test_data_path), _read_csv(df_final_path)
    res = run_layer1_audit(real, gen, config or build_default_layer1_config())
    if output_dir:
        ens = run_layer1_ensemble_audit(real, [gen], config or build_default_layer1_config(), [0])
        _save_outputs(ens, output_dir)
    return res

def run_layer1_ensemble_audit_from_files(real_test_data_path, realizations_csv_path=None, realizations_dir=None, df_final_path=None, output_dir=None, realization_id_col="realization_id", config=None):
    real = _read_csv(real_test_data_path)
    config = config or build_default_layer1_config()
    if sum(x is not None for x in [realizations_csv_path, realizations_dir, df_final_path]) != 1:
        raise ValueError("Specify exactly one of realizations_csv_path, realizations_dir, or df_final_path.")
    if realizations_dir is not None:
        realizations, ids = _load_dir(realizations_dir)
    elif realizations_csv_path is not None:
        realizations, ids = _load_combined(realizations_csv_path, realization_id_col)
    else:
        realizations, ids = [_read_csv(df_final_path)], [0]
    res = run_layer1_ensemble_audit(real, realizations, config, ids)
    if output_dir: _save_outputs(res, output_dir)
    return res

def main(real_test_data=None, df_final=None, realizations_csv=None, realizations_dir=None, output_dir="layer1_outputs"):
    if real_test_data is None:
        ap=argparse.ArgumentParser(description="Run ensemble-aware Layer-1 audit on X_A.")
        ap.add_argument("--real_test_data", required=True)
        g=ap.add_mutually_exclusive_group(required=True)
        g.add_argument("--df_final")
        g.add_argument("--realizations_csv")
        g.add_argument("--realizations_dir")
        ap.add_argument("--realization_id_col", default="realization_id")
        ap.add_argument("--output_dir", default=output_dir)
        ap.add_argument("--support_min_count", type=int, default=1)
        ap.add_argument("--min_count_per_category", type=int, default=5)
        ap.add_argument("--n_sliced_projections", type=int, default=64)
        ap.add_argument("--check_full_attrs", action="store_true")
        a=ap.parse_args()
        cfg=build_default_layer1_config(a.support_min_count, a.min_count_per_category, a.n_sliced_projections, a.check_full_attrs)
        res=run_layer1_ensemble_audit_from_files(a.real_test_data, a.realizations_csv, a.realizations_dir, a.df_final, a.output_dir, a.realization_id_col, cfg)
    else:
        cfg=build_default_layer1_config()
        res=run_layer1_ensemble_audit_from_files(real_test_data, realizations_csv, realizations_dir, df_final, output_dir, config=cfg)
    print("===== Attribute Spaces =====")
    print(f"Full training space X ({len(FULL_ATTRS_X)} attrs): {FULL_ATTRS_X}")
    print(f"Task space X_A ({len(TASK_ATTRS_XA)} attrs): {TASK_ATTRS_XA}")
    print("\n===== Layer 1 Ensemble Summary =====")
    print(json.dumps(_json_safe(res.ensemble_summary), indent=2, ensure_ascii=False))
    print("\n===== Per-Realization Metrics =====")
    print(res.per_realization_frame.to_string(index=False))
    print("\n===== Dependency Pair Summary =====")
    print(res.dependency_pair_summary_frame.to_string(index=False))
    return res

if __name__ == "__main__":
    main()
