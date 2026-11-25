from __future__ import annotations

"""
Layer 3 — Projection Accountability Audit

This module implements realization-level projection accountability for generated
population realizations from a learned generative prior P_theta.

For a future scenario S with marginal targets m_j(S), Layer 3 audits each finite
generated population realization X^(r) = {x_i^(r)} as a candidate input for
downstream transport modeling. For each realization, it solves:

    w^(r)* = argmin_w  D_KL(w || p^(r))
             s.t.     sum_i w_i m_hat_j(x_i^(r)) = m_j(S),  j=1,...,J
                      w_i >= 0, sum_i w_i = 1

where p^(r) is the base prior weight vector, usually uniform.

Scenario targets may be constructed from a reference population CSV. If the
reference population is PUMS, pass --reference_weight_col PWGTP so marginal
shares and means are constructed using population weights. For long-term
planning scenarios, use --sigma_mode scenario_uncertainty with attribute- or
target-level sigma JSON to avoid hard-wiring short-term historical uncertainty.

The script reports:
    - expected finite-realization projection distortion, estimated as the mean
      distortion over feasible generated realizations;
    - feasibility / hard-feasibility rates;
    - Layer-2-aligned Gaussian compatibility loss:
          loss_r = 1/2 * sum_j ((s_j^(r) - m_j(S)) / sigma_j)^2;
    - theorem diagnostics for the bound
          d_proj^(r) >= (4 / B_r^2) * loss_r;
    - agent-level ESS/nESS and weight-concentration diagnostics.

Example:
    python audit/layer3_projection_uncertainty_branches.py \
        --realizations_dir generated_ensembles \
        --scenario_reference_csv pums_future.csv \
        --reference_weight_col PWGTP \
        --target_attrs Age Household_Size Employment_status Household_Income_Class Number_of_cars_of_household \
        --projection_mode hard \
        --solver SCIPY \
        --output_dir layer3_outputs \
        --save_projected_weights

By default, if --target_attrs contains raw Age, the script internally derives
Age_Group using --age_bins/--age_bin_labels and uses Age_Group marginals. Use
--age_binning none to retain exact-age value-share constraints.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
import argparse
import json
import math

try:
    import cvxpy as cp
except Exception:
    cp = None

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None


# ============================================================
# Attribute spaces, aligned with layer2_historical_sigma.py / current framework
# ============================================================

FULL_NUM_COLS: List[str] = [
    "Age",
    "Household_Size",
    "Number_of_children",
    "Number_of_cars_of_household",
]

FULL_CAT_COLS: List[str] = [
    "Region_Category",
    "Gender",
    "Marital_status",
    "Employment_status",
    "Studenthood_status",
    "Household_Type",
    "Commute_Mode",
    "Personal_Income_Class",
    "Household_Income_Class",
]

FULL_ATTRS_X: List[str] = FULL_NUM_COLS + FULL_CAT_COLS

TASK_NUM_COLS_XA: List[str] = [
    "Age",
    "Household_Size",
    "Number_of_children",
    "Number_of_cars_of_household",
]

TASK_CAT_COLS_XA: List[str] = [
    "Employment_status",
    "Household_Income_Class",
    "Commute_Mode",
]

TASK_ATTRS_XA: List[str] = TASK_NUM_COLS_XA + TASK_CAT_COLS_XA

# Canonical 7-attr transport-visible set — MUST match Layer-1 (TASK_ATTRS_XA) and
# Layer-2 so all three layers audit the same attributes (incl. Commute_Mode and
# Number_of_children).
DEFAULT_PROJECTION_TARGET_ATTRS: List[str] = [
    "Age",
    "Household_Size",
    "Number_of_children",
    "Number_of_cars_of_household",
    "Employment_status",
    "Household_Income_Class",
    "Commute_Mode",
]

# Default age grouping used when the input files contain raw Age but the audit
# should use coarse age-group marginals rather than one constraint per age value.
DEFAULT_AGE_BIN_SOURCE_COL = "Age"
DEFAULT_AGE_BIN_COL = "Age_Group"
DEFAULT_AGE_BINS: List[float] = [0, 18, 35, 50, 65, 200]
DEFAULT_AGE_LABELS: List[str] = ["0-17", "18-34", "35-49", "50-64", "65+"]


# ============================================================
# Types
# ============================================================

TargetKind = Literal["share", "mean"]
NumericTargetMode = Literal["value_share", "mean"]
SigmaMode = Literal["fixed", "reference_se", "scenario_uncertainty"]
ProjectionMode = Literal["hard", "soft", "uncertainty_set"]
UncertaintyAggregationMode = Literal["global_scalar", "attribute_normalized", "attribute_constraints"]
AgeBinningMode = Literal["none", "default"]
DecisionStatus = Literal["simulation-ready", "use-with-caution", "not-supported"]


@dataclass
class ScenarioTarget:
    """A scalar scenario target m_j(S)."""
    name: str
    attr: str
    kind: TargetKind
    m: float
    sigma: float = 0.02
    value: Optional[Any] = None
    sigma_source: str = "unspecified"


@dataclass
class ScenarioSpec:
    name: str
    targets: List[ScenarioTarget]
    reference_weight_col: Optional[str] = None


@dataclass
class Layer3ProjectionConfig:
    full_attrs: Sequence[str]
    task_attrs: Sequence[str]
    target_attrs: Sequence[str]
    scenario: ScenarioSpec
    projection_mode: ProjectionMode = "hard"
    soft_lambda: float = 100.0
    uncertainty_tau: float = 1.0
    uncertainty_aggregation: UncertaintyAggregationMode = "global_scalar"
    attribute_tau_json: Optional[str] = None
    attribute_weight_json: Optional[str] = None
    solver: str = "SCS"
    solver_max_iters: int = 20000
    solver_eps: float = 1e-6
    feasibility_tol: float = 1e-4
    realization_id_col: str = "realization_id"
    base_weight_col: Optional[str] = None
    missing_token: str = "__MISSING__"
    check_full_attrs: bool = False
    topk_frac: float = 0.05
    save_projected_weights: bool = False
    projected_weight_col: str = "projected_weight"
    base_weight_out_col: str = "base_weight"
    decision_fr_threshold: float = 0.9
    decision_ness_threshold: float = 0.3
    decision_distortion_threshold: Optional[float] = None
    decision_topk_threshold: Optional[float] = None


@dataclass
class RealizationProjectionResult:
    realization_id: Any
    n_agents: int
    n_targets: int
    optimization_success: bool
    hard_constraint_satisfied: bool
    uncertainty_constraint_satisfied: bool
    feasible: bool
    status: str
    solver_status: str
    projection_mode: ProjectionMode
    uncertainty_aggregation: str
    weights: Optional[np.ndarray]
    base_weights: np.ndarray
    distortion: Optional[float]
    gaussian_loss_prior: Optional[float]
    projected_gaussian_loss: Optional[float]
    uncertainty_tau: Optional[float]
    attribute_tau_json: Optional[str]
    attribute_prior_losses_json: Optional[str]
    attribute_projected_losses_json: Optional[str]
    max_attribute_violation: Optional[float]
    n_attribute_violations: Optional[int]
    projection_objective: Optional[float]
    range_bound_B2: Optional[float]
    distortion_lower_bound: Optional[float]
    objective_lower_bound: Optional[float]
    bound_gap: Optional[float]
    theorem_applicable: bool
    target_l2_error: Optional[float]
    target_max_abs_error: Optional[float]
    target_max_std_error: Optional[float]
    agent_ess: Optional[float]    # TRUE agent-level ESS (uses n_c if cell-aggregated; see §12 of LAYER-3 doc)
    agent_ness: Optional[float]   # = agent_ess / N_total_agents; range [1/N, 1]
    cell_ess: Optional[float]     # Cell-level ESS (= what older script versions reported as agent_ess in cell-aggregated mode)
    cell_ness: Optional[float]    # = cell_ess / K_cells; range [1/K, 1]
    entropy: Optional[float]
    entropy_ratio: Optional[float]
    max_weight: Optional[float]        # max per-row weight (cell weight in cell-aggregated mode)
    max_agent_weight: Optional[float]  # max per-AGENT weight (= max_k w_k / n_k); only differs from max_weight in cell-aggregated mode
    topk_mass: Optional[float]
    per_target_errors: pd.DataFrame

    @property
    def ess(self) -> Optional[float]:
        return self.agent_ess

    @property
    def ness(self) -> Optional[float]:
        return self.agent_ness


@dataclass
class Layer3AuditResult:
    scenario_name: str
    n_realizations: int
    n_feasible: int
    n_hard_satisfied: int
    n_uncertainty_satisfied: int
    n_optimization_success: int
    n_targets: int
    feasibility_rate: float
    hard_feasibility_rate: float
    uncertainty_feasibility_rate: float
    projection_success_rate: float
    mean_distortion: Optional[float]
    std_distortion: Optional[float]
    median_distortion: Optional[float]
    iqr_distortion: Optional[float]
    mean_gaussian_loss_prior: Optional[float]
    std_gaussian_loss_prior: Optional[float]
    median_gaussian_loss_prior: Optional[float]
    mean_projected_gaussian_loss: Optional[float]
    mean_projection_objective: Optional[float]
    mean_distortion_lower_bound: Optional[float]
    mean_objective_lower_bound: Optional[float]
    mean_bound_gap: Optional[float]
    mean_agent_ness: Optional[float]
    std_agent_ness: Optional[float]
    mean_agent_ess: Optional[float]
    mean_cell_ness: Optional[float]
    std_cell_ness: Optional[float]
    mean_cell_ess: Optional[float]
    mean_max_weight: Optional[float]
    mean_max_agent_weight: Optional[float]
    mean_topk_mass: Optional[float]
    decision_status: DecisionStatus
    per_realization_diagnostics: pd.DataFrame
    target_summary: pd.DataFrame
    projection_summary: Dict[str, Any]
    projected_weight_files: List[str]

    @property
    def mean_ess(self) -> Optional[float]:
        return self.mean_agent_ess

    @property
    def mean_ness(self) -> Optional[float]:
        return self.mean_agent_ness

    @property
    def std_ness(self) -> Optional[float]:
        return self.std_agent_ness


# ============================================================
# Basic helpers
# ============================================================

def _check_columns(df: pd.DataFrame, required_cols: Sequence[str], df_name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _prepare_cat_series(s: pd.Series, missing_token: str) -> pd.Series:
    out = s.astype("object").copy()
    out = out.where(~out.isna(), other=missing_token)
    return out.astype(str)


def _is_integer_like_series(s: pd.Series, tol: float = 1e-8) -> bool:
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return False
    return bool(np.all(np.abs(x - np.round(x)) <= tol))


def _target_value_to_str(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _parse_float_list(value: str | Sequence[float] | None, default: Sequence[float]) -> List[float]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        out = [float(x.strip()) for x in value.split(",") if x.strip()]
    else:
        out = [float(x) for x in value]
    if len(out) < 2:
        raise ValueError("At least two bin edges are required.")
    if any(out[i] >= out[i + 1] for i in range(len(out) - 1)):
        raise ValueError(f"Bin edges must be strictly increasing, got {out}.")
    return out


def _parse_str_list(value: str | Sequence[str] | None, default: Sequence[str]) -> List[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [str(x) for x in value]


def add_age_group_column(
    df: pd.DataFrame,
    *,
    source_col: str = DEFAULT_AGE_BIN_SOURCE_COL,
    out_col: str = DEFAULT_AGE_BIN_COL,
    bins: Sequence[float] = DEFAULT_AGE_BINS,
    labels: Sequence[str] = DEFAULT_AGE_LABELS,
    missing_token: str = "__MISSING__",
) -> pd.DataFrame:
    """Return a copy of df with an age-group column derived from raw Age.

    The bins are left-closed and right-open, e.g. [18, 35) -> "18-34".
    Missing or out-of-range ages are mapped to missing_token.
    """
    if source_col not in df.columns:
        return df
    if len(labels) != len(bins) - 1:
        raise ValueError("age_bin_labels length must equal len(age_bins)-1.")
    out = df.copy()
    age = pd.to_numeric(out[source_col], errors="coerce")
    cats = pd.cut(age, bins=list(bins), labels=list(labels), right=False, include_lowest=True)
    out[out_col] = cats.astype("object").where(~cats.isna(), other=missing_token).astype(str)
    return out


def maybe_add_age_group_column(
    df: pd.DataFrame,
    *,
    age_binning: AgeBinningMode = "default",
    source_col: str = DEFAULT_AGE_BIN_SOURCE_COL,
    out_col: str = DEFAULT_AGE_BIN_COL,
    bins: Sequence[float] = DEFAULT_AGE_BINS,
    labels: Sequence[str] = DEFAULT_AGE_LABELS,
    missing_token: str = "__MISSING__",
) -> pd.DataFrame:
    if age_binning == "none":
        return df
    return add_age_group_column(df, source_col=source_col, out_col=out_col, bins=bins, labels=labels, missing_token=missing_token)


def normalize_target_attrs_for_age_binning(
    target_attrs: Sequence[str],
    *,
    age_binning: AgeBinningMode = "default",
    source_col: str = DEFAULT_AGE_BIN_SOURCE_COL,
    out_col: str = DEFAULT_AGE_BIN_COL,
) -> List[str]:
    """Replace raw Age in target_attrs with Age_Group when age binning is enabled."""
    mapped: List[str] = []
    for a in target_attrs:
        if age_binning != "none" and a == source_col:
            mapped.append(out_col)
        else:
            mapped.append(a)
    return list(dict.fromkeys(mapped))


def _normalize_prob(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = np.where(np.isfinite(x), x, 0.0)
    x = np.clip(x, eps, None)
    s = float(x.sum())
    if s <= 0:
        raise ValueError("Probability vector has non-positive sum after clipping.")
    return x / s


def _reference_weights(
    df: pd.DataFrame,
    reference_weight_col: Optional[str] = None,
) -> np.ndarray:
    """
    Return normalized reference weights for scenario target construction.

    For PUMS, use reference_weight_col="PWGTP". If omitted, the reference
    population is treated as an unweighted sample.
    """
    if len(df) == 0:
        raise ValueError("Reference population is empty.")
    if reference_weight_col is None:
        return np.ones(len(df), dtype=float) / len(df)
    if reference_weight_col not in df.columns:
        raise ValueError(f"reference_weight_col='{reference_weight_col}' not found in scenario_reference_csv.")
    w = pd.to_numeric(df[reference_weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if np.sum(w > 0) == 0:
        raise ValueError(f"Reference weight column '{reference_weight_col}' contains no positive weights.")
    return _normalize_prob(w)


def _effective_n_from_weights(w: np.ndarray) -> float:
    w = _normalize_prob(w)
    s2 = float(np.sum(w ** 2))
    if s2 <= 0:
        return float("nan")
    return float(1.0 / s2)


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return float("nan")
    ww = _normalize_prob(w[mask])
    return float(np.sum(ww * x[mask]))


def _weighted_var(x: np.ndarray, w: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return float("nan")
    ww = _normalize_prob(w[mask])
    xx = x[mask]
    mu = float(np.sum(ww * xx))
    return float(np.sum(ww * (xx - mu) ** 2))


def _weighted_value_probs(values: pd.Series, w: np.ndarray) -> pd.Series:
    if len(values) != len(w):
        raise ValueError("values and weights must have the same length.")
    df_tmp = pd.DataFrame({"value": values.astype(str).to_numpy(), "w": np.asarray(w, dtype=float)})
    df_tmp = df_tmp[np.isfinite(df_tmp["w"].to_numpy()) & (df_tmp["w"] > 0)]
    if df_tmp.empty:
        raise ValueError("No positive reference weights available for value probabilities.")
    probs = df_tmp.groupby("value", dropna=False)["w"].sum().sort_index()
    probs = probs / probs.sum()
    return probs


def kl_divergence(q: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    q = _normalize_prob(q, eps=eps)
    p = _normalize_prob(p, eps=eps)
    return float(np.sum(q * np.log(q / p)))


def agent_level_ess_from_cell_weights(
    cell_weights: np.ndarray,
    cell_sizes: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """TRUE agent-level ESS when cells are size-weighted.

    Given per-cell projected weights w_c (sum to 1) and per-cell agent counts n_c,
    each agent in cell c has weight w_c / n_c (uniform within cell). Then:
        sum_a w_a   = sum_c n_c * (w_c / n_c) = sum_c w_c = 1
        sum_a w_a^2 = sum_c n_c * (w_c / n_c)^2 = sum_c (w_c^2 / n_c)
        agent_ESS    = 1 / sum_c (w_c^2 / n_c)

    See §12 of LAYER-3_FORMULATION.md for derivation.
    """
    w = _normalize_prob(cell_weights, eps=eps)
    nc = np.maximum(cell_sizes.astype(float), eps)
    denom = float(np.sum(w ** 2 / nc))
    if denom <= 0:
        return 0.0
    return 1.0 / denom


def effective_sample_size(weights: np.ndarray, eps: float = 1e-12) -> float:
    w = _normalize_prob(weights, eps=eps)
    s2 = float(np.sum(w ** 2))
    if s2 <= 0:
        return float("nan")
    return float(1.0 / s2)


def entropy_of_weights(weights: np.ndarray, eps: float = 1e-300) -> float:
    w = _normalize_prob(weights, eps=1e-12)
    return float(-np.sum(w * np.log(np.clip(w, eps, 1.0))))


def topk_weight_mass(weights: np.ndarray, topk_frac: float = 0.05) -> float:
    if not (0.0 < topk_frac <= 1.0):
        raise ValueError("topk_frac must be in (0, 1].")
    w = _normalize_prob(weights)
    k = max(1, int(math.ceil(len(w) * topk_frac)))
    return float(np.sum(np.sort(w)[::-1][:k]))


def _solve_options(solver: str, eps: float, max_iters: int) -> Dict[str, Any]:
    solver_upper = solver.upper()
    if solver_upper == "SCS":
        return {"eps": eps, "max_iters": max_iters}
    if solver_upper == "CLARABEL":
        return {}
    return {}


# ============================================================
# Input realizations
# ============================================================

def load_realizations_from_csv(
    path: str | Path,
    realization_id_col: str = "realization_id",
) -> List[pd.DataFrame]:
    df = _read_csv(path)
    if realization_id_col not in df.columns:
        raise ValueError(
            f"Combined realizations CSV must contain '{realization_id_col}'. "
            "Alternatively use --realizations_dir."
        )
    realizations: List[pd.DataFrame] = []
    for rid, g in df.groupby(realization_id_col, sort=True):
        out = g.copy()
        out[realization_id_col] = rid
        realizations.append(out.reset_index(drop=True))
    if len(realizations) == 0:
        raise ValueError(f"No realizations found in {path}")
    return realizations


def load_realizations_from_dir(
    path: str | Path,
    realization_id_col: str = "realization_id",
    glob_pattern: str = "*.csv",
) -> List[pd.DataFrame]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Realizations directory not found: {path}")
    files = sorted(path.glob(glob_pattern))
    if not files:
        raise ValueError(f"No CSV files found in {path} with pattern {glob_pattern}")

    realizations: List[pd.DataFrame] = []
    for k, fp in enumerate(files):
        df = pd.read_csv(fp)
        if realization_id_col not in df.columns:
            df[realization_id_col] = k
        realizations.append(df.reset_index(drop=True))
    return realizations


# ============================================================
# Scenario target construction
# ============================================================

def _share_feature(df: pd.DataFrame, attr: str, value: Any, missing_token: str) -> np.ndarray:
    s = _prepare_cat_series(df[attr], missing_token)
    return (s == str(value)).astype(float).to_numpy()


def _mean_feature(df: pd.DataFrame, attr: str) -> np.ndarray:
    x = pd.to_numeric(df[attr], errors="coerce")
    if x.isna().any():
        fill = float(x.dropna().mean()) if x.dropna().shape[0] > 0 else 0.0
        x = x.fillna(fill)
    return x.astype(float).to_numpy()


def compute_target_stat(df: pd.DataFrame, target: ScenarioTarget, missing_token: str = "__MISSING__") -> float:
    if target.kind == "share":
        if target.value is None:
            raise ValueError(f"Share target '{target.name}' must have a value.")
        return float(_share_feature(df, target.attr, target.value, missing_token).mean()) if len(df) > 0 else np.nan
    if target.kind == "mean":
        x = pd.to_numeric(df[target.attr], errors="coerce").dropna()
        return float(x.mean()) if len(x) > 0 else np.nan
    raise ValueError(f"Unsupported target kind: {target.kind}")


def build_scenario_from_reference(
    reference_df: pd.DataFrame,
    target_attrs: Sequence[str],
    task_attrs: Sequence[str],
    missing_token: str = "__MISSING__",
    numeric_mode: NumericTargetMode = "value_share",
    max_numeric_categories: int = 150,
    sigma_mode: SigmaMode = "fixed",
    sigma_fixed: float = 0.02,
    sigma_floor: float = 0.005,
    sigma_scale: float = 1.0,
    scenario_name: str = "scenario_from_reference",
    reference_weight_col: Optional[str] = None,
) -> ScenarioSpec:
    """
    Build scenario marginal targets from a reference population.

    For PUMS reference data, pass reference_weight_col="PWGTP" so target
    shares/means are population-weighted.

    For categorical attributes: creates weighted share targets for each category.
    For numeric attributes:
        - value_share: creates weighted share targets for each integer/category-like value.
        - mean: creates a weighted scalar mean target.
    """
    _check_columns(reference_df, target_attrs, "reference_df")
    for a in target_attrs:
        if a not in task_attrs:
            raise ValueError(f"Target attribute '{a}' is not in task_attrs X_A.")

    ref_w = _reference_weights(reference_df, reference_weight_col=reference_weight_col)
    n_eff = _effective_n_from_weights(ref_w)
    targets: List[ScenarioTarget] = []

    for attr in target_attrs:
        s = reference_df[attr]
        is_numeric = pd.api.types.is_numeric_dtype(s) or pd.to_numeric(s, errors="coerce").notna().mean() > 0.95

        if is_numeric and numeric_mode == "mean":
            x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            m = _weighted_mean(x, ref_w)
            if not np.isfinite(m):
                continue
            if sigma_mode == "reference_se":
                var = _weighted_var(x, ref_w)
                sigma = max(float(math.sqrt(max(var, 0.0) / max(n_eff, 1.0))), sigma_floor)
            else:
                sigma = sigma_fixed
            targets.append(ScenarioTarget(name=f"{attr}__mean", attr=attr, kind="mean", m=float(m), sigma=float(sigma), sigma_source=str(sigma_mode)))
            continue

        if is_numeric:
            x_series = pd.to_numeric(s, errors="coerce")
            x_nonmiss = x_series.dropna()
            if _is_integer_like_series(x_nonmiss) and x_nonmiss.nunique() <= max_numeric_categories:
                s_prepared = x_series.round().astype("Int64").astype("object")
                s_prepared = s_prepared.where(~x_series.isna(), other=missing_token).astype(str)
            else:
                x = x_series.to_numpy(dtype=float)
                m = _weighted_mean(x, ref_w)
                if not np.isfinite(m):
                    continue
                if sigma_mode == "reference_se":
                    var = _weighted_var(x, ref_w)
                    sigma = max(float(math.sqrt(max(var, 0.0) / max(n_eff, 1.0))), sigma_floor)
                else:
                    sigma = sigma_fixed
                targets.append(ScenarioTarget(name=f"{attr}__mean", attr=attr, kind="mean", m=float(m), sigma=float(sigma), sigma_source=str(sigma_mode)))
                continue
        else:
            s_prepared = _prepare_cat_series(s, missing_token)

        probs = _weighted_value_probs(s_prepared, ref_w)
        for value, p in probs.items():
            p = float(p)
            if sigma_mode == "reference_se":
                se = math.sqrt(max(p * (1.0 - p), 0.0) / max(n_eff, 1.0))
                sigma = max(sigma_scale * se, sigma_floor)
            else:
                sigma = sigma_fixed
            value_str = _target_value_to_str(value)
            targets.append(
                ScenarioTarget(
                    name=f"{attr}=={value_str}",
                    attr=attr,
                    kind="share",
                    value=value_str,
                    m=p,
                    sigma=float(sigma),
                    sigma_source=str(sigma_mode),
                )
            )

    if len(targets) == 0:
        raise ValueError("No scenario targets were constructed from the reference data.")
    return ScenarioSpec(name=scenario_name, targets=targets, reference_weight_col=reference_weight_col)


def load_scenario_targets_json(path: str | Path, scenario_name: str = "scenario_from_json") -> ScenarioSpec:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario targets JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    reference_weight_col: Optional[str] = None
    if isinstance(data, dict):
        scenario_name = str(data.get("name", scenario_name))
        reference_weight_col = data.get("reference_weight_col", None)
        raw_targets = data.get("targets", [])
    elif isinstance(data, list):
        raw_targets = data
    else:
        raise ValueError("Scenario targets JSON must be a list or a dict with a 'targets' field.")

    targets: List[ScenarioTarget] = []
    for idx, t in enumerate(raw_targets):
        kind = str(t.get("kind", "share"))
        if kind not in {"share", "mean"}:
            raise ValueError(f"Target {idx} has unsupported kind={kind}")
        name = str(t.get("name", f"target_{idx}"))
        attr = str(t["attr"])
        m = float(t["m"])
        sigma = float(t.get("sigma", 0.02))
        value = t.get("value", None)
        targets.append(ScenarioTarget(name=name, attr=attr, kind=kind, value=value, m=m, sigma=sigma, sigma_source="json"))

    if not targets:
        raise ValueError("Scenario targets JSON contains no targets.")
    return ScenarioSpec(name=scenario_name, targets=targets, reference_weight_col=reference_weight_col)


def apply_scenario_uncertainty_sigmas(
    scenario: ScenarioSpec,
    *,
    attribute_sigma: Optional[Dict[str, float]] = None,
    target_sigma: Optional[Dict[str, float]] = None,
    default_sigma: float = 0.02,
    sigma_floor: float = 1e-12,
    sigma_scale: float = 1.0,
    source: str = "scenario_uncertainty",
) -> pd.DataFrame:
    """Apply planning/scenario-defined diagonal uncertainty to Layer-3 targets.

    This keeps the general projection formulation available when scenario
    uncertainty is defined by a planning horizon, policy tolerance, or expert
    specification rather than historical short-term dynamics. Values in
    target_sigma override values in attribute_sigma; otherwise default_sigma
    is used.
    """
    attribute_sigma = attribute_sigma or {}
    target_sigma = target_sigma or {}
    if default_sigma <= 0:
        raise ValueError("default_sigma must be positive.")
    if sigma_floor <= 0:
        raise ValueError("sigma_floor must be positive.")
    if sigma_scale <= 0:
        raise ValueError("sigma_scale must be positive.")

    rows: List[Dict[str, Any]] = []
    for t in scenario.targets:
        if t.name in target_sigma:
            raw_sigma = float(target_sigma[t.name])
            sigma_key = t.name
            sigma_level = "target"
        elif t.attr in attribute_sigma:
            raw_sigma = float(attribute_sigma[t.attr])
            sigma_key = t.attr
            sigma_level = "attribute"
        elif t.attr == DEFAULT_AGE_BIN_COL and DEFAULT_AGE_BIN_SOURCE_COL in attribute_sigma:
            # Convenience alias: users often specify Age even when age_binning maps it to Age_Group.
            raw_sigma = float(attribute_sigma[DEFAULT_AGE_BIN_SOURCE_COL])
            sigma_key = DEFAULT_AGE_BIN_SOURCE_COL
            sigma_level = "attribute_alias"
        else:
            raw_sigma = float(default_sigma)
            sigma_key = "default_sigma"
            sigma_level = "default"
        if not np.isfinite(raw_sigma) or raw_sigma <= 0:
            raise ValueError(f"Scenario sigma for {t.name!r} must be positive, got {raw_sigma}.")
        sigma = float(max(float(sigma_floor), raw_sigma) * float(sigma_scale))
        t.sigma = sigma
        t.sigma_source = source
        rows.append({
            "target": t.name,
            "attr": t.attr,
            "kind": t.kind,
            "value": t.value,
            "target_m": t.m,
            "sigma": sigma,
            "sigma_source": source,
            "sigma_level": sigma_level,
            "sigma_key": sigma_key,
            "raw_sigma": raw_sigma,
            "sigma_floor": sigma_floor,
            "sigma_scale": sigma_scale,
        })
    return pd.DataFrame(rows)


# ============================================================
# Feature matrix and base weights
# ============================================================

def build_feature_matrix(
    df: pd.DataFrame,
    scenario: ScenarioSpec,
    missing_token: str = "__MISSING__",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    required_cols = sorted({t.attr for t in scenario.targets})
    _check_columns(df, required_cols, "realization")

    cols: List[np.ndarray] = []
    target_values: List[float] = []
    sigma_values: List[float] = []
    rows: List[Dict[str, Any]] = []

    for target in scenario.targets:
        if target.kind == "share":
            if target.value is None:
                raise ValueError(f"Share target '{target.name}' must have a value.")
            feat = _share_feature(df, target.attr, target.value, missing_token)
        elif target.kind == "mean":
            feat = _mean_feature(df, target.attr)
        else:
            raise ValueError(f"Unsupported target kind: {target.kind}")

        cols.append(feat.astype(float))
        target_values.append(float(target.m))
        sigma_values.append(max(float(target.sigma), 1e-12))
        rows.append({
            "target": target.name,
            "attr": target.attr,
            "kind": target.kind,
            "value": target.value,
            "target_m": float(target.m),
            "sigma": float(target.sigma),
            "sigma_source": getattr(target, "sigma_source", None),
            "unweighted_stat": float(np.mean(feat)) if len(feat) > 0 else np.nan,
        })

    if len(cols) == 0:
        raise ValueError("Scenario has no targets.")

    F = np.column_stack(cols)
    t_vec = np.asarray(target_values, dtype=float)
    sigma_vec = np.asarray(sigma_values, dtype=float)
    meta = pd.DataFrame(rows)
    return F, t_vec, sigma_vec, meta


def get_base_weights(df: pd.DataFrame, base_weight_col: Optional[str] = None) -> np.ndarray:
    if len(df) == 0:
        raise ValueError("Population realization is empty.")
    if base_weight_col is None:
        return np.ones(len(df), dtype=float) / len(df)
    if base_weight_col not in df.columns:
        raise ValueError(f"base_weight_col='{base_weight_col}' not found in realization columns.")
    weights = pd.to_numeric(df[base_weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if np.sum(weights > 0) == 0:
        raise ValueError(f"Base weight column '{base_weight_col}' contains no positive weights.")
    return _normalize_prob(weights)


# ============================================================
# Uncertainty aggregation helpers
# ============================================================

def _load_json_mapping(spec: Optional[str], *, name: str) -> Dict[str, float]:
    """Load an attribute mapping from a JSON string or a JSON file path."""
    if spec is None or str(spec).strip() == "":
        return {}
    s = str(spec).strip()
    try:
        if s.startswith("{"):
            data = json.loads(s)
        else:
            fp = Path(s)
            if not fp.exists():
                raise FileNotFoundError(f"{name} file not found: {fp}")
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse {name}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a JSON object mapping attribute names to numbers.")
    out: Dict[str, float] = {}
    for k, v in data.items():
        out[str(k)] = float(v)
    return out


def _attribute_indices(target_meta: pd.DataFrame) -> Dict[str, np.ndarray]:
    if "attr" not in target_meta.columns:
        raise ValueError("target_meta must contain an 'attr' column.")
    groups: Dict[str, List[int]] = {}
    for idx, attr in enumerate(target_meta["attr"].astype(str).tolist()):
        groups.setdefault(attr, []).append(idx)
    return {a: np.asarray(idxs, dtype=int) for a, idxs in groups.items()}


def _resolve_attribute_weights(target_meta: pd.DataFrame, spec: Optional[str]) -> Dict[str, float]:
    provided = _load_json_mapping(spec, name="attribute_weight_json")
    attrs = sorted(_attribute_indices(target_meta).keys())
    weights: Dict[str, float] = {}
    for a in attrs:
        w = float(provided.get(a, 1.0))
        if not np.isfinite(w) or w < 0:
            raise ValueError(f"Attribute weight for '{a}' must be finite and non-negative, got {w}.")
        weights[a] = w
    return weights


def _resolve_attribute_taus(target_meta: pd.DataFrame, default_tau: float, spec: Optional[str]) -> Dict[str, float]:
    if default_tau < 0:
        raise ValueError("uncertainty_tau must be non-negative.")
    provided = _load_json_mapping(spec, name="attribute_tau_json")
    attrs = sorted(_attribute_indices(target_meta).keys())
    taus: Dict[str, float] = {}
    for a in attrs:
        tau = float(provided.get(a, default_tau))
        if not np.isfinite(tau) or tau < 0:
            raise ValueError(f"Attribute tau for '{a}' must be finite and non-negative, got {tau}.")
        taus[a] = tau
    return taus


def _std_error_from_weights(F: np.ndarray, t: np.ndarray, sigma: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (F.T @ _normalize_prob(w) - t) / np.maximum(sigma, 1e-12)


def _attribute_losses_from_std_error(std_err: np.ndarray, target_meta: pd.DataFrame) -> Dict[str, float]:
    groups = _attribute_indices(target_meta)
    e = np.asarray(std_err, dtype=float).reshape(-1)
    out: Dict[str, float] = {}
    for attr, idx in groups.items():
        if len(idx) == 0:
            out[attr] = 0.0
        else:
            out[attr] = float(0.5 * np.mean(e[idx] ** 2))
    return out


def _global_scalar_loss_from_std_error(std_err: np.ndarray) -> float:
    e = np.asarray(std_err, dtype=float).reshape(-1)
    return float(0.5 * np.sum(e ** 2))


def _aggregate_attribute_loss(
    attr_losses: Dict[str, float],
    attr_weights: Dict[str, float],
) -> float:
    return float(sum(float(attr_weights.get(a, 1.0)) * float(v) for a, v in attr_losses.items()))


def _uncertainty_loss_from_std_error(
    std_err: np.ndarray,
    target_meta: pd.DataFrame,
    aggregation: UncertaintyAggregationMode,
    attr_weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    attr_losses = _attribute_losses_from_std_error(std_err, target_meta)
    if aggregation == "global_scalar":
        return _global_scalar_loss_from_std_error(std_err), attr_losses
    if aggregation == "attribute_normalized":
        return _aggregate_attribute_loss(attr_losses, attr_weights), attr_losses
    if aggregation == "attribute_constraints":
        # This aggregate is only for diagnostics/soft penalty. Feasibility is checked
        # component-wise against attribute-specific tau values.
        return _aggregate_attribute_loss(attr_losses, attr_weights), attr_losses
    raise ValueError(f"Unsupported uncertainty_aggregation={aggregation!r}")


def _attribute_violations(
    attr_losses: Dict[str, float],
    attr_taus: Dict[str, float],
) -> Tuple[float, int, Dict[str, float]]:
    violations: Dict[str, float] = {}
    for attr, loss in attr_losses.items():
        tau = float(attr_taus.get(attr, 0.0))
        violations[attr] = float(loss - tau)
    max_violation = max(violations.values()) if violations else 0.0
    n_violations = int(sum(v > 1e-10 for v in violations.values()))
    return float(max_violation), n_violations, violations


def _json_dumps_float_dict(d: Optional[Dict[str, float]]) -> Optional[str]:
    if d is None:
        return None
    return json.dumps({str(k): float(v) for k, v in d.items()}, ensure_ascii=False, sort_keys=True)


def _uncertainty_loss_and_grad_scipy(
    w: np.ndarray,
    *,
    F: np.ndarray,
    t: np.ndarray,
    sigma: np.ndarray,
    target_meta: pd.DataFrame,
    aggregation: UncertaintyAggregationMode,
    attr_weights: Dict[str, float],
) -> Tuple[float, np.ndarray, Dict[str, float]]:
    """Return selected uncertainty loss, its gradient wrt w, and attribute losses."""
    F = np.asarray(F, dtype=float)
    t = np.asarray(t, dtype=float).reshape(-1)
    sigma = np.maximum(np.asarray(sigma, dtype=float).reshape(-1), 1e-12)
    w = _normalize_prob(w)
    err = (F.T @ w - t) / sigma
    groups = _attribute_indices(target_meta)
    coeff = np.zeros_like(err, dtype=float)
    attr_losses: Dict[str, float] = {}

    if aggregation == "global_scalar":
        coeff[:] = 1.0
        loss = float(0.5 * np.sum(err ** 2))
        attr_losses = _attribute_losses_from_std_error(err, target_meta)
    elif aggregation in {"attribute_normalized", "attribute_constraints"}:
        loss = 0.0
        for attr, idx in groups.items():
            if len(idx) == 0:
                attr_losses[attr] = 0.0
                continue
            attr_loss = float(0.5 * np.mean(err[idx] ** 2))
            attr_losses[attr] = attr_loss
            weight = float(attr_weights.get(attr, 1.0))
            loss += weight * attr_loss
            coeff[idx] = weight / float(len(idx))
    else:
        raise ValueError(f"Unsupported uncertainty_aggregation={aggregation!r}")

    grad = F @ (coeff * err / sigma)
    return float(loss), grad, attr_losses


def _attribute_loss_and_grad_scipy(
    w: np.ndarray,
    *,
    F: np.ndarray,
    t: np.ndarray,
    sigma: np.ndarray,
    target_meta: pd.DataFrame,
    attr: str,
) -> Tuple[float, np.ndarray]:
    F = np.asarray(F, dtype=float)
    t = np.asarray(t, dtype=float).reshape(-1)
    sigma = np.maximum(np.asarray(sigma, dtype=float).reshape(-1), 1e-12)
    w = _normalize_prob(w)
    err = (F.T @ w - t) / sigma
    groups = _attribute_indices(target_meta)
    idx = groups[attr]
    coeff = np.zeros_like(err, dtype=float)
    coeff[idx] = 1.0 / float(len(idx))
    loss = float(0.5 * np.mean(err[idx] ** 2))
    grad = F @ (coeff * err / sigma)
    return loss, grad


def _cvx_uncertainty_expr(
    standardized_error: Any,
    target_meta: pd.DataFrame,
    aggregation: UncertaintyAggregationMode,
    attr_weights: Dict[str, float],
) -> Any:
    groups = _attribute_indices(target_meta)
    if aggregation == "global_scalar":
        return 0.5 * cp.sum_squares(standardized_error)
    terms = []
    for attr, idx in groups.items():
        weight = float(attr_weights.get(attr, 1.0))
        terms.append(weight * 0.5 / float(len(idx)) * cp.sum_squares(standardized_error[idx]))
    if not terms:
        return 0.0
    return sum(terms)


def _cvx_attribute_loss_exprs(standardized_error: Any, target_meta: pd.DataFrame) -> Dict[str, Any]:
    groups = _attribute_indices(target_meta)
    return {
        attr: 0.5 / float(len(idx)) * cp.sum_squares(standardized_error[idx])
        for attr, idx in groups.items()
    }


def _range_bound_B2(
    *,
    F: np.ndarray,
    sigma: np.ndarray,
    target_meta: pd.DataFrame,
    aggregation: UncertaintyAggregationMode,
    attr_weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    sigma = np.maximum(np.asarray(sigma, dtype=float).reshape(-1), 1e-12)
    groups = _attribute_indices(target_meta)
    attr_B2: Dict[str, float] = {}
    if aggregation == "global_scalar":
        G = F / sigma.reshape(1, -1)
        ranges = np.max(G, axis=0) - np.min(G, axis=0)
        B2 = float(np.sum(ranges ** 2))
        for attr, idx in groups.items():
            attr_B2[attr] = float(np.sum(ranges[idx] ** 2))
        return B2, attr_B2

    B2_total = 0.0
    for attr, idx in groups.items():
        if len(idx) == 0:
            attr_B2[attr] = 0.0
            continue
        if aggregation == "attribute_normalized":
            scale = math.sqrt(float(attr_weights.get(attr, 1.0)) / float(len(idx)))
        else:
            # Attribute-specific constraints use chi_a = 0.5/|C_a| sum z^2.
            scale = math.sqrt(1.0 / float(len(idx)))
        G = F[:, idx] * (scale / sigma[idx]).reshape(1, -1)
        ranges = np.max(G, axis=0) - np.min(G, axis=0)
        val = float(np.sum(ranges ** 2))
        attr_B2[attr] = val
        B2_total += val
    return float(B2_total), attr_B2


def _projection_lower_bound(
    *,
    prior_loss: float,
    projected_loss_radius: float,
    B2: float,
) -> Optional[float]:
    if B2 <= 0 or not np.isfinite(B2):
        return None
    return float((4.0 / B2) * max(math.sqrt(max(prior_loss, 0.0)) - math.sqrt(max(projected_loss_radius, 0.0)), 0.0) ** 2)


# ============================================================
# Optimization backends
# ============================================================

def _kl_objective_and_grad(w: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> Tuple[float, np.ndarray]:
    w_safe = np.clip(np.asarray(w, dtype=float), eps, None)
    p_safe = np.clip(np.asarray(p, dtype=float), eps, None)
    val = float(np.sum(w_safe * np.log(w_safe / p_safe)))
    grad = np.log(w_safe / p_safe) + 1.0
    return val, grad


def _solve_projection_problem_scipy(
    *,
    p: np.ndarray,
    F: np.ndarray,
    t: np.ndarray,
    sigma: np.ndarray,
    target_meta: pd.DataFrame,
    mode: ProjectionMode,
    soft_lambda: float,
    uncertainty_tau: float,
    uncertainty_aggregation: UncertaintyAggregationMode,
    attr_taus: Dict[str, float],
    attr_weights: Dict[str, float],
    max_iters: int,
    feasibility_tol: float,
) -> Tuple[Optional[np.ndarray], str, str]:
    """Solve the realization-level projection problem with scipy/SLSQP.

    Modes:
        hard:
            minimize KL(w||p) subject to exact target equality F.T @ w = t.
        soft:
            minimize KL(w||p) + 2*soft_lambda*chi_H(w,S), where chi_H is
            selected by --uncertainty_aggregation.
        uncertainty_set:
            minimize KL(w||p) subject to either a global/attribute-normalized
            uncertainty set chi_H(w,S) <= uncertainty_tau, or component-wise
            attribute constraints chi_{H,a}(w,S) <= tau_a.
    """
    if minimize is None:
        return None, "solver_unavailable", "solver_unavailable: neither cvxpy nor scipy.optimize.minimize is available"

    p = _normalize_prob(p)
    N = len(p)
    F = np.asarray(F, dtype=float)
    t = np.asarray(t, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)

    if mode == "uncertainty_set" and uncertainty_tau < 0:
        raise ValueError("uncertainty_tau must be non-negative for projection_mode='uncertainty_set'.")

    def uncertainty_loss_grad(w: np.ndarray) -> Tuple[float, np.ndarray, Dict[str, float]]:
        return _uncertainty_loss_and_grad_scipy(
            w,
            F=F,
            t=t,
            sigma=sigma,
            target_meta=target_meta,
            aggregation=uncertainty_aggregation,
            attr_weights=attr_weights,
        )

    def obj(w: np.ndarray) -> float:
        kl, _ = _kl_objective_and_grad(w, p)
        if mode == "soft":
            chi, _, _ = uncertainty_loss_grad(w)
            return float(kl + 2.0 * soft_lambda * chi)
        return kl

    def jac(w: np.ndarray) -> np.ndarray:
        _, g = _kl_objective_and_grad(w, p)
        if mode == "soft":
            _, grad_chi, _ = uncertainty_loss_grad(w)
            g = g + 2.0 * soft_lambda * grad_chi
        return g

    if mode == "hard":
        A_full = np.vstack([np.ones((1, N)), F.T])
        b_full = np.concatenate([[1.0], t])
        keep: List[int] = []
        current = np.empty((0, N), dtype=float)
        for idx in range(A_full.shape[0]):
            candidate = np.vstack([current, A_full[idx:idx + 1]])
            if np.linalg.matrix_rank(candidate, tol=1e-10) > np.linalg.matrix_rank(current, tol=1e-10):
                keep.append(idx)
                current = candidate
        A_eq = A_full[keep]
        b_eq = b_full[keep]
        constraints: List[Dict[str, Any]] = [
            {"type": "eq", "fun": lambda w, A=A_eq, b=b_eq: A @ w - b, "jac": lambda w, A=A_eq: A},
        ]
    elif mode == "uncertainty_set":
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones(N)},
        ]
        if uncertainty_aggregation == "attribute_constraints":
            for attr in sorted(_attribute_indices(target_meta).keys()):
                tau_a = float(attr_taus[attr])
                constraints.append({
                    "type": "ineq",
                    "fun": lambda w, a=attr, tau=tau_a: float(tau - _attribute_loss_and_grad_scipy(
                        w, F=F, t=t, sigma=sigma, target_meta=target_meta, attr=a
                    )[0]),
                    "jac": lambda w, a=attr: -_attribute_loss_and_grad_scipy(
                        w, F=F, t=t, sigma=sigma, target_meta=target_meta, attr=a
                    )[1],
                })
        else:
            constraints.append({
                "type": "ineq",
                "fun": lambda w: float(uncertainty_tau - uncertainty_loss_grad(w)[0]),
                "jac": lambda w: -uncertainty_loss_grad(w)[1],
            })
    else:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones(N)},
        ]

    bounds = [(0.0, 1.0) for _ in range(N)]
    res = minimize(
        obj,
        p.copy(),
        jac=jac,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": int(max_iters), "ftol": float(feasibility_tol), "disp": False},
    )

    if not res.success or res.x is None:
        return None, f"scipy_{res.status}", f"solver_error: {res.message}"

    w_star = _normalize_prob(np.maximum(np.asarray(res.x, dtype=float), 0.0))
    if mode == "hard":
        max_abs = float(np.max(np.abs(F.T @ w_star - t)))
        if max_abs > max(10.0 * feasibility_tol, 1e-8):
            return None, "scipy_inaccurate", f"target_error_above_tol:{max_abs:.6g}"
    if mode == "uncertainty_set":
        chi, _, attr_losses = uncertainty_loss_grad(w_star)
        if uncertainty_aggregation == "attribute_constraints":
            max_violation, _, _ = _attribute_violations(attr_losses, attr_taus)
            if max_violation > max(10.0 * feasibility_tol, 1e-8):
                return None, "scipy_inaccurate", f"attribute_uncertainty_violation:{max_violation:.6g}"
        else:
            if chi > uncertainty_tau + max(10.0 * feasibility_tol, 1e-8):
                return None, "scipy_inaccurate", f"uncertainty_set_violation:{chi:.6g}>{uncertainty_tau:.6g}"
    return w_star, "optimal", "ok"


def _solve_projection_problem_cvxpy(
    *,
    p: np.ndarray,
    F: np.ndarray,
    t: np.ndarray,
    sigma: np.ndarray,
    target_meta: pd.DataFrame,
    mode: ProjectionMode,
    soft_lambda: float,
    uncertainty_tau: float,
    uncertainty_aggregation: UncertaintyAggregationMode,
    attr_taus: Dict[str, float],
    attr_weights: Dict[str, float],
    solver: str,
    solver_eps: float,
    solver_max_iters: int,
    verbose: bool = False,
) -> Tuple[Optional[np.ndarray], str, str]:
    if cp is None:
        return None, "cvxpy_unavailable", "cvxpy_unavailable"
    if mode == "uncertainty_set" and uncertainty_tau < 0:
        raise ValueError("uncertainty_tau must be non-negative for projection_mode='uncertainty_set'.")

    # --- Exact pre-aggregation of exchangeable cells -------------------------
    # Cells with identical constraint rows F_i are interchangeable: they enter
    # every constraint identically, and the unique KL-optimal solution splits
    # each group's mass proportionally to the prior p (w_i = w_g * p_i / p_g).
    # Hence KL(w||p) = KL(w_group||p_group) and solving the reduced problem then
    # expanding back yields the *identical* optimum with far fewer exponential
    # cones (cp.kl_div emits one exp-cone per variable; CLARABEL is unstable
    # above ~25k of them). This is a result-preserving solver optimization, not
    # a change to the projection's objective, constraints, or optimum.
    p_orig = np.asarray(p, dtype=float)
    N_orig = F.shape[0]
    F_unique, inverse_idx = np.unique(F, axis=0, return_inverse=True)
    inverse_idx = np.asarray(inverse_idx).reshape(-1)
    merged = F_unique.shape[0] < N_orig
    if merged:
        p_merged = np.zeros(F_unique.shape[0], dtype=float)
        np.add.at(p_merged, inverse_idx, p_orig)
        F = F_unique
        p = _normalize_prob(p_merged)

    N, _J = F.shape
    w = cp.Variable(N, nonneg=True)
    kl_obj = cp.sum(cp.kl_div(w, p))
    constraints = [cp.sum(w) == 1]
    standardized_error = cp.multiply(1.0 / sigma, F.T @ w - t)
    chi_expr = _cvx_uncertainty_expr(standardized_error, target_meta, uncertainty_aggregation, attr_weights)

    if mode == "hard":
        constraints.append(F.T @ w == t)
        objective = kl_obj
    elif mode == "soft":
        objective = kl_obj + float(2.0 * soft_lambda) * chi_expr
    elif mode == "uncertainty_set":
        if uncertainty_aggregation == "attribute_constraints":
            attr_exprs = _cvx_attribute_loss_exprs(standardized_error, target_meta)
            for attr, expr in attr_exprs.items():
                constraints.append(expr <= float(attr_taus[attr]))
        else:
            constraints.append(chi_expr <= float(uncertainty_tau))
        objective = kl_obj
    else:
        raise ValueError("mode must be 'hard', 'soft', or 'uncertainty_set'.")

    problem = cp.Problem(cp.Minimize(objective), constraints)
    options = _solve_options(solver=solver, eps=solver_eps, max_iters=solver_max_iters)

    try:
        problem.solve(solver=solver, verbose=verbose, **options)
    except Exception as e:
        return None, "solver_error", f"solver_error: {e}"

    if w.value is None or problem.status not in {"optimal", "optimal_inaccurate"}:
        return None, problem.status, problem.status

    w_star = np.maximum(np.asarray(w.value).reshape(-1), 0.0)
    if merged:
        # expand merged solution back to all original cells (split ∝ prior)
        denom = p[inverse_idx]
        share = np.divide(p_orig, denom, out=np.zeros_like(p_orig), where=denom > 0)
        w_star = w_star[inverse_idx] * share
    w_star = _normalize_prob(w_star)
    return w_star, problem.status, "ok"


def _solve_projection_problem_scalable(
    *,
    p: np.ndarray,
    F: np.ndarray,
    t: np.ndarray,
    sigma: np.ndarray,
    target_meta: pd.DataFrame,
    mode: ProjectionMode,
    uncertainty_tau: float,
    uncertainty_aggregation: UncertaintyAggregationMode,
    attr_taus: Dict[str, float],
    solver_max_iters: int,
    feasibility_tol: float,
    verbose: bool = False,
) -> Tuple[Optional[np.ndarray], str, str]:
    """Exp-cone-free solver for the KL I-projection (uncertainty_set + attribute_constraints).

    Solves the IDENTICAL program as the cvxpy/CLARABEL path
        min_w  KL(w||p)   s.t.   1'w = 1,  w >= 0,
               0.5/|C_a| * sum_{j in C_a} ((F_j'w - t_j)/sigma_j)^2 <= tau_a   for each attr a
    via its low-dimensional dual. Introducing moments y_a = F_a' w and dualizing that coupling
    with lambda_a in R^{|C_a|} gives a smooth concave dual in lambda (dim J = #targets = 63):

        g(lambda) = -log( sum_i p_i exp(-(F lambda)_i) )
                    - sum_a [ lambda_a' t_a + R_a * ||sigma_a (.) lambda_a|| ],   R_a = sqrt(2 tau_a |C_a|)

    (the ball-support term comes from min over the attribute uncertainty ellipsoid). Maximizing
    g is an unconstrained J-dim problem solved by L-BFGS, O(N*J) per step, with no per-cell
    exponential cones (CLARABEL segfaults above ~25k of them) and no N-dim factorization, so it
    scales to any N and any conditioning. The primal optimum is recovered exactly as
    w_i \propto p_i exp(-(F lambda*)_i). Result is identical (to optimizer tolerance) to the
    cvxpy/CLARABEL solve of the same program.
    """
    from scipy.optimize import minimize
    if mode != "uncertainty_set" or uncertainty_aggregation != "attribute_constraints":
        return None, "unsupported", "scalable solver supports uncertainty_set+attribute_constraints only"

    p = _normalize_prob(np.asarray(p, dtype=float))
    F = np.asarray(F, dtype=float)
    t = np.asarray(t, dtype=float).reshape(-1)
    sigma2 = np.maximum(np.asarray(sigma, dtype=float).reshape(-1), 1e-12) ** 2
    J = F.shape[1]
    groups = _attribute_indices(target_meta)
    gidx = {a: np.asarray(idx, dtype=int) for a, idx in groups.items()}
    R = {a: float(np.sqrt(2.0 * float(attr_taus[a]) * len(idx))) for a, idx in groups.items()}
    EPS = 1e-12  # smoothing of the (non-smooth at 0) ball-support norm

    def _w_of(lam):
        a_ = -(F @ lam); m = float(a_.max())
        ex = p * np.exp(a_ - m); Z = float(ex.sum())
        return ex / Z

    def vgh(lam, want_hess):
        w = _w_of(lam)
        a_ = -(F @ lam); m = float(a_.max())
        val = m + np.log(float((p * np.exp(a_ - m)).sum()))
        Fw = F.T @ w
        grad = -Fw.copy()
        for a, idx in gidx.items():
            la = lam[idx]; s2 = sigma2[idx]
            nrm = float(np.sqrt(np.sum(s2 * la * la) + EPS))
            val += float(t[idx] @ la) + R[a] * nrm
            grad[idx] += t[idx] + R[a] * (s2 * la) / nrm
        if not want_hess:
            return val, grad, None
        H = (F.T * w) @ F - np.outer(Fw, Fw)          # Cov_w(F), PSD
        for a, idx in gidx.items():
            la = lam[idx]; s2 = sigma2[idx]
            q = float(np.sum(s2 * la * la) + EPS); nrm = np.sqrt(q)
            g2 = s2 * la
            H[np.ix_(idx, idx)] += R[a] * (np.diag(s2) / nrm - np.outer(g2, g2) / (q * nrm))
        return val, grad, H

    # warm start with L-BFGS (cheap, badly-scaled-robust), then polish with Newton.
    res = minimize(lambda l: vgh(l, False)[:2], np.zeros(J), jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "maxfun": 8000, "ftol": 1e-16, "gtol": 1e-12})
    lam = np.asarray(res.x, dtype=float)
    val, grad, _ = vgh(lam, False)
    for _ in range(60):                                # damped-Newton refinement
        if float(np.max(np.abs(grad))) < 1e-11:
            break
        _, grad, H = vgh(lam, True)
        try:
            step = -np.linalg.solve(H + 1e-12 * np.eye(J), grad)
        except np.linalg.LinAlgError:
            break
        s = 1.0; gTs = float(grad @ step)
        if gTs >= 0:
            break
        while s > 1e-12:                               # backtracking line search
            vnew = vgh(lam + s * step, False)[0]
            if vnew <= val + 1e-4 * s * gTs:
                break
            s *= 0.5
        lam = lam + s * step
        val, grad, _ = vgh(lam, False)

    w_star = _normalize_prob(np.maximum(_w_of(lam), 0.0))
    status = "optimal" if float(np.max(np.abs(grad))) < 1e-7 else "optimal_inaccurate"
    return w_star, status, "ok"


def _solve_projection_problem(
    *,
    p: np.ndarray,
    F: np.ndarray,
    t: np.ndarray,
    sigma: np.ndarray,
    target_meta: pd.DataFrame,
    mode: ProjectionMode,
    soft_lambda: float,
    uncertainty_tau: float,
    uncertainty_aggregation: UncertaintyAggregationMode,
    attr_taus: Dict[str, float],
    attr_weights: Dict[str, float],
    solver: str,
    solver_eps: float,
    solver_max_iters: int,
    feasibility_tol: float,
    verbose: bool = False,
) -> Tuple[Optional[np.ndarray], str, str]:
    p = _normalize_prob(p)
    F = np.asarray(F, dtype=float)
    t = np.asarray(t, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)

    if F.ndim != 2:
        raise ValueError("F must be a 2D array of shape (N, J).")
    N, J = F.shape
    if len(p) != N:
        raise ValueError(f"Length of p ({len(p)}) must match F.shape[0] ({N}).")
    if t.shape[0] != J or sigma.shape[0] != J:
        raise ValueError("t and sigma must have length F.shape[1].")
    if soft_lambda <= 0:
        raise ValueError("soft_lambda must be positive.")
    if mode == "uncertainty_set" and uncertainty_tau < 0:
        raise ValueError("uncertainty_tau must be non-negative for projection_mode='uncertainty_set'.")

    if solver.upper() in ("SCALABLE", "TRUSTKL", "KLTRUST"):
        return _solve_projection_problem_scalable(
            p=p, F=F, t=t, sigma=sigma, target_meta=target_meta, mode=mode,
            uncertainty_tau=uncertainty_tau, uncertainty_aggregation=uncertainty_aggregation,
            attr_taus=attr_taus, solver_max_iters=solver_max_iters,
            feasibility_tol=feasibility_tol, verbose=verbose,
        )

    if solver.upper() == "SCIPY" or cp is None:
        return _solve_projection_problem_scipy(
            p=p,
            F=F,
            t=t,
            sigma=sigma,
            target_meta=target_meta,
            mode=mode,
            soft_lambda=soft_lambda,
            uncertainty_tau=uncertainty_tau,
            uncertainty_aggregation=uncertainty_aggregation,
            attr_taus=attr_taus,
            attr_weights=attr_weights,
            max_iters=solver_max_iters,
            feasibility_tol=feasibility_tol,
        )

    w_star, solver_status, status = _solve_projection_problem_cvxpy(
        p=p,
        F=F,
        t=t,
        sigma=sigma,
        target_meta=target_meta,
        mode=mode,
        soft_lambda=soft_lambda,
        uncertainty_tau=uncertainty_tau,
        uncertainty_aggregation=uncertainty_aggregation,
        attr_taus=attr_taus,
        attr_weights=attr_weights,
        solver=solver,
        solver_eps=solver_eps,
        solver_max_iters=solver_max_iters,
        verbose=verbose,
    )
    if w_star is not None:
        return w_star, solver_status, status

    if status.startswith("solver_error") or solver_status in {"solver_error", "cvxpy_unavailable"}:
        return _solve_projection_problem_scipy(
            p=p,
            F=F,
            t=t,
            sigma=sigma,
            target_meta=target_meta,
            mode=mode,
            soft_lambda=soft_lambda,
            uncertainty_tau=uncertainty_tau,
            uncertainty_aggregation=uncertainty_aggregation,
            attr_taus=attr_taus,
            attr_weights=attr_weights,
            max_iters=solver_max_iters,
            feasibility_tol=feasibility_tol,
        )
    return w_star, solver_status, status


# ============================================================
# Layer-3 realization-level optimization
# ============================================================

def _theorem_diagnostics(F: np.ndarray, t: np.ndarray, sigma: np.ndarray, p: np.ndarray) -> Tuple[float, float, Optional[float], bool]:
    prior_stats = F.T @ _normalize_prob(p)
    std_prior_err = (prior_stats - t) / np.maximum(sigma, 1e-12)
    gaussian_loss = float(0.5 * np.sum(std_prior_err ** 2))
    G = F / np.maximum(sigma, 1e-12).reshape(1, -1)
    ranges = np.max(G, axis=0) - np.min(G, axis=0)
    B2 = float(np.sum(ranges ** 2))
    if B2 > 0 and np.isfinite(B2):
        lower_bound = float((4.0 / B2) * gaussian_loss)
        applicable = True
    else:
        lower_bound = None
        applicable = False
    return gaussian_loss, B2, lower_bound, applicable


def _per_target_error_table(
    *,
    target_meta: pd.DataFrame,
    F: np.ndarray,
    t: np.ndarray,
    sigma: np.ndarray,
    p: np.ndarray,
    w: Optional[np.ndarray],
    realization_id: Any,
) -> pd.DataFrame:
    prior_stats = F.T @ _normalize_prob(p)
    if w is None:
        projected_stats = np.full_like(t, np.nan, dtype=float)
    else:
        projected_stats = F.T @ _normalize_prob(w)

    out = target_meta.copy()
    out.insert(0, "realization_id", realization_id)
    out["prior_stat"] = prior_stats
    out["projected_stat"] = projected_stats
    out["target_m"] = t
    out["sigma"] = sigma
    out["abs_err_prior"] = np.abs(prior_stats - t)
    out["abs_err_projected"] = np.abs(projected_stats - t)
    out["std_err_prior"] = out["abs_err_prior"] / np.maximum(sigma, 1e-12)
    out["std_err_projected"] = out["abs_err_projected"] / np.maximum(sigma, 1e-12)
    out["delta_projected_minus_prior"] = projected_stats - prior_stats
    return out


def project_one_realization(
    df: pd.DataFrame,
    config: Layer3ProjectionConfig,
    realization_index: int,
    verbose: bool = False,
) -> RealizationProjectionResult:
    if config.check_full_attrs:
        _check_columns(df, config.full_attrs, f"realization_{realization_index}")

    required_cols = sorted({t.attr for t in config.scenario.targets})
    for c in required_cols:
        if c not in config.task_attrs:
            raise ValueError(f"Scenario target attr '{c}' is not in task_attrs X_A.")

    rid = df[config.realization_id_col].iloc[0] if config.realization_id_col in df.columns else realization_index
    p = get_base_weights(df, base_weight_col=config.base_weight_col)
    F, t_vec, sigma_vec, target_meta = build_feature_matrix(df, config.scenario, missing_token=config.missing_token)

    aggregation = config.uncertainty_aggregation
    tau = float(config.uncertainty_tau)
    attr_weights = _resolve_attribute_weights(target_meta, config.attribute_weight_json)
    attr_taus = _resolve_attribute_taus(target_meta, tau, config.attribute_tau_json)

    prior_std_err = _std_error_from_weights(F, t_vec, sigma_vec, p)
    gaussian_loss, prior_attr_losses = _uncertainty_loss_from_std_error(
        prior_std_err,
        target_meta,
        aggregation,
        attr_weights,
    )
    B2, attr_B2 = _range_bound_B2(
        F=F,
        sigma=sigma_vec,
        target_meta=target_meta,
        aggregation=aggregation,
        attr_weights=attr_weights,
    )
    theorem_applicable = bool(B2 > 0 and np.isfinite(B2))
    hard_lower_bound = _projection_lower_bound(prior_loss=gaussian_loss, projected_loss_radius=0.0, B2=B2)

    if aggregation == "attribute_constraints":
        attr_trust_bounds: Dict[str, float] = {}
        for attr, prior_loss_a in prior_attr_losses.items():
            B2_a = attr_B2.get(attr, 0.0)
            lb_a = _projection_lower_bound(
                prior_loss=prior_loss_a,
                projected_loss_radius=attr_taus.get(attr, tau),
                B2=B2_a,
            )
            attr_trust_bounds[attr] = 0.0 if lb_a is None else float(lb_a)
        trust_lower_bound = float(max(attr_trust_bounds.values())) if attr_trust_bounds else None
    else:
        trust_lower_bound = _projection_lower_bound(prior_loss=gaussian_loss, projected_loss_radius=tau, B2=B2)

    w_star, solver_status, status = _solve_projection_problem(
        p=p,
        F=F,
        t=t_vec,
        sigma=sigma_vec,
        target_meta=target_meta,
        mode=config.projection_mode,
        soft_lambda=config.soft_lambda,
        uncertainty_tau=tau,
        uncertainty_aggregation=aggregation,
        attr_taus=attr_taus,
        attr_weights=attr_weights,
        solver=config.solver,
        solver_eps=config.solver_eps,
        solver_max_iters=config.solver_max_iters,
        feasibility_tol=config.feasibility_tol,
        verbose=verbose,
    )

    attr_tau_json = _json_dumps_float_dict(attr_taus)
    prior_attr_json = _json_dumps_float_dict(prior_attr_losses)

    if w_star is None:
        per_target_errors = _per_target_error_table(
            target_meta=target_meta, F=F, t=t_vec, sigma=sigma_vec, p=p, w=None, realization_id=rid
        )
        soft_objective_lower_bound = None
        if theorem_applicable:
            # Bound for min_Q KL + eta*chi_H(Q,S), where eta = 2*soft_lambda.
            eta = 2.0 * float(config.soft_lambda)
            soft_objective_lower_bound = float((4.0 * eta * gaussian_loss) / (4.0 + eta * B2))
        return RealizationProjectionResult(
            realization_id=rid,
            n_agents=int(len(df)),
            n_targets=len(config.scenario.targets),
            optimization_success=False,
            hard_constraint_satisfied=False,
            uncertainty_constraint_satisfied=False,
            feasible=False,
            status=status,
            solver_status=solver_status,
            projection_mode=config.projection_mode,
            uncertainty_aggregation=aggregation,
            weights=None,
            base_weights=p,
            distortion=None,
            gaussian_loss_prior=gaussian_loss,
            projected_gaussian_loss=None,
            uncertainty_tau=tau,
            attribute_tau_json=attr_tau_json,
            attribute_prior_losses_json=prior_attr_json,
            attribute_projected_losses_json=None,
            max_attribute_violation=None,
            n_attribute_violations=None,
            projection_objective=None,
            range_bound_B2=B2,
            distortion_lower_bound=None,
            objective_lower_bound=soft_objective_lower_bound if config.projection_mode == "soft" else trust_lower_bound,
            bound_gap=None,
            theorem_applicable=False,
            target_l2_error=None,
            target_max_abs_error=None,
            target_max_std_error=None,
            agent_ess=None,
            agent_ness=None,
            cell_ess=None,
            cell_ness=None,
            entropy=None,
            entropy_ratio=None,
            max_weight=None,
            max_agent_weight=None,
            topk_mass=None,
            per_target_errors=per_target_errors,
        )

    projected_stats = F.T @ w_star
    errors = projected_stats - t_vec
    abs_errors = np.abs(errors)
    std_errors_signed = errors / np.maximum(sigma_vec, 1e-12)
    std_errors = np.abs(std_errors_signed)
    projected_gaussian_loss, projected_attr_losses = _uncertainty_loss_from_std_error(
        std_errors_signed,
        target_meta,
        aggregation,
        attr_weights,
    )
    max_attr_violation, n_attr_violations, _attr_violations = _attribute_violations(projected_attr_losses, attr_taus)

    hard_constraint_satisfied = bool(np.max(abs_errors) <= config.feasibility_tol)
    if aggregation == "attribute_constraints":
        uncertainty_constraint_satisfied = bool(max_attr_violation <= max(config.feasibility_tol, 1e-10))
    else:
        uncertainty_constraint_satisfied = bool(projected_gaussian_loss <= tau + max(config.feasibility_tol, 1e-10))
    optimization_success = True

    if config.projection_mode == "hard":
        feasible = hard_constraint_satisfied
        if not hard_constraint_satisfied:
            status = f"target_error_above_tol:{float(np.max(abs_errors)):.6g}"
    elif config.projection_mode == "uncertainty_set":
        feasible = uncertainty_constraint_satisfied
        if not uncertainty_constraint_satisfied:
            if aggregation == "attribute_constraints":
                status = f"attribute_uncertainty_violation:{max_attr_violation:.6g}"
            else:
                status = f"uncertainty_set_violation:{projected_gaussian_loss:.6g}>{tau:.6g}"
    else:
        feasible = optimization_success

    distortion = kl_divergence(w_star, p)
    if config.projection_mode == "soft":
        projection_objective = float(distortion + 2.0 * config.soft_lambda * projected_gaussian_loss)
    else:
        projection_objective = float(distortion)

    # Mode-specific lower-bound diagnostics.
    distortion_lower_bound_final: Optional[float] = None
    objective_lower_bound: Optional[float] = None
    theorem_applicable_final = False
    if theorem_applicable and B2 is not None and B2 > 0 and np.isfinite(B2):
        if config.projection_mode == "hard" and hard_constraint_satisfied:
            distortion_lower_bound_final = hard_lower_bound
            objective_lower_bound = hard_lower_bound
            theorem_applicable_final = hard_lower_bound is not None
        elif config.projection_mode == "uncertainty_set" and uncertainty_constraint_satisfied:
            distortion_lower_bound_final = trust_lower_bound
            objective_lower_bound = trust_lower_bound
            theorem_applicable_final = trust_lower_bound is not None
        elif config.projection_mode == "soft":
            # Lower bound for KL + eta*chi_H with eta = 2*soft_lambda.
            eta = 2.0 * float(config.soft_lambda)
            objective_lower_bound = float((4.0 * eta * gaussian_loss) / (4.0 + eta * B2))
            theorem_applicable_final = True

    bound_gap = None
    if config.projection_mode in {"hard", "uncertainty_set"}:
        if distortion_lower_bound_final is not None and np.isfinite(distortion):
            bound_gap = float(distortion - distortion_lower_bound_final)
    elif objective_lower_bound is not None and projection_objective is not None:
        bound_gap = float(projection_objective - objective_lower_bound)

    # ----- ESS computations (see §12 of LAYER-3_FORMULATION.md) -----
    # cell-level ESS / nESS: legacy script behavior (one weight per ROW of input)
    cell_ess = effective_sample_size(w_star)
    cell_ness = float(cell_ess / len(w_star)) if len(w_star) > 0 and np.isfinite(cell_ess) else np.nan
    # TRUE agent-level ESS / nESS: account for per-cell agent count n_c when input is cell-aggregated
    if config.base_weight_col is not None and config.base_weight_col in df.columns:
        n_per_row = pd.to_numeric(df[config.base_weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        # agent-level input: each row is one agent
        n_per_row = np.ones(len(w_star), dtype=float)
    N_total_agents = float(n_per_row.sum())
    agent_ess = agent_level_ess_from_cell_weights(w_star, n_per_row)
    agent_ness = float(agent_ess / N_total_agents) if N_total_agents > 0 and np.isfinite(agent_ess) else np.nan
    # max per-AGENT weight (= w_c / n_c maximized over cells)
    w_norm = _normalize_prob(w_star)
    max_agent_weight = float(np.max(w_norm / np.maximum(n_per_row, 1e-12)))

    entropy = entropy_of_weights(w_star)
    entropy_ratio = float(entropy / np.log(len(w_star))) if len(w_star) > 1 else 1.0
    max_weight = float(np.max(w_star))
    topk_mass = topk_weight_mass(w_star, topk_frac=config.topk_frac)

    per_target_errors = _per_target_error_table(
        target_meta=target_meta, F=F, t=t_vec, sigma=sigma_vec, p=p, w=w_star, realization_id=rid
    )

    return RealizationProjectionResult(
        realization_id=rid,
        n_agents=int(len(df)),
        n_targets=len(config.scenario.targets),
        optimization_success=optimization_success,
        hard_constraint_satisfied=hard_constraint_satisfied,
        uncertainty_constraint_satisfied=uncertainty_constraint_satisfied,
        feasible=feasible,
        status=status,
        solver_status=solver_status,
        projection_mode=config.projection_mode,
        uncertainty_aggregation=aggregation,
        weights=w_star,
        base_weights=p,
        distortion=distortion,
        gaussian_loss_prior=gaussian_loss,
        projected_gaussian_loss=projected_gaussian_loss,
        uncertainty_tau=tau,
        attribute_tau_json=attr_tau_json,
        attribute_prior_losses_json=prior_attr_json,
        attribute_projected_losses_json=_json_dumps_float_dict(projected_attr_losses),
        max_attribute_violation=max_attr_violation,
        n_attribute_violations=n_attr_violations,
        projection_objective=projection_objective,
        range_bound_B2=B2,
        distortion_lower_bound=distortion_lower_bound_final,
        objective_lower_bound=objective_lower_bound,
        bound_gap=bound_gap,
        theorem_applicable=theorem_applicable_final,
        target_l2_error=float(np.linalg.norm(errors)),
        target_max_abs_error=float(np.max(abs_errors)),
        target_max_std_error=float(np.max(std_errors)),
        agent_ess=agent_ess,
        agent_ness=agent_ness,
        cell_ess=cell_ess,
        cell_ness=cell_ness,
        entropy=entropy,
        entropy_ratio=entropy_ratio,
        max_weight=max_weight,
        max_agent_weight=max_agent_weight,
        topk_mass=topk_mass,
        per_target_errors=per_target_errors,
    )


# ============================================================
# Layer-3 ensemble-level audit
# ============================================================

def _finite_values(values: Sequence[Optional[float]]) -> np.ndarray:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    return arr[np.isfinite(arr)]


def _mean_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    arr = _finite_values(values)
    return float(np.mean(arr)) if len(arr) else None


def _std_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    arr = _finite_values(values)
    return float(np.std(arr, ddof=1)) if len(arr) > 1 else (0.0 if len(arr) == 1 else None)


def _median_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    arr = _finite_values(values)
    return float(np.median(arr)) if len(arr) else None


def _iqr_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    arr = _finite_values(values)
    if len(arr) == 0:
        return None
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def _determine_decision_status(
    *,
    feasibility_rate: float,
    mean_distortion: Optional[float],
    mean_agent_ness: Optional[float],
    mean_topk_mass: Optional[float],
    config: Layer3ProjectionConfig,
) -> DecisionStatus:
    if feasibility_rate <= 0:
        return "not-supported"
    if feasibility_rate < config.decision_fr_threshold:
        return "not-supported"
    if mean_agent_ness is not None and mean_agent_ness < config.decision_ness_threshold:
        return "use-with-caution"
    if config.decision_distortion_threshold is not None:
        if mean_distortion is not None and mean_distortion > config.decision_distortion_threshold:
            return "use-with-caution"
    if config.decision_topk_threshold is not None:
        if mean_topk_mass is not None and mean_topk_mass > config.decision_topk_threshold:
            return "use-with-caution"
    return "simulation-ready"


def run_layer3_audit(
    realizations: List[pd.DataFrame],
    config: Layer3ProjectionConfig,
    output_dir: Optional[str | Path] = None,
    verbose: bool = False,
) -> Layer3AuditResult:
    if len(realizations) == 0:
        raise ValueError("No realizations provided.")

    realization_results: List[RealizationProjectionResult] = []
    projected_weight_files: List[str] = []
    out: Optional[Path] = None
    weights_out: Optional[Path] = None

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if config.save_projected_weights:
            weights_out = out / "projected_weights"
            weights_out.mkdir(parents=True, exist_ok=True)

    for k, df_k in enumerate(realizations):
        res = project_one_realization(df_k, config=config, realization_index=k, verbose=verbose)
        realization_results.append(res)

        if config.save_projected_weights and weights_out is not None and res.weights is not None:
            df_save = df_k.copy()
            df_save[config.base_weight_out_col] = res.base_weights
            df_save[config.projected_weight_col] = res.weights
            safe_rid = str(res.realization_id).replace("/", "_").replace("\\", "_").replace(" ", "_")
            fp = weights_out / f"projected_weights_realization_{safe_rid}.csv"
            df_save.to_csv(fp, index=False)
            projected_weight_files.append(str(fp))

    diag_rows: List[Dict[str, Any]] = []
    target_tables: List[pd.DataFrame] = []

    for res in realization_results:
        topk_col = f"top_{int(round(config.topk_frac * 100))}_pct_mass"
        diag_rows.append({
            "scenario_name": config.scenario.name,
            "realization_id": res.realization_id,
            "n_agents": res.n_agents,
            "n_targets": res.n_targets,
            "projection_mode": res.projection_mode,
            "uncertainty_aggregation": res.uncertainty_aggregation,
            "attribute_tau_json": res.attribute_tau_json,
            "attribute_prior_losses_json": res.attribute_prior_losses_json,
            "attribute_projected_losses_json": res.attribute_projected_losses_json,
            "max_attribute_violation": res.max_attribute_violation,
            "n_attribute_violations": res.n_attribute_violations,
            "optimization_success": int(res.optimization_success),
            "hard_constraint_satisfied": int(res.hard_constraint_satisfied),
            "uncertainty_constraint_satisfied": int(res.uncertainty_constraint_satisfied),
            "feasible": int(res.feasible),
            "status": res.status,
            "solver_status": res.solver_status,
            "distortion": res.distortion,
            "gaussian_loss_prior": res.gaussian_loss_prior,
            "projected_gaussian_loss": res.projected_gaussian_loss,
            "uncertainty_tau": res.uncertainty_tau,
            "projection_objective": res.projection_objective,
            "range_bound_B2": res.range_bound_B2,
            "distortion_lower_bound": res.distortion_lower_bound,
            "objective_lower_bound": res.objective_lower_bound,
            "bound_gap": res.bound_gap,
            "theorem_applicable": int(res.theorem_applicable),
            "target_l2_error": res.target_l2_error,
            "target_max_abs_error": res.target_max_abs_error,
            "target_max_std_error": res.target_max_std_error,
            "agent_ESS": res.agent_ess,           # TRUE per-agent ESS (uses n_c in cell-aggregated mode)
            "agent_nESS": res.agent_ness,         # = agent_ess / N_total_agents
            "cell_ESS": res.cell_ess,             # per-row (=per-cell in cell-aggregated mode) ESS
            "cell_nESS": res.cell_ness,           # = cell_ess / K_cells
            "ESS": res.agent_ess,                 # legacy alias
            "nESS": res.agent_ness,               # legacy alias
            "entropy": res.entropy,
            "entropy_ratio": res.entropy_ratio,
            "max_weight": res.max_weight,                 # per-row weight
            "max_agent_weight": res.max_agent_weight,     # per-AGENT weight (= max_k w_k / n_k)
            topk_col: res.topk_mass,
        })
        target_tables.append(res.per_target_errors)

    per_diag = pd.DataFrame(diag_rows)
    target_long = pd.concat(target_tables, ignore_index=True) if target_tables else pd.DataFrame()

    n_realizations = len(per_diag)
    n_feasible = int(per_diag["feasible"].astype(bool).sum())
    n_hard_satisfied = int(per_diag["hard_constraint_satisfied"].astype(bool).sum())
    n_uncertainty_satisfied = int(per_diag["uncertainty_constraint_satisfied"].astype(bool).sum())
    n_optimization_success = int(per_diag["optimization_success"].astype(bool).sum())
    feasibility_rate = float(n_feasible / n_realizations) if n_realizations > 0 else 0.0
    hard_feasibility_rate = float(n_hard_satisfied / n_realizations) if n_realizations > 0 else 0.0
    uncertainty_feasibility_rate = float(n_uncertainty_satisfied / n_realizations) if n_realizations > 0 else 0.0
    projection_success_rate = float(n_optimization_success / n_realizations) if n_realizations > 0 else 0.0

    feasible_diag = per_diag.loc[per_diag["feasible"].astype(bool)].copy()
    topk_col = f"top_{int(round(config.topk_frac * 100))}_pct_mass"

    distortion_values = feasible_diag["distortion"].tolist() if not feasible_diag.empty else []
    gaussian_values = feasible_diag["gaussian_loss_prior"].tolist() if not feasible_diag.empty else []
    projected_gaussian_values = feasible_diag["projected_gaussian_loss"].tolist() if not feasible_diag.empty else []
    objective_values = feasible_diag["projection_objective"].tolist() if not feasible_diag.empty else []
    lower_values = feasible_diag["distortion_lower_bound"].tolist() if not feasible_diag.empty else []
    objective_lower_values = feasible_diag["objective_lower_bound"].tolist() if not feasible_diag.empty else []
    gap_values = feasible_diag["bound_gap"].tolist() if not feasible_diag.empty else []
    agent_ness_values = feasible_diag["agent_nESS"].tolist() if not feasible_diag.empty else []
    agent_ess_values = feasible_diag["agent_ESS"].tolist() if not feasible_diag.empty else []
    cell_ness_values = feasible_diag["cell_nESS"].tolist() if not feasible_diag.empty else []
    cell_ess_values = feasible_diag["cell_ESS"].tolist() if not feasible_diag.empty else []
    max_weight_values = feasible_diag["max_weight"].tolist() if not feasible_diag.empty else []
    max_agent_weight_values = feasible_diag["max_agent_weight"].tolist() if not feasible_diag.empty else []
    topk_values = feasible_diag[topk_col].tolist() if not feasible_diag.empty else []

    mean_distortion = _mean_or_none(distortion_values)
    std_distortion = _std_or_none(distortion_values)
    median_distortion = _median_or_none(distortion_values)
    iqr_distortion = _iqr_or_none(distortion_values)
    mean_gaussian_loss_prior = _mean_or_none(gaussian_values)
    std_gaussian_loss_prior = _std_or_none(gaussian_values)
    median_gaussian_loss_prior = _median_or_none(gaussian_values)
    mean_projected_gaussian_loss = _mean_or_none(projected_gaussian_values)
    mean_projection_objective = _mean_or_none(objective_values)
    mean_distortion_lower_bound = _mean_or_none(lower_values)
    mean_objective_lower_bound = _mean_or_none(objective_lower_values)
    mean_bound_gap = _mean_or_none(gap_values)
    mean_agent_ness = _mean_or_none(agent_ness_values)
    std_agent_ness = _std_or_none(agent_ness_values)
    mean_agent_ess = _mean_or_none(agent_ess_values)
    mean_cell_ness = _mean_or_none(cell_ness_values)
    std_cell_ness = _std_or_none(cell_ness_values)
    mean_cell_ess = _mean_or_none(cell_ess_values)
    mean_max_weight = _mean_or_none(max_weight_values)
    mean_max_agent_weight = _mean_or_none(max_agent_weight_values)
    mean_topk_mass = _mean_or_none(topk_values)

    decision_status = _determine_decision_status(
        feasibility_rate=feasibility_rate,
        mean_distortion=mean_distortion,
        mean_agent_ness=mean_agent_ness,
        mean_topk_mass=mean_topk_mass,
        config=config,
    )

    if not target_long.empty:
        target_summary = (
            target_long
            .groupby(["target", "attr", "kind", "value", "target_m", "sigma"], dropna=False)
            .agg(
                prior_mean=("prior_stat", "mean"),
                projected_mean=("projected_stat", "mean"),
                prior_abs_err_mean=("abs_err_prior", "mean"),
                projected_abs_err_mean=("abs_err_projected", "mean"),
                prior_std_err_mean=("std_err_prior", "mean"),
                projected_std_err_mean=("std_err_projected", "mean"),
                projected_abs_err_max=("abs_err_projected", "max"),
            )
            .reset_index()
        )
    else:
        target_summary = pd.DataFrame()

    projection_summary: Dict[str, Any] = {
        "scenario_name": config.scenario.name,
        "reference_weight_col": config.scenario.reference_weight_col,
        "projection_mode": config.projection_mode,
        "uncertainty_aggregation": config.uncertainty_aggregation,
        "soft_lambda": config.soft_lambda,
        "uncertainty_tau": config.uncertainty_tau,
        "attribute_tau_json": config.attribute_tau_json,
        "attribute_weight_json": config.attribute_weight_json,
        "n_realizations": n_realizations,
        "n_feasible": n_feasible,
        "n_hard_satisfied": n_hard_satisfied,
        "n_uncertainty_satisfied": n_uncertainty_satisfied,
        "n_optimization_success": n_optimization_success,
        "n_targets": len(config.scenario.targets),
        "feasibility_rate": feasibility_rate,
        "hard_feasibility_rate": hard_feasibility_rate,
        "uncertainty_feasibility_rate": uncertainty_feasibility_rate,
        "projection_success_rate": projection_success_rate,
        "mean_distortion": mean_distortion,
        "std_distortion": std_distortion,
        "median_distortion": median_distortion,
        "iqr_distortion": iqr_distortion,
        "mean_gaussian_loss_prior": mean_gaussian_loss_prior,
        "std_gaussian_loss_prior": std_gaussian_loss_prior,
        "median_gaussian_loss_prior": median_gaussian_loss_prior,
        "mean_projected_gaussian_loss": mean_projected_gaussian_loss,
        "mean_projection_objective": mean_projection_objective,
        "mean_distortion_lower_bound": mean_distortion_lower_bound,
        "mean_objective_lower_bound": mean_objective_lower_bound,
        "mean_bound_gap": mean_bound_gap,
        "mean_agent_ESS": mean_agent_ess,
        "mean_agent_nESS": mean_agent_ness,
        "std_agent_nESS": std_agent_ness,
        "mean_cell_ESS": mean_cell_ess,
        "mean_cell_nESS": mean_cell_ness,
        "std_cell_nESS": std_cell_ness,
        "mean_ESS": mean_agent_ess,                # legacy alias = agent_ess
        "mean_nESS": mean_agent_ness,              # legacy alias = agent_ness
        "mean_max_weight": mean_max_weight,
        "mean_max_agent_weight": mean_max_agent_weight,
        "mean_topk_mass": mean_topk_mass,
        "topk_frac": config.topk_frac,
        "decision_status": decision_status,
        "decision_thresholds": {
            "feasibility_rate_min": config.decision_fr_threshold,
            "agent_nESS_min": config.decision_ness_threshold,
            "distortion_max": config.decision_distortion_threshold,
            "topk_mass_max": config.decision_topk_threshold,
        },
        "interpretation": {
            "mean_distortion": "Monte Carlo estimate of expected finite-realization projection distortion conditional on feasibility.",
            "gaussian_loss_prior": "Layer-2-aligned prior mismatch, 0.5 * sum standardized target errors squared.",
            "projected_gaussian_loss": "Residual scenario mismatch after projection, measured in the same sigma-normalized units.",
            "uncertainty_tau": "For projection_mode='uncertainty_set', global_scalar/attribute_normalized require projected_gaussian_loss <= uncertainty_tau; attribute_constraints uses per-attribute tau values.",
            "uncertainty_aggregation": "global_scalar: raw target-level sum; attribute_normalized: category-count-normalized attribute losses with optional weights; attribute_constraints: per-attribute uncertainty set constraints.",
            "projection_objective": "For hard/uncertainty_set this equals KL distortion; for soft it equals KL + soft_lambda * sum standardized errors squared.",
            "agent_nESS": "TRUE per-AGENT normalized effective sample size: 1/(N * sum_c w_c^2/n_c). Accounts for cell sizes when input is cell-aggregated. Deployment-relevant: 'how many distinct agents effectively contribute'. See §12 of LAYER-3 doc.",
            "cell_nESS": "Per-ROW (=per-CELL in cell-aggregated mode) normalized ESS: 1/(K * sum_c w_c^2). Measures joint-pattern diversity of the posterior. Equals agent_nESS only when all cells have equal size.",
            "max_agent_weight": "Maximum per-agent weight (= max_k w_k / n_k). Deployment-relevant concentration metric.",
            "distortion_lower_bound": "Mode-specific theorem diagnostic: hard bound or uncertainty-set bound on KL distortion.",
            "objective_lower_bound": "For soft mode, theorem diagnostic for the penalized objective lower bound.",
            "simulation-ready": "projection is feasible and not strongly degenerate under the configured thresholds",
            "use-with-caution": "projection is feasible but shows notable distortion or weight concentration",
            "not-supported": "scenario is not stably supported by the generated prior realizations",
        },
    }

    result = Layer3AuditResult(
        scenario_name=config.scenario.name,
        n_realizations=n_realizations,
        n_feasible=n_feasible,
        n_hard_satisfied=n_hard_satisfied,
        n_uncertainty_satisfied=n_uncertainty_satisfied,
        n_optimization_success=n_optimization_success,
        n_targets=len(config.scenario.targets),
        feasibility_rate=feasibility_rate,
        hard_feasibility_rate=hard_feasibility_rate,
        uncertainty_feasibility_rate=uncertainty_feasibility_rate,
        projection_success_rate=projection_success_rate,
        mean_distortion=mean_distortion,
        std_distortion=std_distortion,
        median_distortion=median_distortion,
        iqr_distortion=iqr_distortion,
        mean_gaussian_loss_prior=mean_gaussian_loss_prior,
        std_gaussian_loss_prior=std_gaussian_loss_prior,
        median_gaussian_loss_prior=median_gaussian_loss_prior,
        mean_projected_gaussian_loss=mean_projected_gaussian_loss,
        mean_projection_objective=mean_projection_objective,
        mean_distortion_lower_bound=mean_distortion_lower_bound,
        mean_objective_lower_bound=mean_objective_lower_bound,
        mean_bound_gap=mean_bound_gap,
        mean_agent_ness=mean_agent_ness,
        std_agent_ness=std_agent_ness,
        mean_agent_ess=mean_agent_ess,
        mean_cell_ness=mean_cell_ness,
        std_cell_ness=std_cell_ness,
        mean_cell_ess=mean_cell_ess,
        mean_max_weight=mean_max_weight,
        mean_max_agent_weight=mean_max_agent_weight,
        mean_topk_mass=mean_topk_mass,
        decision_status=decision_status,
        per_realization_diagnostics=per_diag,
        target_summary=target_summary,
        projection_summary=projection_summary,
        projected_weight_files=projected_weight_files,
    )

    if out is not None:
        save_layer3_outputs(result, out)

    return result


# ============================================================
# Outputs
# ============================================================

def layer3_result_to_summary_dict(result: Layer3AuditResult) -> Dict[str, Any]:
    return {
        "scenario_name": result.scenario_name,
        "n_realizations": result.n_realizations,
        "n_feasible": result.n_feasible,
        "n_hard_satisfied": result.n_hard_satisfied,
        "n_uncertainty_satisfied": result.n_uncertainty_satisfied,
        "n_optimization_success": result.n_optimization_success,
        "n_targets": result.n_targets,
        "feasibility_rate": result.feasibility_rate,
        "hard_feasibility_rate": result.hard_feasibility_rate,
        "uncertainty_feasibility_rate": result.uncertainty_feasibility_rate,
        "projection_success_rate": result.projection_success_rate,
        "mean_distortion": result.mean_distortion,
        "std_distortion": result.std_distortion,
        "median_distortion": result.median_distortion,
        "iqr_distortion": result.iqr_distortion,
        "mean_gaussian_loss_prior": result.mean_gaussian_loss_prior,
        "std_gaussian_loss_prior": result.std_gaussian_loss_prior,
        "median_gaussian_loss_prior": result.median_gaussian_loss_prior,
        "mean_projected_gaussian_loss": result.mean_projected_gaussian_loss,
        "mean_projection_objective": result.mean_projection_objective,
        "mean_distortion_lower_bound": result.mean_distortion_lower_bound,
        "mean_objective_lower_bound": result.mean_objective_lower_bound,
        "mean_bound_gap": result.mean_bound_gap,
        "mean_agent_ESS": result.mean_agent_ess,
        "mean_agent_nESS": result.mean_agent_ness,
        "std_agent_nESS": result.std_agent_ness,
        "mean_cell_ESS": result.mean_cell_ess,
        "mean_cell_nESS": result.mean_cell_ness,
        "std_cell_nESS": result.std_cell_ness,
        "mean_ESS": result.mean_agent_ess,     # legacy alias = agent_ess
        "mean_nESS": result.mean_agent_ness,   # legacy alias = agent_ness
        "mean_max_weight": result.mean_max_weight,
        "mean_max_agent_weight": result.mean_max_agent_weight,
        "mean_topk_mass": result.mean_topk_mass,
        "decision_status": result.decision_status,
    }


def save_layer3_outputs(result: Layer3AuditResult, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = layer3_result_to_summary_dict(result)
    with (out / "layer3_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe({**summary, "projection_summary": result.projection_summary}), f, indent=2, ensure_ascii=False)

    pd.DataFrame([summary]).to_csv(out / "layer3_summary.csv", index=False)
    result.per_realization_diagnostics.to_csv(out / "layer3_per_realization_diagnostics.csv", index=False)
    result.target_summary.to_csv(out / "layer3_target_summary.csv", index=False)

    with (out / "layer3_projected_weight_files.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(result.projected_weight_files), f, indent=2, ensure_ascii=False)


# ============================================================
# CSV-facing interface and main
# ============================================================

def build_layer3_config_from_inputs(
    *,
    scenario: ScenarioSpec,
    target_attrs: Sequence[str],
    projection_mode: ProjectionMode = "hard",
    soft_lambda: float = 100.0,
    uncertainty_tau: float = 1.0,
    uncertainty_aggregation: UncertaintyAggregationMode = "global_scalar",
    attribute_tau_json: Optional[str] = None,
    attribute_weight_json: Optional[str] = None,
    solver: str = "SCS",
    solver_max_iters: int = 20000,
    solver_eps: float = 1e-6,
    feasibility_tol: float = 1e-4,
    realization_id_col: str = "realization_id",
    base_weight_col: Optional[str] = None,
    missing_token: str = "__MISSING__",
    check_full_attrs: bool = False,
    topk_frac: float = 0.05,
    save_projected_weights: bool = False,
    decision_fr_threshold: float = 0.9,
    decision_ness_threshold: float = 0.3,
    decision_distortion_threshold: Optional[float] = None,
    decision_topk_threshold: Optional[float] = None,
) -> Layer3ProjectionConfig:
    scenario_attrs = [t.attr for t in scenario.targets]
    task_attrs_effective = list(dict.fromkeys(list(TASK_ATTRS_XA) + list(target_attrs) + scenario_attrs))
    return Layer3ProjectionConfig(
        full_attrs=FULL_ATTRS_X,
        task_attrs=task_attrs_effective,
        target_attrs=list(target_attrs),
        scenario=scenario,
        projection_mode=projection_mode,
        soft_lambda=soft_lambda,
        uncertainty_tau=uncertainty_tau,
        uncertainty_aggregation=uncertainty_aggregation,
        attribute_tau_json=attribute_tau_json,
        attribute_weight_json=attribute_weight_json,
        solver=solver,
        solver_max_iters=solver_max_iters,
        solver_eps=solver_eps,
        feasibility_tol=feasibility_tol,
        realization_id_col=realization_id_col,
        base_weight_col=base_weight_col,
        missing_token=missing_token,
        check_full_attrs=check_full_attrs,
        topk_frac=topk_frac,
        save_projected_weights=save_projected_weights,
        decision_fr_threshold=decision_fr_threshold,
        decision_ness_threshold=decision_ness_threshold,
        decision_distortion_threshold=decision_distortion_threshold,
        decision_topk_threshold=decision_topk_threshold,
    )


def run_layer3_audit_from_files(
    *,
    realizations_csv: Optional[str | Path] = None,
    realizations_dir: Optional[str | Path] = None,
    scenario_reference_csv: Optional[str | Path] = None,
    scenario_targets_json: Optional[str | Path] = None,
    target_attrs: Optional[Sequence[str]] = None,
    realization_id_col: str = "realization_id",
    base_weight_col: Optional[str] = None,
    reference_weight_col: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
    missing_token: str = "__MISSING__",
    age_binning: AgeBinningMode = "default",
    age_bin_source_col: str = DEFAULT_AGE_BIN_SOURCE_COL,
    age_bin_col: str = DEFAULT_AGE_BIN_COL,
    age_bins: Optional[Sequence[float]] = None,
    age_bin_labels: Optional[Sequence[str]] = None,
    numeric_mode: NumericTargetMode = "value_share",
    sigma_mode: SigmaMode = "fixed",
    sigma_fixed: float = 0.02,
    sigma_floor: float = 0.005,
    sigma_scale: float = 1.0,
    scenario_attribute_sigma: Optional[Dict[str, float]] = None,
    scenario_target_sigma: Optional[Dict[str, float]] = None,
    projection_mode: ProjectionMode = "hard",
    soft_lambda: float = 100.0,
    uncertainty_tau: float = 1.0,
    uncertainty_aggregation: UncertaintyAggregationMode = "global_scalar",
    attribute_tau_json: Optional[str] = None,
    attribute_weight_json: Optional[str] = None,
    solver: str = "SCS",
    solver_max_iters: int = 20000,
    solver_eps: float = 1e-6,
    feasibility_tol: float = 1e-4,
    topk_frac: float = 0.05,
    save_projected_weights: bool = False,
    check_full_attrs: bool = False,
    decision_fr_threshold: float = 0.9,
    decision_ness_threshold: float = 0.3,
    decision_distortion_threshold: Optional[float] = None,
    decision_topk_threshold: Optional[float] = None,
    verbose: bool = False,
) -> Layer3AuditResult:
    if (realizations_csv is None) == (realizations_dir is None):
        raise ValueError("Provide exactly one of --realizations_csv or --realizations_dir.")

    if realizations_csv is not None:
        realizations = load_realizations_from_csv(realizations_csv, realization_id_col=realization_id_col)
    else:
        realizations = load_realizations_from_dir(realizations_dir, realization_id_col=realization_id_col)

    if target_attrs is None or len(list(target_attrs)) == 0:
        target_attrs = DEFAULT_PROJECTION_TARGET_ATTRS
    raw_target_attrs = list(target_attrs)
    age_bins_resolved = _parse_float_list(age_bins, DEFAULT_AGE_BINS)
    age_labels_resolved = _parse_str_list(age_bin_labels, DEFAULT_AGE_LABELS)

    # Add Age_Group internally when raw Age is used. This keeps input files unchanged
    # while avoiding one hard projection constraint per exact age value.
    realizations = [
        maybe_add_age_group_column(
            df,
            age_binning=age_binning,
            source_col=age_bin_source_col,
            out_col=age_bin_col,
            bins=age_bins_resolved,
            labels=age_labels_resolved,
            missing_token=missing_token,
        )
        for df in realizations
    ]
    target_attrs = normalize_target_attrs_for_age_binning(
        raw_target_attrs,
        age_binning=age_binning,
        source_col=age_bin_source_col,
        out_col=age_bin_col,
    )

    if (scenario_reference_csv is None) == (scenario_targets_json is None):
        raise ValueError("Provide exactly one of --scenario_reference_csv or --scenario_targets_json.")

    if scenario_targets_json is not None:
        scenario = load_scenario_targets_json(scenario_targets_json)
    else:
        ref = _read_csv(scenario_reference_csv)
        ref = maybe_add_age_group_column(
            ref,
            age_binning=age_binning,
            source_col=age_bin_source_col,
            out_col=age_bin_col,
            bins=age_bins_resolved,
            labels=age_labels_resolved,
            missing_token=missing_token,
        )
        task_attrs_effective = list(dict.fromkeys(list(TASK_ATTRS_XA) + list(target_attrs)))
        scenario = build_scenario_from_reference(
            reference_df=ref,
            target_attrs=target_attrs,
            task_attrs=task_attrs_effective,
            missing_token=missing_token,
            numeric_mode=numeric_mode,
            sigma_mode="fixed" if sigma_mode == "scenario_uncertainty" else sigma_mode,
            sigma_fixed=sigma_fixed,
            sigma_floor=sigma_floor,
            sigma_scale=sigma_scale,
            scenario_name="retrospective_reference_scenario",
            reference_weight_col=reference_weight_col,
        )

    scenario_sigma_diag = None
    if sigma_mode == "scenario_uncertainty" or scenario_attribute_sigma or scenario_target_sigma:
        scenario_sigma_diag = apply_scenario_uncertainty_sigmas(
            scenario,
            attribute_sigma=scenario_attribute_sigma,
            target_sigma=scenario_target_sigma,
            default_sigma=sigma_fixed,
            sigma_floor=sigma_floor,
            sigma_scale=sigma_scale,
            source="scenario_uncertainty",
        )

    config = build_layer3_config_from_inputs(
        scenario=scenario,
        target_attrs=target_attrs,
        projection_mode=projection_mode,
        soft_lambda=soft_lambda,
        uncertainty_tau=uncertainty_tau,
        uncertainty_aggregation=uncertainty_aggregation,
        attribute_tau_json=attribute_tau_json,
        attribute_weight_json=attribute_weight_json,
        solver=solver,
        solver_max_iters=solver_max_iters,
        solver_eps=solver_eps,
        feasibility_tol=feasibility_tol,
        realization_id_col=realization_id_col,
        base_weight_col=base_weight_col,
        missing_token=missing_token,
        check_full_attrs=check_full_attrs,
        topk_frac=topk_frac,
        save_projected_weights=save_projected_weights,
        decision_fr_threshold=decision_fr_threshold,
        decision_ness_threshold=decision_ness_threshold,
        decision_distortion_threshold=decision_distortion_threshold,
        decision_topk_threshold=decision_topk_threshold,
    )

    result = run_layer3_audit(realizations=realizations, config=config, output_dir=output_dir, verbose=verbose)
    if output_dir is not None and scenario_sigma_diag is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        scenario_sigma_diag.to_csv(out / "layer3_scenario_sigma_summary.csv", index=False)
    return result


def main() -> Layer3AuditResult:
    parser = argparse.ArgumentParser(description="Run Layer-3 projection accountability audit over population realizations.")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--realizations_csv", help="Combined generated realizations CSV with realization_id column.")
    src.add_argument("--realizations_dir", help="Directory containing one CSV per generated population realization.")

    scen = parser.add_mutually_exclusive_group(required=True)
    scen.add_argument("--scenario_reference_csv", help="Future/reference population CSV used to construct scenario marginal targets.")
    scen.add_argument("--scenario_targets_json", help="JSON file with explicit scenario targets.")

    parser.add_argument("--target_attrs", nargs="+", default=DEFAULT_PROJECTION_TARGET_ATTRS, help="Attributes in X_A used as scenario marginal targets.")
    parser.add_argument("--realization_id_col", default="realization_id")
    parser.add_argument("--base_weight_col", default=None, help="Optional base-weight column for generated realizations. If omitted, generated agents are equally weighted.")
    parser.add_argument("--reference_weight_col", default=None, help="Optional weight column in scenario_reference_csv, e.g. PWGTP for PUMS. Used only to construct scenario targets.")
    parser.add_argument("--output_dir", default="layer3_outputs")
    parser.add_argument("--missing_token", default="__MISSING__")
    parser.add_argument("--age_binning", choices=["none", "default"], default="default", help="If default, raw Age in target_attrs is internally replaced by Age_Group.")
    parser.add_argument("--age_bin_source_col", default=DEFAULT_AGE_BIN_SOURCE_COL)
    parser.add_argument("--age_bin_col", default=DEFAULT_AGE_BIN_COL)
    parser.add_argument("--age_bins", default=",".join(str(x) for x in DEFAULT_AGE_BINS), help="Comma-separated age bin edges, e.g. 0,18,35,50,65,200.")
    parser.add_argument("--age_bin_labels", default=",".join(DEFAULT_AGE_LABELS), help="Comma-separated age bin labels, length must be len(age_bins)-1.")
    parser.add_argument("--numeric_mode", choices=["value_share", "mean"], default="value_share")
    parser.add_argument("--sigma_mode", choices=["fixed", "reference_se", "scenario_uncertainty"], default="fixed")
    parser.add_argument("--sigma_fixed", type=float, default=0.02)
    parser.add_argument("--sigma_floor", type=float, default=0.005)
    parser.add_argument("--sigma_scale", type=float, default=1.0)
    parser.add_argument(
        "--scenario_attribute_sigma_json",
        default=None,
        help=(
            "JSON dict or JSON file mapping attribute names to planning-defined sigma values; "
            "used with --sigma_mode scenario_uncertainty. Example: "
            '{"Age_Group":0.05,"Household_Income_Class":0.08}'
        ),
    )
    parser.add_argument(
        "--scenario_target_sigma_json",
        default=None,
        help=(
            "JSON dict or JSON file mapping exact target names to planning-defined sigma values. "
            "Target-level entries override --scenario_attribute_sigma_json."
        ),
    )

    parser.add_argument(
        "--projection_mode",
        choices=["hard", "soft", "uncertainty_set"],
        default="hard",
        help=(
            "hard: exact marginal constraints; soft: KL plus standardized mismatch penalty; "
            "uncertainty_set: minimize KL while requiring 0.5*sum standardized errors^2 <= --uncertainty_tau."
        ),
    )
    parser.add_argument("--soft_lambda", type=float, default=100.0, help="Penalty strength for projection_mode='soft'.")
    parser.add_argument("--uncertainty_tau", type=float, default=1.0, help="Scenario uncertainty-set radius for projection_mode='uncertainty_set'.")
    parser.add_argument(
        "--uncertainty_aggregation",
        choices=["global_scalar", "attribute_normalized", "attribute_constraints"],
        default="global_scalar",
        help=(
            "How standardized target mismatches are aggregated. "
            "global_scalar uses 0.5*sum_j z_j^2; "
            "attribute_normalized uses sum_a weight_a*(0.5/|C_a|)*sum_{j in a} z_j^2; "
            "attribute_constraints imposes chi_a <= tau_a for each attribute in uncertainty_set mode."
        ),
    )
    parser.add_argument("--attribute_tau_json", default=None, help="JSON string or JSON file mapping attribute names to tau_a values; used by attribute_constraints. Defaults to --uncertainty_tau for every attribute.")
    parser.add_argument("--attribute_weight_json", default=None, help="JSON string or JSON file mapping attribute names to weights for attribute_normalized aggregation. Defaults to 1 for every attribute.")
    parser.add_argument("--solver", default="SCS")
    parser.add_argument("--solver_max_iters", type=int, default=20000)
    parser.add_argument("--solver_eps", type=float, default=1e-6)
    parser.add_argument("--feasibility_tol", type=float, default=1e-4)
    parser.add_argument("--topk_frac", type=float, default=0.05)
    parser.add_argument("--save_projected_weights", action="store_true")
    parser.add_argument("--check_full_attrs", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--decision_fr_threshold", type=float, default=0.9)
    parser.add_argument("--decision_ness_threshold", type=float, default=0.3)
    parser.add_argument("--decision_distortion_threshold", type=float, default=None)
    parser.add_argument("--decision_topk_threshold", type=float, default=None)

    args = parser.parse_args()

    result = run_layer3_audit_from_files(
        realizations_csv=args.realizations_csv,
        realizations_dir=args.realizations_dir,
        scenario_reference_csv=args.scenario_reference_csv,
        scenario_targets_json=args.scenario_targets_json,
        target_attrs=args.target_attrs,
        realization_id_col=args.realization_id_col,
        base_weight_col=args.base_weight_col,
        reference_weight_col=args.reference_weight_col,
        output_dir=args.output_dir,
        missing_token=args.missing_token,
        age_binning=args.age_binning,
        age_bin_source_col=args.age_bin_source_col,
        age_bin_col=args.age_bin_col,
        age_bins=_parse_float_list(args.age_bins, DEFAULT_AGE_BINS),
        age_bin_labels=_parse_str_list(args.age_bin_labels, DEFAULT_AGE_LABELS),
        numeric_mode=args.numeric_mode,
        sigma_mode=args.sigma_mode,
        sigma_fixed=args.sigma_fixed,
        sigma_floor=args.sigma_floor,
        sigma_scale=args.sigma_scale,
        scenario_attribute_sigma=_load_json_mapping(args.scenario_attribute_sigma_json, name="scenario_attribute_sigma_json"),
        scenario_target_sigma=_load_json_mapping(args.scenario_target_sigma_json, name="scenario_target_sigma_json"),
        projection_mode=args.projection_mode,
        soft_lambda=args.soft_lambda,
        uncertainty_tau=args.uncertainty_tau,
        uncertainty_aggregation=args.uncertainty_aggregation,
        attribute_tau_json=args.attribute_tau_json,
        attribute_weight_json=args.attribute_weight_json,
        solver=args.solver,
        solver_max_iters=args.solver_max_iters,
        solver_eps=args.solver_eps,
        feasibility_tol=args.feasibility_tol,
        topk_frac=args.topk_frac,
        save_projected_weights=args.save_projected_weights,
        check_full_attrs=args.check_full_attrs,
        decision_fr_threshold=args.decision_fr_threshold,
        decision_ness_threshold=args.decision_ness_threshold,
        decision_distortion_threshold=args.decision_distortion_threshold,
        decision_topk_threshold=args.decision_topk_threshold,
        verbose=args.verbose,
    )

    print("===== Attribute Spaces =====")
    print(f"Full training space X ({len(FULL_ATTRS_X)} attrs): {FULL_ATTRS_X}")
    print(f"Task space X_A ({len(TASK_ATTRS_XA)} attrs): {TASK_ATTRS_XA}")
    mapped_target_attrs = normalize_target_attrs_for_age_binning(
        list(args.target_attrs),
        age_binning=args.age_binning,
        source_col=args.age_bin_source_col,
        out_col=args.age_bin_col,
    )
    print(f"Scenario target attrs input ({len(args.target_attrs)} attrs): {list(args.target_attrs)}")
    print(f"Scenario target attrs effective ({len(mapped_target_attrs)} attrs): {mapped_target_attrs}")
    print(f"Age binning: {args.age_binning}; {args.age_bin_source_col} -> {args.age_bin_col}")
    print(f"Reference weight col: {args.reference_weight_col}")
    print(f"Sigma mode: {args.sigma_mode}")
    print(f"Scenario attribute sigma: {args.scenario_attribute_sigma_json}")
    print(f"Scenario target sigma: {args.scenario_target_sigma_json}")

    print("\n===== Layer 3 Projection Accountability Summary =====")
    print(json.dumps(_json_safe(layer3_result_to_summary_dict(result)), indent=2, ensure_ascii=False))

    print("\n===== Scenario Target Summary =====")
    if not result.target_summary.empty:
        print(result.target_summary.to_string(index=False))
    else:
        print("No target summary available.")

    return result


if __name__ == "__main__":
    main()
