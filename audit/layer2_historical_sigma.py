from __future__ import annotations

"""
Layer 2 — Scenario Compatibility Audit

This module implements Layer-2 compatibility diagnostics under the distinction:

    X   = full attribute space used to train the generative population model
    X_A = task-relevant attribute subset used for audit / projection / validation

Layer 2 evaluates whether an ensemble of generated populations from the learned
prior P_theta is compatible with scenario marginal constraints over X_A.

Core idea:
    - Each generated population realization is one sample from the prior over
      full synthetic populations.
    - A scenario S is represented by marginal target statistics m_j with
      uncertainty sigma_j. sigma_j can come from fixed/reference SE,
      historical dynamics, explicit JSON targets, or planning-defined
      scenario uncertainty overrides.
    - Each realization receives an importance weight according to how closely
      its statistics match the scenario. The Gaussian loss can be computed as
      either a raw scalar-target sum or an attribute-normalized loss that first
      averages standardized errors within each target attribute.
    - Compatibility is diagnosed by the Effective Sample Size (ESS) and related
      weight-concentration metrics.

Expected command-line inputs:
    1) generated population realizations, either as:
         --realizations_csv combined.csv with a realization_id column, or
         --realizations_dir folder_of_csv_files
    2) scenario targets, either as:
         --scenario_targets_json targets.json, or
         --scenario_reference_csv future_real_population.csv from which marginal
         targets are constructed automatically.

Example:
    python audit/layer2_historical_sigma.py \
        --realizations_csv generated_ensembles.csv \
        --scenario_reference_csv real_future.csv \
        --target_attrs Age Household_Size Employment_status Household_Income_Class Number_of_cars_of_household \
        --reference_weight_col PWGTP \
        --output_dir layer2_outputs

By default, raw Age in --target_attrs is internally mapped to Age_Group using
--age_bins/--age_bin_labels. Use --age_binning none to retain exact-age targets.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple
import argparse
import json
import math

import numpy as np
import pandas as pd


# ============================================================
# Attribute spaces
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

# Default task-relevant attribute set X_A, aligned with the current Layer-1 code.
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
# Layer-3 so all three layers audit the same attributes. Commute_Mode and
# Number_of_children are load-bearing (e.g. WFH/commute drift in the 2021 pandemic
# horizon); dropping them silently makes the scenario test too easy.
DEFAULT_PROJECTION_TARGET_ATTRS: List[str] = [
    "Age",
    "Household_Size",
    "Number_of_children",
    "Number_of_cars_of_household",
    "Employment_status",
    "Household_Income_Class",
    "Commute_Mode",
]

# Default age grouping used when input files contain raw Age but the audit should
# use coarse age-group marginals rather than one target per exact age value.
DEFAULT_AGE_BIN_SOURCE_COL = "Age"
DEFAULT_AGE_BIN_COL = "Age_Group"
DEFAULT_AGE_BINS: List[float] = [0, 18, 35, 50, 65, 200]
DEFAULT_AGE_LABELS: List[str] = ["0-17", "18-34", "35-49", "50-64", "65+"]


# ============================================================
# Types
# ============================================================

TargetKind = Literal["share", "mean", "joint_share"]
SigmaMode = Literal["fixed", "reference_se", "historical_dynamics", "scenario_uncertainty"]
NumericTargetMode = Literal["value_share", "mean"]
AgeBinningMode = Literal["none", "default"]
LossAggregationMode = Literal["global_scalar", "attribute_normalized"]


@dataclass
class ScenarioTarget:
    """
    A scalar scenario target m_j with uncertainty sigma_j.

    kind="share":
        target statistic is P(attr == value).
    kind="mean":
        target statistic is E[attr].
    kind="joint_share":
        target statistic is P(attr_1 == v_1 AND attr_2 == v_2 AND ...) where the
        conjunctive event is given by `conditions = [{"attr":..., "value":...}, ...]`.
        The dataclass `attr` field is used purely as the attribute-group label for
        attribute_normalized loss aggregation (e.g. "equity_joint"); it does NOT have
        to match any column in the dataframe. The `value` field is unused.

    The optional sigma_* fields are diagnostics for historical-dynamics-informed
    uncertainty estimation. They do not change the target statistic itself; they
    record how the Gaussian kernel bandwidth was obtained.
    """
    name: str
    attr: str
    kind: TargetKind
    m: float
    sigma: float
    value: Optional[Any] = None
    conditions: Optional[List[Dict[str, Any]]] = None
    sigma_source: str = "unspecified"
    historical_mean_delta: Optional[float] = None
    historical_sd_delta: Optional[float] = None
    historical_projection_horizon: Optional[float] = None
    historical_n_files: Optional[int] = None
    historical_sampling_se: Optional[float] = None
    historical_sigma_floor: Optional[float] = None


@dataclass
class ScenarioSpec:
    name: str
    targets: List[ScenarioTarget]


@dataclass
class Layer2AuditConfig:
    full_attrs: Sequence[str]
    task_attrs: Sequence[str]
    target_attrs: Sequence[str]
    scenario: ScenarioSpec
    realization_id_col: str = "realization_id"
    missing_token: str = "__MISSING__"
    check_full_attrs: bool = False
    loss_aggregation: LossAggregationMode = "attribute_normalized"
    attribute_weights: Optional[Dict[str, float]] = None


@dataclass
class Layer2AuditResult:
    scenario_name: str
    n_realizations: int
    n_targets: int
    ess: float
    ess_ratio: float
    entropy: float
    entropy_ratio: float
    perplexity: float
    max_weight: float
    top_5_weight: float
    prior_mean_loss: float
    posterior_mean_loss: float
    per_realization_stats: pd.DataFrame
    target_summary: pd.DataFrame
    weight_summary: Dict[str, Any]


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


def _reference_weights(df: pd.DataFrame, reference_weight_col: Optional[str] = None) -> np.ndarray:
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
    return float(1.0 / s2) if s2 > 0 else float("nan")


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
    return probs / probs.sum()


# ============================================================
# Historical-dynamics-informed sigma estimation
# ============================================================

def _parse_json_float_dict(value: Optional[str], *, name: str) -> Dict[str, float]:
    """Parse a JSON object or JSON file path mapping names to numeric values."""
    if value is None or str(value).strip() == "":
        return {}
    s = str(value).strip()
    try:
        if s.startswith("{"):
            data = json.loads(s)
        else:
            fp = Path(s)
            if fp.exists():
                with fp.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = json.loads(s)
    except Exception as e:
        raise ValueError(f"{name} must be a valid JSON object or JSON file path, got: {value}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a JSON object mapping names to numbers.")
    out: Dict[str, float] = {}
    for k, v in data.items():
        try:
            out[str(k)] = float(v)
        except Exception as e:
            raise ValueError(f"{name}[{k!r}] must be numeric, got {v!r}.") from e
    return out


def apply_scenario_uncertainty_sigmas(
    scenario: ScenarioSpec,
    *,
    attribute_sigma: Optional[Dict[str, float]] = None,
    target_sigma: Optional[Dict[str, float]] = None,
    default_sigma: float = 0.02,
    sigma_floor: float = 1e-12,
    sigma_scale: float = 1.0,
    source: str = "scenario_uncertainty",
    compose: Literal["replace", "quadrature_add"] = "replace",
) -> pd.DataFrame:
    """Apply planning/scenario-defined diagonal uncertainty to targets.

    This is the general Layer-2 branch for long-term or policy-defined
    scenarios where sigma should not be inferred from short-term historical
    dynamics. Values in target_sigma override values in attribute_sigma; if
    neither is provided for a target, default_sigma is used.

    compose:
      - "replace" (default, backward-compatible): overwrite t.sigma with the
        resolved scenario sigma.
      - "quadrature_add": combine with any existing t.sigma additively in
        quadrature, i.e. t.sigma <- sqrt(t.sigma**2 + raw_sigma**2). This
        implements the two-part decomposition sigma**2 = sigma_epistemic**2
        + sigma_decision**2, with the base assumed to be epistemic
        (historical_dynamics / reference_se / fixed) and the add-on the
        planning/scenario uncertainty.
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
        addon_sigma = float(max(float(sigma_floor), raw_sigma) * float(sigma_scale))
        if compose == "replace":
            new_sigma = addon_sigma
            new_source = source
            base_sigma_used = None
        elif compose == "quadrature_add":
            base_sigma = float(t.sigma) if (t.sigma is not None and np.isfinite(t.sigma) and t.sigma > 0) else 0.0
            new_sigma = float(np.sqrt(base_sigma ** 2 + addon_sigma ** 2))
            base_source = str(t.sigma_source) if t.sigma_source else "uninitialized"
            new_source = f"{base_source}+{source}"
            base_sigma_used = base_sigma
        else:
            raise ValueError(f"Unsupported compose mode: {compose!r}")
        t.sigma = new_sigma
        t.sigma_source = new_source
        # Preserve historical_* diagnostics if we composed on top of historical_dynamics;
        # otherwise reset them to None (consistent with previous behavior under replace).
        if compose == "replace":
            t.historical_mean_delta = None
            t.historical_sd_delta = None
            t.historical_projection_horizon = None
            t.historical_n_files = None
            t.historical_sampling_se = None
            t.historical_sigma_floor = sigma_floor
        rows.append({
            "target": t.name,
            "attr": t.attr,
            "kind": t.kind,
            "value": t.value,
            "target_m": t.m,
            "sigma": new_sigma,
            "sigma_source": new_source,
            "sigma_level": sigma_level,
            "sigma_key": sigma_key,
            "raw_sigma": raw_sigma,
            "addon_sigma": addon_sigma,
            "base_sigma_used": base_sigma_used,
            "compose": compose,
            "sigma_floor": sigma_floor,
            "sigma_scale": sigma_scale,
        })
    return pd.DataFrame(rows)


def _infer_year_from_path(path: str | Path) -> Optional[int]:
    """Best-effort year extraction from a file/path name.

    It first looks for a four-digit year in [1900, 2099]. If not found, it
    recognizes common compact path fragments such as pums13, acs18, year_2013,
    and converts two-digit years to 2000+YY.
    """
    import re

    text = str(path)
    four = re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text)
    if four:
        return int(four[-1])
    two = re.findall(r"(?i)(?:pums|acs|year|yr|y)[_\- ]?(\d{2})(?!\d)", text)
    if two:
        yy = int(two[-1])
        return 2000 + yy if yy < 80 else 1900 + yy
    return None


def load_historical_reference_files(
    historical_reference_dir: str | Path,
    *,
    glob_pattern: str = "*.csv",
) -> Tuple[List[Path], List[Optional[int]]]:
    path = Path(historical_reference_dir)
    if not path.exists():
        raise FileNotFoundError(f"historical_reference_dir not found: {path}")
    files = sorted(path.glob(glob_pattern))
    if not files:
        raise ValueError(f"No historical CSV files found in {path} with pattern {glob_pattern}")
    years = [_infer_year_from_path(fp) for fp in files]
    # If all years are available, sort chronologically. Otherwise keep filename order.
    if all(y is not None for y in years):
        paired = sorted(zip(files, years), key=lambda x: (int(x[1]), str(x[0])))
        files = [p for p, _ in paired]
        years = [y for _, y in paired]
    return files, years


def _joint_event_mask(df: pd.DataFrame, conditions: Sequence[Dict[str, Any]], missing_token: str) -> np.ndarray:
    """Boolean mask for the AND-conjunction of per-attribute (==value or in [values]) tests.

    Each condition is {"attr": col, "value": v_or_list}. If `value` is a list/tuple,
    it acts as an OR within that attribute: P(attr in values). The outer AND across
    conditions remains.
    """
    if not conditions:
        raise ValueError("joint_share target requires non-empty 'conditions'.")
    mask = np.ones(len(df), dtype=bool)
    for c in conditions:
        col = str(c["attr"]) if isinstance(c, dict) else str(c[0])
        val = c["value"] if isinstance(c, dict) else c[1]
        if col not in df.columns:
            raise ValueError(f"joint_share condition refers to missing column '{col}'.")
        s = _prepare_cat_series(df[col], missing_token)
        if isinstance(val, (list, tuple, set)):
            vals_str = [str(v) for v in val]
            mask &= s.isin(vals_str).to_numpy()
        else:
            mask &= (s == str(val)).to_numpy()
    return mask


def compute_weighted_target_stat(
    df: pd.DataFrame,
    target: ScenarioTarget,
    weights: np.ndarray,
    *,
    missing_token: str = "__MISSING__",
) -> float:
    """Compute a target statistic using normalized population weights."""
    w = _normalize_prob(weights)
    if target.kind == "share":
        if target.attr not in df.columns:
            raise ValueError(f"historical/reference dataframe is missing target attr '{target.attr}'.")
        if target.value is None:
            raise ValueError(f"Share target '{target.name}' must have a value.")
        s = _prepare_cat_series(df[target.attr], missing_token)
        return float(np.sum(w * (s == str(target.value)).to_numpy(dtype=float)))
    if target.kind == "mean":
        if target.attr not in df.columns:
            raise ValueError(f"historical/reference dataframe is missing target attr '{target.attr}'.")
        x = pd.to_numeric(df[target.attr], errors="coerce").to_numpy(dtype=float)
        return _weighted_mean(x, w)
    if target.kind == "joint_share":
        mask = _joint_event_mask(df, target.conditions or [], missing_token)
        return float(np.sum(w * mask.astype(float)))
    raise ValueError(f"Unsupported target kind: {target.kind}")


def reference_sampling_se_for_target(
    df: pd.DataFrame,
    target: ScenarioTarget,
    weights: np.ndarray,
    *,
    missing_token: str = "__MISSING__",
) -> float:
    """Approximate sampling SE using the effective sample size of normalized weights."""
    w = _normalize_prob(weights)
    n_eff = max(_effective_n_from_weights(w), 1.0)
    if target.kind in ("share", "joint_share"):
        p = compute_weighted_target_stat(df, target, w, missing_token=missing_token)
        return float(math.sqrt(max(p * (1.0 - p), 0.0) / n_eff))
    if target.kind == "mean":
        x = pd.to_numeric(df[target.attr], errors="coerce").to_numpy(dtype=float)
        var = _weighted_var(x, w)
        if not np.isfinite(var):
            return 0.0
        return float(math.sqrt(max(var, 0.0) / n_eff))
    return 0.0


def apply_historical_dynamics_sigmas(
    scenario: ScenarioSpec,
    *,
    historical_reference_dir: str | Path,
    target_attrs: Sequence[str],
    missing_token: str = "__MISSING__",
    reference_weight_col: Optional[str] = None,
    historical_weight_col: Optional[str] = None,
    historical_glob_pattern: str = "*.csv",
    age_binning: AgeBinningMode = "default",
    age_bin_source_col: str = DEFAULT_AGE_BIN_SOURCE_COL,
    age_bin_col: str = DEFAULT_AGE_BIN_COL,
    age_bins: Sequence[float] = DEFAULT_AGE_BINS,
    age_bin_labels: Sequence[str] = DEFAULT_AGE_LABELS,
    projection_horizon: Optional[float] = None,
    scenario_reference_csv: Optional[str | Path] = None,
    scenario_reference_df: Optional[pd.DataFrame] = None,
    sigma_floor: float = 0.005,
    sigma_scale: float = 1.0,
    historical_drift_scale: float = 1.0,
    historical_vol_scale: float = 1.0,
    historical_sampling_scale: float = 1.0,
    attribute_sigma_floor: Optional[Dict[str, float]] = None,
    attribute_drift_scale: Optional[Dict[str, float]] = None,
    attribute_vol_scale: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Estimate target-specific Gaussian kernel sigmas from historical dynamics.

    For each target j=(a,c), we compute weighted historical marginals over a
    pre-baseline reference directory, estimate annual marginal changes, and set

        sigma_j(H) = max(
            sigma_floor_a,
            gamma_a * H * |mean_delta_j|
            + lambda_a * sqrt(H) * sd_delta_j
            + historical_sampling_scale * sampling_se_j
        ) * sigma_scale

    where H is the projection horizon. If years can be inferred from filenames,
    annual deltas are divided by the year gaps; otherwise sorted-file order is
    treated as equally spaced annual data.
    """
    attribute_sigma_floor = attribute_sigma_floor or {}
    attribute_drift_scale = attribute_drift_scale or {}
    attribute_vol_scale = attribute_vol_scale or {}

    hist_weight_col = historical_weight_col if historical_weight_col is not None else reference_weight_col
    files, years = load_historical_reference_files(historical_reference_dir, glob_pattern=historical_glob_pattern)

    hist_dfs: List[pd.DataFrame] = []
    for fp in files:
        df = _read_csv(fp)
        df = maybe_add_age_group_column(
            df,
            age_binning=age_binning,
            source_col=age_bin_source_col,
            out_col=age_bin_col,
            bins=age_bins,
            labels=age_bin_labels,
            missing_token=missing_token,
        )
        _check_columns(df, target_attrs, f"historical_reference_file={fp}")
        hist_dfs.append(df)

    if len(hist_dfs) < 2:
        raise ValueError("At least two historical reference CSV files are required to estimate dynamics.")

    # Resolve projection horizon.
    if projection_horizon is None:
        hist_years_available = all(y is not None for y in years)
        scenario_year = _infer_year_from_path(scenario_reference_csv) if scenario_reference_csv is not None else None
        if hist_years_available and scenario_year is not None:
            projection_horizon = float(scenario_year - int(max(y for y in years if y is not None)))
        else:
            projection_horizon = 1.0
    H = float(projection_horizon)
    if not np.isfinite(H) or H <= 0:
        raise ValueError(f"projection_horizon must be positive, got {projection_horizon}.")

    # Reference sampling SE is computed from projection-year scenario_reference_df when available.
    ref_w_for_sampling: Optional[np.ndarray] = None
    if scenario_reference_df is not None:
        ref_w_for_sampling = _reference_weights(scenario_reference_df, reference_weight_col=reference_weight_col)

    diag_rows: List[Dict[str, Any]] = []
    years_numeric = [y if y is not None else None for y in years]

    for target in scenario.targets:
        vals: List[float] = []
        for df in hist_dfs:
            hw = _reference_weights(df, reference_weight_col=hist_weight_col)
            vals.append(compute_weighted_target_stat(df, target, hw, missing_token=missing_token))
        arr = np.asarray(vals, dtype=float)
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"Historical target values contain non-finite values for target {target.name}.")

        if all(y is not None for y in years_numeric):
            gaps = np.diff(np.asarray([int(y) for y in years_numeric], dtype=float))
            # Fallback to unit gaps if any inferred year gaps are invalid.
            if np.any(gaps <= 0):
                gaps = np.ones(len(arr) - 1, dtype=float)
        else:
            gaps = np.ones(len(arr) - 1, dtype=float)
        deltas = np.diff(arr) / gaps
        mean_delta = float(np.mean(deltas)) if len(deltas) else 0.0
        sd_delta = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0

        sampling_se = 0.0
        if scenario_reference_df is not None and ref_w_for_sampling is not None:
            sampling_se = reference_sampling_se_for_target(
                scenario_reference_df,
                target,
                ref_w_for_sampling,
                missing_token=missing_token,
            )

        floor_a = float(attribute_sigma_floor.get(target.attr, sigma_floor))
        gamma_a = float(attribute_drift_scale.get(target.attr, historical_drift_scale))
        lambda_a = float(attribute_vol_scale.get(target.attr, historical_vol_scale))

        sigma_dyn = (
            gamma_a * H * abs(mean_delta)
            + lambda_a * math.sqrt(H) * sd_delta
            + float(historical_sampling_scale) * sampling_se
        )
        sigma = max(floor_a, sigma_dyn)
        sigma = float(sigma_scale * sigma)
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = float(max(floor_a, sigma_floor))

        target.sigma = sigma
        target.sigma_source = "historical_dynamics"
        target.historical_mean_delta = mean_delta
        target.historical_sd_delta = sd_delta
        target.historical_projection_horizon = H
        target.historical_n_files = len(hist_dfs)
        target.historical_sampling_se = sampling_se
        target.historical_sigma_floor = floor_a

        diag_rows.append({
            "target": target.name,
            "attr": target.attr,
            "kind": target.kind,
            "value": target.value,
            "target_m": target.m,
            "sigma": target.sigma,
            "sigma_source": target.sigma_source,
            "historical_projection_horizon": H,
            "historical_n_files": len(hist_dfs),
            "historical_files": "|".join(str(x) for x in files),
            "historical_years": "|".join("" if y is None else str(y) for y in years_numeric),
            "historical_values": "|".join(f"{v:.12g}" for v in arr.tolist()),
            "historical_mean_delta": mean_delta,
            "historical_sd_delta": sd_delta,
            "historical_sampling_se": sampling_se,
            "historical_sigma_floor": floor_a,
            "historical_drift_scale": gamma_a,
            "historical_vol_scale": lambda_a,
            "historical_sampling_scale": historical_sampling_scale,
        })

    return pd.DataFrame(diag_rows)


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
    # NOTE (2026-11-23): realization_id = POSITION in this lexicographically sorted list
    # (_0, _1, _10, ...), not the file-name suffix.  Consumers that need the file must
    # resolve it through the same sorted list (rerun_audit_new.resolve_rstar_file).
    for k, fp in enumerate(files):
        df = pd.read_csv(fp)
        if realization_id_col not in df.columns:
            df[realization_id_col] = k
        realizations.append(df.reset_index(drop=True))
    return realizations


# ============================================================
# Scenario target construction
# ============================================================

def _share_stat(df: pd.DataFrame, attr: str, value: Any, missing_token: str) -> float:
    s = _prepare_cat_series(df[attr], missing_token)
    return float((s == str(value)).mean()) if len(s) > 0 else np.nan


def _mean_stat(df: pd.DataFrame, attr: str) -> float:
    x = pd.to_numeric(df[attr], errors="coerce").dropna()
    return float(x.mean()) if len(x) > 0 else np.nan


def compute_target_stat(df: pd.DataFrame, target: ScenarioTarget, missing_token: str = "__MISSING__") -> float:
    if target.kind == "share":
        if target.value is None:
            raise ValueError(f"Share target '{target.name}' must have a value.")
        return _share_stat(df, target.attr, target.value, missing_token)
    if target.kind == "mean":
        return _mean_stat(df, target.attr)
    if target.kind == "joint_share":
        mask = _joint_event_mask(df, target.conditions or [], missing_token)
        return float(mask.mean()) if len(mask) else np.nan
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
    shares/means are population-weighted. If Age is internally mapped to
    Age_Group before this function is called, value_share constraints are built
    over the coarse age groups rather than exact ages.
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

            targets.append(
                ScenarioTarget(
                    name=f"{attr}=={_target_value_to_str(value)}",
                    attr=attr,
                    kind="share",
                    value=_target_value_to_str(value),
                    m=p,
                    sigma=float(sigma),
                    sigma_source=str(sigma_mode),
                )
            )

    if len(targets) == 0:
        raise ValueError("No scenario targets were constructed from the reference data.")
    return ScenarioSpec(name=scenario_name, targets=targets)

def load_scenario_targets_json(path: str | Path, scenario_name: str = "scenario_from_json") -> ScenarioSpec:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario targets JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        scenario_name = str(data.get("name", scenario_name))
        raw_targets = data.get("targets", [])
    elif isinstance(data, list):
        raw_targets = data
    else:
        raise ValueError("Scenario targets JSON must be a list or a dict with a 'targets' field.")

    targets: List[ScenarioTarget] = []
    for idx, t in enumerate(raw_targets):
        kind = str(t.get("kind", "share"))
        if kind not in {"share", "mean", "joint_share"}:
            raise ValueError(f"Target {idx} has unsupported kind={kind}")
        name = str(t.get("name", f"target_{idx}"))
        attr = str(t["attr"])
        m = float(t["m"])
        sigma = float(t["sigma"])
        value = t.get("value", None)
        conditions = t.get("conditions", None)
        if kind == "joint_share":
            if not conditions or not isinstance(conditions, list):
                raise ValueError(f"Target {idx} (kind=joint_share) requires non-empty 'conditions' list of {{'attr':..., 'value':...}} dicts.")
            for c in conditions:
                if not isinstance(c, dict) or "attr" not in c or "value" not in c:
                    raise ValueError(f"Target {idx} condition entries must be dicts with 'attr' and 'value' keys.")
        targets.append(ScenarioTarget(name=name, attr=attr, kind=kind, value=value, m=m, sigma=sigma, conditions=conditions, sigma_source="json"))

    if not targets:
        raise ValueError("Scenario targets JSON contains no targets.")
    return ScenarioSpec(name=scenario_name, targets=targets)


# ============================================================
# Layer-2 core computation
# ============================================================

def effective_sample_size_from_normalized_weights(pi: np.ndarray) -> float:
    pi = np.asarray(pi, dtype=float)
    pi = pi[np.isfinite(pi)]
    if len(pi) == 0:
        return np.nan
    s2 = float(np.sum(pi ** 2))
    if s2 <= 0:
        return np.nan
    return float(1.0 / s2)


def _safe_column_name(value: str) -> str:
    """Return a compact safe suffix for diagnostic columns."""
    out = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _targets_by_attribute(scenario: ScenarioSpec) -> Dict[str, List[ScenarioTarget]]:
    groups: Dict[str, List[ScenarioTarget]] = {}
    for t in scenario.targets:
        groups.setdefault(str(t.attr), []).append(t)
    return groups


def compute_realization_attribute_losses(
    per_realization_stats: pd.DataFrame,
    scenario: ScenarioSpec,
) -> pd.DataFrame:
    """Compute attribute-normalized mismatch components for each realization.

    For each attribute a with scalar targets C_a, this computes

        chi_a(k) = 0.5 / |C_a| * sum_{j in C_a} z_{k,j}^2,

    where z_{k,j} = (s_j^{(k)} - m_j) / sigma_j.

    These columns are diagnostics and are also used by the
    attribute_normalized Gaussian kernel.
    """
    K = len(per_realization_stats)
    rows: Dict[str, np.ndarray] = {}

    for attr, targets in _targets_by_attribute(scenario).items():
        z2_sum = np.zeros(K, dtype=np.float64)
        count = 0
        for t in targets:
            if t.name not in per_realization_stats.columns:
                raise KeyError(f"Target statistic '{t.name}' not found in per_realization_stats.")
            sigma = float(t.sigma)
            if not np.isfinite(sigma) or sigma <= 0:
                raise ValueError(f"Target '{t.name}' has non-positive sigma={sigma}.")
            s = per_realization_stats[t.name].astype(np.float64).to_numpy()
            z = (s - float(t.m)) / sigma
            z2_sum += z ** 2
            count += 1
        if count <= 0:
            continue
        rows[attr] = 0.5 * z2_sum / float(count)

    if not rows:
        return pd.DataFrame(index=per_realization_stats.index)

    out = pd.DataFrame(index=per_realization_stats.index)
    for attr, vals in rows.items():
        out[f"attribute_loss__{_safe_column_name(attr)}"] = vals
    return out


def stable_scenario_weights(
    per_realization_stats: pd.DataFrame,
    scenario: ScenarioSpec,
    *,
    loss_aggregation: LossAggregationMode = "attribute_normalized",
    attribute_weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Compute stable Gaussian scenario weights over population realizations.

    Two aggregation modes are supported.

    global_scalar:
        loss_k = 1/2 * sum_j ((s_j^(k) - m_j) / sigma_j)^2

    attribute_normalized:
        loss_k = sum_a omega_a * [1/(2|C_a|) * sum_{j in C_a}
                 ((s_j^(k) - m_j) / sigma_j)^2]

    The attribute-normalized mode is the recommended historical-uncertainty
    formulation because it prevents high-cardinality attributes such as
    Household_Size or car ownership from dominating the Gaussian kernel simply
    because they contribute more scalar category targets.
    """
    K = len(per_realization_stats)
    if K == 0:
        raise ValueError("No population realizations provided.")
    if loss_aggregation not in {"global_scalar", "attribute_normalized"}:
        raise ValueError(
            "loss_aggregation must be one of {'global_scalar', 'attribute_normalized'}."
        )

    attribute_weights = attribute_weights or {}
    loss = np.zeros(K, dtype=np.float64)
    attr_loss_df = compute_realization_attribute_losses(per_realization_stats, scenario)

    if loss_aggregation == "global_scalar":
        for t in scenario.targets:
            if t.name not in per_realization_stats.columns:
                raise KeyError(f"Target statistic '{t.name}' not found in per_realization_stats.")
            sigma = float(t.sigma)
            if not np.isfinite(sigma) or sigma <= 0:
                raise ValueError(f"Target '{t.name}' has non-positive sigma={sigma}.")
            s = per_realization_stats[t.name].astype(np.float64).to_numpy()
            z = (s - float(t.m)) / sigma
            loss += 0.5 * (z ** 2)
    else:
        for attr in _targets_by_attribute(scenario).keys():
            col = f"attribute_loss__{_safe_column_name(attr)}"
            if col not in attr_loss_df.columns:
                continue
            omega = float(attribute_weights.get(attr, 1.0))
            loss += omega * attr_loss_df[col].astype(np.float64).to_numpy()

    # Constants in Gaussian likelihood do not affect normalized realization
    # weights. Using logw=-loss keeps the diagnostics directly aligned with the
    # configured compatibility loss.
    logw = -loss
    max_logw = float(np.nanmax(logw))
    logw_stable = logw - max_logw
    w_unnorm = np.exp(logw_stable)
    w_sum = float(np.sum(w_unnorm))

    if (not np.isfinite(w_sum)) or w_sum <= 0:
        pi = np.ones(K, dtype=float) / K
        w_unnorm = np.ones(K, dtype=float)
    else:
        pi = w_unnorm / w_sum

    return pi, w_unnorm, logw, loss, attr_loss_df


def compute_per_realization_target_stats(
    realizations: List[pd.DataFrame],
    scenario: ScenarioSpec,
    realization_id_col: str = "realization_id",
    missing_token: str = "__MISSING__",
) -> pd.DataFrame:
    required_set: set = set()
    for t in scenario.targets:
        if t.kind == "joint_share":
            for c in (t.conditions or []):
                required_set.add(str(c["attr"]) if isinstance(c, dict) else str(c[0]))
        else:
            required_set.add(t.attr)
    required_cols = sorted(required_set)
    rows: List[Dict[str, Any]] = []

    for k, df_k in enumerate(realizations):
        _check_columns(df_k, required_cols, f"realization_{k}")
        if realization_id_col in df_k.columns:
            rid = df_k[realization_id_col].iloc[0]
        else:
            rid = k
        row: Dict[str, Any] = {"realization_id": rid, "n_agents": int(len(df_k))}
        for t in scenario.targets:
            row[t.name] = compute_target_stat(df_k, t, missing_token=missing_token)
        rows.append(row)

    return pd.DataFrame(rows)


def build_target_summary(
    per_realization_stats: pd.DataFrame,
    scenario: ScenarioSpec,
    pi: np.ndarray,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for t in scenario.targets:
        vals = per_realization_stats[t.name].astype(float).to_numpy()
        prior_mean = float(np.mean(vals))
        posterior_mean = float(np.sum(pi * vals))
        target_m = float(t.m)
        sigma = float(t.sigma)
        rows.append({
            "target": t.name,
            "attr": t.attr,
            "kind": t.kind,
            "value": t.value,
            "target_m": target_m,
            "sigma": sigma,
            "sigma_source": getattr(t, "sigma_source", None),
            "historical_projection_horizon": getattr(t, "historical_projection_horizon", None),
            "historical_mean_delta": getattr(t, "historical_mean_delta", None),
            "historical_sd_delta": getattr(t, "historical_sd_delta", None),
            "historical_sampling_se": getattr(t, "historical_sampling_se", None),
            "historical_sigma_floor": getattr(t, "historical_sigma_floor", None),
            "prior_mean": prior_mean,
            "posterior_weighted_mean": posterior_mean,
            "abs_err_prior": abs(prior_mean - target_m),
            "abs_err_posterior": abs(posterior_mean - target_m),
            "z_err_prior": abs(prior_mean - target_m) / sigma if sigma > 0 else np.nan,
            "z_err_posterior": abs(posterior_mean - target_m) / sigma if sigma > 0 else np.nan,
            "delta_posterior_minus_prior": posterior_mean - prior_mean,
        })
    return pd.DataFrame(rows)


def run_layer2_audit(
    realizations: List[pd.DataFrame],
    config: Layer2AuditConfig,
) -> Layer2AuditResult:
    if len(realizations) == 0:
        raise ValueError("No realizations provided.")

    if config.check_full_attrs:
        for k, df_k in enumerate(realizations):
            _check_columns(df_k, config.full_attrs, f"realization_{k}")

    # For joint_share targets, t.attr is a virtual group label, not a column;
    # the actually-required columns are listed in t.conditions.
    required_cols_set: set = set()
    for t in config.scenario.targets:
        if t.kind == "joint_share":
            for c in (t.conditions or []):
                required_cols_set.add(str(c["attr"]) if isinstance(c, dict) else str(c[0]))
        else:
            required_cols_set.add(t.attr)
    required_cols = sorted(required_cols_set)
    for c in required_cols:
        if c not in config.task_attrs:
            raise ValueError(f"Scenario target attr '{c}' is not in task_attrs X_A.")

    per_stats = compute_per_realization_target_stats(
        realizations=realizations,
        scenario=config.scenario,
        realization_id_col=config.realization_id_col,
        missing_token=config.missing_token,
    )

    pi, w_unnorm, logw, loss, attr_loss_df = stable_scenario_weights(
        per_stats,
        config.scenario,
        loss_aggregation=config.loss_aggregation,
        attribute_weights=config.attribute_weights,
    )
    K = len(per_stats)
    ess = effective_sample_size_from_normalized_weights(pi)
    ess_ratio = float(ess / K) if K > 0 and np.isfinite(ess) else np.nan

    entropy = float(-np.sum(pi * np.log(np.clip(pi, 1e-300, 1.0))))
    entropy_ratio = float(entropy / np.log(K)) if K > 1 else 1.0
    perplexity = float(np.exp(entropy))
    max_weight = float(np.max(pi))
    top_5_weight = float(np.sum(np.sort(pi)[::-1][: min(5, K)]))

    per_stats = per_stats.copy()
    per_stats["w_norm"] = pi
    per_stats["w_unnorm_stable"] = w_unnorm
    per_stats["logw"] = logw
    per_stats["scenario_loss"] = loss
    if attr_loss_df is not None and not attr_loss_df.empty:
        for col in attr_loss_df.columns:
            per_stats[col] = attr_loss_df[col].to_numpy()

    target_summary = build_target_summary(per_stats, config.scenario, pi)
    prior_mean_loss = float(np.mean(loss))
    posterior_mean_loss = float(np.sum(pi * loss))

    weight_summary = {
        "scenario_name": config.scenario.name,
        "n_realizations": K,
        "n_targets": len(config.scenario.targets),
        "n_attributes": len(_targets_by_attribute(config.scenario)),
        "loss_aggregation": config.loss_aggregation,
        "attribute_weights": config.attribute_weights or {},
        "ESS": ess,
        "ESS_ratio": ess_ratio,
        "entropy": entropy,
        "entropy_ratio": entropy_ratio,
        "perplexity": perplexity,
        "max_weight": max_weight,
        "top_5_weight": top_5_weight,
        "prior_mean_scenario_loss": prior_mean_loss,
        "posterior_mean_scenario_loss": posterior_mean_loss,
        "interpretation": {
            "ESS_ratio_close_to_1": "posterior weights are diffuse across realizations under the configured scenario kernel",
            "ESS_ratio_close_to_0": "scenario posterior weights concentrate on very few realizations",
            "attribute_normalized_loss": "sum over attributes of 0.5 times the mean squared standardized error within each attribute",
        },
    }

    return Layer2AuditResult(
        scenario_name=config.scenario.name,
        n_realizations=K,
        n_targets=len(config.scenario.targets),
        ess=ess,
        ess_ratio=ess_ratio,
        entropy=entropy,
        entropy_ratio=entropy_ratio,
        perplexity=perplexity,
        max_weight=max_weight,
        top_5_weight=top_5_weight,
        prior_mean_loss=prior_mean_loss,
        posterior_mean_loss=posterior_mean_loss,
        per_realization_stats=per_stats,
        target_summary=target_summary,
        weight_summary=weight_summary,
    )


# ============================================================
# Outputs
# ============================================================

def layer2_result_to_summary_dict(result: Layer2AuditResult) -> Dict[str, Any]:
    return {
        "scenario_name": result.scenario_name,
        "n_realizations": result.n_realizations,
        "n_targets": result.n_targets,
        "loss_aggregation": result.weight_summary.get("loss_aggregation"),
        "n_attributes": result.weight_summary.get("n_attributes"),
        "ESS": result.ess,
        "ESS_ratio": result.ess_ratio,
        "entropy": result.entropy,
        "entropy_ratio": result.entropy_ratio,
        "perplexity": result.perplexity,
        "max_weight": result.max_weight,
        "top_5_weight": result.top_5_weight,
        "prior_mean_scenario_loss": result.prior_mean_loss,
        "posterior_mean_scenario_loss": result.posterior_mean_loss,
    }


def save_layer2_outputs(result: Layer2AuditResult, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = layer2_result_to_summary_dict(result)
    with (out / "layer2_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe({**summary, "weight_summary": result.weight_summary}), f, indent=2, ensure_ascii=False)

    pd.DataFrame([summary]).to_csv(out / "layer2_summary.csv", index=False)
    result.per_realization_stats.to_csv(out / "layer2_per_realization_weights.csv", index=False)
    result.target_summary.to_csv(out / "layer2_target_summary.csv", index=False)


# ============================================================
# CSV-facing interface and main
# ============================================================

def build_layer2_config_from_inputs(
    *,
    scenario: ScenarioSpec,
    target_attrs: Sequence[str],
    realization_id_col: str = "realization_id",
    missing_token: str = "__MISSING__",
    check_full_attrs: bool = False,
    loss_aggregation: LossAggregationMode = "attribute_normalized",
    attribute_weights: Optional[Dict[str, float]] = None,
) -> Layer2AuditConfig:
    task_attrs_effective = list(dict.fromkeys(list(TASK_ATTRS_XA) + list(target_attrs)))
    return Layer2AuditConfig(
        full_attrs=FULL_ATTRS_X,
        task_attrs=task_attrs_effective,
        target_attrs=target_attrs,
        scenario=scenario,
        realization_id_col=realization_id_col,
        missing_token=missing_token,
        check_full_attrs=check_full_attrs,
        loss_aggregation=loss_aggregation,
        attribute_weights=attribute_weights,
    )


def run_layer2_audit_from_files(
    *,
    realizations_csv: Optional[str | Path] = None,
    realizations_dir: Optional[str | Path] = None,
    scenario_reference_csv: Optional[str | Path] = None,
    scenario_targets_json: Optional[str | Path] = None,
    target_attrs: Optional[Sequence[str]] = None,
    realization_id_col: str = "realization_id",
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
    scenario_sigma_compose: Literal["replace", "quadrature_add"] = "replace",
    historical_reference_dir: Optional[str | Path] = None,
    historical_reference_glob: str = "*.csv",
    historical_weight_col: Optional[str] = None,
    projection_horizon: Optional[float] = None,
    historical_drift_scale: float = 1.0,
    historical_vol_scale: float = 1.0,
    historical_sampling_scale: float = 1.0,
    attribute_sigma_floor: Optional[Dict[str, float]] = None,
    attribute_drift_scale: Optional[Dict[str, float]] = None,
    attribute_vol_scale: Optional[Dict[str, float]] = None,
    loss_aggregation: LossAggregationMode = "attribute_normalized",
    attribute_weights: Optional[Dict[str, float]] = None,
    check_full_attrs: bool = False,
) -> Layer2AuditResult:
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

    ref: Optional[pd.DataFrame] = None
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
        # In historical_dynamics or scenario_uncertainty mode, initial sigmas
        # are placeholders and are overwritten below by the selected uncertainty model.
        initial_sigma_mode = "reference_se" if sigma_mode == "historical_dynamics" else "fixed"
        scenario = build_scenario_from_reference(
            reference_df=ref,
            target_attrs=target_attrs,
            task_attrs=task_attrs_effective,
            missing_token=missing_token,
            numeric_mode=numeric_mode,
            sigma_mode=initial_sigma_mode,
            sigma_fixed=sigma_fixed,
            sigma_floor=sigma_floor,
            sigma_scale=sigma_scale,
            scenario_name="retrospective_reference_scenario",
            reference_weight_col=reference_weight_col,
        )

    historical_sigma_diag = None
    scenario_sigma_diag = None
    base_set = False
    if sigma_mode == "historical_dynamics" or historical_reference_dir is not None:
        if historical_reference_dir is None:
            raise ValueError("--historical_reference_dir is required when --sigma_mode historical_dynamics.")
        historical_sigma_diag = apply_historical_dynamics_sigmas(
            scenario,
            historical_reference_dir=historical_reference_dir,
            target_attrs=target_attrs,
            missing_token=missing_token,
            reference_weight_col=reference_weight_col,
            historical_weight_col=historical_weight_col,
            historical_glob_pattern=historical_reference_glob,
            age_binning=age_binning,
            age_bin_source_col=age_bin_source_col,
            age_bin_col=age_bin_col,
            age_bins=age_bins_resolved,
            age_bin_labels=age_labels_resolved,
            projection_horizon=projection_horizon,
            scenario_reference_csv=scenario_reference_csv,
            scenario_reference_df=ref if scenario_reference_csv is not None else None,
            sigma_floor=sigma_floor,
            sigma_scale=sigma_scale,
            historical_drift_scale=historical_drift_scale,
            historical_vol_scale=historical_vol_scale,
            historical_sampling_scale=historical_sampling_scale,
            attribute_sigma_floor=attribute_sigma_floor,
            attribute_drift_scale=attribute_drift_scale,
            attribute_vol_scale=attribute_vol_scale,
        )
        base_set = True
    # Apply scenario uncertainty either as the sole sigma (no base set) or
    # composed additively (quadrature) on top of the historical/base sigma.
    if sigma_mode == "scenario_uncertainty" or scenario_attribute_sigma or scenario_target_sigma:
        if base_set and scenario_sigma_compose == "quadrature_add":
            effective_compose: Literal["replace", "quadrature_add"] = "quadrature_add"
        else:
            effective_compose = "replace"
        scenario_sigma_diag = apply_scenario_uncertainty_sigmas(
            scenario,
            attribute_sigma=scenario_attribute_sigma,
            target_sigma=scenario_target_sigma,
            default_sigma=sigma_fixed,
            sigma_floor=sigma_floor,
            sigma_scale=sigma_scale,
            source="scenario_uncertainty",
            compose=effective_compose,
        )

    config = build_layer2_config_from_inputs(
        scenario=scenario,
        target_attrs=target_attrs,
        realization_id_col=realization_id_col,
        missing_token=missing_token,
        check_full_attrs=check_full_attrs,
        loss_aggregation=loss_aggregation,
        attribute_weights=attribute_weights,
    )

    result = run_layer2_audit(realizations=realizations, config=config)

    if output_dir is not None:
        save_layer2_outputs(result, output_dir)
        out = Path(output_dir)
        if historical_sigma_diag is not None:
            historical_sigma_diag.to_csv(out / "layer2_historical_sigma_summary.csv", index=False)
        if scenario_sigma_diag is not None:
            scenario_sigma_diag.to_csv(out / "layer2_scenario_sigma_summary.csv", index=False)

    return result


def main() -> Layer2AuditResult:
    parser = argparse.ArgumentParser(description="Run Layer-2 scenario compatibility audit over population realizations.")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--realizations_csv", help="Combined generated realizations CSV with realization_id column.")
    src.add_argument("--realizations_dir", help="Directory containing one CSV per generated population realization.")

    scen = parser.add_mutually_exclusive_group(required=True)
    scen.add_argument("--scenario_reference_csv", help="Future/reference real population CSV used to construct scenario marginal targets.")
    scen.add_argument("--scenario_targets_json", help="JSON file with explicit scenario targets.")

    parser.add_argument("--target_attrs", nargs="+", default=DEFAULT_PROJECTION_TARGET_ATTRS, help="Attributes in X_A used as scenario marginal targets.")
    parser.add_argument("--realization_id_col", default="realization_id")
    parser.add_argument("--reference_weight_col", default=None, help="Optional weight column in scenario_reference_csv, e.g. PWGTP for PUMS.")
    parser.add_argument("--output_dir", default="layer2_outputs")
    parser.add_argument("--missing_token", default="__MISSING__")
    parser.add_argument("--age_binning", choices=["none", "default"], default="default", help="If default, raw Age in target_attrs is internally replaced by Age_Group.")
    parser.add_argument("--age_bin_source_col", default=DEFAULT_AGE_BIN_SOURCE_COL)
    parser.add_argument("--age_bin_col", default=DEFAULT_AGE_BIN_COL)
    parser.add_argument("--age_bins", default=",".join(str(x) for x in DEFAULT_AGE_BINS), help="Comma-separated age bin edges, e.g. 0,18,35,50,65,200.")
    parser.add_argument("--age_bin_labels", default=",".join(DEFAULT_AGE_LABELS), help="Comma-separated age bin labels, length must be len(age_bins)-1.")
    parser.add_argument("--numeric_mode", choices=["value_share", "mean"], default="value_share")
    parser.add_argument("--sigma_mode", choices=["fixed", "reference_se", "historical_dynamics", "scenario_uncertainty"], default="fixed")
    parser.add_argument("--sigma_fixed", type=float, default=0.02)
    parser.add_argument("--sigma_floor", type=float, default=0.005)
    parser.add_argument("--sigma_scale", type=float, default=1.0)
    parser.add_argument(
        "--scenario_attribute_sigma_json",
        default=None,
        help=(
            "JSON dict or JSON file mapping attribute names to planning-defined sigma values; "
            "used with --sigma_mode scenario_uncertainty. Example: "
            '{\"Age_Group\":0.05,\"Household_Income_Class\":0.08}'
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
        "--scenario_sigma_compose",
        choices=["replace", "quadrature_add"],
        default="replace",
        help=(
            "How scenario_uncertainty sigma combines with a base sigma (historical_dynamics, "
            "reference_se, fixed). 'replace' (default) overwrites; 'quadrature_add' implements "
            "the two-part decomposition sigma^2 = sigma_epistemic^2 + sigma_decision^2, where "
            "the base acts as epistemic uncertainty and the scenario JSON adds decision-level "
            "uncertainty. Has no effect when no scenario_attribute_sigma / scenario_target_sigma "
            "is provided."
        ),
    )
    parser.add_argument(
        "--loss_aggregation",
        choices=["global_scalar", "attribute_normalized"],
        default="attribute_normalized",
        help=(
            "Gaussian-kernel loss aggregation. global_scalar uses 0.5*sum_j z_j^2; "
            "attribute_normalized uses sum_a omega_a*(0.5/|C_a|)*sum_{j in C_a} z_j^2. "
            "The latter is the recommended formulation."
        ),
    )
    parser.add_argument(
        "--attribute_weight_json",
        default=None,
        help='JSON dict of attribute weights for --loss_aggregation attribute_normalized, e.g. {"Household_Income_Class":1.0}.',
    )
    parser.add_argument("--historical_reference_dir", default=None, help="Directory of pre-baseline historical reference CSVs used to estimate historical-dynamics-informed sigmas.")
    parser.add_argument("--historical_reference_glob", default="*.csv", help="Glob pattern for historical reference CSVs.")
    parser.add_argument("--historical_weight_col", default=None, help="Optional weight column for historical reference CSVs. Defaults to --reference_weight_col when omitted.")
    parser.add_argument("--projection_horizon", type=float, default=None, help="Projection horizon H. If omitted, the script tries to infer it from path years; otherwise uses 1.0.")
    parser.add_argument("--historical_drift_scale", type=float, default=1.0, help="Scale gamma for H*|mean annual historical delta| in historical sigma.")
    parser.add_argument("--historical_vol_scale", type=float, default=1.0, help="Scale lambda for sqrt(H)*annual historical volatility in historical sigma.")
    parser.add_argument("--historical_sampling_scale", type=float, default=1.0, help="Scale applied to reference sampling SE in historical sigma.")
    parser.add_argument("--attribute_sigma_floor_json", default=None, help='JSON dict of attribute-specific sigma floors, e.g. {"Household_Income_Class":0.04}.')
    parser.add_argument("--attribute_drift_scale_json", default=None, help="JSON dict of attribute-specific drift scales.")
    parser.add_argument("--attribute_vol_scale_json", default=None, help="JSON dict of attribute-specific volatility scales.")
    parser.add_argument("--check_full_attrs", action="store_true")

    args = parser.parse_args()

    result = run_layer2_audit_from_files(
        realizations_csv=args.realizations_csv,
        realizations_dir=args.realizations_dir,
        scenario_reference_csv=args.scenario_reference_csv,
        scenario_targets_json=args.scenario_targets_json,
        target_attrs=args.target_attrs,
        realization_id_col=args.realization_id_col,
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
        scenario_attribute_sigma=_parse_json_float_dict(args.scenario_attribute_sigma_json, name="scenario_attribute_sigma_json"),
        scenario_target_sigma=_parse_json_float_dict(args.scenario_target_sigma_json, name="scenario_target_sigma_json"),
        scenario_sigma_compose=args.scenario_sigma_compose,
        historical_reference_dir=args.historical_reference_dir,
        historical_reference_glob=args.historical_reference_glob,
        historical_weight_col=args.historical_weight_col,
        projection_horizon=args.projection_horizon,
        historical_drift_scale=args.historical_drift_scale,
        historical_vol_scale=args.historical_vol_scale,
        historical_sampling_scale=args.historical_sampling_scale,
        attribute_sigma_floor=_parse_json_float_dict(args.attribute_sigma_floor_json, name="attribute_sigma_floor_json"),
        attribute_drift_scale=_parse_json_float_dict(args.attribute_drift_scale_json, name="attribute_drift_scale_json"),
        attribute_vol_scale=_parse_json_float_dict(args.attribute_vol_scale_json, name="attribute_vol_scale_json"),
        loss_aggregation=args.loss_aggregation,
        attribute_weights=_parse_json_float_dict(args.attribute_weight_json, name="attribute_weight_json"),
        check_full_attrs=args.check_full_attrs,
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
    print(f"Historical reference dir: {args.historical_reference_dir}")
    print(f"Projection horizon: {args.projection_horizon}")
    print(f"Loss aggregation: {args.loss_aggregation}")
    print(f"Attribute weights: {args.attribute_weight_json}")

    print("\n===== Layer 2 Compatibility Summary =====")
    print(json.dumps(_json_safe(layer2_result_to_summary_dict(result)), indent=2, ensure_ascii=False))

    print("\n===== Scenario Target Summary =====")
    print(result.target_summary.to_string(index=False))

    return result


if __name__ == "__main__":
    main()
