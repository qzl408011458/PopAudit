"""SemaPop evaluation protocol v2.

Replaces ``utils/evals.py``, whose ``evaluate_marg`` / ``evaluate_bivar`` built
(n-1)/(n-2)-way joint tables, accumulated MSE across attribute sets and never
applied the sqrt(N_j) factor of Eq. 11 correctly.  Everything here is written
against Eq. 11 directly:

    SRMSE_j = sqrt( (1/N_j) * sum_c (p_hat_c - p_c)^2 ) / ( (1/N_j) * sum_c p_c )
            = sqrt( N_j * sum_c (p_hat_c - p_c)^2 ) / ( sum_c p_c )

with N_j the number of cells of the *fixed* k-way table (Cartesian product of
the stored per-column category lists, so structural zeros are counted) and
sum_c p_c == 1 whenever the reference lies fully inside the fixed cell set.
There is no accumulation across attribute sets: each j gives one SRMSE_j and
the reported number is the plain mean over j.

Public API
----------
    BinningSpec, EvalConfig            fixed, storable discretisation
    srmse_kway                          Eq. 11, k = 1, 2, ...
    block_precision_recall              exact-tuple P/R/F1 per block and 'all'
    support_coverage                    k-way support coverage (+ train split)
    oracle_rows                         the three mandatory ceilings
    record_validity_rate                universe rules of plan sec.5.3
    conditional_eval                    E5 conditional protocol
    aggregate_seeds, welch_ttest, permutation_test, cohens_d, ess
    evaluate_all                        one dict with everything
    CLI: python -m utils.eval_v2 --real X.csv --gen Y.csv --config cfg.yaml --out metrics.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # optional
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:  # optional
    from scipy import stats as _sps
except Exception:  # pragma: no cover
    _sps = None


OOS_CODE = -1
TAIL_LABEL = "__TAIL__"
OTHER_LABEL = "__OTHER__"
NA_LABEL = "__NA__"

_KIND_CATEGORICAL = "categorical"
_KIND_INTEGER = "integer"
_KIND_BINNED = "binned"
_KINDS = (_KIND_CATEGORICAL, _KIND_INTEGER, _KIND_BINNED)

_DENSE_MAX_CELLS = 20_000_000
_PACK_MAX = 1 << 40


# --------------------------------------------------------------------------- #
# 1. Binning / config
# --------------------------------------------------------------------------- #
@dataclass
class BinningSpec:
    """Discretisation of a single column.

    The ``categories`` list is the *fixed* category list: once fitted on the
    reference it is stored in the yaml so that N_j is byte-identical across
    models, seeds and runs.  Shared with ``utils.MARG_LOSS``.
    """

    name: str
    kind: str = _KIND_CATEGORICAL
    # binned
    bin_width: Optional[float] = None
    edges: Optional[List[float]] = None
    # integer
    tail_merge_at: Optional[int] = None      # values >= this are merged into '>=t'
    low_merge_at: Optional[int] = None       # values <= this are merged into '<=l'
    int_lo: Optional[int] = None
    int_hi: Optional[int] = None
    # categorical
    min_count: int = 0                       # reference categories rarer than this -> __TAIL__
    tail_values: Optional[List[str]] = None
    # common
    clip: bool = True                        # out-of-range numeric -> nearest bin
    #: binned only: carry an explicit NA_LABEL cell instead of dropping NaN as OOS.
    #:
    #: Off by default, so every config frozen before 2026-08-26 encodes byte-identically.
    #: The mixed tier's selection config turns it on: without it a generated frame full of
    #: NaN in a continuous column would encode to OOS, and OOS cells are dropped from
    #: SRMSE rather than charged -- a model that emits no continuous value at all would
    #: score as if it had emitted the right one.
    na_cell: bool = False
    categories: Optional[List[str]] = None   # FIXED cell labels, stored

    # ---------------- fit ---------------- #
    def fit(self, s: pd.Series, unseen_policy: str = "oos") -> "BinningSpec":
        if self.kind not in _KINDS:
            raise ValueError(f"{self.name}: unknown kind {self.kind!r}")
        if self.kind == _KIND_CATEGORICAL:
            self._fit_categorical(s)
        elif self.kind == _KIND_INTEGER:
            self._fit_integer(s)
        else:
            self._fit_binned(s)
        if unseen_policy == "other" and OTHER_LABEL not in self.categories:
            self.categories = list(self.categories) + [OTHER_LABEL]
        return self

    def _fit_categorical(self, s: pd.Series) -> None:
        lab = _as_label(s)
        vc = lab.value_counts()
        tail = sorted(str(v) for v in vc.index[vc.values < max(int(self.min_count), 0)])
        keep = sorted(str(v) for v in vc.index if str(v) not in set(tail))
        cats = list(keep)
        if tail:
            cats.append(TAIL_LABEL)
        self.tail_values = tail
        self.categories = cats

    def _fit_integer(self, s: pd.Series) -> None:
        v = _as_int(s)
        v = v[np.isfinite(v)]
        if v.size == 0:
            raise ValueError(f"{self.name}: no finite values to fit")
        lo = int(self.low_merge_at) if self.low_merge_at is not None else int(v.min())
        hi = int(self.tail_merge_at) if self.tail_merge_at is not None else int(v.max())
        if hi < lo:
            raise ValueError(f"{self.name}: tail_merge_at {hi} < low_merge_at {lo}")
        labels: List[str] = []
        for x in range(lo, hi + 1):
            if x == lo and self.low_merge_at is not None:
                labels.append(f"<={lo}")
            elif x == hi and self.tail_merge_at is not None:
                labels.append(f">={hi}")
            else:
                labels.append(str(x))
        self.int_lo, self.int_hi = lo, hi
        self.categories = labels

    def _fit_binned(self, s: pd.Series) -> None:
        v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            raise ValueError(f"{self.name}: no finite values to fit")
        if self.edges is None:
            if not self.bin_width:
                raise ValueError(f"{self.name}: binned column needs bin_width or edges")
            bw = float(self.bin_width)
            lo = math.floor(float(v.min()) / bw) * bw
            hi = math.floor(float(v.max()) / bw) * bw + bw
            n = int(round((hi - lo) / bw))
            self.edges = [float(lo + i * bw) for i in range(n + 1)]
        e = [float(x) for x in self.edges]
        self.edges = e
        cats = [_bin_label(e[i], e[i + 1]) for i in range(len(e) - 1)]
        if self.na_cell:
            cats.append(NA_LABEL)
        self.categories = cats

    # ---------------- transform ---------------- #
    @property
    def n_categories(self) -> int:
        if self.categories is None:
            raise RuntimeError(f"{self.name}: spec not fitted")
        return len(self.categories)

    def encode(self, s: pd.Series, unseen_policy: str = "oos") -> np.ndarray:
        """Return int32 codes in [0, n_categories); OOS_CODE for out-of-support."""
        if self.categories is None:
            raise RuntimeError(f"{self.name}: spec not fitted")
        if self.kind == _KIND_CATEGORICAL:
            codes = self._encode_categorical(s)
        elif self.kind == _KIND_INTEGER:
            codes = self._encode_integer(s)
        else:
            codes = self._encode_binned(s)
        if unseen_policy == "other":
            if OTHER_LABEL not in self.categories:
                raise RuntimeError(f"{self.name}: unseen_policy='other' but no {OTHER_LABEL}")
            other = self.categories.index(OTHER_LABEL)
            codes = np.where(codes == OOS_CODE, other, codes)
        return codes.astype(np.int32, copy=False)

    def _encode_categorical(self, s: pd.Series) -> np.ndarray:
        lab = _as_label(s)
        if self.tail_values:
            lab = lab.where(~lab.isin(set(self.tail_values)), TAIL_LABEL)
        return pd.Categorical(lab, categories=self.categories).codes.astype(np.int32)

    def _encode_integer(self, s: pd.Series) -> np.ndarray:
        v = _as_int(s)
        lo, hi = int(self.int_lo), int(self.int_hi)
        bad = ~np.isfinite(v)
        vi = np.where(bad, lo, v).astype(np.int64)
        if self.low_merge_at is not None:
            vi = np.maximum(vi, lo)
        if self.tail_merge_at is not None:
            vi = np.minimum(vi, hi)
        if self.clip:
            vi = np.clip(vi, lo, hi)
        codes = vi - lo
        out = np.where((codes < 0) | (codes > hi - lo) | bad, OOS_CODE, codes)
        return out.astype(np.int32)

    def _encode_binned(self, s: pd.Series) -> np.ndarray:
        v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
        e = np.asarray(self.edges, dtype=np.float64)
        nb = len(e) - 1
        bad = ~np.isfinite(v)
        vv = np.where(bad, e[0], v)
        codes = np.searchsorted(e, vv, side="right") - 1
        if self.clip:
            codes = np.clip(codes, 0, nb - 1)
        miss = nb if self.na_cell else OOS_CODE
        out = np.where((codes < 0) | (codes >= nb), OOS_CODE, codes)
        out = np.where(bad, miss, out)
        return out.astype(np.int32)

    # ---------------- (de)serialise ---------------- #
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"kind": self.kind}
        for k in ("bin_width", "edges", "tail_merge_at", "low_merge_at", "int_lo",
                  "int_hi", "tail_values", "categories"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.min_count:
            d["min_count"] = int(self.min_count)
        if self.na_cell:
            d["na_cell"] = True
        if not self.clip:
            d["clip"] = False
        return d

    @classmethod
    def from_dict(cls, name: str, d: Dict[str, Any]) -> "BinningSpec":
        d = dict(d or {})
        return cls(
            name=name,
            kind=str(d.get("kind", _KIND_CATEGORICAL)),
            bin_width=d.get("bin_width"),
            edges=list(d["edges"]) if d.get("edges") is not None else None,
            tail_merge_at=d.get("tail_merge_at"),
            low_merge_at=d.get("low_merge_at"),
            int_lo=d.get("int_lo"),
            int_hi=d.get("int_hi"),
            min_count=int(d.get("min_count", 0)),
            tail_values=list(d["tail_values"]) if d.get("tail_values") is not None else None,
            clip=bool(d.get("clip", True)),
            na_cell=bool(d.get("na_cell", False)),
            categories=list(d["categories"]) if d.get("categories") is not None else None,
        )


def _as_label(s: pd.Series) -> pd.Series:
    out = s.astype("object")
    out = out.where(~pd.isna(out), NA_LABEL)
    return out.astype(str)


def _as_int(s: pd.Series) -> np.ndarray:
    v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.rint(v)


def _bin_label(a: float, b: float) -> str:
    fa = f"{a:g}"
    fb = f"{b:g}"
    return f"[{fa},{fb})"


@dataclass
class Encoded:
    """Integer-code view of a dataframe under a fitted EvalConfig."""

    codes: np.ndarray            # (n, m) int32, OOS_CODE for out-of-support
    valid: np.ndarray            # (n, m) bool
    sizes: np.ndarray            # (m,) int64
    cols: List[str]
    weights: Optional[np.ndarray] = None   # (n,) float64 or None (=> unit weights)

    @property
    def n(self) -> int:
        return int(self.codes.shape[0])

    def col_index(self, name: str) -> int:
        return self.cols.index(name)

    def total_weight(self) -> float:
        return float(self.n if self.weights is None else self.weights.sum())

    def take(self, mask: np.ndarray) -> "Encoded":
        return Encoded(
            codes=self.codes[mask],
            valid=self.valid[mask],
            sizes=self.sizes,
            cols=list(self.cols),
            weights=None if self.weights is None else self.weights[mask],
        )

    def subset_cols(self, cols: Sequence[str]) -> "Encoded":
        idx = [self.cols.index(c) for c in cols]
        return Encoded(
            codes=np.ascontiguousarray(self.codes[:, idx]),
            valid=np.ascontiguousarray(self.valid[:, idx]),
            sizes=self.sizes[idx],
            cols=list(cols),
            weights=self.weights,
        )


@dataclass
class EvalConfig:
    """Full evaluation configuration (columns, binning, blocks, weights)."""

    specs: Dict[str, BinningSpec]
    order: List[str]
    blocks: Dict[str, List[str]] = field(default_factory=dict)
    weight_col: Optional[str] = None
    unseen_policy: str = "oos"     # 'oos' (dropped from the table, kept in denominator) | 'other'
    name: str = "eval_v2"

    # ---------------- construction ---------------- #
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalConfig":
        cols = d.get("columns") or {}
        if isinstance(cols, list):  # [{name: ..., kind: ...}, ...]
            order = [c["name"] for c in cols]
            specs = {c["name"]: BinningSpec.from_dict(c["name"], c) for c in cols}
        else:
            order = list(d.get("order") or cols.keys())
            specs = {k: BinningSpec.from_dict(k, v) for k, v in cols.items()}
        blocks = {k: list(v) for k, v in (d.get("blocks") or {}).items()}
        for b, bc in blocks.items():
            missing = [c for c in bc if c not in specs]
            if missing:
                raise ValueError(f"block {b!r} references unknown columns {missing}")
        return cls(
            specs=specs,
            order=order,
            blocks=blocks,
            weight_col=d.get("weight_col"),
            unseen_policy=str(d.get("unseen_policy", "oos")),
            name=str(d.get("name", "eval_v2")),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "EvalConfig":
        if yaml is None:
            raise RuntimeError("PyYAML is required to read eval configs")
        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "weight_col": self.weight_col,
            "unseen_policy": self.unseen_policy,
            "order": list(self.order),
            "blocks": {k: list(v) for k, v in self.blocks.items()},
            "columns": {c: self.specs[c].to_dict() for c in self.order},
        }

    def to_yaml(self, path: str) -> None:
        if yaml is None:
            raise RuntimeError("PyYAML is required to write eval configs")
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)

    # ---------------- fit / encode ---------------- #
    @property
    def is_fitted(self) -> bool:
        return all(self.specs[c].categories for c in self.order)

    def fit(self, ref_df: pd.DataFrame, force: bool = False) -> "EvalConfig":
        """Derive and STORE the fixed category list of every column from the reference."""
        missing = [c for c in self.order if c not in ref_df.columns]
        if missing:
            raise KeyError(f"reference is missing columns {missing}")
        for c in self.order:
            sp = self.specs[c]
            if sp.categories and not force:
                continue
            sp.fit(ref_df[c], unseen_policy=self.unseen_policy)
        return self

    def sizes(self) -> np.ndarray:
        return np.array([self.specs[c].n_categories for c in self.order], dtype=np.int64)

    def n_cells(self, cols: Sequence[str]) -> int:
        out = 1
        for c in cols:
            out *= self.specs[c].n_categories
        return int(out)

    def encode(self, df: pd.DataFrame, weights: Optional[np.ndarray] = None) -> Encoded:
        if not self.is_fitted:
            raise RuntimeError("EvalConfig is not fitted; call fit(reference_df) first")
        missing = [c for c in self.order if c not in df.columns]
        if missing:
            raise KeyError(f"dataframe is missing columns {missing}")
        n = len(df)
        m = len(self.order)
        codes = np.empty((n, m), dtype=np.int32)
        for j, c in enumerate(self.order):
            codes[:, j] = self.specs[c].encode(df[c], unseen_policy=self.unseen_policy)
        w = _resolve_weights(df, weights, self.weight_col)
        return Encoded(codes=codes, valid=codes != OOS_CODE, sizes=self.sizes(),
                       cols=list(self.order), weights=w)

    def subset(self, cols: Sequence[str]) -> "EvalConfig":
        cols = [c for c in self.order if c in set(cols)]
        blocks = {b: [c for c in bc if c in set(cols)] for b, bc in self.blocks.items()}
        blocks = {b: bc for b, bc in blocks.items() if bc}
        return EvalConfig(
            specs={c: self.specs[c] for c in cols},
            order=list(cols),
            blocks=blocks,
            weight_col=self.weight_col,
            unseen_policy=self.unseen_policy,
            name=self.name + "|subset",
        )

    def block_map(self) -> Dict[str, List[str]]:
        out = {b: list(c) for b, c in self.blocks.items()}
        out["all"] = list(self.order)
        return out


def _resolve_weights(df: pd.DataFrame, weights, weight_col: Optional[str]) -> Optional[np.ndarray]:
    if weights is None:
        if weight_col and weight_col in df.columns:
            w = df[weight_col].to_numpy(dtype=np.float64)
        else:
            return None
    elif isinstance(weights, str):
        w = df[weights].to_numpy(dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
    if w.shape[0] != len(df):
        raise ValueError(f"weights length {w.shape[0]} != n rows {len(df)}")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and non-negative")
    return w


def _as_encoded(x, cfg: EvalConfig, weights=None) -> Encoded:
    if isinstance(x, Encoded):
        if weights is not None:
            x = Encoded(x.codes, x.valid, x.sizes, x.cols,
                        np.asarray(weights, dtype=np.float64))
        return x
    return cfg.encode(x, weights=weights)


# --------------------------------------------------------------------------- #
# 2. cell counting
# --------------------------------------------------------------------------- #
def _dense_counts(enc: Encoded, idx: Sequence[int]) -> Tuple[np.ndarray, int]:
    """Dense weighted counts over the full Cartesian cell set of columns ``idx``."""
    sizes = enc.sizes
    ncell = 1
    for j in idx:
        ncell *= int(sizes[j])
    mask = enc.valid[:, idx[0]]
    for j in idx[1:]:
        mask = mask & enc.valid[:, j]
    if mask.all():
        key = enc.codes[:, idx[0]].astype(np.int64)
        for j in idx[1:]:
            key = key * int(sizes[j]) + enc.codes[:, j]
        w = enc.weights
    else:
        key = enc.codes[mask, idx[0]].astype(np.int64)
        for j in idx[1:]:
            key = key * int(sizes[j]) + enc.codes[mask, j]
        w = None if enc.weights is None else enc.weights[mask]
    cnt = np.bincount(key, weights=w, minlength=ncell)
    return cnt.astype(np.float64, copy=False), int(ncell)


def _sparse_counts(enc: Encoded, idx: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, int]:
    """Sparse weighted counts: (unique_keys, counts, n_cells)."""
    sizes = enc.sizes
    ncell = 1
    for j in idx:
        ncell *= int(sizes[j])
    mask = enc.valid[:, idx[0]]
    for j in idx[1:]:
        mask = mask & enc.valid[:, j]
    if mask.all():
        key = enc.codes[:, idx[0]].astype(np.int64)
        for j in idx[1:]:
            key = key * int(sizes[j]) + enc.codes[:, j]
        w = enc.weights
    else:
        key = enc.codes[mask, idx[0]].astype(np.int64)
        for j in idx[1:]:
            key = key * int(sizes[j]) + enc.codes[mask, j]
        w = None if enc.weights is None else enc.weights[mask]
    if key.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64), int(ncell)
    uk, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv, weights=w, minlength=uk.size).astype(np.float64)
    return uk, cnt, int(ncell)


def _pack_exact(enc: Encoded, cols: Sequence[str]) -> Tuple[np.ndarray, int]:
    """Exact row-tuple keys over ``cols`` (OOS shifted into its own slot).

    Falls back to ``pd.factorize`` whenever the running product would overflow,
    so arbitrarily wide tuples (all 20 attributes) stay exact.
    """
    idx = [enc.col_index(c) for c in cols]
    sizes = enc.sizes
    key = enc.codes[:, idx[0]].astype(np.int64) + 1
    n_key = int(sizes[idx[0]]) + 1
    for j in idx[1:]:
        s = int(sizes[j]) + 1
        key = key * s + (enc.codes[:, j].astype(np.int64) + 1)
        n_key *= s
        if n_key > _PACK_MAX:
            codes, uniq = pd.factorize(key, sort=False)
            key = codes.astype(np.int64)
            n_key = int(len(uniq))
    return key, n_key


# --------------------------------------------------------------------------- #
# 3. Eq. 11
# --------------------------------------------------------------------------- #
def _subsets(cols: Sequence[str], k: int, subsets=None, max_subsets: Optional[int] = None,
             rng: Optional[np.random.Generator] = None) -> List[Tuple[str, ...]]:
    if subsets is not None:
        return [tuple(s) for s in subsets]
    all_s = list(itertools.combinations(list(cols), k))
    if max_subsets is not None and len(all_s) > max_subsets:
        rng = rng or np.random.default_rng(0)
        pick = rng.choice(len(all_s), size=max_subsets, replace=False)
        return [all_s[i] for i in sorted(pick.tolist())]
    return all_s


def srmse_kway(real_df, gen_df, cfg: EvalConfig, k: int = 1,
               real_weights=None, gen_weights=None,
               subsets: Optional[Sequence[Sequence[str]]] = None,
               max_subsets: Optional[int] = None,
               rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
    """Eq. 11 SRMSE over every attribute set of size ``k``.

    SRMSE_j = sqrt(N_j * sum_c (p_hat_c - p_c)^2) / (sum_c p_c), with p / p_hat
    normalised by the *total* (weighted) row count of each side, and the cell
    set fixed by the stored category lists (zero cells included in N_j).

    Returns ``{'k', 'mean', 'median', 'max', 'n_subsets', 'per_j'}`` where
    ``per_j`` is a DataFrame (attrs, n_cells, srmse, tv, sum_p, sum_p_hat).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    er = _as_encoded(real_df, cfg, real_weights)
    eg = _as_encoded(gen_df, cfg, gen_weights)
    combos = _subsets(cfg.order, k, subsets, max_subsets, rng)
    wr_tot = er.total_weight()
    wg_tot = eg.total_weight()
    if wr_tot <= 0 or wg_tot <= 0:
        raise ValueError("empty (or zero-weight) reference / generated sample")

    rows = []
    for cs in combos:
        idx = [er.col_index(c) for c in cs]
        ncell = 1
        for j in idx:
            ncell *= int(er.sizes[j])
        if ncell <= _DENSE_MAX_CELLS:
            cr, _ = _dense_counts(er, idx)
            cg, _ = _dense_counts(eg, idx)
            p = cr / wr_tot
            q = cg / wg_tot
            d = q - p
            ssd = float(np.dot(d, d))
            tv = 0.5 * float(np.abs(d).sum())
            sum_p = float(p.sum())
            sum_q = float(q.sum())
        else:
            kr, cr, _ = _sparse_counts(er, idx)
            kg, cg, _ = _sparse_counts(eg, idx)
            union = np.union1d(kr, kg)
            p = np.zeros(union.size, dtype=np.float64)
            q = np.zeros(union.size, dtype=np.float64)
            p[np.searchsorted(union, kr)] = cr / wr_tot
            q[np.searchsorted(union, kg)] = cg / wg_tot
            d = q - p
            ssd = float(np.dot(d, d))
            tv = 0.5 * float(np.abs(d).sum())
            sum_p = float(p.sum())
            sum_q = float(q.sum())
        srmse = math.sqrt(ncell * ssd) / sum_p if sum_p > 0 else float("nan")
        rows.append({"attrs": "|".join(cs), "n_cells": int(ncell), "srmse": srmse,
                     "tv": tv, "sum_p": sum_p, "sum_p_hat": sum_q})

    per_j = pd.DataFrame(rows)
    vals = per_j["srmse"].to_numpy(dtype=np.float64)
    return {
        "k": int(k),
        "mean": float(np.nanmean(vals)) if vals.size else float("nan"),
        "median": float(np.nanmedian(vals)) if vals.size else float("nan"),
        "max": float(np.nanmax(vals)) if vals.size else float("nan"),
        "n_subsets": int(len(combos)),
        "per_j": per_j,
    }


# --------------------------------------------------------------------------- #
# 4. block precision / recall
# --------------------------------------------------------------------------- #
def block_precision_recall(real_df, gen_df, cfg: EvalConfig,
                           real_weights=None, gen_weights=None,
                           blocks: Optional[Dict[str, Sequence[str]]] = None) -> Dict[str, Any]:
    """Exact-tuple precision / recall / F1 per matching block and for 'all'.

    precision = (weighted) share of generated rows whose tuple exists in the
                reference;
    recall    = (weighted) share of reference rows whose tuple exists in the
                generated sample.
    """
    er = _as_encoded(real_df, cfg, real_weights)
    eg = _as_encoded(gen_df, cfg, gen_weights)
    bm = {k: list(v) for k, v in (blocks or cfg.block_map()).items()}
    wr = er.weights
    wg = eg.weights
    wr_tot = er.total_weight()
    wg_tot = eg.total_weight()

    combined = Encoded(
        codes=np.vstack([er.codes, eg.codes]),
        valid=np.vstack([er.valid, eg.valid]),
        sizes=er.sizes, cols=list(er.cols), weights=None,
    )
    nr = er.n
    out: Dict[str, Any] = {}
    for bname, cols in bm.items():
        if not cols:
            continue
        key, _ = _pack_exact(combined, cols)
        kr = key[:nr]
        kg = key[nr:]
        ur, cnt_r = np.unique(kr, return_counts=True)
        ug = np.unique(kg)
        in_ref = np.isin(kg, ur, assume_unique=False)
        in_gen = np.isin(kr, ug, assume_unique=False)
        prec = (float(wg[in_ref].sum()) / wg_tot) if wg is not None \
            else float(in_ref.sum()) / kg.size
        rec = (float(wr[in_gen].sum()) / wr_tot) if wr is not None \
            else float(in_gen.sum()) / kr.size
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        n_single = int((cnt_r == 1).sum())
        out[bname] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "n_unique_ref": int(ur.size),
            "n_unique_gen": int(ug.size),
            "singleton_share_ref": float(n_single / ur.size) if ur.size else float("nan"),
            "singleton_row_share_ref": float(n_single / kr.size) if kr.size else float("nan"),
            "n_cols": len(cols),
        }
    return out


# --------------------------------------------------------------------------- #
# 5. support coverage
# --------------------------------------------------------------------------- #
def support_coverage(real_df, gen_df, cfg: EvalConfig, k: int = 2,
                     train_df=None, real_weights=None, gen_weights=None,
                     max_subsets: Optional[int] = None,
                     rng: Optional[np.random.Generator] = None,
                     subsets: Optional[Sequence[Sequence[str]]] = None) -> Dict[str, Any]:
    """Fraction of reference cells with non-zero mass that the sample also covers.

    When ``train_df`` is given the reference cells are decomposed into
    'inside training support' vs 'outside'.
    """
    er = _as_encoded(real_df, cfg, real_weights)
    eg = _as_encoded(gen_df, cfg, gen_weights)
    et = None if train_df is None else _as_encoded(train_df, cfg)
    combos = _subsets(cfg.order, k, subsets, max_subsets, rng)
    wr_tot = er.total_weight()

    rows = []
    for cs in combos:
        idx_r = [er.col_index(c) for c in cs]
        kr, cr, ncell = _sparse_counts(er, idx_r)
        kg, _, _ = _sparse_counts(eg, [eg.col_index(c) for c in cs])
        covered = np.isin(kr, kg)
        mass = cr / wr_tot
        rec = {
            "attrs": "|".join(cs),
            "n_cells": int(ncell),
            "n_ref_nonzero": int(kr.size),
            "cell_coverage": float(covered.mean()) if kr.size else float("nan"),
            "mass_coverage": float(mass[covered].sum() / mass.sum()) if mass.sum() > 0 else float("nan"),
        }
        if et is not None:
            kt, _, _ = _sparse_counts(et, [et.col_index(c) for c in cs])
            inside = np.isin(kr, kt)
            n_in = int(inside.sum())
            n_out = int(kr.size - n_in)
            rec["n_ref_inside_train"] = n_in
            rec["n_ref_outside_train"] = n_out
            rec["cell_coverage_inside"] = float(covered[inside].mean()) if n_in else float("nan")
            rec["cell_coverage_outside"] = float(covered[~inside].mean()) if n_out else float("nan")
        rows.append(rec)

    per_j = pd.DataFrame(rows)
    out: Dict[str, Any] = {
        "k": int(k),
        "n_subsets": int(len(combos)),
        "cell_coverage_mean": float(per_j["cell_coverage"].mean()) if len(per_j) else float("nan"),
        "mass_coverage_mean": float(per_j["mass_coverage"].mean()) if len(per_j) else float("nan"),
        "per_j": per_j,
    }
    if et is not None and len(per_j):
        out["cell_coverage_inside_mean"] = float(per_j["cell_coverage_inside"].mean())
        out["cell_coverage_outside_mean"] = float(np.nanmean(per_j["cell_coverage_outside"].to_numpy()))
        out["ref_cells_outside_train_share"] = float(
            per_j["n_ref_outside_train"].sum() / max(per_j["n_ref_nonzero"].sum(), 1))
    return out


# --------------------------------------------------------------------------- #
# 6. core metric bundle + oracle rows
# --------------------------------------------------------------------------- #
def core_metrics(real_df, gen_df, cfg: EvalConfig, real_weights=None, gen_weights=None,
                 train_df=None, coverage_k: Sequence[int] = (2,),
                 coverage_max_subsets: Optional[int] = None,
                 rng: Optional[np.random.Generator] = None,
                 with_per_j: bool = False) -> Dict[str, Any]:
    """SRMSE-1/2 + block P/R/F1 + coverage in one dict (used by tables and oracles)."""
    er = _as_encoded(real_df, cfg, real_weights)
    eg = _as_encoded(gen_df, cfg, gen_weights)
    t0 = time.time()
    s1 = srmse_kway(er, eg, cfg, k=1)
    t1 = time.time()
    s2 = srmse_kway(er, eg, cfg, k=2)
    t2 = time.time()
    pr = block_precision_recall(er, eg, cfg)
    t3 = time.time()
    out: Dict[str, Any] = {
        "n_real": er.n,
        "n_gen": eg.n,
        "srmse1": s1["mean"],
        "srmse1_max": s1["max"],
        "srmse2": s2["mean"],
        "srmse2_max": s2["max"],
        "n_pairs": s2["n_subsets"],
        "blocks": pr,
        "timing_s": {"srmse1": t1 - t0, "srmse2": t2 - t1, "block_pr": t3 - t2},
    }
    if with_per_j:
        out["srmse1_per_j"] = s1["per_j"]
        out["srmse2_per_j"] = s2["per_j"]
    cov = {}
    for k in coverage_k:
        t = time.time()
        c = support_coverage(er, eg, cfg, k=k, train_df=train_df,
                             max_subsets=coverage_max_subsets, rng=rng)
        if not with_per_j:
            c.pop("per_j", None)
        c["timing_s"] = time.time() - t
        cov[f"k{k}"] = c
    out["coverage"] = cov
    return out


def oracle_rows(real_ref: pd.DataFrame, train_df: pd.DataFrame, cfg: EvalConfig,
                rng: Optional[np.random.Generator] = None,
                real_weights=None, train_weights=None,
                coverage_k: Sequence[int] = (2,),
                coverage_max_subsets: Optional[int] = None) -> Dict[str, Any]:
    """The three mandatory ceilings of plan sec.6.1.

    (a) ``train_vs_ref``      real training sample scored against the reference;
    (b) ``ref_half_vs_half``  reference split in half, one half scoring the other;
    (c) ``train_resample``    empirical resample (with replacement) of the
                              training sample to reference size.
    All three return the same metric dict as a model row, so tables can print
    them side by side.
    """
    rng = rng or np.random.default_rng(42)
    er = _as_encoded(real_ref, cfg, real_weights)
    et = _as_encoded(train_df, cfg, train_weights)

    out: Dict[str, Any] = {}
    out["train_vs_ref"] = core_metrics(er, et, cfg, coverage_k=coverage_k,
                                       coverage_max_subsets=coverage_max_subsets, rng=rng)
    perm = rng.permutation(er.n)
    half = er.n // 2
    a = er.take(perm[:half])
    b = er.take(perm[half:2 * half])
    out["ref_half_vs_half"] = core_metrics(a, b, cfg, coverage_k=coverage_k,
                                           coverage_max_subsets=coverage_max_subsets, rng=rng)
    if et.weights is None:
        pick = rng.integers(0, et.n, size=er.n)
    else:
        p = et.weights / et.weights.sum()
        pick = rng.choice(et.n, size=er.n, replace=True, p=p)
    res = et.take(pick)
    res = Encoded(res.codes, res.valid, res.sizes, res.cols, None)
    out["train_resample"] = core_metrics(er, res, cfg, coverage_k=coverage_k,
                                         coverage_max_subsets=coverage_max_subsets, rng=rng)
    return out


# --------------------------------------------------------------------------- #
# 7. record validity (universe rules, plan sec.5.3)
# --------------------------------------------------------------------------- #
@dataclass
class UniverseRule:
    name: str
    requires: Tuple[str, ...]
    fn: Callable[[pd.DataFrame], np.ndarray]   # -> bool array, True == violation
    note: str = ""


DEFAULT_UNIVERSE_VOCAB: Dict[str, Any] = {
    # column names
    "age": "Age",
    "marital": "Marital_status",
    "education": "Education",
    "employment": "Employment_status",
    "studenthood": "Studenthood_status",
    "work_hours": "Work_hours_class",
    "income_p": "Personal_Income_Class",
    "commute_mode": "Commute_Mode",
    "commute_time": "Commute_Time_Class",
    "hh_type": "Household_Type",
    "hh_size": "Household_Size",
    "hh_income": "Household_Income_Class",
    "tenure": "Tenure",
    "vehicles": "Number_of_vehicles",
    # category labels
    "no_commute": "No_Commute",
    "wfh": "Worked_at_home",
    "not_applicable": "Not_applicable",
    "employed": "Employed",
    "unemployed": "Unemployed",
    "nilf": "Not_in_labor_force",
    "under_16": "Under_16",
    "under_15": "Under_15",            # Marital_status label for AGEP<15
    "under_15_income": "Under_15",     # Personal_Income_Class label for AGEP<15
    "under_3": "Under_3",
    "no_work_12m": "No_work_past_12_months",
    "gq": "Group_quarters",
}


def _eq(df: pd.DataFrame, col: str, val) -> np.ndarray:
    return (df[col].astype(str).to_numpy() == str(val))


def _isin(df: pd.DataFrame, col: str, vals: Iterable) -> np.ndarray:
    s = {str(v) for v in vals}
    return np.isin(df[col].astype(str).to_numpy(), list(s))


def _implies(ante: np.ndarray, cons: np.ndarray) -> np.ndarray:
    return ante & ~cons


def universe_rules_for(cfg: "EvalConfig",
                       vocab: Optional[Dict[str, Any]] = None) -> List[UniverseRule]:
    """The rule set matching a config's attribute vocabulary.

    The 20-attribute build encodes out-of-universe as a sentinel CATEGORY ("No_Commute",
    "Under_16"); the extended S/M/L build uses `Not_in_universe`.  Running one set against
    the other data is not merely wrong, it is quietly wrong: the 20-attribute rules on
    50-attribute data report record_validity_rate = 0.0 with one rule violated by every
    single row.  Dispatch on the config so that cannot happen by omission.
    """
    if str(getattr(cfg, "name", "")).startswith("eval_sml"):
        from utils.universe_rules_sml import universe_rules_sml
        return universe_rules_sml(list(cfg.order))
    return default_universe_rules(vocab)


def default_universe_rules(vocab: Optional[Dict[str, Any]] = None) -> List[UniverseRule]:
    """One-directional universe-dependency rules of plan sec.5.3.

    Only used as a *record-validity diagnostic* (never called feasibility or
    audit).  Rules whose columns are absent from the frame are reported as
    skipped rather than failing.
    """
    v = dict(DEFAULT_UNIVERSE_VOCAB)
    v.update(vocab or {})
    R: List[UniverseRule] = []

    R.append(UniverseRule(
        "nonworker_implies_no_commute", (v["employment"], v["commute_mode"]),
        lambda d: _implies(_isin(d, v["employment"], [v["under_16"], v["unemployed"], v["nilf"]]),
                           _eq(d, v["commute_mode"], v["no_commute"])),
        "Under_16 / Unemployed / NILF => Commute_Mode = No_Commute"))
    R.append(UniverseRule(
        "commute_implies_employed", (v["employment"], v["commute_mode"]),
        lambda d: _implies(~_eq(d, v["commute_mode"], v["no_commute"]),
                           _eq(d, v["employment"], v["employed"])),
        "Commute_Mode != No_Commute => Employed (reverse does NOT hold: ESR 2/5)"))
    R.append(UniverseRule(
        "commute_time_na_implies_no_trip", (v["commute_time"], v["commute_mode"]),
        lambda d: _implies(_eq(d, v["commute_time"], v["not_applicable"]),
                           _isin(d, v["commute_mode"], [v["no_commute"], v["wfh"]])),
        "Commute_Time = Not_applicable => Commute_Mode in {No_Commute, Worked_at_home}"))
    R.append(UniverseRule(
        "no_trip_implies_commute_time_na", (v["commute_time"], v["commute_mode"]),
        lambda d: _implies(_isin(d, v["commute_mode"], [v["no_commute"], v["wfh"]]),
                           _eq(d, v["commute_time"], v["not_applicable"])),
        "other half of the iff"))
    R.append(UniverseRule(
        "under16_implies_no_work_hours", (v["employment"], v["work_hours"]),
        lambda d: _implies(_eq(d, v["employment"], v["under_16"]),
                           _eq(d, v["work_hours"], v["no_work_12m"])),
        "Under_16 => Work_hours = No_work_past_12_months"))
    R.append(UniverseRule(
        "employed_implies_work_hours", (v["employment"], v["work_hours"]),
        lambda d: _implies(_eq(d, v["employment"], v["employed"]),
                           ~_eq(d, v["work_hours"], v["no_work_12m"])),
        "Employed => Work_hours != NA (reverse does NOT hold)"))
    R.append(UniverseRule(
        "age_lt15_implies_marital_under15", (v["age"], v["marital"]),
        lambda d: _implies(pd.to_numeric(d[v["age"]], errors="coerce").to_numpy() < 15,
                           _eq(d, v["marital"], v["under_15"])),
        "AGEP<15 => Marital = Under_15"))
    R.append(UniverseRule(
        "age_lt15_implies_income_under15", (v["age"], v["income_p"]),
        lambda d: _implies(pd.to_numeric(d[v["age"]], errors="coerce").to_numpy() < 15,
                           _eq(d, v["income_p"], v["under_15_income"])),
        "AGEP<15 => Personal_Income = Under_15"))
    R.append(UniverseRule(
        "age_lt3_implies_education_under3", (v["age"], v["education"]),
        lambda d: _implies(pd.to_numeric(d[v["age"]], errors="coerce").to_numpy() < 3,
                           _eq(d, v["education"], v["under_3"])),
        "AGEP<3 => Education = Under_3"))
    R.append(UniverseRule(
        "age_lt3_implies_school_under3", (v["age"], v["studenthood"]),
        lambda d: _implies(pd.to_numeric(d[v["age"]], errors="coerce").to_numpy() < 3,
                           _eq(d, v["studenthood"], v["under_3"])),
        "AGEP<3 => Studenthood = Under_3"))
    for tgt, key in (("hh_income", "hh_income"), ("tenure", "tenure"), ("vehicles", "vehicles")):
        col = v[key]
        R.append(UniverseRule(
            f"gq_implies_{key}_gq", (v["hh_type"], col),
            (lambda c: (lambda d: _implies(_eq(d, v["hh_type"], v["gq"]), _eq(d, c, v["gq"]))))(col),
            f"Household_Type = Group_quarters => {col} = Group_quarters"))
        R.append(UniverseRule(
            f"{key}_gq_implies_gq", (v["hh_type"], col),
            (lambda c: (lambda d: _implies(_eq(d, c, v["gq"]), _eq(d, v["hh_type"], v["gq"]))))(col),
            "other half of the iff"))
    R.append(UniverseRule(
        "gq_implies_hh_size_1", (v["hh_type"], v["hh_size"]),
        lambda d: _implies(_eq(d, v["hh_type"], v["gq"]),
                           pd.to_numeric(d[v["hh_size"]], errors="coerce").to_numpy() == 1),
        "Household_Type = Group_quarters => Household_Size = 1"))
    return R


def record_validity_rate(df: pd.DataFrame,
                         rules: Optional[Sequence] = None,
                         vocab: Optional[Dict[str, Any]] = None,
                         weights=None) -> Dict[str, Any]:
    """Per-rule violation share and overall record-validity rate.

    ``rules`` may be a list of :class:`UniverseRule`, or of plain
    ``(name, callable)`` pairs where the callable returns a boolean violation
    mask.  Rules whose required columns are missing are skipped (and counted).
    """
    if rules is None:
        rules = default_universe_rules(vocab)
    w = None if weights is None else np.asarray(weights, dtype=np.float64)
    n = len(df)
    w_tot = float(n if w is None else w.sum())
    any_viol = np.zeros(n, dtype=bool)
    per_rule: Dict[str, Any] = {}
    n_applied = 0
    n_skipped = 0
    for r in rules:
        if isinstance(r, UniverseRule):
            name, req, fn, note = r.name, r.requires, r.fn, r.note
        else:
            name, fn = r[0], r[1]
            req, note = tuple(), ""
        missing = [c for c in req if c not in df.columns]
        if missing:
            per_rule[name] = {"status": "skipped", "missing_columns": missing, "note": note}
            n_skipped += 1
            continue
        try:
            m = np.asarray(fn(df), dtype=bool).reshape(-1)
        except KeyError as e:  # column referenced but not declared
            per_rule[name] = {"status": "skipped", "missing_columns": [str(e)], "note": note}
            n_skipped += 1
            continue
        if m.shape[0] != n:
            raise ValueError(f"rule {name}: mask length {m.shape[0]} != {n}")
        share = float(m.sum() / n) if w is None else float(w[m].sum() / w_tot)
        per_rule[name] = {"status": "ok", "violation_share": share,
                          "n_violations": int(m.sum()), "note": note}
        any_viol |= m
        n_applied += 1
    total = float(any_viol.sum() / n) if w is None else float(w[any_viol].sum() / w_tot)
    return {
        "n_rows": int(n),
        "n_rules_applied": n_applied,
        "n_rules_skipped": n_skipped,
        "total_violation_share": total,
        "record_validity_rate": 1.0 - total,
        "per_rule": per_rule,
    }


# --------------------------------------------------------------------------- #
# 8. conditional evaluation (E5)
# --------------------------------------------------------------------------- #
def constraint_mask(df: pd.DataFrame, constraints: Dict[str, Any]) -> np.ndarray:
    """``{col: set(values)}`` or ``{col: callable(series) -> bool mask}``."""
    m = np.ones(len(df), dtype=bool)
    for col, cond in constraints.items():
        if col not in df.columns:
            raise KeyError(f"constraint column {col!r} not in dataframe")
        if callable(cond):
            m &= np.asarray(cond(df[col]), dtype=bool).reshape(-1)
        else:
            vals = {str(v) for v in (cond if isinstance(cond, (set, list, tuple)) else [cond])}
            m &= np.isin(df[col].astype(str).to_numpy(), list(vals))
    return m


def ess(w) -> float:
    """Effective sample size (sum w)^2 / sum w^2."""
    w = np.asarray(w, dtype=np.float64)
    s2 = float(np.sum(w ** 2))
    return float((float(w.sum()) ** 2) / s2) if s2 > 0 else 0.0


def conditional_eval(real_ref: pd.DataFrame, gen: pd.DataFrame,
                     constraints: Dict[str, Any], cfg: EvalConfig,
                     real_weights=None, gen_weights=None,
                     coverage_k: Sequence[int] = (2,),
                     coverage_max_subsets: Optional[int] = None,
                     rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
    """E5 protocol: yield + everything else computed on the satisfying subsets.

    Free attributes = configured columns not constrained.  Conditional SRMSE-1/2,
    per-attribute TV, block P/R and coverage are computed on the restricted
    config; ESS is reported whenever weights are supplied.
    """
    rng = rng or np.random.default_rng(0)
    wr = _resolve_weights(real_ref, real_weights, cfg.weight_col)
    wg = _resolve_weights(gen, gen_weights, cfg.weight_col)

    mr = constraint_mask(real_ref, constraints)
    mg = constraint_mask(gen, constraints)
    n_gen = len(gen)
    yield_unw = float(mg.sum() / n_gen) if n_gen else float("nan")
    yield_w = float(wg[mg].sum() / wg.sum()) if wg is not None and wg.sum() > 0 else None
    ref_share = float(mr.sum() / len(real_ref)) if len(real_ref) else float("nan")

    out: Dict[str, Any] = {
        "constraints": {k: (sorted(str(x) for x in v) if not callable(v) else "<callable>")
                        for k, v in constraints.items()},
        "n_gen": int(n_gen),
        "n_gen_satisfying": int(mg.sum()),
        "yield": yield_unw,
        "yield_weighted": yield_w,
        "n_ref": int(len(real_ref)),
        "n_ref_satisfying": int(mr.sum()),
        "ref_satisfying_share": ref_share,
    }
    if wg is not None:
        out["ess_gen"] = ess(wg)
        out["ess_gen_satisfying"] = ess(wg[mg]) if mg.any() else 0.0
        out["ess_ratio_satisfying"] = (out["ess_gen_satisfying"] / max(int(mg.sum()), 1))
    if wr is not None:
        out["ess_ref_satisfying"] = ess(wr[mr]) if mr.any() else 0.0

    free = [c for c in cfg.order if c not in constraints]
    out["free_attributes"] = free
    if mg.sum() == 0 or mr.sum() == 0 or not free:
        out["status"] = "empty_subset" if (mg.sum() == 0 or mr.sum() == 0) else "no_free_attributes"
        return out

    sub = cfg.subset(free)
    rs = real_ref.loc[mr]
    gs = gen.loc[mg]
    wrs = None if wr is None else wr[mr]
    wgs = None if wg is None else wg[mg]

    er = sub.encode(rs, weights=wrs)
    eg = sub.encode(gs, weights=wgs)
    s1 = srmse_kway(er, eg, sub, k=1)
    s2 = srmse_kway(er, eg, sub, k=2) if len(free) >= 2 else None
    out["status"] = "ok"
    out["cond_srmse1"] = s1["mean"]
    out["cond_srmse2"] = s2["mean"] if s2 else float("nan")
    out["tv_per_attribute"] = {r["attrs"]: float(r["tv"]) for _, r in s1["per_j"].iterrows()}
    out["tv_mean"] = float(s1["per_j"]["tv"].mean())
    out["blocks"] = block_precision_recall(er, eg, sub)
    cov = {}
    for k in coverage_k:
        if len(free) < k:
            continue
        c = support_coverage(er, eg, sub, k=k, max_subsets=coverage_max_subsets, rng=rng)
        c.pop("per_j", None)
        cov[f"k{k}"] = c
    out["coverage"] = cov
    return out


# --------------------------------------------------------------------------- #
# 9. multi-seed aggregation and tests
# --------------------------------------------------------------------------- #
def flatten_metrics(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """Flatten a nested metric dict into dotted float paths (non-numerics dropped)."""
    out: Dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_metrics(v, key + "."))
        elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def _t_crit(n: int) -> float:
    if n < 2:
        return float("nan")
    if _sps is not None:
        return float(_sps.t.ppf(0.975, n - 1))
    return 1.959963984540054


def aggregate_seeds(list_of_metric_dicts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean / sd / n / 95% CI over seeds, keyed by dotted metric path."""
    flat = [flatten_metrics(d) for d in list_of_metric_dicts]
    keys = sorted({k for f in flat for k in f})
    out: Dict[str, Any] = {}
    for k in keys:
        vals = np.array([f[k] for f in flat if k in f], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        n = int(vals.size)
        mean = float(vals.mean()) if n else float("nan")
        sd = float(vals.std(ddof=1)) if n > 1 else float("nan")
        half = _t_crit(n) * sd / math.sqrt(n) if n > 1 else float("nan")
        out[k] = {"mean": mean, "sd": sd, "n": n,
                  "ci95_lo": mean - half if n > 1 else float("nan"),
                  "ci95_hi": mean + half if n > 1 else float("nan"),
                  "values": vals.tolist()}
    return out


def cohens_d(a, b) -> float:
    """Pooled-sd standardised mean difference (a - b)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float("nan")
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if sp2 <= 0:
        return 0.0 if float(a.mean()) == float(b.mean()) else float("inf")
    return float((a.mean() - b.mean()) / math.sqrt(sp2))


def welch_ttest(a, b) -> Dict[str, float]:
    """Welch two-sample t test on seed-level means, with effect size and CI."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return {"t": float("nan"), "df": float("nan"), "p": float("nan"),
                "mean_diff": float(a.mean() - b.mean()) if na and nb else float("nan"),
                "cohens_d": float("nan"), "ci95_lo": float("nan"), "ci95_hi": float("nan"),
                "n_a": na, "n_b": nb}
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    diff = float(a.mean() - b.mean())
    if se == 0:
        t, dfree, p = float("inf") if diff else 0.0, float(na + nb - 2), 0.0 if diff else 1.0
        half = 0.0
    else:
        t = diff / se
        dfree = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
        if _sps is not None:
            p = float(2 * _sps.t.sf(abs(t), dfree))
            half = float(_sps.t.ppf(0.975, dfree)) * se
        else:
            p = float(math.erfc(abs(t) / math.sqrt(2)))
            half = 1.959963984540054 * se
    return {"t": float(t), "df": float(dfree), "p": float(p), "mean_diff": diff,
            "cohens_d": cohens_d(a, b), "ci95_lo": diff - half, "ci95_hi": diff + half,
            "n_a": na, "n_b": nb}


def permutation_test(a, b, n_perm: int = 10000, rng: Optional[np.random.Generator] = None,
                     alternative: str = "two-sided") -> Dict[str, float]:
    """Exact-ish permutation test on the difference of means (seed-level)."""
    rng = rng or np.random.default_rng(0)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = a.size, b.size
    if na == 0 or nb == 0:
        return {"p": float("nan"), "observed_diff": float("nan"), "cohens_d": float("nan"),
                "n_perm": 0}
    pool = np.concatenate([a, b])
    obs = float(a.mean() - b.mean())
    idx = np.argsort(rng.random((n_perm, pool.size)), axis=1)
    perm = pool[idx]
    diffs = perm[:, :na].mean(axis=1) - perm[:, na:].mean(axis=1)
    if alternative == "two-sided":
        hits = int(np.sum(np.abs(diffs) >= abs(obs) - 1e-15))
    elif alternative == "greater":
        hits = int(np.sum(diffs >= obs - 1e-15))
    else:
        hits = int(np.sum(diffs <= obs + 1e-15))
    return {"p": float((hits + 1) / (n_perm + 1)), "observed_diff": obs,
            "cohens_d": cohens_d(a, b), "n_perm": int(n_perm)}


# --------------------------------------------------------------------------- #
# 10. one-shot evaluation + CLI
# --------------------------------------------------------------------------- #
def evaluate_all(real_df: pd.DataFrame, gen_df: pd.DataFrame, cfg: EvalConfig,
                 train_df: Optional[pd.DataFrame] = None,
                 real_weights=None, gen_weights=None,
                 coverage_k: Sequence[int] = (2, 3),
                 coverage_max_subsets: Optional[int] = 200,
                 validity_rules: Optional[Sequence] = None,
                 validity_vocab: Optional[Dict[str, Any]] = None,
                 with_oracles: bool = False,
                 rng: Optional[np.random.Generator] = None,
                 with_per_j: bool = False) -> Dict[str, Any]:
    rng = rng or np.random.default_rng(42)
    t0 = time.time()
    er = cfg.encode(real_df, weights=real_weights)
    eg = cfg.encode(gen_df, weights=gen_weights)
    et = None if train_df is None else cfg.encode(train_df)
    t_enc = time.time() - t0

    out = core_metrics(er, eg, cfg, train_df=et, coverage_k=coverage_k,
                       coverage_max_subsets=coverage_max_subsets, rng=rng,
                       with_per_j=with_per_j)
    out["config_name"] = cfg.name
    out["weighted_real"] = er.weights is not None
    out["weighted_gen"] = eg.weights is not None
    out["oos_row_share_gen"] = float(1.0 - eg.valid.all(axis=1).mean())
    out["oos_row_share_real"] = float(1.0 - er.valid.all(axis=1).mean())
    out["oos_share_by_col_gen"] = {c: float(1.0 - eg.valid[:, j].mean())
                                   for j, c in enumerate(eg.cols)}
    out["timing_s"]["encode"] = t_enc

    t = time.time()
    # real_df's rate is the ACHIEVABLE CEILING, not 1.0: three PUMS rows in the CA
    # reference carry a blank Moved_in inside an otherwise-complete housing unit, so the
    # co-missing rule holds at 0.999991 on real people.  Models are read against that.
    _rules = validity_rules if validity_rules is not None else universe_rules_for(cfg)
    out["record_validity_gen"] = record_validity_rate(gen_df, _rules, validity_vocab)
    out["record_validity_real"] = record_validity_rate(real_df, _rules, validity_vocab)
    out["timing_s"]["record_validity"] = time.time() - t

    if with_oracles and train_df is not None:
        t = time.time()
        out["oracles"] = oracle_rows(er, et, cfg, rng=rng, coverage_k=coverage_k,
                                     coverage_max_subsets=coverage_max_subsets)
        out["timing_s"]["oracles"] = time.time() - t
    out["timing_s"]["total"] = time.time() - t0
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SemaPop eval protocol v2 (Eq. 11 SRMSE, "
                                            "block P/R, coverage, validity)")
    p.add_argument("--real", required=True, help="reference population csv")
    p.add_argument("--gen", required=True, help="generated population csv")
    p.add_argument("--config", required=True, help="eval config yaml")
    p.add_argument("--out", default="metrics.json")
    p.add_argument("--train", default=None, help="training csv (support decomposition + oracles)")
    p.add_argument("--real-weight-col", default=None)
    p.add_argument("--gen-weight-col", default=None)
    p.add_argument("--coverage-k", default="2,3")
    p.add_argument("--coverage-max-subsets", type=int, default=200)
    p.add_argument("--oracles", action="store_true")
    p.add_argument("--per-j-csv", default=None, help="prefix for per-attribute-set csv dumps")
    p.add_argument("--fit-config-out", default=None,
                   help="if the config has no stored categories, fit on --real and save here")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> Dict[str, Any]:
    args = parse_args(argv)
    cfg = EvalConfig.from_yaml(args.config)
    real = pd.read_csv(args.real)
    gen = pd.read_csv(args.gen)
    train = pd.read_csv(args.train) if args.train else None
    if not cfg.is_fitted:
        cfg.fit(real)
        if args.fit_config_out:
            cfg.to_yaml(args.fit_config_out)
            print(f"[eval_v2] fitted category lists written to {args.fit_config_out}")
        else:
            print("[eval_v2] WARNING: config had no stored categories; fitted on --real. "
                  "Pass --fit-config-out to freeze them so N_j is identical across models.")
    ks = tuple(int(x) for x in str(args.coverage_k).split(",") if x.strip())
    res = evaluate_all(
        real, gen, cfg, train_df=train,
        real_weights=args.real_weight_col, gen_weights=args.gen_weight_col,
        coverage_k=ks, coverage_max_subsets=args.coverage_max_subsets,
        with_oracles=args.oracles, rng=np.random.default_rng(args.seed),
        with_per_j=bool(args.per_j_csv),
    )
    if args.per_j_csv:
        for key in ("srmse1_per_j", "srmse2_per_j"):
            df = res.pop(key, None)
            if isinstance(df, pd.DataFrame):
                path = f"{args.per_j_csv}_{key}.csv"
                df.to_csv(path, index=False)
                print(f"[eval_v2] wrote {path}")
        for k, c in res.get("coverage", {}).items():
            df = c.pop("per_j", None)
            if isinstance(df, pd.DataFrame):
                path = f"{args.per_j_csv}_coverage_{k}.csv"
                df.to_csv(path, index=False)
                print(f"[eval_v2] wrote {path}")
    with open(args.out, "w") as f:
        json.dump(_jsonable(res), f, indent=2)
    print(f"[eval_v2] SRMSE-1 {res['srmse1']:.6f}  SRMSE-2 {res['srmse2']:.6f}  "
          f"all-tuple F1 {res['blocks']['all']['f1']:.4f}  -> {args.out}")
    return res


if __name__ == "__main__":
    main()


def tail_metrics(history, k: int = 5, prefix: str = "val_"):
    """Mean of the last ``k`` in-training evaluations, as a metrics dict.

    This, not the single final evaluation, is what HPO should tune on.  Measured
    run-to-run sigma on val SRMSE-1 for the SemaPop trainer: 9.3% for the last
    single evaluation, 5.2% for the mean of the last five, at zero extra cost.

    It must be emitted by EVERY iterative trainer, not just ours.  When only the
    SemaPop trainer produced it, `hpo/objective.py` fell back to
    ``final_val_metrics`` for the baselines -- so SemaPop was tuned on a metric
    with 5.2% noise while the baselines were tuned on one with 9.3%, and a
    cleaner objective is a real advantage to TPE, i.e. an unfair asymmetry.

    Returns ``{}`` when the history holds no evaluation records, which is the
    correct answer for a one-shot fitter such as the BN.
    """
    import numpy as _np
    recs = [h for h in history if any(str(key).startswith(prefix) for key in h)]
    recs = recs[-max(int(k), 1):]
    if not recs:
        return {}
    keys = sorted({str(key)[len(prefix):] for h in recs for key, val in h.items()
                   if str(key).startswith(prefix) and isinstance(val, (int, float))})
    out = {key: float(_np.mean([h[prefix + key] for h in recs if prefix + key in h]))
           for key in keys}
    out["n_evals_averaged"] = len(recs)
    return out
