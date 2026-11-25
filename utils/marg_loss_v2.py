"""
utils/marg_loss_v2.py
=====================
v2 of the 1-way marginal regulariser used by SemaPop generators.

This module replaces ``utils/MARG_LOSS.py`` (kept unchanged as the historical
record).  The three defects fixed here are:

F3(a)  ``MARG_LOSS.soft_histogram_probs`` was decorated with ``@torch.no_grad``,
       so the continuous marginal term contributed *no gradient at all*.  Here
       every op on the generated tensor is differentiable and unit-tested.

F3(b)  Bins were built on z-scored values with quantile-trimmed linspace, so
       minute-of-day columns collapsed to ~1 bin and Age to ~5 bins.  Here the
       binning spec lives in the **original** scale; the generator output is
       mapped back with the scaler's ``mean``/``std`` used as *differentiable
       affine constants* (``x_orig = x_scaled * std + mean``).

F3(c)  The categorical term normalised **raw generator logits** (epoch-1
       marg = 8.5e10 in the old logs).  The fix is *not* to put a softmax inside
       the loss: the critic would still score raw blocks against one-hot real
       rows, and softmax of an already-one-hot vector with K=7 peaks at 0.312,
       so the loss could never reach 0.  Instead **the generator must emit a
       probability simplex per categorical block** (per-block softmax with
       temperature, or straight-through Gumbel-softmax as in CTGAN) so that the
       critic, this loss and the sampler all consume the *same* object.  This
       module therefore assumes ``x_fake[:, s:e]`` is already a simplex and only
       renormalises defensively.

Irreducible-floor note
----------------------
A soft histogram of the *generated* values compared against a *hard* histogram
of the real values has a floor (~0.044 at bw = 0.7 x bin step) that no generator
can beat.  Default here is ``soft_target=True``: the target is built with the
**same** soft kernel on the real data, and the default bandwidth is
0.35 x bin step, so a perfect generator reaches exactly 0.  ``soft_target=False``
is kept only for diagnosing that floor.

SRMSE convention
----------------
``srmse_from_probs`` implements Eq. 11 of the paper,
``SRMSE = sqrt(N_j * sum_k (p_hat_k - p_k)^2)`` with ``N_j`` = number of cells of
the table, matching ``utils/eval_v2.py``.  ``mode="l2"`` reproduces the old
``sqrt(sum d^2)`` (no sqrt(N_j)) if a run needs to be compared with the
historical loss scale.

The ``BinningSpec`` produced here is JSON-serialisable and is meant to be shared
with ``utils/eval_v2.py`` and stored in every checkpoint.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

#: sensible per-column minute-bin widths; durations get finer bins than
#: minute-of-day columns.  Override via ``time_step_overrides``.
DEFAULT_TIME_STEPS: Dict[str, int] = {
    "Commute_Travel_Time_Minutes": 5,
    "Commute_Arrival_Time_Minutes": 15,
}

__all__ = [
    "DEFAULT_TIME_STEPS",
    "ColumnBins",
    "BinningSpec",
    "build_cat_slices",
    "srmse_from_probs",
    "soft_hist_probs",
    "is_time_col",
    "build_column_bins",
    "build_binning_spec",
    "cat_target_probs_from_onehot",
    "MarginalLossV2",
    "build_marginal_loss",
]


# --------------------------------------------------------------------------- #
# slices
# --------------------------------------------------------------------------- #
def build_cat_slices(
    num_dim: int,
    cat_cols: Sequence[str],
    cat_dims: Sequence[int],
) -> Dict[str, Tuple[int, int]]:
    """Column ranges of each one-hot block inside ``[num | cat1 | cat2 | ...]``."""
    if len(cat_cols) != len(cat_dims):
        raise ValueError("cat_cols and cat_dims length mismatch")
    out: Dict[str, Tuple[int, int]] = {}
    s = int(num_dim)
    for name, k in zip(cat_cols, cat_dims):
        out[name] = (s, s + int(k))
        s += int(k)
    return out


# --------------------------------------------------------------------------- #
# SRMSE on probability vectors
# --------------------------------------------------------------------------- #
def srmse_from_probs(
    p_hat: torch.Tensor,
    p_real: torch.Tensor,
    mode: str = "eq11",
    eps: float = 1e-12,
) -> torch.Tensor:
    """SRMSE between two probability vectors.

    mode="eq11": sqrt(N_j * sum_k (p_hat_k - p_k)^2)   (paper Eq. 11)
    mode="l2":   sqrt(sum_k (p_hat_k - p_k)^2)         (historical MARG_LOSS)
    """
    p_hat = p_hat.reshape(-1).float()
    p_real = p_real.reshape(-1).float().to(p_hat.device)
    if p_hat.numel() != p_real.numel():
        raise ValueError(f"prob length mismatch: {p_hat.numel()} vs {p_real.numel()}")
    d = p_hat - p_real
    sq = torch.sum(d * d)
    if mode == "eq11":
        sq = sq * float(p_hat.numel())
    elif mode != "l2":
        raise ValueError(f"unknown srmse mode {mode!r}")
    # +eps keeps sqrt finite; it also makes the gradient 0 exactly at the optimum
    # (unavoidable for any loss whose minimum value is 0).
    return torch.sqrt(sq + eps)


# --------------------------------------------------------------------------- #
# differentiable soft histogram  (F3a)
# --------------------------------------------------------------------------- #
def soft_hist_probs(
    x: torch.Tensor,
    centers: torch.Tensor,
    bw: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Differentiable soft histogram of ``x`` over ``centers``.

    Each sample is softly assigned to bins with a Gaussian kernel of bandwidth
    ``bw`` and its weights are normalised to sum to 1 (so every sample
    contributes exactly one unit of mass); the batch mean is the bin
    distribution.  **No** ``torch.no_grad`` anywhere -- this is the F3(a) fix.
    """
    x = x.reshape(-1).float()
    centers = centers.reshape(-1).float().to(x.device)
    bw = max(float(bw), 1e-6)

    z = (x[:, None] - centers[None, :]) / bw          # [B, K]
    logw = -0.5 * z * z
    logw = logw - torch.logsumexp(logw, dim=1, keepdim=True)
    w = torch.exp(logw)                                # [B, K], rows sum to 1
    p = w.mean(dim=0)
    p = p / p.sum().clamp_min(eps)
    return p


def hard_hist_probs(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Nearest-center hard histogram (diagnostics / eval only, non-differentiable)."""
    with torch.no_grad():
        x = x.reshape(-1).float()
        centers = centers.reshape(-1).float().to(x.device)
        idx = torch.argmin((x[:, None] - centers[None, :]).abs(), dim=1)
        counts = torch.bincount(idx, minlength=centers.numel()).float()
        return counts / counts.sum().clamp_min(1.0)


# --------------------------------------------------------------------------- #
# binning spec (original scale)  (F3b)
# --------------------------------------------------------------------------- #
_TIME_RE = re.compile(r"(time|minutes_of_day|minute_of_day)", re.IGNORECASE)


def is_time_col(name: str) -> bool:
    """True for minute-of-day / duration-in-minutes columns."""
    return bool(_TIME_RE.search(name))


@dataclass
class ColumnBins:
    """Bins for one numeric column, in the **original** (un-scaled) data units."""

    name: str
    kind: str                    # "time" | "integer" | "integer_tail" | "linspace"
    centers: np.ndarray          # [K] float64, original units
    bw: float                    # kernel bandwidth, original units
    step: float                  # nominal bin step, original units
    support_lo: float            # observed min on train (used for F6 clipping)
    support_hi: float            # observed max on train
    tail_k: Optional[int] = None # for kind="integer_tail": last exact integer

    @property
    def n_bins(self) -> int:
        return int(self.centers.shape[0])

    @property
    def edges(self) -> np.ndarray:
        """Bin edges = midpoints between centers, extended by half a step."""
        c = np.asarray(self.centers, dtype=np.float64)
        if c.size == 1:
            return np.array([c[0] - 0.5 * self.step, c[0] + 0.5 * self.step])
        mids = 0.5 * (c[:-1] + c[1:])
        return np.concatenate([[c[0] - 0.5 * self.step], mids, [c[-1] + 0.5 * self.step]])

    def digitize(self, values: np.ndarray) -> np.ndarray:
        """Hard bin index of ``values`` (nearest center). For eval_v2 / diagnostics."""
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        c = np.asarray(self.centers, dtype=np.float64)
        return np.abs(v[:, None] - c[None, :]).argmin(axis=1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "centers": [float(v) for v in np.asarray(self.centers).ravel()],
            "bw": float(self.bw),
            "step": float(self.step),
            "support_lo": float(self.support_lo),
            "support_hi": float(self.support_hi),
            "tail_k": None if self.tail_k is None else int(self.tail_k),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ColumnBins":
        return ColumnBins(
            name=str(d["name"]),
            kind=str(d["kind"]),
            centers=np.asarray(d["centers"], dtype=np.float64),
            bw=float(d["bw"]),
            step=float(d["step"]),
            support_lo=float(d["support_lo"]),
            support_hi=float(d["support_hi"]),
            tail_k=None if d.get("tail_k") is None else int(d["tail_k"]),
        )


@dataclass
class BinningSpec:
    """Per-column bins for all numeric columns; shared with ``utils/eval_v2.py``."""

    columns: Dict[str, ColumnBins] = field(default_factory=dict)

    def __contains__(self, k: str) -> bool:
        return k in self.columns

    def __getitem__(self, k: str) -> ColumnBins:
        return self.columns[k]

    def names(self) -> List[str]:
        return list(self.columns.keys())

    def support(self, name: str) -> Tuple[float, float]:
        cb = self.columns[name]
        return float(cb.support_lo), float(cb.support_hi)

    def support_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {n: self.support(n) for n in self.columns}

    def to_dict(self) -> Dict[str, Any]:
        return {"columns": {n: cb.to_dict() for n, cb in self.columns.items()}}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "BinningSpec":
        return BinningSpec(
            columns={n: ColumnBins.from_dict(v) for n, v in d["columns"].items()}
        )


def _looks_integer(x: np.ndarray, rtol: float = 1e-4) -> bool:
    """True if every value is (numerically) an integer.

    The tolerance is scaled by the magnitude of the column: values reach this
    function after a float32 round-trip through the StandardScaler
    (``x_scaled * std + mean``), so an Age of 95 carries ~1e-5 of absolute
    error.  A fixed 1e-6 tolerance would misclassify every integer column as
    continuous and silently fall back to 32 linspace bins.
    """
    if x.size == 0:
        return False
    scale = max(1.0, float(np.max(np.abs(x))))
    return bool(np.max(np.abs(x - np.round(x))) < rtol * scale)


def build_column_bins(
    x_orig: np.ndarray,
    name: str,
    *,
    num_bins: int = 32,
    q_low: float = 0.001,
    q_high: float = 0.999,
    bw_scale: float = 0.35,
    time_bin_step_min: int = 15,
    max_integer_bins: int = 120,
    tail_k: Optional[int] = None,
    time_cols: Optional[Sequence[str]] = None,
) -> ColumnBins:
    """Build bins for one numeric column from **original-scale** training values.

    ``time_cols``: explicit set of minute-of-day columns.  When ``None`` the legacy
    name heuristic (:func:`is_time_col`) is used for backward compatibility -- that
    heuristic also matches e.g. ``Commute_Time_Class``, so v2 callers should pass an
    explicit (possibly empty) set.
    """
    x = np.asarray(x_orig, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        centers = np.linspace(0.0, 1.0, num=num_bins)
        step = float(centers[1] - centers[0]) if num_bins > 1 else 1.0
        return ColumnBins(name, "linspace", centers, max(1e-6, bw_scale * step),
                          step, 0.0, 1.0)

    obs_lo, obs_hi = float(x.min()), float(x.max())

    # ---- time columns: fixed-width minute bins -----------------------------
    if (time_cols is not None and name in time_cols) or (time_cols is None and is_time_col(name)):
        step = float(time_bin_step_min)
        t_min = float(math.floor(obs_lo / step) * step)
        t_max = float(math.ceil(obs_hi / step) * step)
        if t_max <= t_min:
            t_max = t_min + step
        n = int(round((t_max - t_min) / step))
        centers = t_min + (np.arange(n, dtype=np.float64) + 0.5) * step
        return ColumnBins(name, "time", centers, max(1e-6, bw_scale * step),
                          step, obs_lo, obs_hi)

    # ---- integer domain ----------------------------------------------------
    if _looks_integer(x):
        lo_i, hi_i = int(round(obs_lo)), int(round(obs_hi))
        # snap the stored support to the exact integers: the values arrive here
        # after a float32 round-trip, and a support_hi of 93.99999965 would make
        # "clip then round then clip" return 93 for a generated 94.
        obs_lo, obs_hi = float(lo_i), float(hi_i)
        if tail_k is not None and lo_i >= 0:
            k = int(max(lo_i, min(int(tail_k), hi_i)))
            centers = np.concatenate([
                np.arange(lo_i, k + 1, dtype=np.float64),
                np.array([float(k + 1)]),   # "k+" merged tail bin
            ])
            return ColumnBins(name, "integer_tail", centers, max(1e-6, bw_scale * 1.0),
                              1.0, obs_lo, obs_hi, tail_k=k)
        n_int = hi_i - lo_i + 1
        if 0 < n_int <= max_integer_bins:
            centers = np.arange(lo_i, hi_i + 1, dtype=np.float64)
            return ColumnBins(name, "integer", centers, max(1e-6, bw_scale * 1.0),
                              1.0, obs_lo, obs_hi)

    # ---- fallback: quantile-trimmed linspace -------------------------------
    lo_f = float(np.quantile(x, q_low))
    hi_f = float(np.quantile(x, q_high))
    if hi_f - lo_f < 1e-9:
        lo_f, hi_f = lo_f - 0.5, hi_f + 0.5
    centers = np.linspace(lo_f, hi_f, num=num_bins)
    step = float(centers[1] - centers[0]) if num_bins > 1 else 1.0
    return ColumnBins(name, "linspace", centers, max(1e-6, bw_scale * step),
                      step, obs_lo, obs_hi)


def build_binning_spec(
    x_num_orig: np.ndarray,
    num_cols: Sequence[str],
    *,
    num_bins: int = 32,
    q_low: float = 0.001,
    q_high: float = 0.999,
    bw_scale: float = 0.35,
    time_bin_step_min: int = 15,
    time_step_overrides: Optional[Dict[str, int]] = None,
    max_integer_bins: int = 120,
    tail_merge: Optional[Dict[str, int]] = None,
    time_cols: Optional[Sequence[str]] = None,
) -> BinningSpec:
    """Build a :class:`BinningSpec` from an ``[N, M]`` array of ORIGINAL-scale values."""
    x_num_orig = np.asarray(x_num_orig)
    if x_num_orig.ndim != 2 or x_num_orig.shape[1] != len(num_cols):
        raise ValueError(
            f"x_num_orig shape {x_num_orig.shape} incompatible with {len(num_cols)} num_cols"
        )
    tail_merge = dict(tail_merge or {})
    time_step_overrides = dict(time_step_overrides or {})
    cols: Dict[str, ColumnBins] = {}
    for i, name in enumerate(num_cols):
        cols[name] = build_column_bins(
            x_num_orig[:, i],
            name,
            num_bins=num_bins,
            q_low=q_low,
            q_high=q_high,
            bw_scale=bw_scale,
            time_bin_step_min=int(time_step_overrides.get(name, time_bin_step_min)),
            max_integer_bins=max_integer_bins,
            tail_k=tail_merge.get(name),
            time_cols=time_cols,
        )
    return BinningSpec(columns=cols)


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #
@torch.no_grad()
def cat_target_probs_from_onehot(
    x_real: torch.Tensor,
    cat_slices: Dict[str, Tuple[int, int]],
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """Real 1-way categorical marginals from the one-hot part of the real matrix."""
    X = x_real.float()
    out: Dict[str, torch.Tensor] = {}
    for name, (s, e) in cat_slices.items():
        block = X[:, s:e]
        p = block.mean(dim=0)
        out[name] = p / p.sum().clamp_min(eps)
    return out


@torch.no_grad()
def cont_target_probs(
    x_num_orig: torch.Tensor,
    num_cols: Sequence[str],
    spec: BinningSpec,
    soft_target: bool = True,
) -> Dict[str, torch.Tensor]:
    """Real 1-way numeric marginals, built with the SAME soft kernel by default.

    ``soft_target=True`` is what removes the irreducible floor: a perfect
    generator (identical values) then reaches exactly 0.
    """
    out: Dict[str, torch.Tensor] = {}
    dev = x_num_orig.device
    for i, name in enumerate(num_cols):
        cb = spec[name]
        centers = torch.as_tensor(cb.centers, dtype=torch.float32, device=dev)
        x = x_num_orig[:, i].float()
        out[name] = (
            soft_hist_probs(x, centers, cb.bw) if soft_target
            else hard_hist_probs(x, centers)
        )
    return out


# --------------------------------------------------------------------------- #
# the loss module
# --------------------------------------------------------------------------- #
def _safe(name: str) -> str:
    return re.sub(r"\W", "_", name)


class MarginalLossV2(nn.Module):
    """1-way marginal SRMSE regulariser (categorical + numeric).

    Expected input ``x_fake`` of shape ``[B, D]`` laid out as
    ``[num_1..num_M | cat_1 one-hot/simplex | cat_2 | ...]`` where

    * the numeric part is in the **scaled** space of the training
      ``StandardScaler`` (the module un-scales it with ``mean``/``std`` held as
      constant buffers), and
    * every categorical block is a **probability simplex** (F3c).  Pass
      ``strict_simplex=True`` to hard-fail when a block is not one.
    """

    def __init__(
        self,
        num_cols: Sequence[str],
        cat_cols: Sequence[str],
        cat_slices: Dict[str, Tuple[int, int]],
        num_mean: np.ndarray,
        num_std: np.ndarray,
        spec: BinningSpec,
        cat_target: Dict[str, torch.Tensor],
        cont_target: Dict[str, torch.Tensor],
        *,
        srmse_mode: str = "eq11",
        num_weights: Optional[Dict[str, float]] = None,
        cat_weights: Optional[Dict[str, float]] = None,
        reduce: str = "mean",
        strict_simplex: bool = False,
        simplex_tol: float = 1e-3,
    ) -> None:
        super().__init__()
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.cat_slices = {k: (int(v[0]), int(v[1])) for k, v in cat_slices.items()}
        self.spec = spec
        self.srmse_mode = str(srmse_mode)
        self.reduce = str(reduce)
        self.strict_simplex = bool(strict_simplex)
        self.simplex_tol = float(simplex_tol)
        self.num_weights = dict(num_weights or {})
        self.cat_weights = dict(cat_weights or {})

        mean = np.asarray(num_mean, dtype=np.float32).reshape(-1)
        std = np.asarray(num_std, dtype=np.float32).reshape(-1)
        if mean.size != len(self.num_cols) or std.size != len(self.num_cols):
            raise ValueError("num_mean/num_std must have one entry per numeric column")
        std = np.where(np.abs(std) < 1e-8, 1.0, std).astype(np.float32)
        # constant differentiable affine: x_orig = x_scaled * std + mean
        self.register_buffer("num_mean", torch.from_numpy(mean))
        self.register_buffer("num_std", torch.from_numpy(std))

        for i, name in enumerate(self.num_cols):
            cb = spec[name]
            self.register_buffer(
                f"centers_{i}", torch.as_tensor(cb.centers, dtype=torch.float32)
            )
            self.register_buffer(
                f"ctgt_{i}", cont_target[name].detach().float().cpu().clone()
            )
        self._bw = [float(spec[n].bw) for n in self.num_cols]

        for j, name in enumerate(self.cat_cols):
            self.register_buffer(
                f"kt_{j}", cat_target[name].detach().float().cpu().clone()
            )

    # ------------------------------------------------------------------ #
    def to_original(self, x_num_scaled: torch.Tensor) -> torch.Tensor:
        """Differentiable inverse of the StandardScaler (mean/std are constants)."""
        return x_num_scaled * self.num_std.to(x_num_scaled.device) + self.num_mean.to(
            x_num_scaled.device
        )

    def categorical_term(self, x_fake: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        per_col: Dict[str, float] = {}
        losses = []
        for j, name in enumerate(self.cat_cols):
            s, e = self.cat_slices[name]
            block = x_fake[:, s:e].float()
            if block.numel() == 0:
                continue
            if self.strict_simplex:
                with torch.no_grad():
                    row_sum = block.sum(dim=1)
                    ok = bool(
                        torch.all(block >= -self.simplex_tol)
                        and torch.all((row_sum - 1.0).abs() < self.simplex_tol)
                    )
                if not ok:
                    raise ValueError(
                        f"categorical block '{name}' is not a probability simplex; the "
                        "generator must apply a per-block softmax / Gumbel-softmax (F3c)"
                    )
            # defensive renormalisation only (blocks are already simplices)
            p_hat = block.mean(dim=0)
            p_hat = p_hat / p_hat.sum().clamp_min(1e-12)
            p_real = getattr(self, f"kt_{j}").to(x_fake.device)
            lv = srmse_from_probs(p_hat, p_real, mode=self.srmse_mode)
            w = float(self.cat_weights.get(name, 1.0))
            per_col[name] = float(lv.detach().item())
            losses.append(lv * w)
        if not losses:
            return x_fake.sum() * 0.0, per_col
        stacked = torch.stack(losses)
        return (stacked.mean() if self.reduce == "mean" else stacked.sum()), per_col

    def continuous_term(self, x_fake: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        per_col: Dict[str, float] = {}
        losses = []
        if len(self.num_cols) == 0:
            return x_fake.sum() * 0.0, per_col
        x_orig = self.to_original(x_fake[:, : len(self.num_cols)])
        for i, name in enumerate(self.num_cols):
            centers = getattr(self, f"centers_{i}").to(x_fake.device)
            p_hat = soft_hist_probs(x_orig[:, i], centers, self._bw[i])
            p_real = getattr(self, f"ctgt_{i}").to(x_fake.device)
            lv = srmse_from_probs(p_hat, p_real, mode=self.srmse_mode)
            w = float(self.num_weights.get(name, 1.0))
            per_col[name] = float(lv.detach().item())
            losses.append(lv * w)
        stacked = torch.stack(losses)
        return (stacked.mean() if self.reduce == "mean" else stacked.sum()), per_col

    def forward(self, x_fake: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Returns ``(loss_cat, loss_cont, info)``. Both terms carry gradient."""
        loss_cat, per_cat = self.categorical_term(x_fake)
        loss_cont, per_cont = self.continuous_term(x_fake)
        info = {
            "cat": float(loss_cat.detach().item()),
            "cont": float(loss_cont.detach().item()),
            "total": float((loss_cat + loss_cont).detach().item()),
            "per_col_cat": per_cat,
            "per_col_num": per_cont,
        }
        return loss_cat, loss_cont, info


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #
def build_marginal_loss(
    x_train_scaled: np.ndarray,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    cat_dims: Sequence[int],
    num_mean: np.ndarray,
    num_std: np.ndarray,
    *,
    num_bins: int = 32,
    bw_scale: float = 0.35,
    time_bin_step_min: int = 15,
    time_step_overrides: Optional[Dict[str, int]] = None,
    max_integer_bins: int = 120,
    tail_merge: Optional[Dict[str, int]] = None,
    soft_target: bool = True,
    srmse_mode: str = "eq11",
    num_weights: Optional[Dict[str, float]] = None,
    cat_weights: Optional[Dict[str, float]] = None,
    strict_simplex: bool = False,
    max_target_rows: Optional[int] = 200_000,
    seed: int = 0,
) -> Tuple[MarginalLossV2, BinningSpec]:
    """Build the loss module from the **training** processed matrix.

    ``x_train_scaled`` is ``[N, D]`` = ``[num (z-scored) | one-hot cats]``.
    ``num_mean``/``num_std`` come from the fitted ``StandardScaler``.
    """
    X = np.asarray(x_train_scaled, dtype=np.float32)
    num_dim = len(num_cols)
    slices = build_cat_slices(num_dim, cat_cols, cat_dims)
    if X.shape[1] != num_dim + int(sum(cat_dims)):
        raise ValueError(
            f"x_train_scaled has {X.shape[1]} cols, expected {num_dim + int(sum(cat_dims))}"
        )

    mean = np.asarray(num_mean, dtype=np.float64).reshape(-1)
    std = np.asarray(num_std, dtype=np.float64).reshape(-1)
    std = np.where(np.abs(std) < 1e-8, 1.0, std)
    x_num_orig = X[:, :num_dim].astype(np.float64) * std[None, :] + mean[None, :]

    spec = build_binning_spec(
        x_num_orig,
        num_cols,
        num_bins=num_bins,
        bw_scale=bw_scale,
        time_bin_step_min=time_bin_step_min,
        time_step_overrides=time_step_overrides,
        max_integer_bins=max_integer_bins,
        tail_merge=tail_merge,
    )

    # targets are computed on (a subsample of) the full training data
    if max_target_rows is not None and X.shape[0] > max_target_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_target_rows, replace=False)
        Xt = X[idx]
        xnum_t = x_num_orig[idx]
    else:
        Xt, xnum_t = X, x_num_orig

    t_all = torch.from_numpy(Xt)
    cat_tgt = cat_target_probs_from_onehot(t_all, slices)
    cont_tgt = cont_target_probs(
        torch.from_numpy(xnum_t.astype(np.float32)), num_cols, spec, soft_target=soft_target
    )

    loss = MarginalLossV2(
        num_cols=num_cols,
        cat_cols=cat_cols,
        cat_slices=slices,
        num_mean=mean,
        num_std=std,
        spec=spec,
        cat_target=cat_tgt,
        cont_target=cont_tgt,
        srmse_mode=srmse_mode,
        num_weights=num_weights,
        cat_weights=cat_weights,
        strict_simplex=strict_simplex,
    )
    return loss, spec
