#!/usr/bin/env python
"""
train_ctgan_v2.py
=================
Faithful **CTGAN baseline** for the v2 PUMS data.

Why this file exists
--------------------
This trainer replaces an earlier in-house implementation (``train_ctgan.py`` +
``model/ctgan.py``, not shipped), which was *not* CTGAN.  Verified defects:

(a) **no mode-specific (VGM) normalisation** -- numeric columns went through a
    plain ``StandardScaler``, so the "CT" of CTGAN was missing entirely;
(b) **training-by-sampling is broken** -- the conditional vector was drawn from
    ``CondSampler`` while the real minibatch came from an *independent* shuffled
    DataLoader (``train_ctgan.py:109-124``).  Real rows therefore did not match
    the sampled condition, which is exactly the mechanism CTGAN uses to handle
    imbalanced categoricals;
(c) **no clipping before rounding** -- numeric outputs were inverse transformed
    and ``.round().astype(int)`` with no clamp to the training support, so 77.5%
    of generated rows carried at least one value never seen in the reference
    (precision 3.8%).

This trainer throws all of that away and calls the **reference implementation**
(``ctgan.CTGAN`` from the ``ctgan`` package, the code released with the CTGAN
paper and vendored by SDV), wrapped in the same CLI / seeding / checkpoint /
``metrics.json`` conventions as ``train_semapop_gan_v2.py``:

* F2  the validation split is **held out** -- the synthesiser is fit on train
      only; val is used exclusively for model selection.
* F5  ``--seed`` seeds python/numpy/torch/cuda/cudnn, ``--sample-seed`` is
      separate, the checkpoint stores the full effective config *and* the fitted
      ``DataTransformer`` (the VGM + one-hot preprocessor), every run writes
      ``metrics.json``.
* F6  numeric outputs are **clipped to the training support, then rounded**
      (integer columns only), and the categorical outputs are asserted to lie in
      the training vocabulary.  ``metrics.json`` records the out-of-support rate
      *before* clipping as evidence.
* model selection on the **val** split with the frozen ``utils.eval_v2``
      protocol (SRMSE-1/2 + block P/R), never on the reference.

Model selection without forking the library
-------------------------------------------
``ctgan.CTGAN.fit`` is a single monolithic method with no per-epoch hook and it
re-initialises the transformer/generator on every call, so it cannot be resumed.
Instead of copying the training loop (which would defeat the purpose of using
the reference implementation) we temporarily replace the module-global ``tqdm``
symbol that the fit loop iterates over -- ``for i in tqdm(range(epochs), ...)``
-- with a thin generator that fires a callback after each epoch (see
:class:`_EpochProbe`).  The library's optimisation code runs **verbatim**; the
callback only samples, scores on val and snapshots the generator state, with the
RNG state saved/restored around it so that the training trajectory is identical
to a plain ``fit`` regardless of ``--eval-every``.

Conditional sampling (for E5)
-----------------------------
Be precise about what the library actually supports:

* ``sdv.single_table.CTGANSynthesizer._sample`` raises
  ``NotImplementedError("CTGANSynthesizer doesn't support conditional
  sampling.")``; SDV's ``sample_from_conditions`` then degrades to
  unconditional sampling + rejection.
* raw ``ctgan.CTGAN.sample(n, condition_column=, condition_value=)`` supports
  **one** discrete condition column, and even then it only *biases* the
  conditional vector -- it does not guarantee the constraint holds.

:meth:`CTGANBaseline.sample_conditional` therefore uses the native
single-column condition vector when the constraint is a single discrete column
with a single value, and plain rejection sampling otherwise; **either way** the
returned rows are filtered to actually satisfy the constraints, and the metadata
says which path was taken (``path`` = ``native_condvec+rejection`` or
``rejection_only``) together with the observed yield.

Example
-------
    python training/train_ctgan_v2.py \
        --data-dir <semapop_repo>/data/pums_v2/CA \
        --out-root <semapop_repo>/modelSave \
        --tag ctgan_ca --device cuda:0 --epochs 300 --seed 42
"""

from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); import popaudit_paths as _PA  # release bootstrap: repository root first on sys.path (see popaudit_paths.py)

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, _PA.R(""))

REPO_DIR = _PA.R("")
DEFAULT_EVAL_CONFIG = os.path.join(REPO_DIR, "configs", "eval_v3_pums12.yaml")


# --------------------------------------------------------------------------- #
# F5: determinism (same helper as train_semapop_gan_v2.py)
# --------------------------------------------------------------------------- #
def set_torch_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python / numpy / torch / cuda and (optionally) pin cudnn."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def _rng_snapshot() -> Tuple[Any, ...]:
    # NOTE: the cudnn flags are part of the snapshot because set_torch_seed(...,
    # deterministic=False) flips them globally; restoring only the RNG streams
    # would leave cudnn non-deterministic for the REST of training after the
    # first in-training sampling/eval call (invisible on CPU, real on GPU).
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )


def _rng_restore(state: Tuple[Any, ...]) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state[3])
    if len(state) >= 6:
        torch.backends.cudnn.deterministic = state[4]
        torch.backends.cudnn.benchmark = state[5]


@contextmanager
def frozen_rng():
    """Run a block without disturbing the global RNG streams."""
    s = _rng_snapshot()
    try:
        yield
    finally:
        _rng_restore(s)


# --------------------------------------------------------------------------- #
# per-epoch hook into the unmodified library training loop
# --------------------------------------------------------------------------- #
class _EpochProbe:
    """Stand-in for ``tqdm`` inside ``ctgan.synthesizers.ctgan``.

    The library's loop is ``epoch_iterator = tqdm(range(epochs), disable=...)``
    followed by ``for i in epoch_iterator``.  Yielding from a generator lets us
    run ``callback(i)`` *after* epoch ``i`` has finished, without touching a
    single line of the library's optimisation code.
    """

    def __init__(self, callback) -> None:
        self.callback = callback

    def __call__(self, iterable: Iterable, *args, **kwargs):
        return self._iter(iterable)

    def _iter(self, iterable: Iterable):
        for i in iterable:
            yield i
            self.callback(int(i) + 1)          # 1-based epoch number


@contextmanager
def epoch_hook(callback):
    """Temporarily install :class:`_EpochProbe` as the fit loop's ``tqdm``."""
    import ctgan.synthesizers.ctgan as _mod

    original = _mod.tqdm
    _mod.tqdm = _EpochProbe(callback)
    try:
        yield
    finally:
        _mod.tqdm = original


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _read_split(path: str, num_cols: Sequence[str], cat_cols: Sequence[str],
                weight_col: Optional[str], subsample: int = 0,
                seed: int = 42) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    use = list(num_cols) + list(cat_cols)
    missing = [c for c in use if c not in header]
    if missing:
        raise KeyError(f"{path} is missing modelled columns {missing}")
    if weight_col and weight_col in header:
        use = use + [weight_col]
    df = pd.read_csv(path, usecols=use)
    df = df[[c for c in (list(num_cols) + list(cat_cols)) if c in df.columns]
            + ([weight_col] if (weight_col and weight_col in df.columns) else [])]
    for c in cat_cols:
        df[c] = df[c].astype(str)
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="raise").astype(float)
    if subsample and subsample < len(df):
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(df), size=subsample, replace=False))
        df = df.iloc[keep].reset_index(drop=True)
    return df


def numeric_support(df: pd.DataFrame, num_cols: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    """F6: the training support each generated numeric value is clipped into."""
    return {c: (float(df[c].min()), float(df[c].max())) for c in num_cols}


def integer_flags(df: pd.DataFrame, num_cols: Sequence[str]) -> Dict[str, bool]:
    return {c: bool(np.allclose(df[c].values, np.rint(df[c].values))) for c in num_cols}


def cat_vocabulary(df: pd.DataFrame, cat_cols: Sequence[str]) -> Dict[str, List[str]]:
    return {c: sorted(df[c].astype(str).unique().tolist()) for c in cat_cols}


# --------------------------------------------------------------------------- #
# F6: post-processing
# --------------------------------------------------------------------------- #
def postprocess(
    df_raw: pd.DataFrame,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    support: Dict[str, Tuple[float, float]],
    integer_cols: Dict[str, bool],
    vocab: Dict[str, List[str]],
    train_median: Dict[str, float],
    strict_vocab: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clip numeric outputs to the training support, **then** round (F6).

    Returns the cleaned frame and a diagnostics dict holding the out-of-support
    rate *before* clipping (the thing the old implementation never measured) and
    the categorical vocabulary check.
    """
    out = df_raw.copy()
    diag: Dict[str, Any] = {
        "n_rows": int(len(out)),
        "out_of_support_rate_before_clip": {},
        "n_out_of_support_before_clip": {},
        "out_of_support_rate_after_clip": {},
        "n_nan_numeric_before_clip": {},
        "cat_oov_counts": {},
    }
    n = len(out)
    row_bad = np.zeros(n, dtype=bool)

    for c in num_cols:
        lo, hi = support[c]
        v = out[c].to_numpy(dtype=np.float64)
        nan_mask = ~np.isfinite(v)
        diag["n_nan_numeric_before_clip"][c] = int(nan_mask.sum())
        if nan_mask.any():
            # the VGM inverse transform can emit NaN for degenerate components;
            # recorded above so the paper can state it, filled with the training
            # median so the row is still evaluable.
            v = np.where(nan_mask, float(train_median[c]), v)
        bad = (v < lo - 1e-9) | (v > hi + 1e-9)
        diag["n_out_of_support_before_clip"][c] = int(bad.sum())
        diag["out_of_support_rate_before_clip"][c] = float(bad.mean()) if n else float("nan")
        row_bad |= bad

        v = np.clip(v, lo, hi)
        if integer_cols.get(c, False):
            ilo = int(math.ceil(lo - 1e-6))
            ihi = int(math.floor(hi + 1e-6))
            v = np.clip(np.rint(v), ilo, ihi).astype(np.int64)
        out[c] = v

        w = out[c].to_numpy(dtype=np.float64)
        diag["out_of_support_rate_after_clip"][c] = (
            float(np.mean((w < lo - 1e-9) | (w > hi + 1e-9))) if n else float("nan")
        )

    diag["n_rows_out_of_support_before_clip"] = int(row_bad.sum())
    diag["row_out_of_support_rate_before_clip"] = float(row_bad.mean()) if n else float("nan")

    oov_total = 0
    for c in cat_cols:
        out[c] = out[c].astype(str)
        known = set(vocab[c])
        bad = ~out[c].isin(known).to_numpy()
        diag["cat_oov_counts"][c] = int(bad.sum())
        oov_total += int(bad.sum())
    diag["n_cat_oov_total"] = oov_total
    if strict_vocab and oov_total:
        offenders = {c: k for c, k in diag["cat_oov_counts"].items() if k}
        raise AssertionError(
            f"generated categorical values outside the training vocabulary: {offenders}"
        )

    # invariant: after clipping nothing may sit outside the training support
    for c in num_cols:
        assert diag["out_of_support_rate_after_clip"][c] == 0.0, (
            f"post-clip out-of-support rows remain in {c}"
        )
    return out, diag


# --------------------------------------------------------------------------- #
# evaluation helpers (frozen protocol)
# --------------------------------------------------------------------------- #
def load_eval_module():
    import utils.eval_v2 as ev  # noqa: WPS433 -- deliberate local import

    return ev


def load_eval_config(path: Optional[str], ev, fit_df: Optional[pd.DataFrame] = None):
    if not path:
        raise ValueError(
            "--eval-config is required: the paper's numbers must come from the frozen "
            f"protocol ({DEFAULT_EVAL_CONFIG})"
        )
    cfg = ev.EvalConfig.from_yaml(path)
    if not cfg.is_fitted:
        if fit_df is None:
            raise ValueError(f"eval config {path} is not fitted and no reference frame was given")
        cfg.fit(fit_df)
    return cfg


def val_metrics(df_ref: pd.DataFrame, df_gen: pd.DataFrame, cfg, ev,
                use_weights: bool = True, do_2way: bool = True) -> Dict[str, float]:
    """Selection metrics: SRMSE-1/2 + block P/R/F1 on the **val** split."""
    rw = None if use_weights else np.ones(len(df_ref), dtype=np.float64)
    m: Dict[str, float] = {}
    m["srmse_1"] = float(ev.srmse_kway(df_ref, df_gen, cfg, k=1, real_weights=rw)["mean"])
    m["srmse_2"] = (
        float(ev.srmse_kway(df_ref, df_gen, cfg, k=2, real_weights=rw)["mean"])
        if do_2way else float("nan")
    )
    pr = ev.block_precision_recall(df_ref, df_gen, cfg, real_weights=rw)
    f1s = []
    for b, v in pr.items():
        if b == "all" or not isinstance(v, dict) or "f1" not in v:
            continue
        m[f"block_f1_{b}"] = float(v["f1"])
        m[f"block_precision_{b}"] = float(v["precision"])
        m[f"block_recall_{b}"] = float(v["recall"])
        f1s.append(float(v["f1"]))
    if f1s:
        m["block_f1"] = float(np.mean(f1s))
    return m


def selection_score(m: Dict[str, float], f1_weight: float = 1.0,
                    select_on: str = "val_srmse") -> float:
    """Lower is better.  The formula lives in utils.criteria so that this per-epoch
    checkpoint pick and Optuna's between-trial pick read the SAME definition -- a model
    selected on one quantity and ranked on another is not the model that was searched.
    ``select_on`` is one of utils.criteria.CRITERIA: val_srmse (default), if1, ml_eff.
    """
    import utils.criteria as _CR
    return _CR.selection_score(select_on, m, f1_weight)


# --------------------------------------------------------------------------- #
# the baseline itself
# --------------------------------------------------------------------------- #
class CTGANBaseline:
    """Thin wrapper around ``ctgan.CTGAN`` holding everything needed to sample.

    The wrapper owns the post-processing contract (F6) so that *every* sampling
    path -- final sample, val evaluation, conditional sampling -- goes through
    the same clip-then-round + vocabulary assertion.
    """

    def __init__(self, num_cols, cat_cols, support, integer_cols, vocab,
                 train_median, model=None) -> None:
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.support = dict(support)
        self.integer_cols = dict(integer_cols)
        self.vocab = {k: list(v) for k, v in vocab.items()}
        self.train_median = dict(train_median)
        self.model = model

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_train(cls, df_train: pd.DataFrame, num_cols, cat_cols, args) -> "CTGANBaseline":
        from ctgan import CTGAN  # installed reference implementation

        dev = torch.device(args.device)
        enable_gpu = dev.type != "cpu"
        model = CTGAN(
            embedding_dim=args.embedding_dim,
            generator_dim=tuple(args.generator_dim),
            discriminator_dim=tuple(args.discriminator_dim),
            generator_lr=args.generator_lr,
            generator_decay=args.generator_decay,
            discriminator_lr=args.discriminator_lr,
            discriminator_decay=args.discriminator_decay,
            batch_size=args.batch_size,
            discriminator_steps=args.discriminator_steps,
            log_frequency=not args.no_log_frequency,
            verbose=False,           # we do our own per-epoch logging
            epochs=args.epochs,
            pac=args.pac,
            enable_gpu=enable_gpu,
        )
        model.set_device(dev)
        return cls(
            num_cols=num_cols,
            cat_cols=cat_cols,
            support=numeric_support(df_train, num_cols),
            integer_cols=integer_flags(df_train, num_cols),
            vocab=cat_vocabulary(df_train, cat_cols),
            train_median={c: float(df_train[c].median()) for c in num_cols},
            model=model,
        )

    # -- sampling ---------------------------------------------------------- #
    def _sample_raw(self, n: int, condition_column: Optional[str] = None,
                    condition_value: Optional[str] = None) -> pd.DataFrame:
        with torch.no_grad():
            # NOTE: the library never calls ``generator.eval()``; its BatchNorm
            # layers use batch statistics at sampling time.  We keep that
            # behaviour so the baseline is the reference implementation, not a
            # variant of it.
            df = self.model.sample(n, condition_column=condition_column,
                                   condition_value=condition_value)
        return pd.DataFrame(df)[self.num_cols + self.cat_cols]

    def sample(self, n: int, strict_vocab: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Unconditional sample, post-processed (F6)."""
        raw = self._sample_raw(n)
        return postprocess(raw, self.num_cols, self.cat_cols, self.support,
                           self.integer_cols, self.vocab, self.train_median,
                           strict_vocab=strict_vocab)

    # -- E5 conditional sampling ------------------------------------------- #
    def sample_conditional(
        self,
        constraints: Dict[str, Any],
        n: int,
        seed: Optional[int] = None,
        max_rounds: int = 20,
        oversample: float = 2.0,
        max_draw_per_round: int = 200_000,
        probe_size: int = 4096,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Draw ``n`` rows satisfying ``constraints`` = ``{col: value | [values]}``.

        Honest summary of library support (checked against ctgan 0.11.1 /
        sdv 1.32.0):

        * ``sdv.single_table.CTGANSynthesizer._sample`` raises
          ``NotImplementedError`` for conditions, so SDV's conditional API is a
          rejection sampler in disguise -- we never route through it.
        * ``ctgan.CTGAN.sample`` accepts **one** ``condition_column`` /
          ``condition_value`` pair, restricted to a *discrete* column, and only
          biases the conditional vector: rows violating the condition still
          occur.  We therefore always filter afterwards.

        ``meta['path']`` is ``native_condvec+rejection`` when the native
        single-column conditioning was used and ``rejection_only`` otherwise;
        ``meta['reason']`` says why.
        """
        if self.model is None:
            raise RuntimeError("no fitted CTGAN model")
        if not constraints:
            raise ValueError("constraints must be a non-empty {column: value(s)} mapping")
        ev = load_eval_module()

        norm = {c: ([str(x) for x in v] if isinstance(v, (list, tuple, set)) else [str(v)])
                for c, v in constraints.items()}
        for c in norm:
            if c not in self.num_cols + self.cat_cols:
                raise KeyError(f"constraint column {c!r} is not a modelled attribute")

        # values that the training data never contained can never be produced by
        # the one-hot decoder -> refuse to burn sampling rounds on them
        unknown = {c: sorted(set(v) - set(self.vocab[c]))
                   for c, v in norm.items() if c in self.cat_cols}
        unknown = {c: v for c, v in unknown.items() if v}
        impossible = [c for c, v in unknown.items()
                      if not (set(norm[c]) & set(self.vocab[c]))]

        cond_col = cond_val = None
        reason = ""
        if len(norm) != 1:
            reason = (f"{len(norm)} constraint columns; ctgan.CTGAN.sample supports at most one "
                      "condition column")
        else:
            (c, vals), = norm.items()
            if c not in self.cat_cols:
                reason = f"{c!r} is a continuous column; native conditioning requires a discrete one"
            elif len(vals) != 1:
                reason = (f"{len(vals)} values requested for {c!r}; native conditioning takes a "
                          "single category")
            elif vals[0] not in self.vocab[c]:
                reason = f"value {vals[0]!r} is not in the training vocabulary of {c!r}"
            else:
                try:
                    self.model._transformer.convert_column_name_value_to_id(c, vals[0])
                    cond_col, cond_val = c, vals[0]
                    reason = "single discrete column with a single value -> native condition vector"
                except Exception as exc:  # noqa: BLE001
                    reason = (f"ctgan could not build a condition vector for {c}={vals[0]!r} "
                              f"({type(exc).__name__}: {exc})")

        path = "native_condvec+rejection" if cond_col else "rejection_only"
        meta: Dict[str, Any] = {
            "path": path,
            "reason": reason,
            "constraints": {k: list(v) for k, v in norm.items()},
            "n_requested": int(n),
            "implementation": "ctgan.CTGAN.sample",
            "library_supports_multi_column_conditions": False,
            "sdv_ctgansynthesizer_conditional": (
                "NotImplementedError -- SDV falls back to unconditional sampling + rejection"
            ),
            "native_condition_column": cond_col,
            "native_condition_value": cond_val,
            "values_outside_training_vocabulary": unknown,
        }

        if impossible:
            meta.update({
                "status": "unsatisfiable_constraint",
                "rounds": 0, "n_drawn": 0, "n_satisfying": 0, "n_returned": 0,
                "yield": 0.0, "complete": False,
                "reason": (f"no requested value of {impossible} occurs in the training data; "
                           "the decoder cannot emit it, so no sampling was attempted"),
            })
            return pd.DataFrame(columns=self.num_cols + self.cat_cols), meta

        if seed is not None:
            set_torch_seed(seed, deterministic=False)

        # cheap yield probe so the first real round is sized sensibly
        probe = self._sample_raw(min(probe_size, max_draw_per_round), cond_col, cond_val)
        probe, _ = postprocess(probe, self.num_cols, self.cat_cols, self.support,
                               self.integer_cols, self.vocab, self.train_median)
        probe_mask = ev.constraint_mask(probe, {k: set(v) for k, v in norm.items()})
        probe_yield = float(probe_mask.mean()) if len(probe) else 0.0
        meta["probe_n"] = int(len(probe))
        meta["probe_yield"] = probe_yield

        kept: List[pd.DataFrame] = []
        if probe_mask.any():
            kept.append(probe.loc[probe_mask])
        n_kept = int(probe_mask.sum())
        n_drawn = int(len(probe))
        rounds = 0
        rate = max(probe_yield, 1e-4)

        while n_kept < n and rounds < max_rounds:
            need = n - n_kept
            draw = int(min(max_draw_per_round, max(1024, math.ceil(need / rate * oversample))))
            df = self._sample_raw(draw, cond_col, cond_val)
            df, _ = postprocess(df, self.num_cols, self.cat_cols, self.support,
                                self.integer_cols, self.vocab, self.train_median)
            mask = ev.constraint_mask(df, {k: set(v) for k, v in norm.items()})
            if mask.any():
                kept.append(df.loc[mask])
            n_kept += int(mask.sum())
            n_drawn += draw
            rounds += 1
            rate = max(n_kept / max(n_drawn, 1), 1e-4)

        out = (pd.concat(kept, ignore_index=True).iloc[:n].reset_index(drop=True)
               if kept else pd.DataFrame(columns=self.num_cols + self.cat_cols))
        meta.update({
            "rounds": rounds,
            "n_drawn": int(n_drawn),
            "n_satisfying": int(n_kept),
            "n_returned": int(len(out)),
            "yield": float(n_kept / n_drawn) if n_drawn else float("nan"),
            "complete": bool(len(out) >= n),
            "status": "ok" if len(out) >= n else "incomplete",
        })
        return out, meta


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def implementation_record(model) -> Dict[str, Any]:
    """What ``metrics.json`` states in the paper: which CTGAN this really is."""
    import ctgan as _ctgan

    rec: Dict[str, Any] = {
        "package": "ctgan",
        "version": getattr(_ctgan, "__version__", "unknown"),
        "class": "ctgan.synthesizers.ctgan.CTGAN",
        "module_file": getattr(_ctgan, "__file__", None),
        "custom_reimplementation_used": False,
        "note": "reference CTGAN (mode-specific VGM normalisation + training-by-sampling)",
    }
    try:
        import sdv as _sdv

        rec["sdv_version"] = getattr(_sdv, "__version__", "unknown")
        rec["sdv_wrapper_used"] = False
        rec["sdv_note"] = ("sdv.single_table.CTGANSynthesizer wraps this same ctgan.CTGAN; we call "
                           "the raw synthesiser so that single-column conditional sampling is "
                           "available (the SDV wrapper raises NotImplementedError).")
    except Exception:  # noqa: BLE001 -- sdv is optional here
        rec["sdv_version"] = None
    tr = getattr(model, "_transformer", None)
    if tr is not None and getattr(tr, "_column_transform_info_list", None):
        rec["transformer"] = {
            "class": type(tr).__name__,
            "output_dimensions": int(tr.output_dimensions),
            "columns": [
                {
                    "name": info.column_name,
                    "type": info.column_type,
                    "output_dimensions": int(info.output_dimensions),
                    # continuous: 1 scalar + n VGM mode indicators
                    "vgm_modes": (int(info.output_dimensions) - 1
                                  if info.column_type == "continuous" else None),
                }
                for info in tr._column_transform_info_list
            ],
        }
    return rec


def train(args: argparse.Namespace) -> Dict[str, Any]:
    ev = load_eval_module()

    with open(os.path.join(args.data_dir, args.cols_config), encoding="utf-8") as f:
        cols = yaml.safe_load(f)
    num_cols = list(args.num_cols) if args.num_cols else list(cols["num_cols"])
    cat_cols = list(args.cat_cols) if args.cat_cols else list(cols["cat_cols"])

    cfg = load_eval_config(args.eval_config, ev)
    weight_col = cfg.weight_col if (args.use_weights and cfg.weight_col) else None

    df_train = _read_split(os.path.join(args.data_dir, args.train_file), num_cols, cat_cols,
                           weight_col, args.subsample, args.data_seed)
    df_val = _read_split(os.path.join(args.data_dir, args.val_file), num_cols, cat_cols,
                         weight_col, args.subsample_val, args.data_seed + 1)
    ref_path = os.path.join(args.data_dir, args.ref_file)
    df_ref = (_read_split(ref_path, num_cols, cat_cols, weight_col,
                          args.subsample_ref, args.data_seed + 2)
              if args.ref_eval else None)
    print(f"[data] train={df_train.shape} val={df_val.shape} "
          f"ref={None if df_ref is None else df_ref.shape} weight_col={weight_col}")

    # F2: the synthesiser only ever sees the training split
    fit_cols = num_cols + cat_cols
    df_fit = df_train[fit_cols].copy()

    run_name = args.run_name or f"{args.tag}_s{args.seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    save_dir = os.path.join(args.out_root, run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[run] {save_dir}")

    set_torch_seed(args.seed, deterministic=not args.no_deterministic)
    base = CTGANBaseline.from_train(df_fit, num_cols, cat_cols, args)

    n_eval = min(len(df_val), args.eval_n) if args.eval_n > 0 else len(df_val)
    df_val_sel = df_val.iloc[:n_eval].reset_index(drop=True)

    history: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {"score": float("inf"), "epoch": -1, "metrics": {}, "state": None}
    t0 = time.time()

    def on_epoch(ep: int) -> None:
        rec: Dict[str, Any] = {"epoch": ep, "elapsed_s": round(time.time() - t0, 2)}
        lv = getattr(base.model, "loss_values", None)
        if lv is not None and len(lv):
            row = lv.iloc[-1]
            rec["g_loss"] = float(row.get("Generator Loss", float("nan")))
            rec["d_loss"] = float(row.get("Discriminator Loss",
                                          row.get("Distriminator Loss", float("nan"))))
        do_eval = (ep % args.eval_every == 0) or (ep == args.epochs) or (ep == 1)
        if do_eval:
            # F5: model selection must not change the training RNG stream
            with frozen_rng():
                set_torch_seed(args.sample_seed + ep, deterministic=False)
                df_gen, _ = base.sample(len(df_val_sel), strict_vocab=not args.no_strict_vocab)
                m = val_metrics(df_val_sel, df_gen, cfg, ev,
                                use_weights=weight_col is not None,
                                do_2way=not args.no_2way)
            score = selection_score(m, args.f1_weight, args.select_on)
            rec.update({f"val_{k}": v for k, v in m.items()})
            rec["val_score"] = score
            if score < best["score"]:
                best.update({
                    "score": score, "epoch": ep, "metrics": m,
                    "state": copy.deepcopy(
                        {k: v.detach().cpu().clone()
                         for k, v in base.model._generator.state_dict().items()}
                    ),
                })
                rec["is_best"] = True
            print(f"[ep {ep:04d}] G={rec.get('g_loss', float('nan')):+.4f} "
                  f"D={rec.get('d_loss', float('nan')):+.4f} | "
                  f"val_srmse1={m.get('srmse_1', float('nan')):.4f} "
                  f"val_srmse2={m.get('srmse_2', float('nan')):.4f} "
                  f"val_blockF1={m.get('block_f1', float('nan')):.4f} score={score:.4f}"
                  + ("  <- best" if rec.get("is_best") else ""), flush=True)
        elif ep % args.log_every == 0:
            print(f"[ep {ep:04d}] G={rec.get('g_loss', float('nan')):+.4f} "
                  f"D={rec.get('d_loss', float('nan')):+.4f}", flush=True)
        history.append(rec)

    print(f"[fit] ctgan.CTGAN epochs={args.epochs} batch={args.batch_size} pac={args.pac} "
          f"disc_steps={args.discriminator_steps} device={args.device}", flush=True)
    with epoch_hook(on_epoch):
        base.model.fit(df_fit, discrete_columns=cat_cols)
    fit_seconds = time.time() - t0

    impl = implementation_record(base.model)
    print("[impl] " + json.dumps({k: impl[k] for k in ("package", "version", "class")}))

    def save_ckpt(path: str, epoch: int, metrics: Dict[str, Any]) -> None:
        # F5: full effective config + the fitted preprocessor travel with the model
        torch.save(
            {
                "ctgan_model": base.model,          # includes the fitted DataTransformer (VGM)
                "config": vars(args),
                "num_cols": num_cols,
                "cat_cols": cat_cols,
                "num_support": base.support,
                "integer_cols": base.integer_cols,
                "cat_vocab": base.vocab,
                "train_median": base.train_median,
                "implementation": impl,
                "epoch": epoch,
                "val_metrics": metrics,
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
            },
            path,
        )

    last_path = os.path.join(save_dir, "ctgan_last.pt")
    best_path = os.path.join(save_dir, "ctgan_best.pt")
    save_ckpt(last_path, args.epochs, history[-1] if history else {})

    loaded = "ctgan_last.pt"
    if best["state"] is not None and not args.load_final:
        base.model._generator.load_state_dict(best["state"])
        base.model._generator.to(base.model._device)
        save_ckpt(best_path, int(best["epoch"]), best["metrics"])
        loaded = "ctgan_best.pt"
    print(f"[select] loaded {loaded} (best val score={best['score']:.4f} @ epoch {best['epoch']})")

    # ---- final sample ------------------------------------------------------ #
    if args.n_samples > 0:
        n_out = args.n_samples
    elif df_ref is not None:
        n_out = len(df_ref)
    else:
        n_out = len(df_val)
    set_torch_seed(args.sample_seed, deterministic=not args.no_deterministic)
    t_s = time.time()
    df_out, post_diag = base.sample(n_out, strict_vocab=not args.no_strict_vocab)
    sample_seconds = time.time() - t_s
    sample_csv = os.path.join(save_dir, "generated_samples.csv")
    df_out.to_csv(sample_csv, index=False)
    print(f"[sample] n={n_out} in {sample_seconds:.1f}s -> {sample_csv}")
    print(f"[F6] rows out of numeric support BEFORE clipping: "
          f"{post_diag['n_rows_out_of_support_before_clip']}/{post_diag['n_rows']} "
          f"({100 * post_diag['row_out_of_support_rate_before_clip']:.2f}%); "
          f"categorical OOV after decoding: {post_diag['n_cat_oov_total']}")

    # ---- frozen-protocol evaluation --------------------------------------- #
    final_val = val_metrics(df_val, df_out.iloc[:len(df_val)], cfg, ev,
                            use_weights=weight_col is not None, do_2way=not args.no_2way)
    ref_metrics = None
    if df_ref is not None:
        t_e = time.time()
        ref_metrics = ev.evaluate_all(
            df_ref, df_out, cfg,
            train_df=df_train[fit_cols],
            coverage_k=tuple(args.coverage_k),
            coverage_max_subsets=args.coverage_max_subsets,
            with_oracles=args.with_oracles,
            rng=np.random.default_rng(args.seed),
        )
        print(f"[eval] reference: srmse1={ref_metrics['srmse1']:.4f} "
              f"srmse2={ref_metrics['srmse2']:.4f} "
              + " ".join(f"{b}_F1={v['f1']:.4f}" for b, v in ref_metrics["blocks"].items())
              + f"  ({time.time() - t_e:.1f}s)")

    cond_demo = None
    if args.cond_demo:
        constraints = json.loads(args.cond_demo)
        n_cd = args.cond_demo_n
        df_cd, cd_meta = base.sample_conditional(constraints, n_cd, seed=args.sample_seed + 1)
        cond_demo = cd_meta
        if df_ref is not None and len(df_cd):
            try:
                cond_demo["conditional_eval_note"] = (
                    "computed on the already-filtered conditional sample, so its 'yield' is 1.0 "
                    "by construction; the sampler's true acceptance rate is the top-level 'yield'"
                )
                cond_demo["conditional_eval"] = ev.conditional_eval(
                    df_ref, df_cd, {k: set(v) for k, v in cd_meta["constraints"].items()}, cfg,
                    coverage_k=(2,), coverage_max_subsets=args.coverage_max_subsets,
                )
            except Exception as exc:  # noqa: BLE001
                cond_demo["conditional_eval_error"] = f"{type(exc).__name__}: {exc}"
        cd_csv = os.path.join(save_dir, "generated_conditional.csv")
        df_cd.to_csv(cd_csv, index=False)
        print(f"[cond] path={cd_meta['path']} yield={cd_meta['yield']:.4f} "
              f"returned={cd_meta['n_returned']}/{n_cd} -> {cd_csv}")

    metrics = {
        "run": {
            "save_dir": save_dir,
            "started": datetime.now().isoformat(timespec="seconds"),
            "fit_wall_clock_s": round(fit_seconds, 2),
            "sample_wall_clock_s": round(sample_seconds, 2),
            "device": args.device,
            "torch_version": torch.__version__,
            "python": sys.version.split()[0],
        },
        "implementation": impl,
        "config": vars(args),
        "data": {
            "data_dir": args.data_dir,
            "n_train": int(len(df_train)),
            "n_val": int(len(df_val)),
            "n_ref": (None if df_ref is None else int(len(df_ref))),
            "n_generated": int(len(df_out)),
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "discrete_columns_declared": cat_cols,
            "held_out_val": True,               # F2
            "weight_col": weight_col,
            "num_support_train": {c: list(v) for c, v in base.support.items()},
            "integer_cols": base.integer_cols,
            "cat_vocab_sizes": {c: len(v) for c, v in base.vocab.items()},
        },
        "eval_config": cfg.to_dict(),
        "selection": {
            "rule": "min 0.5*(val SRMSE-1 + SRMSE-2) - w*mean block F1, on the held-out val split",
            "metric_backend": "utils.eval_v2",
            "eval_config_path": args.eval_config,
            "best_epoch": best["epoch"],
            "best_score": best["score"],
            "best_metrics": best["metrics"],
            "loaded_checkpoint": loaded,
        },
        # The HPO objective reads this in preference to final_val_metrics; see
        # utils.eval_v2.tail_metrics for why a single last evaluation is the
        # noisier choice, and why every iterative trainer must emit it.
        "tail_val_metrics": ev.tail_metrics(history, k=int(args.tail_k)),
        "tail_k": int(args.tail_k),
        "final_val_metrics": final_val,
        "reference_metrics": ref_metrics,
        "postprocessing": post_diag,
        "conditional_demo": cond_demo,
        "history": history,
        "outputs": {
            "samples_csv": sample_csv,
            "last_ckpt": last_path,
            "best_ckpt": best_path if loaded == "ctgan_best.pt" else None,
        },
    }
    out_json = os.path.join(save_dir, "metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(ev._jsonable(metrics), f, indent=2, default=str)
    print(f"[done] metrics -> {out_json}")

    # ---- K=50 audit ensemble (utils/ensemble_v2) --------------------------- #
    # Runs AFTER metrics.json so the ensemble wall clock never leaks into the
    # recorded run timings and a mid-ensemble failure still leaves a complete
    # metrics record.
    if args.ensemble:
        from utils.ensemble_v2 import write_ensemble

        if args.ensemble_n and args.ensemble_n > 0:
            ens_n = int(args.ensemble_n)
        elif df_ref is not None:
            ens_n = int(len(df_ref))
        else:
            raise SystemExit(
                "--ensemble needs a row count: the reference split was not loaded "
                "(--no-ref-eval), so N cannot default to len(reference). "
                "Pass an explicit --ensemble-n > 0."
            )

        def ensemble_sample_fn(seed: int, n: int) -> pd.DataFrame:
            # identical code path to the final sample above: reseed
            # python/numpy/torch/cuda from `seed` (set_torch_seed), then
            # CTGANBaseline.sample -> clip-then-round + strict-vocab assertion
            # (F6).  Columns are exactly the 12 modelled attributes of
            # cols_config_v3 in num+cat order.
            set_torch_seed(int(seed), deterministic=not args.no_deterministic)
            df_e, _ = base.sample(int(n), strict_vocab=not args.no_strict_vocab)
            return df_e[num_cols + cat_cols]

        write_ensemble(ensemble_sample_fn, n_samples=ens_n, save_dir=save_dir,
                       k=args.ensemble_k)

    return metrics


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CTGAN baseline (reference ctgan.CTGAN) for the v2 PUMS data")
    # data
    p.add_argument("--data-dir", default=_PA.S("/data/pums_states/CA"))
    p.add_argument("--cols-config", default="cols_config.yaml")
    p.add_argument("--train-file", default="traindata_pers.csv")
    p.add_argument("--val-file", default="valdata_pers.csv")
    p.add_argument("--ref-file", default="referencedata_pers.csv")
    p.add_argument("--num-cols", nargs="*", default=None)
    p.add_argument("--cat-cols", nargs="*", default=None)
    p.add_argument("--subsample", type=int, default=0, help="subsample train (smoke runs)")
    p.add_argument("--subsample-val", type=int, default=0)
    p.add_argument("--subsample-ref", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=42)
    # CTGAN hyper-parameters (library defaults unless stated)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--generator-dim", type=int, nargs="+", default=[256, 256])
    p.add_argument("--discriminator-dim", type=int, nargs="+", default=[256, 256])
    p.add_argument("--generator-lr", type=float, default=2e-4)
    p.add_argument("--discriminator-lr", type=float, default=2e-4)
    p.add_argument("--generator-decay", type=float, default=1e-6)
    p.add_argument("--discriminator-decay", type=float, default=1e-6)
    p.add_argument("--discriminator-steps", type=int, default=1)
    p.add_argument("--pac", type=int, default=10)
    p.add_argument("--no-log-frequency", action="store_true",
                   help="disable CTGAN's log-frequency training-by-sampling weighting")
    # eval / selection
    p.add_argument("--eval-config", default=DEFAULT_EVAL_CONFIG,
                   help="frozen utils.eval_v2 EvalConfig yaml (paper protocol)")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-n", type=int, default=0, help="0 = whole val split")
    p.add_argument("--no-2way", action="store_true")
    p.add_argument("--f1-weight", type=float, default=1.0)
    p.add_argument("--load-final", action="store_true",
                   help="keep the final instead of the val-selected weights")
    p.add_argument("--select-on", default="val_srmse",
        choices=["val_srmse", "if1", "ml_eff"],
        help="hyperparameter-selection criterion (utils/criteria.py): each model is\n"
             "selected the way its own authors select, not by one shared metric")
    p.add_argument("--no-weights", dest="use_weights", action="store_false",
                   help="ignore the eval config's weight column (PWGTP)")
    p.add_argument("--ref-eval", dest="ref_eval", action="store_true", default=True)
    p.add_argument("--no-ref-eval", dest="ref_eval", action="store_false")
    p.add_argument("--coverage-k", type=int, nargs="+", default=[2, 3])
    p.add_argument("--coverage-max-subsets", type=int, default=200)
    p.add_argument("--with-oracles", action="store_true")
    # sampling
    p.add_argument("--n-samples", type=int, default=0, help="0 = len(reference)")
    p.add_argument("--no-strict-vocab", action="store_true",
                   help="report instead of raising when a generated category is unseen in train")
    p.add_argument("--cond-demo", default=None,
                   help='JSON constraints for a sample_conditional smoke, e.g. \'{"Gender": "Female"}\'')
    p.add_argument("--cond-demo-n", type=int, default=2000)
    # reproducibility / io
    p.add_argument("--tail-k", type=int, default=5,
                   help="trailing in-training evaluations averaged into "
                        "`tail_val_metrics`, the quantity HPO tunes on.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-seed", type=int, default=1234)
    p.add_argument("--no-deterministic", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-root", default=_PA.S("/modelSave"))
    p.add_argument("--tag", default="ctgan_v2")
    p.add_argument("--run-name", default=None)
    # audit-layer ensemble output (utils/ensemble_v2): --ensemble / --ensemble-k /
    # --ensemble-n.  Finalized runs only -- HPO trials never pass --ensemble.
    from utils.ensemble_v2 import add_ensemble_args
    add_ensemble_args(p)
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.batch_size % 2 != 0:
        raise SystemExit("--batch-size must be even (ctgan.CTGAN asserts this)")
    if args.batch_size % args.pac != 0:
        raise SystemExit(f"--batch-size ({args.batch_size}) must be divisible by --pac ({args.pac})")
    if args.ensemble and args.ensemble_n <= 0 and not args.ref_eval:
        raise SystemExit(
            "--ensemble with --no-ref-eval: the reference split will not be loaded, "
            "so N cannot default to len(reference). Pass an explicit --ensemble-n > 0."
        )
    return train(args)


if __name__ == "__main__":
    main()
