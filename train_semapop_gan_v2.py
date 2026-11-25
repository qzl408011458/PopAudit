#!/usr/bin/env python
"""
train_semapop_gan_v2.py
=======================
Clean v2 trainer for SemaPop-GAN.

This trainer replaces an earlier in-house implementation (``train_wgan_llm.py`` /
``train_wgan_llm2.py`` / ``utils/MARG_LOSS.py``, not shipped).  It fixes the
following defects (F2-F6 are the authors' internal item numbers):

F2  **held-out validation.**  The old code set ``processed_train = processed_all``
    and ``embeddings_pers_train = concat(train, val)``
    (``train_wgan_llm2.py:367-370``), i.e. the validation set was inside the
    training set.  Here the preprocessor, the label encoders, the marginal
    targets and the binning spec are fit on **train only**; val is used solely
    for model selection, for the M-R conditioning pool and for diagnostics.

F3  **marginal regulariser.**  Uses ``utils.marg_loss_v2`` (differentiable soft
    histogram, bins in the original scale, Eq.-11 SRMSE).  The generator emits a
    **probability simplex per categorical block** (``SimplexGenerator`` below,
    per-block softmax with temperature or straight-through Gumbel-softmax as in
    CTGAN) so that the critic, the marginal loss and the sampler all consume the
    *same* object.  Merely adding a softmax inside the loss would not work: the
    critic would still score raw blocks against one-hot real rows, and softmax
    of a perfect one-hot with K=7 caps at 0.312.

F4  **model selection.**  The best checkpoint is now the one that is *loaded*.
    Selection is on val SRMSE-1/2 (+ block precision/recall when
    ``utils.eval_v2`` is available), never on the critic gap.

F5  **reproducibility.**  ``--seed`` (training) and ``--sample-seed`` (sampling)
    are separate; ``set_torch_seed`` covers python/numpy/torch/cuda/cudnn; the
    checkpoint stores the **full effective config** plus the preprocessor, the
    label encoders and the binning spec; every run writes ``metrics.json``.

F6  **numeric post-processing.**  Generated numeric values are inverse
    transformed, **clipped to the training support**, and only then rounded
    (integer columns only).

Conditioning
------------
``--cond-source``  persona | rawattr_full | rawattr_demo | none | gaussian
``--cond-mode``    mr  = conditioning vectors drawn from held-out val records
                   mu  = conditioning vectors drawn from a persona prior.
                         ``model.persona_prior`` is imported **lazily**: pass a
                         pre-fitted, val-selected prior with ``--prior-path``
                         (the only variant allowed for paper M-U numbers); with
                         no path the trainer fits ``--prior-family`` in process
                         (smoke runs), and if the module is missing altogether it
                         falls back to a diagonal Gaussian in a PCA subspace.
                         Which of the three happened is recorded under
                         ``prior.kind`` / ``prior.val_selected`` in metrics.json.
                   train = resample the training conditioning pool
                         (record-anchored; appendix ablation only)

Example
-------
    python train_semapop_gan_v2.py \
        --data-dir <repo>/data/pums_states/CA \
        --out-root <semapop_repo>/modelSave \
        --tag ca13 --device cuda:2 --epochs 300 --seed 42
"""

from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[0])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)

import argparse
import copy
import json
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.wgan_llm import (  # noqa: E402
    CondWGANCriticProj,
    CondWGANGeneratorFiLM,
    TabularSchema,
    gradient_penalty_cond,
)
from model.zcr import sample_bank, zcr_terms   # noqa: E402  (C+Z / C+M+Z cells)
from utils.marg_loss_v2 import (  # noqa: E402

    DEFAULT_TIME_STEPS,
    BinningSpec,
    build_cat_slices,
    build_marginal_loss,
)

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
#: the frozen paper protocol; every paper run must use it.
DEFAULT_EVAL_CONFIG = os.path.join(_REPO_ROOT, "configs", "eval_v3_pums12.yaml")

COND_SOURCES = ("persona", "rawattr_full", "rawattr_demo", "none", "gaussian")
COND_MODES = ("mr", "mu", "train")

#: default "demographic" block for ``--cond-source rawattr_demo`` (matched-
#: information control of plan section 7.2 / E3).  Intersected with the columns
#: actually present in the data.
DEFAULT_DEMO_COLS = (
    "Age",
    "Gender",
    "Marital_status",
    "Employment_status",
    "Studenthood_status",
    "Personal_Income_Class",
)

DEFAULT_TAIL_MERGE = {"Household_Size": 10, "Number_of_children": 8}


# --------------------------------------------------------------------------- #
# F5: determinism
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


# --------------------------------------------------------------------------- #
# F3c: simplex generator head
# --------------------------------------------------------------------------- #
class SimplexGenerator(nn.Module):
    """Wraps the FiLM generator so every categorical block is a simplex.

    The FiLM backbone (``model.wgan_llm.CondWGANGeneratorFiLM``) is kept
    unchanged; only the output head changes.  ``cat_mode``:

    ``gumbel_st``   straight-through Gumbel-softmax (CTGAN).  The forward value
                    is an exact one-hot, so the critic sees the same kind of
                    object for real and fake rows; the gradient flows through
                    the soft relaxation.  **Default.**
    ``gumbel_soft`` soft Gumbel-softmax (no straight-through).
    ``softmax``     deterministic per-block softmax with temperature.
    """

    def __init__(
        self,
        base: CondWGANGeneratorFiLM,
        schema: TabularSchema,
        cat_mode: str = "gumbel_st",
        temperature: float = 0.667,
    ) -> None:
        super().__init__()
        if cat_mode not in ("gumbel_st", "gumbel_soft", "softmax"):
            raise ValueError(f"unknown cat_mode {cat_mode!r}")
        self.base = base
        self.num_dim = int(schema.num_dim)
        self.cat_slices = list(schema.cat_slices)
        self.cat_mode = cat_mode
        self.temperature = float(temperature)

    @property
    def noise_dim(self) -> int:
        return self.base.noise_dim

    def raw(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.base(z, cond)

    def _blocks(self, raw: torch.Tensor, mode: str, tau: float) -> List[torch.Tensor]:
        out = []
        for s, e in self.cat_slices:
            logits = raw[:, s:e]
            if mode == "softmax":
                out.append(F.softmax(logits / max(tau, 1e-6), dim=1))
            elif mode == "gumbel_soft":
                out.append(F.gumbel_softmax(logits, tau=tau, hard=False, dim=1))
            elif mode == "gumbel_st":
                out.append(F.gumbel_softmax(logits, tau=tau, hard=True, dim=1))
            elif mode == "argmax":
                idx = logits.argmax(dim=1)
                out.append(F.one_hot(idx, num_classes=e - s).float())
            elif mode == "multinomial":
                p = F.softmax(logits / max(tau, 1e-6), dim=1)
                idx = torch.multinomial(p, num_samples=1).squeeze(1)
                out.append(F.one_hot(idx, num_classes=e - s).float())
            else:
                raise ValueError(f"unknown block mode {mode!r}")
        return out

    def forward(
        self,
        z: torch.Tensor,
        cond: torch.Tensor,
        mode: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        raw = self.raw(z, cond)
        mode = mode or self.cat_mode
        tau = self.temperature if temperature is None else float(temperature)
        blocks = self._blocks(raw, mode, tau)
        return torch.cat([raw[:, : self.num_dim]] + blocks, dim=1)

    def sample_mode(self, requested: str) -> str:
        """Resolve ``--sample-cat-mode auto`` to a mode matching how we trained."""
        if requested != "auto":
            return requested
        return "multinomial" if self.cat_mode.startswith("gumbel") else "argmax"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@dataclass
class PreparedData:
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    x_train: np.ndarray            # [N, D] = [num z-scored | one-hot cats]
    x_val: np.ndarray
    cond_train: np.ndarray         # [N, cond_in_dim]
    cond_val: np.ndarray
    num_cols: List[str]
    cat_cols: List[str]
    cat_dims: List[int]
    schema: TabularSchema
    preprocessor: ColumnTransformer
    label_encoders: Dict[str, LabelEncoder]
    num_mean: np.ndarray
    num_std: np.ndarray
    cond_meta: Dict[str, Any]


def _fit_label_encoders(df: pd.DataFrame, cat_cols: Sequence[str]) -> Dict[str, LabelEncoder]:
    les: Dict[str, LabelEncoder] = {}
    for c in cat_cols:
        le = LabelEncoder()
        le.fit(df[c].astype(str).values)
        les[c] = le
    return les


def _encode_cats(
    df: pd.DataFrame,
    cat_cols: Sequence[str],
    les: Dict[str, LabelEncoder],
) -> Tuple[pd.DataFrame, int]:
    """Label-encode with **train-fitted** encoders; unseen levels -> class 0."""
    out = df.copy()
    n_unseen = 0
    for c in cat_cols:
        le = les[c]
        known = {v: i for i, v in enumerate(le.classes_)}
        vals = out[c].astype(str).values
        codes = np.fromiter((known.get(v, -1) for v in vals), dtype=np.int64, count=len(vals))
        n_unseen += int((codes < 0).sum())
        out[c] = np.where(codes < 0, 0, codes)
    return out, n_unseen


def _build_preprocessor(num_cols: Sequence[str], cat_cols: Sequence[str],
                        cat_dims: Sequence[int]) -> ColumnTransformer:
    cats = [np.arange(k) for k in cat_dims]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(num_cols)),
            ("cat", OneHotEncoder(categories=cats, handle_unknown="ignore",
                                  sparse_output=False), list(cat_cols)),
        ],
        remainder="drop",
    )


def _gaussian_cond(n: int, dim: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def build_cond_vectors(
    source: str,
    n: int,
    processed: np.ndarray,
    embeddings: Optional[np.ndarray],
    demo_slice_idx: Optional[np.ndarray],
    gauss_dim: int,
    seed: int,
) -> np.ndarray:
    """Conditioning vectors for one split."""
    if source == "persona":
        if embeddings is None:
            raise ValueError("--cond-source persona requires persona embeddings")
        e = np.asarray(embeddings, dtype=np.float32)
        if e.shape[0] != n:
            raise ValueError(f"persona embeddings have {e.shape[0]} rows, expected {n}")
        return e
    if source == "rawattr_full":
        return np.asarray(processed, dtype=np.float32)
    if source == "rawattr_demo":
        if demo_slice_idx is None or demo_slice_idx.size == 0:
            raise ValueError("--cond-source rawattr_demo: no demographic columns resolved")
        return np.asarray(processed[:, demo_slice_idx], dtype=np.float32)
    if source == "none":
        return np.zeros((n, 8), dtype=np.float32)
    if source == "gaussian":
        return _gaussian_cond(n, gauss_dim, seed)
    raise ValueError(f"unknown cond source {source!r}")


def _resolve_demo_indices(
    demo_cols: Sequence[str],
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    cat_dims: Sequence[int],
) -> np.ndarray:
    slices = build_cat_slices(len(num_cols), cat_cols, cat_dims)
    idx: List[int] = []
    for c in demo_cols:
        if c in num_cols:
            idx.append(num_cols.index(c))
        elif c in slices:
            s, e = slices[c]
            idx.extend(range(s, e))
    return np.asarray(sorted(set(idx)), dtype=np.int64)


def prepare_data(args: argparse.Namespace) -> PreparedData:
    d = args.data_dir
    with open(os.path.join(d, getattr(args, "cols_config", "cols_config.yaml")), encoding="utf-8") as f:
        cols = yaml.safe_load(f)
    num_cols = list(args.num_cols) if args.num_cols else list(cols["num_cols"])
    cat_cols = list(args.cat_cols) if args.cat_cols else list(cols["cat_cols"])

    df_train = pd.read_csv(os.path.join(d, args.train_file), usecols=num_cols + cat_cols)
    df_val = pd.read_csv(os.path.join(d, args.val_file), usecols=num_cols + cat_cols)

    emb_train = emb_val = None
    if args.cond_source == "persona":
        with open(os.path.join(d, args.emb_file), "rb") as f:
            emb_train = np.asarray(pickle.load(f), dtype=np.float32)
        with open(os.path.join(d, args.emb_val_file), "rb") as f:
            emb_val = np.asarray(pickle.load(f), dtype=np.float32)

    # ---- optional subsample (smoke runs) ----------------------------------
    if args.subsample and args.subsample < len(df_train):
        rng = np.random.default_rng(args.data_seed)
        keep = np.sort(rng.choice(len(df_train), size=args.subsample, replace=False))
        df_train = df_train.iloc[keep].reset_index(drop=True)
        if emb_train is not None:
            emb_train = emb_train[keep]
    if args.subsample_val and args.subsample_val < len(df_val):
        rng = np.random.default_rng(args.data_seed + 1)
        keep = np.sort(rng.choice(len(df_val), size=args.subsample_val, replace=False))
        df_val = df_val.iloc[keep].reset_index(drop=True)
        if emb_val is not None:
            emb_val = emb_val[keep]

    # ---- F2: everything is fit on TRAIN only -------------------------------
    les = _fit_label_encoders(df_train, cat_cols)
    cat_dims = [int(len(les[c].classes_)) for c in cat_cols]
    tr_enc, _ = _encode_cats(df_train, cat_cols, les)
    va_enc, n_unseen = _encode_cats(df_val, cat_cols, les)
    if n_unseen:
        print(f"[data] {n_unseen} val cells had a category unseen in train -> mapped to class 0")

    pre = _build_preprocessor(num_cols, cat_cols, cat_dims)
    x_train = np.asarray(pre.fit_transform(tr_enc), dtype=np.float32)
    x_val = np.asarray(pre.transform(va_enc), dtype=np.float32)

    scaler: StandardScaler = pre.named_transformers_["num"]
    num_mean = np.asarray(scaler.mean_, dtype=np.float64)
    num_std = np.asarray(scaler.scale_, dtype=np.float64)

    schema = TabularSchema(num_dim=len(num_cols), cat_dims=cat_dims)
    if x_train.shape[1] != schema.data_dim:
        raise RuntimeError(f"processed dim {x_train.shape[1]} != schema {schema.data_dim}")

    demo_cols = list(args.demo_cols) if args.demo_cols else list(DEFAULT_DEMO_COLS)
    demo_idx = _resolve_demo_indices(demo_cols, num_cols, cat_cols, cat_dims)

    cond_train = build_cond_vectors(args.cond_source, len(df_train), x_train, emb_train,
                                    demo_idx, args.gauss_dim, args.data_seed)
    cond_val = build_cond_vectors(args.cond_source, len(df_val), x_val, emb_val,
                                  demo_idx, args.gauss_dim, args.data_seed + 7)

    # --- optional subspace consistency -------------------
    # M-U samples come from a prior fitted in a PCA subspace and inverse-mapped
    # back, so the generator would otherwise be TRAINED on raw embeddings and
    # SAMPLED off a slightly different manifold.  With --cond-subspace the same
    # project-and-invert map is applied to the training/val conditioning too, so
    # both live in one space.  Default off: existing runs are unaffected.
    cond_subspace_meta = {"applied": False}
    if getattr(args, "cond_subspace", ""):
        import model.persona_prior as _pp
        _tf = _pp.SubspaceTransform.load(args.cond_subspace)
        _before = cond_train.copy()
        cond_train = _tf.project_and_invert(cond_train).astype(np.float32)
        cond_val = _tf.project_and_invert(cond_val).astype(np.float32)
        _rel = float(np.linalg.norm(cond_train - _before) / (np.linalg.norm(_before) + 1e-12))
        cond_subspace_meta = {
            "applied": True,
            "path": args.cond_subspace,
            "n_components": int(getattr(_tf, "n_components", -1)),
            "train_rel_l2_shift": _rel,
        }
        print(f"[data] cond subspace applied: {args.cond_subspace} "
              f"(k={cond_subspace_meta['n_components']}, rel shift {_rel:.4f})")

    cond_meta = {
        "source": args.cond_source,
        "subspace": cond_subspace_meta,
        "dim": int(cond_train.shape[1]),
        "demo_cols_used": [c for c in demo_cols if c in num_cols or c in cat_cols],
        "demo_indices": demo_idx.tolist() if args.cond_source == "rawattr_demo" else [],
        "n_val_unseen_cells": int(n_unseen),
    }
    print(f"[data] train={x_train.shape} val={x_val.shape} cond={cond_train.shape} "
          f"({args.cond_source}) cat_dims={cat_dims}")
    return PreparedData(df_train, df_val, x_train, x_val, cond_train, cond_val,
                        num_cols, cat_cols, cat_dims, schema, pre, les,
                        num_mean, num_std, cond_meta)


# --------------------------------------------------------------------------- #
# F4: validation metrics
# --------------------------------------------------------------------------- #
def _try_eval_v2():
    """Lazy import of ``utils.eval_v2`` (may not exist yet)."""
    try:
        import utils.eval_v2 as ev  # type: ignore
        return ev
    except Exception:
        return None


def default_blocks(num_cols: Sequence[str], cat_cols: Sequence[str]) -> Dict[str, List[str]]:
    """Demographic / Household / Commute matching blocks."""
    blocks: Dict[str, List[str]] = {"Demographic": [], "Household": [], "Commute": []}
    for c in list(num_cols) + list(cat_cols):
        low = c.lower()
        if "commute" in low or "travel" in low:
            blocks["Commute"].append(c)
        elif "household" in low or "children" in low or "cars" in low:
            blocks["Household"].append(c)
        else:
            blocks["Demographic"].append(c)
    return {k: v for k, v in blocks.items() if v}


def make_eval_config(num_cols: Sequence[str], cat_cols: Sequence[str], spec: BinningSpec,
                     ref_df: pd.DataFrame, ev):
    """Auto-build an ``eval_v2.EvalConfig`` from the training BinningSpec.

    This is the **selection** config, not the frozen paper protocol:
    pass ``--eval-config <yaml>`` to use the frozen one once it exists.
    """
    specs = {}
    for c in cat_cols:
        specs[c] = ev.BinningSpec(name=c, kind="categorical")
    for c in num_cols:
        cb = spec[c]
        if cb.kind in ("integer", "integer_tail"):
            specs[c] = ev.BinningSpec(name=c, kind="integer",
                                      tail_merge_at=cb.tail_k)
        else:
            specs[c] = ev.BinningSpec(name=c, kind="binned", bin_width=float(cb.step))
    order = list(num_cols) + list(cat_cols)
    cfg = ev.EvalConfig(specs=specs, order=order,
                        blocks=default_blocks(num_cols, cat_cols),
                        name="semapop_gan_v2_selection")
    cfg.fit(ref_df)
    return cfg


def _shared_codes(df_ref: pd.DataFrame, df_gen: pd.DataFrame,
                  num_cols: Sequence[str], cat_cols: Sequence[str],
                  spec: BinningSpec) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Integer codes for both frames over a shared cell space (fallback path)."""
    ca: Dict[str, np.ndarray] = {}
    cb: Dict[str, np.ndarray] = {}
    for c in num_cols:
        ca[c] = spec[c].digitize(df_ref[c].values.astype(np.float64))
        cb[c] = spec[c].digitize(df_gen[c].values.astype(np.float64))
    for c in cat_cols:
        va = df_ref[c].astype(str).values
        vb = df_gen[c].astype(str).values
        cats = {v: i for i, v in enumerate(sorted(set(va.tolist()) | set(vb.tolist())))}
        ca[c] = np.fromiter((cats[v] for v in va), dtype=np.int64, count=len(va))
        cb[c] = np.fromiter((cats[v] for v in vb), dtype=np.int64, count=len(vb))
    return ca, cb


def _srmse_table(a: np.ndarray, b: np.ndarray, n_cells_hint: int) -> float:
    """Eq. 11 on two integer-coded tables sharing the same cell space."""
    k = int(max(a.max(), b.max())) + 1 if a.size and b.size else n_cells_hint
    k = max(k, 1)
    pa = np.bincount(a, minlength=k).astype(np.float64)
    pb = np.bincount(b, minlength=k).astype(np.float64)
    pa /= max(pa.sum(), 1.0)
    pb /= max(pb.sum(), 1.0)
    n_j = int(max((pa > 0).sum(), 1))          # observed cells of the reference
    return float(np.sqrt(n_j * np.sum((pb - pa) ** 2)))


def fallback_val_metrics(
    df_ref: pd.DataFrame,
    df_gen: pd.DataFrame,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    spec: BinningSpec,
    do_2way: bool = True,
) -> Dict[str, float]:
    """Local SRMSE-1 / SRMSE-2 fallback used when ``utils/eval_v2.py`` is
    unavailable or its API does not match.

    TODO(eval_v2): this covers only the two headline numbers needed for
    checkpoint selection (F4); it is *not* the paper's evaluation protocol (no
    block P/R, no fixed category lists, no weighting).  Runs that land here are
    flagged ``metric_backend == "fallback"`` in ``metrics.json``.
    """
    cols = list(num_cols) + list(cat_cols)
    ca, cb = _shared_codes(df_ref, df_gen, num_cols, cat_cols, spec)

    s1 = [_srmse_table(ca[c], cb[c], 1) for c in cols]
    out = {"srmse_1": float(np.mean(s1))}

    if do_2way:
        ks = {c: int(max(ca[c].max(), cb[c].max())) + 1 for c in cols}
        s2 = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                ci, cj = cols[i], cols[j]
                a = ca[ci] * ks[cj] + ca[cj]
                b = cb[ci] * ks[cj] + cb[cj]
                s2.append(_srmse_table(a, b, ks[ci] * ks[cj]))
        out["srmse_2"] = float(np.mean(s2))
    else:
        out["srmse_2"] = float("nan")
    return out


def val_metrics(
    df_ref: pd.DataFrame,
    df_gen: pd.DataFrame,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    spec: BinningSpec,
    do_2way: bool = True,
    ev=None,
    eval_cfg=None,
    with_pr: bool = False,
) -> Tuple[Dict[str, float], str]:
    """Selection metrics (F4).

    Uses ``utils.eval_v2`` (Eq.-11 ``srmse_kway`` + ``block_precision_recall``)
    when it is importable and an ``EvalConfig`` could be built; otherwise falls
    back to the local implementation.  The all-tuple F1 is deliberately excluded
    from the selection score (no full-tuple F1).

    ``with_pr`` adds NIP and NIR (utils.coverage_pr), which the ``if1`` criterion needs.
    It is OFF by default because the isolation forest costs ~2.6 s per evaluation, which
    at an 800-epoch cap and eval_every=5 is ~7 min a trial -- worth paying only for the
    model whose authors select on a precision/recall harmonic mean (WGAN-GP-ZCR, Kim &
    Bansal Eq. 16).  Failure to compute it is left as a MISSING KEY rather than a NaN, so
    utils.criteria raises instead of silently scoring the trial on something else.
    """
    if ev is not None and eval_cfg is not None:
        try:
            m: Dict[str, float] = {}
            m["srmse_1"] = float(ev.srmse_kway(df_ref, df_gen, eval_cfg, k=1)["mean"])
            m["srmse_2"] = (
                float(ev.srmse_kway(df_ref, df_gen, eval_cfg, k=2)["mean"])
                if do_2way else float("nan")
            )
            pr = ev.block_precision_recall(df_ref, df_gen, eval_cfg)
            f1s = [v["f1"] for b, v in pr.items()
                   if b != "all" and isinstance(v, dict) and "f1" in v]
            if f1s:
                m["block_f1"] = float(np.mean(f1s))
                for b, v in pr.items():
                    if b != "all":
                        m[f"block_f1_{b}"] = float(v["f1"])
            if with_pr:
                from utils.coverage_pr import alpha_precision_beta_recall as _apbr
                cols = list(eval_cfg.order)
                _num = [c for c in num_cols if c in cols]
                _cat = [c for c in cols if c not in _num]
                _pr = _apbr(df_ref[cols], df_gen[cols], _cat, _num,
                            subsample=8000, random_state=0)
                m["NIP"], m["NIR"] = float(_pr["NIP"]), float(_pr["NIR"])
            return m, "eval_v2"
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] utils.eval_v2 call failed ({type(exc).__name__}: {exc}); "
                  "falling back to the local SRMSE implementation")
    return fallback_val_metrics(df_ref, df_gen, num_cols, cat_cols, spec, do_2way), "fallback"


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
# priors (M-U)
# --------------------------------------------------------------------------- #
class _FallbackGaussianPrior:
    """Diagonal Gaussian in a PCA subspace of the training conditioning pool.

    TODO(persona_prior): this is a stand-in so that ``--cond-mode mu`` is
    runnable before ``model/persona_prior.py`` exists.  The paper's M-U results
    must use ``model.persona_prior`` (P-GMM / P-Flow selected on val
    log-likelihood, plan section 7.1); every run that falls back is marked
    ``prior.kind == "fallback_pca_gaussian"`` in ``metrics.json``.
    """

    kind = "fallback_pca_gaussian"

    def __init__(self, cond_train: np.ndarray, n_components: int = 128, seed: int = 0):
        from sklearn.decomposition import PCA

        n_components = int(min(n_components, cond_train.shape[1], cond_train.shape[0] - 1))
        self.pca = PCA(n_components=n_components, random_state=seed)
        z = self.pca.fit_transform(cond_train.astype(np.float32))
        self.mu = z.mean(axis=0)
        self.sd = z.std(axis=0) + 1e-6
        self.n_components = n_components

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n, self.n_components)) * self.sd + self.mu
        return self.pca.inverse_transform(z).astype(np.float32)


def load_persona_prior(
    prior_path: Optional[str],
    cond_train: np.ndarray,
    seed: int,
    family: str = "gmm",
    n_components: int = 128,
    cond_val: Optional[np.ndarray] = None,
):
    """Get the M-U prior.  ``model.persona_prior`` is imported **lazily** so that
    this trainer works even before that module exists.

    Preference order:
      1. ``model.persona_prior.load_prior(--prior-path)`` -- the only variant the
         paper's M-U numbers may use (fit on train, selected on val, plan 7.1);
      2. ``model.persona_prior.build_prior(family).fit(cond_train, cond_val)`` --
         convenience for smoke runs / ablations; the family itself is *not*
         selected here, which is recorded in ``metrics.json``;
      3. :class:`_FallbackGaussianPrior`.
    """
    try:
        import model.persona_prior as pp  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"[prior] model.persona_prior unavailable ({type(exc).__name__}: {exc}); "
              "using the PCA-Gaussian fallback")
        return _FallbackGaussianPrior(cond_train, seed=seed)

    if prior_path:
        obj = pp.load_prior(prior_path)
        print(f"[prior] loaded {type(obj).__name__} from {prior_path}")
        return obj

    try:
        tf = pp.SubspaceTransform(n_components=n_components, whiten=True, random_state=seed)
        obj = pp.build_prior(family, transform=tf, random_state=seed)
        obj.fit(cond_train, cond_val)
        print(f"[prior] fitted {type(obj).__name__} (family={family}, k={n_components} PCs) "
              "in-process -- NOT a val-selected prior, use --prior-path for paper runs")
        return obj
    except Exception as exc:  # noqa: BLE001
        print(f"[prior] in-process fit of family {family!r} failed "
              f"({type(exc).__name__}: {exc}); using the PCA-Gaussian fallback")
        return _FallbackGaussianPrior(cond_train, seed=seed)


def prior_sample(prior, n: int, seed: int) -> np.ndarray:
    """Draw ``n`` conditioning vectors from a prior object, seeded reproducibly."""
    if not hasattr(prior, "sample"):
        raise TypeError(f"prior object {type(prior)!r} has no .sample method")
    for kw in ({"rng": seed}, {"seed": seed}, {}):
        try:
            out = prior.sample(n, **kw)
            break
        except TypeError:
            continue
    else:  # pragma: no cover
        raise TypeError(f"could not call {type(prior)!r}.sample with a seed")
    if isinstance(out, tuple):
        out = out[0]
    return np.asarray(out, dtype=np.float32)


def draw_cond(
    mode: str,
    n: int,
    cond_train: np.ndarray,
    cond_val: np.ndarray,
    prior,
    seed: int,
) -> np.ndarray:
    """Conditioning vectors used at sampling time."""
    rng = np.random.default_rng(seed)
    if mode == "mr":
        idx = rng.integers(0, cond_val.shape[0], size=n)
        return cond_val[idx]
    if mode == "train":
        idx = rng.integers(0, cond_train.shape[0], size=n)
        return cond_train[idx]
    if mode == "mu":
        if prior is None:
            raise ValueError("--cond-mode mu requires a prior")
        return prior_sample(prior, n, seed)
    raise ValueError(f"unknown cond mode {mode!r}")


# --------------------------------------------------------------------------- #
# sampling + decoding (F6)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_processed(
    gen: SimplexGenerator,
    cond: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
    cat_mode: str = "argmax",
    temperature: float = 1.0,
) -> np.ndarray:
    was_training = gen.training
    gen.eval()
    outs = []
    for s in range(0, cond.shape[0], batch_size):
        c = torch.from_numpy(cond[s : s + batch_size].astype(np.float32)).to(device)
        z = torch.randn(c.shape[0], gen.noise_dim, device=device)
        x = gen(z, c, mode=cat_mode, temperature=temperature)
        outs.append(x.detach().cpu().numpy())
    if was_training:
        gen.train()
    return np.concatenate(outs, axis=0)


def decode_processed(
    x: np.ndarray,
    schema: TabularSchema,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    num_mean: np.ndarray,
    num_std: np.ndarray,
    spec: BinningSpec,
    integer_cols: Dict[str, bool],
    label_encoders: Optional[Dict[str, LabelEncoder]] = None,
) -> pd.DataFrame:
    """Processed matrix -> dataframe.  **F6**: clip to the training support first,
    round afterwards, and only for columns that are integer-valued in train."""
    out: Dict[str, np.ndarray] = {}
    num = x[:, : schema.num_dim].astype(np.float64) * num_std[None, :] + num_mean[None, :]
    for i, c in enumerate(num_cols):
        lo, hi = spec.support(c)
        v = np.clip(num[:, i], lo, hi)
        if integer_cols.get(c, False):
            # integer support bounds, so that rounding cannot push a value back
            # outside the clip range (e.g. hi = 93.99999965 -> 94 -> 93)
            ilo = int(np.ceil(lo - 1e-6))
            ihi = int(np.floor(hi + 1e-6))
            v = np.clip(np.rint(v), ilo, ihi).astype(np.int64)
        out[c] = v

    s = schema.num_dim
    for c, k in zip(cat_cols, schema.cat_dims):
        codes = x[:, s : s + k].argmax(axis=1).astype(np.int64)
        s += k
        out[c] = (
            label_encoders[c].inverse_transform(codes)
            if label_encoders is not None else codes
        )
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# training loop
# --------------------------------------------------------------------------- #
def train(args: argparse.Namespace, data: PreparedData, save_dir: str) -> Dict[str, Any]:
    dev = torch.device(args.device)
    set_torch_seed(args.seed, deterministic=not args.no_deterministic)

    schema = data.schema
    cond_dim_in = int(data.cond_train.shape[1])

    # ---- F3: marginal loss built on TRAIN only ----------------------------
    tail_merge = {k: v for k, v in DEFAULT_TAIL_MERGE.items() if k in data.num_cols}
    marg, spec = build_marginal_loss(
        data.x_train, data.num_cols, data.cat_cols, data.cat_dims,
        data.num_mean, data.num_std,
        bw_scale=args.bw_scale,
        time_step_overrides=DEFAULT_TIME_STEPS,
        tail_merge=tail_merge,
        soft_target=not args.hard_marg_target,
        srmse_mode=args.srmse_mode,
        strict_simplex=args.strict_simplex,
        seed=args.seed,
    )
    marg = marg.to(dev)
    print("[marg] bins per numeric column: "
          + ", ".join(f"{c}={spec[c].n_bins}({spec[c].kind})" for c in data.num_cols))

    integer_cols = {
        c: spec[c].kind in ("integer", "integer_tail", "time")
        or bool(np.allclose(data.df_train[c].values, np.rint(data.df_train[c].values)))
        for c in data.num_cols
    }

    # ---- models ------------------------------------------------------------
    base_gen = CondWGANGeneratorFiLM(
        data_dim=schema.data_dim, noise_dim=args.noise_dim, persona_dim=cond_dim_in,
        cond_dim=args.cond_dim, hidden=tuple(args.g_hidden), dropout=args.dropout_g,
        use_tanh=False, adapter_hidden=args.adapter_hidden,
        adapter_dropout=args.adapter_dropout,
    ).to(dev)
    # Straight-through Gumbel training draws each block's argmax from softmax(logits)
    # independently of tau, whereas the multinomial sampler draws from
    # softmax(logits / sample_temperature).  The two coincide only at 1.0, so a
    # different sampling temperature silently changes the sampled distribution.
    if args.cat_mode.startswith("gumbel") and abs(args.sample_temperature - 1.0) > 1e-9:
        raise ValueError(
            f"--sample-temperature {args.sample_temperature} is incompatible with "
            f"--cat-mode {args.cat_mode}: straight-through Gumbel training is "
            "temperature-invariant, so sampling must use temperature 1.0. "
            "Use --cat-mode softmax if you intend to change the sampling temperature."
        )
    if not getattr(args, "eval_config", None):
        print("[WARN] no --eval-config given: model selection uses an auto-derived "
              "matching space, NOT the frozen paper protocol "
              "(configs/eval_v2_pums20.yaml). Do not use such a run for paper tables.",
              flush=True)
    gen = SimplexGenerator(base_gen, schema, cat_mode=args.cat_mode,
                           temperature=args.gumbel_tau).to(dev)
    critic = CondWGANCriticProj(
        data_dim=schema.data_dim, persona_dim=cond_dim_in, cond_dim=args.cond_dim,
        hidden=tuple(args.d_hidden), feat_dim=args.cond_dim, dropout=args.dropout_d,
        adapter_hidden=args.adapter_hidden, adapter_dropout=args.adapter_dropout,
    ).to(dev)

    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(critic.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    # per-dimension scale of the conditioning vectors, used as the unit for --cond-jitter
    cond_scale = torch.from_numpy(
        data.cond_train.std(axis=0, keepdims=True).astype(np.float32)
    ).to(dev)

    ds = TensorDataset(torch.from_numpy(data.x_train), torch.from_numpy(data.cond_train))
    gen_cpu = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=0, generator=gen_cpu)

    # ---- prior for M-U -----------------------------------------------------
    prior = None
    prior_kind = "none"
    if args.cond_mode == "mu":
        prior = load_persona_prior(args.prior_path, data.cond_train, args.seed,
                                   family=args.prior_family,
                                   n_components=args.prior_components,
                                   cond_val=data.cond_val)
        prior_kind = getattr(prior, "kind", None) or getattr(prior, "family", None) \
            or type(prior).__name__

    # ---- F4: evaluation backend (utils.eval_v2 when available) -------------
    ev = _try_eval_v2()
    eval_cfg = None
    if ev is not None:
        try:
            eval_cfg = (
                ev.EvalConfig.from_yaml(args.eval_config) if args.eval_config
                else make_eval_config(data.num_cols, data.cat_cols, spec, data.df_val, ev)
            )
            if args.eval_config:
                eval_cfg.fit(data.df_val)
            print(f"[eval] utils.eval_v2 config '{eval_cfg.name}' "
                  f"blocks={list(eval_cfg.blocks)}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not build an eval_v2 config ({type(exc).__name__}: {exc}); "
                  "using the local SRMSE fallback")
            eval_cfg = None

    history: List[Dict[str, Any]] = []
    best = {"score": float("inf"), "epoch": -1, "metrics": {}}
    best_path = os.path.join(save_dir, "generator_best.pt")
    last_path = os.path.join(save_dir, "generator_last.pt")
    metric_backend = "n/a"
    t0 = time.time()

    def build_ckpt(epoch: int, metrics: Dict[str, Any]) -> Dict[str, Any]:
        # F5: the checkpoint carries the FULL effective config + preprocessor
        return {
            "gen_state": (gen if gen_ema is None else gen_ema).state_dict(),
            "critic_state": critic.state_dict(),
            "config": vars(args),
            "schema": {"num_dim": schema.num_dim, "cat_dims": list(schema.cat_dims)},
            "num_cols": data.num_cols,
            "cat_cols": data.cat_cols,
            "cat_dims": data.cat_dims,
            "preprocessor": data.preprocessor,
            "label_encoders": data.label_encoders,
            "num_mean": data.num_mean,
            "num_std": data.num_std,
            "integer_cols": integer_cols,
            "binning_spec": spec.to_dict(),
            "cond_meta": data.cond_meta,
            "cat_mode": args.cat_mode,
            "gumbel_tau": args.gumbel_tau,
            "epoch": epoch,
            "val_metrics": metrics,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        }

    # --- v9: EMA of the generator weights -----------------------------------
    # Standard GAN practice we had been missing.  Measured motivation: within ONE run
    # the last five evaluations swing 5.7-13.2% on SRMSE-1, and that end-of-training
    # oscillation -- not seed-to-seed difference -- is most of the 9.3% run-to-run
    # sigma that destroyed six single-seed conclusions.  Averaging the weights also
    # removes the reason to pick a "best" checkpoint out of ~30 noisy evaluations:
    # a best-of-30 selection on a metric with 9.3% noise carries a ~2-sigma
    # optimistic bias in whatever it reports.
    gen_ema = None
    if args.ema_decay > 0.0:
        gen_ema = copy.deepcopy(gen).eval()
        for _p in gen_ema.parameters():
            _p.requires_grad_(False)

    @torch.no_grad()
    def ema_update() -> None:
        if gen_ema is None:
            return
        d = args.ema_decay
        for pe, pg in zip(gen_ema.parameters(), gen.parameters()):
            pe.mul_(d).add_(pg.detach(), alpha=1.0 - d)
        for be, bg in zip(gen_ema.buffers(), gen.buffers()):
            be.copy_(bg)

    def eval_gen():
        """The module the reported metrics and the saved checkpoint come from."""
        return gen if gen_ema is None else gen_ema

    def evaluate(epoch: int) -> Dict[str, Any]:
        nonlocal metric_backend
        n = min(len(data.df_val), args.eval_n) if args.eval_n > 0 else len(data.df_val)
        cond = draw_cond(args.cond_mode, n, data.cond_train, data.cond_val, prior,
                         args.sample_seed + epoch)
        g_ev = eval_gen()
        xs = sample_processed(g_ev, cond, dev, batch_size=args.sample_batch,
                              cat_mode=g_ev.sample_mode(args.sample_cat_mode),
                              temperature=args.sample_temperature)
        df_gen = decode_processed(xs, schema, data.num_cols, data.cat_cols,
                                  data.num_mean, data.num_std, spec, integer_cols,
                                  label_encoders=data.label_encoders)
        m, backend = val_metrics(data.df_val, df_gen, data.num_cols, data.cat_cols, spec,
                                 do_2way=not args.no_2way, ev=ev, eval_cfg=eval_cfg)
        metric_backend = backend
        return m

    use_zcr = (args.bd_weight != 0.0 or args.ad_weight != 0.0)
    want_bd, want_ad = use_zcr and args.bd_weight != 0.0, use_zcr and args.ad_weight != 0.0
    if use_zcr:
        zcr_bank = torch.from_numpy(data.x_train).to(dev)
        n_nb = min(args.zcr_neighbors if args.zcr_neighbors > 0 else zcr_bank.shape[0],
                   zcr_bank.shape[0])
        print(f"[zcr] bd_w={args.bd_weight} ad_w={args.ad_weight} "
              f"neighbours={n_nb}/{zcr_bank.shape[0]} chunk={args.zcr_chunk} "
              f"every={args.zcr_every} cat_scale={args.zcr_cat_scale}", flush=True)
    g_step = 0
    for ep in range(1, args.epochs + 1):
        gen.train()
        critic.train()
        d_ls, g_adv_ls, m_cat_ls, m_cont_ls, gp_ls = [], [], [], [], []
        bd_ls, ad_ls = [], []
        m_cat_p_ls, m_cont_p_ls = [], []
        cal_ls, ent_ls = [], []

        for _step, (x_real, c_real) in enumerate(loader):
            x_real = x_real.to(dev, non_blocking=True)
            c_real = c_real.to(dev, non_blocking=True)
            B = x_real.size(0)

            for _ in range(args.n_critic):
                z = torch.randn(B, args.noise_dim, device=dev)
                with torch.no_grad():
                    x_fake = gen(z, c_real)
                d_real = critic(x_real, c_real).mean()
                d_fake = critic(x_fake, c_real).mean()
                gp = gradient_penalty_cond(critic, x_real, x_fake, c_real, dev,
                                           lambda_gp=args.lambda_gp)
                loss_d = (d_fake - d_real) + gp
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()
                d_ls.append(float(loss_d.item()))
                gp_ls.append(float(gp.item()))

            z = torch.randn(B, args.noise_dim, device=dev)
            c_g = c_real
            if args.cond_jitter > 0.0:
                # 1b: widen the conditioning support the generator is trained on,
                # WITHOUT breaking the (x, e) pairing the critic relies on.
                c_g = c_real + args.cond_jitter * cond_scale * torch.randn_like(c_real)
            x_fake = gen(z, c_g)
            adv = -critic(x_fake, c_g).mean()
            l_cat, l_cont, _ = marg(x_fake)

            # 1a: prior-conditioned marginal regularisation.  The marginal loss
            # compares generated marginals with the reference marginals and needs
            # no paired real rows, so it can be applied to a batch generated from
            # prior-sampled conditioning.  This optimises exactly the quantity that
            # degrades under M-U (the marginals) while leaving the adversarial
            # pairing -- which REQUIRES e to be the record's own embedding -- intact.
            l_cat_p = l_cont_p = None
            if args.lambda_m_prior > 0.0 and prior is not None:
                c_p = torch.as_tensor(
                    draw_cond("mu", B, data.cond_train, data.cond_val, prior,
                              args.sample_seed + 100000 + ep * 1000 + len(m_cat_ls)),
                    dtype=torch.float32, device=dev)
                x_fake_p = gen(torch.randn(B, args.noise_dim, device=dev), c_p)
                l_cat_p, l_cont_p, _ = marg(x_fake_p)

            # v8: conditional calibration.  The critic only asks that the joint be
            # indistinguishable; nothing rewards p(x|e) being CALIBRATED per record.
            # Measured motivation: a plain logistic probe on e, sampled from its
            # predicted probabilities, attains SRMSE-1 0.0169 vs the generator's
            # 0.0413 -- the information is in e, the generator under-uses it.
            # Small weight only: this term is a maximum-likelihood nudge and at large
            # weight it would suppress the z-diversity and collapse p(x|e) to a point.
            l_cal = torch.zeros((), device=dev)
            if args.lambda_cal > 0.0:
                # NOTE: cat_mode 'gumbel_st' makes the FORWARD output a hard one-hot, so an
                # NLL on x_fake would be pinned at the clamp floor and carry almost no
                # signal (smoke test: cal_loss 13.8, block_entropy 0.0).  The calibration
                # term must read the SOFT block probabilities, so we take one extra forward
                # in softmax mode with the same (z, c).
                x_soft = gen(z, c_g, mode="softmax")
                off = schema.num_dim
                for dim in schema.cat_dims:
                    blk = x_soft[:, off:off + dim].clamp_min(1e-8)
                    tgt = x_real[:, off:off + dim].argmax(dim=1)
                    l_cal = l_cal + torch.nn.functional.nll_loss(blk.log(), tgt)
                    off += dim
                l_cal = l_cal / max(len(schema.cat_dims), 1)

            loss_g = adv + args.lambda_m * (args.w_cat * l_cat + args.w_cont * l_cont)
            if use_zcr and (g_step % max(1, args.zcr_every) == 0):
                zsub = sample_bank(zcr_bank, args.zcr_neighbors)
                bd, ad = zcr_terms(x_fake, zsub, num_dim=schema.num_dim,
                                   cat_scale=args.zcr_cat_scale, chunk=args.zcr_chunk,
                                   want_bd=want_bd, want_ad=want_ad,
                                   ad_clamp=args.ad_clamp)
                loss_g = loss_g + args.bd_weight * bd + args.ad_weight * ad
                bd_ls.append(float(bd.item()))
                ad_ls.append(float(ad.item()))
            if args.lambda_cal > 0.0:
                loss_g = loss_g + args.lambda_cal * l_cal
            if l_cat_p is not None:
                loss_g = loss_g + args.lambda_m_prior * (args.w_cat * l_cat_p
                                                         + args.w_cont * l_cont_p)
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            g_step += 1
            ema_update()

            g_adv_ls.append(float(adv.item()))
            m_cat_ls.append(float(l_cat.item()))
            m_cont_ls.append(float(l_cont.item()))
            if args.lambda_cal > 0.0:
                cal_ls.append(float(l_cal.item()))
            # Diversity guard: mean per-block entropy of p(x|e,z), under the SOFT block
            # probabilities.  Logged for EVERY run, not only calibrated ones, so the v8
            # ablation has a matched lambda_cal=0 baseline -- without it the entropy
            # collapse can only be compared against a smoke run of a different config.
            # Subsampled: with lambda_cal=0 the softmax pass is an extra generator forward.
            if _step % 20 == 0:
                # RNG ISOLATION (bug found 2026-08-19): the generator's adapter carries
                # dropout (--adapter-dropout default 0.1), so an extra train-mode forward
                # CONSUMES the random stream and shifts every subsequent draw.  Without the
                # fork below, adding this diagnostic changed a fixed-seed run's SRMSE-1 from
                # 0.05683 to 0.05235 -- i.e. the "matched baseline" was not matched at all.
                with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []), \
                     torch.no_grad():
                    xs = x_soft if args.lambda_cal > 0.0 else gen(z, c_g, mode="softmax")
                    off = schema.num_dim; ent = 0.0
                    for dim in schema.cat_dims:
                        b = xs[:, off:off + dim].clamp_min(1e-8)
                        ent += float(-(b * b.log()).sum(1).mean()); off += dim
                    ent_ls.append(ent / max(len(schema.cat_dims), 1))
            if l_cat_p is not None:
                m_cat_p_ls.append(float(l_cat_p.item()))
                m_cont_p_ls.append(float(l_cont_p.item()))

        rec: Dict[str, Any] = {
            "epoch": ep,
            "d_loss": float(np.mean(d_ls)),
            "gp": float(np.mean(gp_ls)),
            "g_adv": float(np.mean(g_adv_ls)),
            "marg_cat": float(np.mean(m_cat_ls)),
            "marg_cont": float(np.mean(m_cont_ls)),
            "elapsed_s": round(time.time() - t0, 2),
        }
        rec["marg_total"] = rec["marg_cat"] + rec["marg_cont"]
        if bd_ls:
            rec["zcr_bd"] = float(np.mean(bd_ls))
            rec["zcr_ad"] = float(np.mean(ad_ls))
        if cal_ls:
            rec["cal_loss"] = float(np.mean(cal_ls))
        if ent_ls:
            rec["block_entropy"] = float(np.mean(ent_ls))
        if m_cat_p_ls:
            rec["marg_cat_prior"] = float(np.mean(m_cat_p_ls))
            rec["marg_cont_prior"] = float(np.mean(m_cont_p_ls))

        if ep % args.eval_every == 0 or ep == args.epochs or ep == 1:
            m = evaluate(ep)
            score = selection_score(m, args.f1_weight, args.select_on)
            rec.update({f"val_{k}": v for k, v in m.items()})
            rec["val_score"] = score
            if score < best["score"]:
                best = {"score": score, "epoch": ep, "metrics": m}
                torch.save(build_ckpt(ep, m), best_path)
                rec["is_best"] = True
            print(f"[ep {ep:04d}] D={rec['d_loss']:+.4f} gp={rec['gp']:.4f} "
                  f"G_adv={rec['g_adv']:+.4f} marg_cat={rec['marg_cat']:.4f} "
                  f"marg_cont={rec['marg_cont']:.4f} | val_srmse1={m.get('srmse_1', float('nan')):.4f} "
                  f"val_srmse2={m.get('srmse_2', float('nan')):.4f} score={score:.4f}"
                  + ("  <- best" if rec.get("is_best") else ""))
        else:
            print(f"[ep {ep:04d}] D={rec['d_loss']:+.4f} G_adv={rec['g_adv']:+.4f} "
                  f"marg_cat={rec['marg_cat']:.4f} marg_cont={rec['marg_cont']:.4f}")
        history.append(rec)
        torch.save(build_ckpt(ep, rec), last_path)

    # ---- F4: LOAD THE BEST CHECKPOINT -------------------------------------
    load_path = best_path if (os.path.exists(best_path) and not args.load_final) else last_path
    ck = torch.load(load_path, map_location=dev, weights_only=False)
    gen.load_state_dict(ck["gen_state"])
    critic.load_state_dict(ck["critic_state"])
    print(f"[select] loaded {os.path.basename(load_path)} from epoch {ck['epoch']} "
          f"(best score={best['score']:.4f} @ epoch {best['epoch']})")

    # ---- final sample ------------------------------------------------------
    n_out = args.n_samples if args.n_samples > 0 else len(data.df_val)
    cond = draw_cond(args.cond_mode, n_out, data.cond_train, data.cond_val, prior,
                     args.sample_seed)
    xs = sample_processed(gen, cond, dev, batch_size=args.sample_batch,
                          cat_mode=gen.sample_mode(args.sample_cat_mode),
                          temperature=args.sample_temperature)
    df_out = decode_processed(xs, schema, data.num_cols, data.cat_cols,
                              data.num_mean, data.num_std, spec, integer_cols,
                              label_encoders=data.label_encoders)
    sample_csv = os.path.join(save_dir, "generated_samples.csv")
    df_out.to_csv(sample_csv, index=False)

    tail_recs = [h for h in history if "val_srmse_1" in h][-max(int(args.tail_k), 1):]
    tail_keys = sorted({k[4:] for h in tail_recs for k in h if k.startswith("val_")
                        and isinstance(h[k], (int, float))})
    tail_m = {k: float(np.mean([h["val_" + k] for h in tail_recs if "val_" + k in h]))
              for k in tail_keys}
    tail_m["n_evals_averaged"] = len(tail_recs)

    final_m, backend = val_metrics(data.df_val, df_out, data.num_cols, data.cat_cols,
                                   spec, do_2way=not args.no_2way,
                                   ev=ev, eval_cfg=eval_cfg)

    # F6 evidence: out-of-support rate BEFORE clipping (what the generator
    # actually produced) and AFTER clipping (must be 0 -- invariant check).
    raw_num = (xs[:, : schema.num_dim].astype(np.float64) * data.num_std[None, :]
               + data.num_mean[None, :])
    oos_pre, oos_post = {}, {}
    for i, c in enumerate(data.num_cols):
        lo, hi = spec.support(c)
        oos_pre[c] = float(np.mean((raw_num[:, i] < lo - 1e-9) | (raw_num[:, i] > hi + 1e-9)))
        v = df_out[c].values.astype(np.float64)
        oos_post[c] = float(np.mean((v < lo - 1e-9) | (v > hi + 1e-9)))

    metrics = {
        "run": {
            "save_dir": save_dir,
            "started": datetime.now().isoformat(timespec="seconds"),
            "wall_clock_s": round(time.time() - t0, 2),
            "device": args.device,
            "torch_version": torch.__version__,
        },
        "config": vars(args),
        "data": {
            "n_train": int(len(data.df_train)),
            "n_val": int(len(data.df_val)),
            "num_cols": data.num_cols,
            "cat_cols": data.cat_cols,
            "cat_dims": data.cat_dims,
            "data_dim": int(schema.data_dim),
            "held_out_val": True,          # F2
            **data.cond_meta,
        },
        "binning": {c: {"kind": spec[c].kind, "n_bins": spec[c].n_bins,
                        "bw": spec[c].bw, "support": list(spec.support(c))}
                    for c in data.num_cols},
        "prior": {
            "kind": prior_kind,
            "path": args.prior_path,
            "val_selected": bool(args.prior_path),
            "best_k": getattr(prior, "best_k_", None),
            "best_cov": getattr(prior, "best_cov_", None),
            "best_val_ll": getattr(prior, "best_val_ll_", None),
        },
        "eval_config": (eval_cfg.to_dict() if eval_cfg is not None else None),
        "selection": {
            "rule": "min 0.5*(val SRMSE-1 + SRMSE-2) - w*block_F1 "
                    "(block F1 excludes the all-tuple block)",
            "metric_backend": metric_backend,
            "best_epoch": best["epoch"],
            "best_score": best["score"],
            "best_metrics": best["metrics"],
            "loaded_checkpoint": os.path.basename(load_path),
        },
        # v9: the TAIL-AVERAGED validation metrics -- the mean of the last
        # `tail_k` in-training evaluations.  This, not the single last evaluation,
        # is what HPO tunes on.  Measured: run-to-run sigma on SRMSE-1 is 9.3% for
        # the last single evaluation but 5.2% for the mean of the last five, at zero
        # extra cost -- better than a 2-seed mean (6.6%) for twice the money.
        "tail_val_metrics": tail_m,
        "tail_k": int(args.tail_k),
        "final_val_metrics": final_m,
        "final_metric_backend": backend,
        "out_of_support_rate_before_clip": oos_pre,
        "out_of_support_rate_after_clip": oos_post,
        "history": history,
        "outputs": {"samples_csv": sample_csv, "best_ckpt": best_path,
                    "last_ckpt": last_path},
    }
    with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"[done] samples -> {sample_csv}")
    print(f"[done] metrics -> {os.path.join(save_dir, 'metrics.json')}")
    return metrics


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SemaPop-GAN v2 trainer (F2-F6 fixed)")
    # data
    p.add_argument("--data-dir", default=_PA.S("/data/pums_states/CA"))
    p.add_argument("--train-file", default="traindata_pers.csv")
    p.add_argument("--val-file", default="valdata_pers.csv")
    p.add_argument("--emb-file", default="embPers.pkl")
    p.add_argument("--emb-val-file", default="embPers_val.pkl")
    p.add_argument("--cols-config", default="cols_config.yaml")
    p.add_argument("--num-cols", nargs="*", default=None,
                   help="override cols_config.yaml num_cols")
    p.add_argument("--cat-cols", nargs="*", default=None)
    p.add_argument("--demo-cols", nargs="*", default=None,
                   help=f"columns used by --cond-source rawattr_demo (default {DEFAULT_DEMO_COLS})")
    p.add_argument("--subsample", type=int, default=0, help="subsample the train split (smoke runs)")
    p.add_argument("--subsample-val", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=42, help="seed for subsampling / gaussian cond")
    # conditioning
    p.add_argument("--cond-source", choices=COND_SOURCES, default="persona")
    # DEFAULT IS 'mu', THE DELIVERABLE PATH.  'mr' draws conditioning vectors from the
    # validation embeddings, which is an oracle setting (record-level side information
    # derived from held-out data), and it has no business being what you get by forgetting a flag.  Every caller in the repo
    # passes this explicitly, so the change moves no existing result; it only stops a
    # future invocation from silently reporting oracle numbers.
    p.add_argument("--cond-mode", choices=COND_MODES, default="mu")
    p.add_argument("--prior-path", default=None,
                   help="pre-fitted model.persona_prior checkpoint for --cond-mode mu "
                        "(REQUIRED for paper M-U runs: the prior must be fit on train "
                        "and selected on val beforehand)")
    p.add_argument("--prior-family", choices=["gmm", "flow", "diff"], default="gmm",
                   help="family used when --prior-path is not given (smoke runs only)")
    p.add_argument("--prior-components", type=int, default=128,
                   help="PCA subspace dimension of the in-process prior fit")
    p.add_argument("--lambda-cal", type=float, default=0.0,
                   help="weight of the per-record conditional calibration term (NLL of the "
                        "generated categorical block probabilities against the conditioning "
                        "record's true labels). 0 = off. Keep SMALL: large values collapse "
                        "p(x|e) and destroy z-diversity; block_entropy is logged as a guard.")
    # D24 (adopted 2026-08-21): 0.4, not 0.  Measured under EMA at n=3 on the
    # sweep-selected prior: lambda=0 -> 0.05044, lambda=0.4 -> 0.04195 (-16.8%,
    # t=-8.78), lambda=1.0 -> 0.04281.  At lambda=0 the M-U arm is WORSE than the
    # unconditional baseline (0.05044 vs 0.04558), i.e. prior-sampled conditioning
    # is a net liability without this term.  An earlier retraction of D24 was an
    # artefact of the pre-EMA setup, where run-to-run sigma was 8.9% and buried it.
    p.add_argument("--lambda-m-prior", type=float, default=0.4,
                   help="weight of the marginal loss applied to a batch generated from "
                        "PRIOR-sampled conditioning (train/sample distribution matching). "
                        "0 = off (default). Requires a prior (--prior-path / --prior-family).")
    p.add_argument("--cond-jitter", type=float, default=0.0,
                   help="std of Gaussian jitter added to the training conditioning vectors, "
                        "in units of their per-dimension std; widens the conditioning support "
                        "the generator sees without breaking the (x, e) critic pairing. 0 = off.")
    p.add_argument("--cond-subspace", default="",
                   help="path to a fitted SubspaceTransform .npz; when given, the "
                        "training/val conditioning vectors are project-and-inverted "
                        "through it so the generator trains in the same subspace the "
                        "M-U prior samples from (plan sec.7.1). Default off.")
    p.add_argument("--gauss-dim", type=int, default=128)
    # architecture
    p.add_argument("--noise-dim", type=int, default=128)
    p.add_argument("--cond-dim", type=int, default=128)
    p.add_argument("--g-hidden", type=int, nargs="+", default=[256, 512, 256])
    p.add_argument("--d-hidden", type=int, nargs="+", default=[256, 512, 256])
    p.add_argument("--adapter-hidden", type=int, default=1024)
    p.add_argument("--tail-k", type=int, default=5,
                   help="how many trailing in-training evaluations to average into "
                        "`tail_val_metrics` (the HPO objective).")
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="EMA decay for the generator weights used at eval/checkpoint "
                        "time; 0 disables (pre-v9 behaviour).")
    p.add_argument("--adapter-dropout", type=float, default=0.1)
    p.add_argument("--dropout-g", type=float, default=0.0)
    p.add_argument("--dropout-d", type=float, default=0.0)
    p.add_argument("--cat-mode", choices=["gumbel_st", "gumbel_soft", "softmax"],
                   default="gumbel_st", help="F3c: how categorical blocks become a simplex")
    p.add_argument("--gumbel-tau", type=float, default=0.667)
    # optimisation
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-critic", type=int, default=5)
    p.add_argument("--lr-g", type=float, default=2e-4)
    p.add_argument("--lr-d", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.0)
    p.add_argument("--beta2", type=float, default=0.9)
    p.add_argument("--lambda-gp", type=float, default=10.0)
    # ---- ZCR (Kim & Bansal 2023), OFF by default -------------------------------
    # Ported 2026-08-29 for the regulariser factorial: the C+Z and C+M+Z cells need
    # this trainer to carry the SAME bd/ad terms train_wgan_zcr_v2.py has, or the
    # ladder compares implementations rather than components.  bd pulls each
    # generated row toward its nearest real row (feasibility), ad is the negated
    # mean distance to the bank (an explicit anti-collapse push) -- the one term
    # that acts directly on the conditional variance collapse diagnosed on this
    # tier.  Defaults 0/0 leave every existing SemaPop-GAN result byte-identical.
    p.add_argument("--bd-weight", type=float, default=0.0)
    p.add_argument("--ad-weight", type=float, default=0.0)
    p.add_argument("--zcr-neighbors", type=int, default=0, help="0 = the full train bank")
    p.add_argument("--zcr-chunk", type=int, default=20000)
    p.add_argument("--zcr-every", type=int, default=1)
    p.add_argument("--zcr-cat-scale", type=float, default=1.0)
    p.add_argument("--ad-clamp", type=float, default=0.0)
    # marginal loss
    p.add_argument("--lambda-m", type=float, default=0.4)
    p.add_argument("--w-cat", type=float, default=1.0)
    p.add_argument("--w-cont", type=float, default=1.0)
    p.add_argument("--bw-scale", type=float, default=0.35)
    p.add_argument("--srmse-mode", choices=["eq11", "l2"], default="eq11")
    p.add_argument("--hard-marg-target", action="store_true",
                   help="build numeric targets with a hard histogram (reintroduces the "
                        "irreducible floor; diagnostics only)")
    p.add_argument("--strict-simplex", action="store_true", default=True)
    p.add_argument("--no-strict-simplex", dest="strict_simplex", action="store_false")
    # eval / selection
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-n", type=int, default=0, help="0 = use the whole val split")
    p.add_argument("--no-2way", action="store_true")
    p.add_argument("--eval-config", default=DEFAULT_EVAL_CONFIG,
                   help="frozen utils.eval_v2 EvalConfig yaml; auto-built from the "
                        "training BinningSpec when omitted")
    p.add_argument("--f1-weight", type=float, default=1.0)
    p.add_argument("--load-final", action="store_true",
                   help="load the final instead of the best checkpoint (F4 regression switch)")
    p.add_argument("--select-on", default="val_srmse",
        choices=["val_srmse", "if1", "ml_eff"],
        help="hyperparameter-selection criterion (utils/criteria.py): each model is\n"
             "selected the way its own authors select, not by one shared metric")
    # sampling
    p.add_argument("--n-samples", type=int, default=0, help="0 = size of the val split")
    p.add_argument("--sample-batch", type=int, default=4096)
    p.add_argument("--sample-cat-mode",
                   choices=["auto", "argmax", "multinomial", "gumbel_st", "softmax"],
                   default="auto")
    p.add_argument("--sample-temperature", type=float, default=1.0)
    # reproducibility / io
    p.add_argument("--seed", type=int, default=42, help="training seed")
    p.add_argument("--sample-seed", type=int, default=1234, help="sampling seed (separate, F5)")
    p.add_argument("--no-deterministic", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-root", default=_PA.S("/modelSave"))
    p.add_argument("--tag", default="semapop_gan_v2")
    p.add_argument("--run-name", default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    run_name = args.run_name or (
        f"{args.tag}_{args.cond_source}_{args.cond_mode}_lam{args.lambda_m}"
        f"_s{args.seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    save_dir = os.path.join(args.out_root, run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[run] {save_dir}")

    set_torch_seed(args.seed, deterministic=not args.no_deterministic)
    data = prepare_data(args)
    return train(args, data, save_dir)


if __name__ == "__main__":
    main()
