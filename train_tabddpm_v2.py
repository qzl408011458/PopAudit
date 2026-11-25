#!/usr/bin/env python
"""
train_tabddpm_v2.py
===================
Clean v2 trainer for the **TabDDPM baseline**.

This trainer replaces an earlier in-house implementation (``train_tabddpm.py`` +
``model/tabddpm.py``, not shipped).  That implementation is a Gaussian
DDPM over one-hot *relaxations* with an auxiliary cross-entropy term and a
per-step ``argmax`` projection of the categorical slices -- i.e. its forward
process and its reverse process do not match, and it is not the model of
Kotelnikov et al. (2023).  This trainer drives :mod:`model.tabddpm_v2`, which
implements the real thing: Gaussian diffusion on the numeric block and
**multinomial diffusion** (uniform-mixing kernel + KL variational bound) on
each categorical block.

Conventions are copied from ``train_semapop_gan_v2.py`` so that every v2
baseline is comparable:

F2  the validation split is **held out**: the numeric transform, the label
    encoders and the binning spec are fit on *train only*; val is used solely
    for model selection and diagnostics, and is never concatenated into train.
F4  the best checkpoint is the one that is **loaded** at the end.  Selection is
    on val SRMSE-1/2 (+ block precision/recall/F1 through ``utils.eval_v2``
    with the frozen ``configs/eval_v2_pums20.yaml``), never on the training
    loss.
F5  ``--seed`` seeds python/numpy/torch/cuda/cudnn; ``--sample-seed`` is a
    separate sampling seed; the checkpoint carries the **full effective
    config** plus the numeric transformer and the label encoders; every run
    writes ``metrics.json``.
F6  generated numeric values are inverse-transformed, **clipped to the training
    support**, and only then rounded (integer columns only).  ``metrics.json``
    records the out-of-support rate before and after clipping.

Conditioning hook
-----------------
``model.tabddpm_v2.TabDenoiserV2`` accepts a ``cond`` tensor (FiLM), and
``cond`` is threaded through ``mixed_loss`` / ``p_sample`` / ``sample``.  This
trainer is the **unconditional baseline** (``cond_dim=0``, ``cond=None``);
SemaPop-DDPM will pass persona embeddings through the same hook.

Two documented stabilisations (see ``model/tabddpm_v2.py``)
-----------------------------------------------------------
The *diffusion process and objective* here are exactly TabDDPM's.  Two knobs
control things the paper's code left unguarded, and both matter a lot in
practice; both default to the faithful/official behaviour and both are recorded
in ``metrics.json``:

``--cat-input onehot``
    the official denoiser is fed the categorical part as a **log** one-hot
    (values in ``{0, -69}``), which drowns out the numeric inputs and the time
    embedding.  ``onehot`` feeds ``exp`` of the same vector -- identical
    information, identical diffusion, far better conditioned.
``--clip-x0 5``
    the official ``gaussian_p_mean_variance`` accepts ``clip_denoised`` but
    never uses it.  The ``eps -> x_0`` map has gain ``1/sqrt(abar_{T-1})``
    (~2e4 at T=1000), so an imperfectly trained denoiser makes a large share of
    numeric reverse chains explode.  With quantile-normal inputs, clipping the
    reconstructed ``x_0`` at +-5 sigma is a no-op for well-behaved chains.

Measured on a CPU smoke run (CA, 4k train rows, T=100, 3k steps, val SRMSE):

===================================  =========  =========  =====================
setting                              SRMSE-1    SRMSE-2    |z|>5 rate (Age/HH/Nch)
===================================  =========  =========  =====================
faithful defaults                    0.662      1.716      0.41 / 0.44 / 0.71
``--cat-input onehot``               0.257      0.678      0.10 / 0.24 / 0.53
``--clip-x0 5``                      0.149      0.348      0 / 0 / 0
``--cat-input onehot --clip-x0 5``   0.101      0.268      0 / 0 / 0
===================================  =========  =========  =====================

Example
-------
    python train_tabddpm_v2.py \
        --data-dir <semapop_repo>/data/pums_v2/CA \
        --eval-config configs/eval_v2_pums20.yaml \
        --cat-input onehot --clip-x0 5 \
        --tag tabddpm_ca --device cuda:0 --timesteps 1000 --epochs 200 --seed 42
"""

from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[0])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)

import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.tabddpm_v2 import (  # noqa: E402
    GaussianMultinomialDiffusion,
    TabDenoiserV2,
    TabularSchema,
)
from utils.ensemble_v2 import add_ensemble_args, write_ensemble  # noqa: E402
from utils.marg_loss_v2 import DEFAULT_TIME_STEPS, BinningSpec, build_binning_spec  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
#: the frozen paper protocol; every paper run must use it.
DEFAULT_EVAL_CONFIG = os.path.join(_REPO_ROOT, "configs", "eval_v3_pums12.yaml")


DEFAULT_TAIL_MERGE = {"Household_Size": 10, "Number_of_children": 8}


# --------------------------------------------------------------------------- #
# F5: determinism  (same implementation as train_semapop_gan_v2.set_torch_seed)
# --------------------------------------------------------------------------- #
def set_torch_seed(seed: int, deterministic: bool = True) -> None:
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
# shared evaluation helpers
# --------------------------------------------------------------------------- #
def _shared_eval_helpers():
    """Reuse the selection helpers of the reference trainer when importable.

    ``train_semapop_gan_v2`` owns ``default_blocks`` / ``make_eval_config`` /
    ``val_metrics`` / ``selection_score``.  Importing them (rather than
    re-implementing) guarantees the baseline and SemaPop are selected with
    *identical* semantics.  The import is lazy so this trainer still runs if
    that file moves.
    """
    try:
        import train_semapop_gan_v2 as ref  # type: ignore
        return ref
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] train_semapop_gan_v2 not importable ({type(exc).__name__}: {exc}); "
              "using the built-in metric fallback")
        return None


def _try_eval_v2():
    try:
        import utils.eval_v2 as ev  # type: ignore
        return ev
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# data (F2)
# --------------------------------------------------------------------------- #
@dataclass
class PreparedData:
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    x_num_train: np.ndarray        # [N, num_dim] scaled
    x_num_val: np.ndarray
    x_cat_train: np.ndarray        # [N, n_cat] int64 codes
    x_cat_val: np.ndarray
    num_cols: List[str]
    cat_cols: List[str]
    cat_dims: List[int]
    schema: TabularSchema
    num_transformer: Any
    label_encoders: Dict[str, LabelEncoder]
    spec: BinningSpec
    integer_cols: Dict[str, bool]
    n_val_unseen_cells: int
    # Optional persona conditioning, attached by train_semapop_ddpm_v2.  Left None here,
    # and every use below is guarded, so the unconditional baseline is unchanged.
    cond_train: Optional[np.ndarray] = None
    cond_val: Optional[np.ndarray] = None
    #: Which ORIGINAL rows survived --subsample, or None when the split was taken whole.
    #: Needed because anything aligned to the split by POSITION -- the persona embeddings
    #: are -- has to be cut with the same index array, and only this function knows it.
    #: Without it train_semapop_ddpm_v2 could not run subsampled at all: the .pkl still
    #: held 40,000 rows against a subsampled 8,192, and its guard refused (correctly) to
    #: pair every record with someone else's persona.  The GAN arm has always done this;
    #: this makes the DDPM arm able to.
    keep_train: Optional[np.ndarray] = None
    keep_val: Optional[np.ndarray] = None


def _fit_label_encoders(df: pd.DataFrame, cat_cols: Sequence[str]) -> Dict[str, LabelEncoder]:
    les: Dict[str, LabelEncoder] = {}
    for c in cat_cols:
        le = LabelEncoder()
        le.fit(df[c].astype(str).values)
        les[c] = le
    return les


def _encode_cats(df: pd.DataFrame, cat_cols: Sequence[str],
                 les: Dict[str, LabelEncoder]) -> Tuple[np.ndarray, int]:
    """Label-encode with **train-fitted** encoders; unseen levels -> class 0."""
    cols, n_unseen = [], 0
    for c in cat_cols:
        known = {v: i for i, v in enumerate(les[c].classes_)}
        vals = df[c].astype(str).values
        codes = np.fromiter((known.get(v, -1) for v in vals), dtype=np.int64, count=len(vals))
        n_unseen += int((codes < 0).sum())
        cols.append(np.where(codes < 0, 0, codes))
    if not cols:
        return np.zeros((len(df), 0), dtype=np.int64), 0
    return np.stack(cols, axis=1).astype(np.int64), n_unseen


def build_num_transformer(kind: str, n_rows: int, seed: int):
    """``quantile`` reproduces TabDDPM's default normalisation; ``standard`` is
    the z-scoring used by the other v2 baselines."""
    if kind == "standard":
        return StandardScaler()
    if kind == "quantile":
        n_q = max(min(n_rows // 30, 1000), 10)
        return QuantileTransformer(
            output_distribution="normal", n_quantiles=n_q,
            subsample=1_000_000_000, random_state=seed,
        )
    raise ValueError(f"unknown --num-transform {kind!r}")


def prepare_data(args: argparse.Namespace) -> PreparedData:
    d = args.data_dir
    with open(os.path.join(d, args.cols_config), encoding="utf-8") as f:
        cols = yaml.safe_load(f)
    num_cols = list(args.num_cols) if args.num_cols else list(cols["num_cols"])
    cat_cols = list(args.cat_cols) if args.cat_cols else list(cols["cat_cols"])

    df_train = pd.read_csv(os.path.join(d, args.train_file), usecols=num_cols + cat_cols)
    df_val = pd.read_csv(os.path.join(d, args.val_file), usecols=num_cols + cat_cols)

    keep_train = keep_val = None
    if args.subsample and args.subsample < len(df_train):
        rng = np.random.default_rng(args.data_seed)
        keep_train = np.sort(rng.choice(len(df_train), size=args.subsample, replace=False))
        df_train = df_train.iloc[keep_train].reset_index(drop=True)
    if args.subsample_val and args.subsample_val < len(df_val):
        rng = np.random.default_rng(args.data_seed + 1)
        keep_val = np.sort(rng.choice(len(df_val), size=args.subsample_val, replace=False))
        df_val = df_val.iloc[keep_val].reset_index(drop=True)

    # ---- F2: fit on TRAIN only --------------------------------------------
    les = _fit_label_encoders(df_train, cat_cols)
    cat_dims = [int(len(les[c].classes_)) for c in cat_cols]
    x_cat_train, _ = _encode_cats(df_train, cat_cols, les)
    x_cat_val, n_unseen = _encode_cats(df_val, cat_cols, les)
    if n_unseen:
        print(f"[data] {n_unseen} val cells had a category unseen in train -> class 0")

    raw_train = df_train[num_cols].to_numpy(dtype=np.float64) if num_cols else \
        np.zeros((len(df_train), 0))
    raw_val = df_val[num_cols].to_numpy(dtype=np.float64) if num_cols else \
        np.zeros((len(df_val), 0))

    tr = build_num_transformer(args.num_transform, len(df_train), args.seed)
    x_num_train = np.asarray(tr.fit_transform(raw_train), dtype=np.float32) \
        if num_cols else np.zeros((len(df_train), 0), dtype=np.float32)
    x_num_val = np.asarray(tr.transform(raw_val), dtype=np.float32) \
        if num_cols else np.zeros((len(df_val), 0), dtype=np.float32)

    tail_merge = {k: v for k, v in DEFAULT_TAIL_MERGE.items() if k in num_cols}
    spec = build_binning_spec(raw_train, num_cols, bw_scale=args.bw_scale,
                              time_step_overrides=DEFAULT_TIME_STEPS,
                              tail_merge=tail_merge)
    integer_cols = {
        c: spec[c].kind in ("integer", "integer_tail", "time")
        or bool(np.allclose(raw_train[:, i], np.rint(raw_train[:, i])))
        for i, c in enumerate(num_cols)
    }

    schema = TabularSchema(num_dim=len(num_cols), cat_dims=cat_dims)
    print(f"[data] train num={x_num_train.shape} cat={x_cat_train.shape} "
          f"val num={x_num_val.shape} cat={x_cat_val.shape} cat_dims={cat_dims}")
    return PreparedData(df_train, df_val, x_num_train, x_num_val, x_cat_train, x_cat_val,
                        num_cols, cat_cols, cat_dims, schema, tr, les, spec,
                        integer_cols, n_unseen,
                        keep_train=keep_train, keep_val=keep_val)


# --------------------------------------------------------------------------- #
# F6: decoding
# --------------------------------------------------------------------------- #
def decode(
    x_num: np.ndarray,
    x_cat: np.ndarray,
    data: PreparedData,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Scaled model output -> dataframe in the ORIGINAL units.

    Returns ``(df, raw_num_before_clip)``.  **F6**: inverse transform, clip to
    the training support, and only then round integer columns.
    """
    out: Dict[str, np.ndarray] = {}
    if data.num_cols:
        raw = np.asarray(data.num_transformer.inverse_transform(
            np.asarray(x_num, dtype=np.float64)), dtype=np.float64)
    else:
        raw = np.zeros((x_cat.shape[0], 0), dtype=np.float64)
    for i, c in enumerate(data.num_cols):
        lo, hi = data.spec.support(c)
        v = np.clip(raw[:, i], lo, hi)
        if data.integer_cols.get(c, False):
            ilo = int(np.ceil(lo - 1e-6))
            ihi = int(np.floor(hi + 1e-6))
            v = np.clip(np.rint(v), ilo, ihi).astype(np.int64)
        out[c] = v
    for j, c in enumerate(data.cat_cols):
        codes = np.clip(np.asarray(x_cat[:, j], dtype=np.int64), 0, data.cat_dims[j] - 1)
        out[c] = data.label_encoders[c].inverse_transform(codes)
    return pd.DataFrame(out), raw


# --------------------------------------------------------------------------- #
# metrics fallback (only used when train_semapop_gan_v2 is not importable)
# --------------------------------------------------------------------------- #
def _local_val_metrics(df_ref, df_gen, num_cols, cat_cols, spec, do_2way, ev, cfg):
    if ev is not None and cfg is not None:
        m: Dict[str, float] = {}
        m["srmse_1"] = float(ev.srmse_kway(df_ref, df_gen, cfg, k=1)["mean"])
        m["srmse_2"] = float(ev.srmse_kway(df_ref, df_gen, cfg, k=2)["mean"]) \
            if do_2way else float("nan")
        pr = ev.block_precision_recall(df_ref, df_gen, cfg)
        f1s = [v["f1"] for b, v in pr.items() if b != "all" and isinstance(v, dict)]
        if f1s:
            m["block_f1"] = float(np.mean(f1s))
            for b, v in pr.items():
                if b != "all":
                    m[f"block_f1_{b}"] = float(v["f1"])
        return m, "eval_v2"
    raise RuntimeError("no evaluation backend available")


def _local_selection_score(m: Dict[str, float], f1_weight: float = 1.0,
                           select_on: str = "val_srmse") -> float:
    """Lower is better.  The formula lives in utils.criteria so that this per-epoch
    checkpoint pick and Optuna's between-trial pick read the SAME definition -- a model
    selected on one quantity and ranked on another is not the model that was searched.
    ``select_on`` is one of utils.criteria.CRITERIA: val_srmse (default), if1, ml_eff.
    """
    import utils.criteria as _CR
    return _CR.selection_score(select_on, m, f1_weight)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

def build_cat_marginal_targets(x_cat_train: np.ndarray, cat_dims: Sequence[int],
                               dev) -> List[torch.Tensor]:
    """Per-block training category frequencies, the target of the marginal regulariser."""
    out = []
    for j, k in enumerate(cat_dims):
        cnt = np.bincount(x_cat_train[:, j].astype(np.int64), minlength=int(k)).astype(np.float64)
        out.append(torch.tensor(cnt / max(cnt.sum(), 1.0), dtype=torch.float32, device=dev))
    return out


def _srmse_cat(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Eq. 11 SRMSE for one block: sqrt(N * sum (p_hat - p)^2), sum p == 1."""
    n = target.numel()
    return torch.sqrt(torch.clamp(n * ((pred - target) ** 2).sum(), min=1e-12))


def marginal_terms(diffusion, out: Dict[str, torch.Tensor], c_real, args,
                   marg_cat: List[torch.Tensor], dev, ep: int) -> torch.Tensor:
    """SemaPop's two marginal regularisers, adapted to a diffusion backbone.

    A GAN can differentiate through a generated sample, so the GAN arm applies the
    marginal loss to `gen(z, c)` directly.  A diffusion model cannot: producing a sample
    means running the whole reverse chain.  The analogue here is the denoiser's own x0
    prediction at the sampled timestep, which is differentiable and costs nothing extra.

    CATEGORICAL BLOCKS ONLY, and this asymmetry is deliberate rather than an oversight:
    MarginalLossV2 expects z-scored numerics, while TabDDPM's numeric transform is a
    searched hyperparameter (quantile by default), whose inverse is not differentiable in
    torch.  Getting the numeric mapping subtly wrong would be worse than covering 48 of
    the 50 L-tier attributes and saying so.

    lambda_m_prior re-denoises the SAME noised batch under a PRIOR-SAMPLED condition:
    whatever condition it is handed, the marginals of what it reconstructs should match
    the population's.  That is the quantity which degrades under M-U, and sharing x_t
    with the paired term is the same trick as the GAN sharing z.
    """
    def block_marginals(out_cat: torch.Tensor) -> List[torch.Tensor]:
        logp = diffusion.predict_start(out_cat, out["log_x_cat_t"], out["t"])
        probs, off, res = logp.exp(), 0, []
        for k in diffusion.num_classes:
            res.append(probs[:, off:off + int(k)].mean(0)); off += int(k)
        return res

    total = torch.zeros((), device=dev)
    if getattr(args, "lambda_m", 0.0) > 0.0:
        pm = block_marginals(out["out_cat"])
        l = sum(_srmse_cat(p, t) for p, t in zip(pm, marg_cat)) / max(len(marg_cat), 1)
        total = total + args.lambda_m * l
    if getattr(args, "lambda_m_prior", 0.0) > 0.0 and getattr(args, "_prior_obj", None) is not None:
        c_p = torch.from_numpy(
            args._draw_cond("mu", c_real.shape[0], None, None, args._prior_obj,
                            args.sample_seed + 100000 + ep * 1000)).float().to(dev)
        mo = diffusion._denoise_fn(
            torch.cat([out["x_num_t"], out["log_x_cat_t"]], dim=1).float(), out["t"], c_p)
        pm = block_marginals(mo[:, diffusion.num_numerical_features:])
        l = sum(_srmse_cat(p, t) for p, t in zip(pm, marg_cat)) / max(len(marg_cat), 1)
        total = total + args.lambda_m_prior * l
    return total


def train(args: argparse.Namespace, data: PreparedData, save_dir: str) -> Dict[str, Any]:
    dev = torch.device(args.device)
    set_torch_seed(args.seed, deterministic=not args.no_deterministic)

    ref = _shared_eval_helpers()
    ev = _try_eval_v2()
    eval_cfg = None
    if ev is not None:
        try:
            if args.eval_config:
                eval_cfg = ev.EvalConfig.from_yaml(args.eval_config)
                eval_cfg.fit(data.df_val)
            elif ref is not None:
                eval_cfg = ref.make_eval_config(data.num_cols, data.cat_cols,
                                                data.spec, data.df_val, ev)
            print(f"[eval] utils.eval_v2 config "
                  f"'{eval_cfg.name if eval_cfg else None}' "
                  f"blocks={list(eval_cfg.blocks) if eval_cfg else None}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not build an eval_v2 config "
                  f"({type(exc).__name__}: {exc})")
            eval_cfg = None
    if args.cat_input == "logprob" or args.clip_x0 <= 0:
        print("[WARN] running with the *unstabilised* official TabDDPM settings "
              f"(--cat-input {args.cat_input}, --clip-x0 {args.clip_x0}). On a CA "
              "CPU smoke run this cost 6.5x SRMSE-1 and left 10-70% of the numeric "
              "reverse chains diverged; see the module docstring. Consider "
              "'--cat-input onehot --clip-x0 5'.", flush=True)
    if not args.eval_config:
        print("[WARN] no --eval-config given: model selection does NOT use the frozen "
              "paper protocol (configs/eval_v2_pums20.yaml). Do not use such a run "
              "for paper tables.", flush=True)

    def metrics_of(df_gen: pd.DataFrame) -> Tuple[Dict[str, float], str]:
        if ref is not None:
            return ref.val_metrics(data.df_val, df_gen, data.num_cols, data.cat_cols,
                                   data.spec, do_2way=not args.no_2way,
                                   ev=ev, eval_cfg=eval_cfg)
        return _local_val_metrics(data.df_val, df_gen, data.num_cols, data.cat_cols,
                                  data.spec, not args.no_2way, ev, eval_cfg)

    score_of = ref.selection_score if ref is not None else _local_selection_score

    # ---- model -------------------------------------------------------------
    denoiser = TabDenoiserV2(
        num_dim=data.schema.num_dim,
        cat_dims=data.cat_dims,
        hidden=tuple(args.hidden),
        dim_t=args.dim_t,
        dropout=args.dropout,
        cond_dim=args.cond_dim,            # 0 for the unconditional baseline
        cat_input=args.cat_input,
        persona_dim=int(getattr(args, "_persona_dim", 0)),
        adapter_hidden=int(getattr(args, "adapter_hidden", 1024)),
        adapter_dropout=float(getattr(args, "adapter_dropout", 0.1)),
    ).to(dev)
    diffusion = GaussianMultinomialDiffusion(
        denoise_fn=denoiser,
        num_numerical_features=data.schema.num_dim,
        num_classes=data.cat_dims,
        num_timesteps=args.timesteps,
        scheduler=args.schedule,
        gaussian_loss_type=args.gaussian_loss,
        gaussian_parametrization=args.gaussian_parametrization,
        parametrization=args.parametrization,
        clip_x0=args.clip_x0,
    ).to(dev)
    if args.cond_dim > 0:
        print("[WARN] --cond-dim > 0 builds the FiLM adapter, but this trainer is the "
              "UNCONDITIONAL D13 baseline and always passes cond=None, so the adapter "
              "stays at its zero init and is dead weight. It exists for SemaPop-DDPM.",
              flush=True)
    n_params = sum(p.numel() for p in diffusion.parameters() if p.requires_grad)
    print(f"[model] TabDDPM T={args.timesteps} schedule={args.schedule} "
          f"hidden={list(args.hidden)} params={n_params}")

    opt = torch.optim.AdamW(diffusion.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    conditioned = data.cond_train is not None
    cond_scale, marg_cat, prior_obj, draw_cond_ddpm = 1.0, None, None, None
    if conditioned:
        from train_semapop_gan_v2 import draw_cond as draw_cond_ddpm   # one definition
        prior_obj = getattr(args, "_prior_obj", None)
        args._draw_cond = draw_cond_ddpm
        cond_scale = float(np.std(data.cond_train)) or 1.0
        marg_cat = build_cat_marginal_targets(data.x_cat_train, data.cat_dims, dev)
        print(f"[cond] persona conditioning ON: dim={data.cond_train.shape[1]} "
              f"mode={args.cond_mode} prior={'yes' if prior_obj else 'no'} "
              f"lambda_m={getattr(args,'lambda_m',0.0)} "
              f"lambda_m_prior={getattr(args,'lambda_m_prior',0.0)}", flush=True)
    _tensors = [torch.from_numpy(data.x_num_train).float(),
                torch.from_numpy(data.x_cat_train).long()]
    if conditioned:
        _tensors.append(torch.from_numpy(np.asarray(data.cond_train, np.float32)))
    ds = TensorDataset(*_tensors)
    gen_cpu = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=0, generator=gen_cpu)
    if len(loader) == 0:
        # drop_last=True with batch_size > n_train yields NO batches at all, and the run
        # then completes normally: the loss accumulators stay empty, np.mean([]) prints
        # nan, and an untouched network is written out as "best".  A smoke run at
        # --subsample 4000 against the 4096 default hit exactly that.
        raise SystemExit(f"[fatal] batch_size={args.batch_size} exceeds the "
                         f"{len(ds):,}-row training split, so drop_last leaves no "
                         f"batches; lower --batch-size or raise --subsample")
    total_steps = max(len(loader) * args.epochs, 1)

    history: List[Dict[str, Any]] = []
    best = {"score": float("inf"), "epoch": -1, "metrics": {}}
    best_path = os.path.join(save_dir, "tabddpm_best.pt")
    last_path = os.path.join(save_dir, "tabddpm_last.pt")
    metric_backend = "n/a"
    t0 = time.time()
    step = 0

    def build_ckpt(epoch: int, metrics: Dict[str, Any]) -> Dict[str, Any]:
        # F5: full effective config + preprocessing objects travel with the model
        return {
            "model_state": (diffusion if diff_ema is None else diff_ema).state_dict(),
            "config": vars(args),
            "schema": {"num_dim": data.schema.num_dim, "cat_dims": list(data.cat_dims)},
            "num_cols": data.num_cols,
            "cat_cols": data.cat_cols,
            "cat_dims": data.cat_dims,
            "num_transform": args.num_transform,
            "num_transformer": data.num_transformer,
            "label_encoders": data.label_encoders,
            "binning_spec": data.spec.to_dict(),
            "integer_cols": data.integer_cols,
            "epoch": epoch,
            "val_metrics": metrics,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        }

    # --- EMA of the diffusion weights ------------------------------------
    # Added for FAIRNESS, not to flatter this baseline.  EMA is standard practice
    # for diffusion models -- omitting it here while SemaPop's arms use it (worth
    # 27.7% on SRMSE-1, more than the semantic conditioning contributes) would be
    # unfavourable preprocessing or insufficient tuning of the benchmark.
    diff_ema = None
    if getattr(args, "ema_decay", 0.0) > 0.0:
        diff_ema = copy.deepcopy(diffusion).eval()
        for _p in diff_ema.parameters():
            _p.requires_grad_(False)

    @torch.no_grad()
    def ema_update() -> None:
        if diff_ema is None:
            return
        d = args.ema_decay
        for pe, pg in zip(diff_ema.parameters(), diffusion.parameters()):
            pe.mul_(d).add_(pg.detach(), alpha=1.0 - d)
        for be, bg in zip(diff_ema.buffers(), diffusion.buffers()):
            be.copy_(bg)

    def eval_model():
        return diffusion if diff_ema is None else diff_ema

    def sample_df(n: int, seed: int) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        g = torch.Generator(device=dev).manual_seed(int(seed))
        c = None
        if conditioned:
            # stays on the HOST; sample_batched moves it one batch at a time
            c = torch.from_numpy(
                draw_cond_ddpm(args.cond_mode, n, data.cond_train, data.cond_val,
                               prior_obj, int(seed))).float()
        xn, xc = eval_model().sample_batched(n, cond=c, batch_size=args.sample_batch,
                                             generator=g)
        xn = xn.numpy()
        df, raw = decode(xn, xc.numpy(), data)
        return df, raw, xn

    def evaluate(epoch: int) -> Dict[str, Any]:
        nonlocal metric_backend
        n = min(len(data.df_val), args.eval_n) if args.eval_n > 0 else len(data.df_val)
        df_gen, _, _ = sample_df(n, args.sample_seed + epoch)
        m, backend = metrics_of(df_gen)
        metric_backend = backend
        if args.select_on == "ml_eff":
            # Kotelnikov et al. select on downstream ML efficiency; the surrogate target
            # panel is pre-registered in utils/ml_efficacy.py.  ~7 s per call, and their
            # protocol does NO in-training selection anyway -- a trial should run with
            # --load-final and read the five-seed block written after training.
            from utils.ml_efficacy import ml_efficiency
            m.update({k: v for k, v in ml_efficiency(df_gen, data.df_val).items()
                      if k != "ml_eff_targets"})
        return m

    # --generate-only: skip training entirely and go straight to the checkpoint the
    # run already wrote.  Added because national runs died AT GENERATION with the
    # model fully trained -- retraining to recover a sampling bug would be waste.
    for ep in range(1, (0 if args.generate_only else args.epochs) + 1):
        diffusion.train()
        lm, lg, kl_raw = [], [], []
        for batch in loader:
            x_num, x_cat = batch[0], batch[1]
            x_num = x_num.to(dev, non_blocking=True)
            x_cat = x_cat.to(dev, non_blocking=True)
            c_real = batch[2].to(dev, non_blocking=True) if conditioned else None
            c_g = c_real
            if conditioned and getattr(args, "cond_jitter", 0.0) > 0.0:
                c_g = c_real + args.cond_jitter * cond_scale * torch.randn_like(c_real)
            if args.anneal_lr:
                for pg in opt.param_groups:
                    pg["lr"] = args.lr * (1.0 - step / total_steps)
            want_parts = conditioned and (getattr(args, "lambda_m", 0.0) > 0.0
                                          or getattr(args, "lambda_m_prior", 0.0) > 0.0)
            out = diffusion.mixed_loss(x_num, x_cat, cond=c_g,
                                       time_method=args.time_sampler,
                                       return_parts=want_parts)
            loss = args.lambda_multi * out["loss_multi"] + args.lambda_gauss * out["loss_gauss"]
            if want_parts:
                loss = loss + marginal_terms(diffusion, out, c_real, args, marg_cat, dev, ep)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), args.grad_clip)
            opt.step()
            ema_update()
            step += 1
            lm.append(float(out["loss_multi"]))
            lg.append(float(out["loss_gauss"]))
            kl_raw.append(float(out["kl_multi"]))

        rec: Dict[str, Any] = {
            "epoch": ep,
            "loss_multi": float(np.mean(lm)),
            "loss_gauss": float(np.mean(lg)),
            "kl_multi_raw": float(np.mean(kl_raw)),
            "elapsed_s": round(time.time() - t0, 2),
        }
        rec["loss_total"] = args.lambda_multi * rec["loss_multi"] \
            + args.lambda_gauss * rec["loss_gauss"]

        if ep % args.eval_every == 0 or ep == args.epochs or ep == 1:
            m = evaluate(ep)
            s = score_of(m, args.f1_weight, args.select_on)
            rec.update({f"val_{k}": v for k, v in m.items()})
            rec["val_score"] = s
            if s < best["score"]:
                best = {"score": s, "epoch": ep, "metrics": m}
                torch.save(build_ckpt(ep, m), best_path)
                rec["is_best"] = True
            print(f"[ep {ep:04d}] multi={rec['loss_multi']:.4f} "
                  f"gauss={rec['loss_gauss']:.4f} kl={rec['kl_multi_raw']:.4f} | "
                  f"val_srmse1={m.get('srmse_1', float('nan')):.4f} "
                  f"val_srmse2={m.get('srmse_2', float('nan')):.4f} score={s:.4f}"
                  + ("  <- best" if rec.get("is_best") else ""), flush=True)
        else:
            print(f"[ep {ep:04d}] multi={rec['loss_multi']:.4f} "
                  f"gauss={rec['loss_gauss']:.4f} kl={rec['kl_multi_raw']:.4f}",
                  flush=True)
        history.append(rec)
        torch.save(build_ckpt(ep, rec), last_path)

    # ---- F4: LOAD THE BEST CHECKPOINT --------------------------------------
    load_path = best_path if (os.path.exists(best_path) and not args.load_final) else last_path
    ck = torch.load(load_path, map_location=dev, weights_only=False)
    diffusion.load_state_dict(ck["model_state"])
    if diff_ema is not None:
        # build_ckpt stores the EMA weights, and sample_df draws from eval_model(), which
        # returns diff_ema -- so without this line the checkpoint just loaded never reaches
        # sampling at all.  A no-op for a run that trained in this process AND passed
        # --load-final (diff_ema already holds exactly these weights, which is why every
        # published ddpm row is unaffected); load-bearing for --generate-only, and it also
        # restores the intended contract for a run that selects an EARLIER best epoch.
        diff_ema.load_state_dict(ck["model_state"])
    if args.generate_only:
        best = {"score": float("nan"), "epoch": ck.get("epoch", -1),
                "metrics": ck.get("val_metrics", {})}
    print(f"[select] loaded {os.path.basename(load_path)} from epoch {ck['epoch']} "
          f"(best score={best['score']:.4f} @ epoch {best['epoch']})")

    # ---- final sample ------------------------------------------------------
    n_out = args.n_samples if args.n_samples > 0 else len(data.df_val)
    df_out, raw_num, scaled_num = sample_df(n_out, args.sample_seed)
    sample_csv = os.path.join(save_dir, "generated_samples.csv")
    df_out.to_csv(sample_csv, index=False)
    final_m, backend = metrics_of(df_out)
    if args.select_on == "ml_eff":
        # Kotelnikov et al. average the criterion over FIVE sampling seeds at the trained
        # checkpoint, which is what removes the draw noise from a metric whose classifier
        # is refit each time.  Also closes a hole in hpo/objective.objective_from_run's
        # fallback chain: without this the middle rung (final_val_metrics) cannot compute
        # ml_eff and every trial that loses its tail block would fall through to
        # selection.best_metrics, a best-of-N block carrying a ~2-sigma optimistic bias.
        from utils.ml_efficacy import ml_efficiency_multiseed
        final_m.update({k: v for k, v in ml_efficiency_multiseed(
            lambda sd: sample_df(n_out, args.sample_seed + 1000 + sd)[0],
            data.df_val, n_seeds=5).items() if k != "ml_eff_targets"})

    oos_pre, oos_post, diverged = {}, {}, {}
    for i, c in enumerate(data.num_cols):
        lo, hi = data.spec.support(c)
        oos_pre[c] = float(np.mean((raw_num[:, i] < lo - 1e-9) | (raw_num[:, i] > hi + 1e-9)))
        v = df_out[c].values.astype(np.float64)
        oos_post[c] = float(np.mean((v < lo - 1e-9) | (v > hi + 1e-9)))
        # divergence of the Gaussian reverse chain, measured in the SCALED space.
        # This is the honest diagnostic: with --num-transform quantile the
        # inverse transform already saturates at the fitted quantile range, so
        # `out_of_support_rate_before_clip` is 0 by construction and hides the
        # exploded rows (they pile up on the support endpoints instead).
        diverged[c] = float(np.mean(np.abs(scaled_num[:, i]) > args.diverged_z))

    metrics = {
        "run": {
            "save_dir": save_dir,
            "started": datetime.now().isoformat(timespec="seconds"),
            "wall_clock_s": round(time.time() - t0, 2),
            "device": args.device,
            "torch_version": torch.__version__,
            "n_params": int(n_params),
        },
        "model": {
            "family": "TabDDPM",
            "reference": "Kotelnikov et al., ICML 2023; "
                         "github.com/yandex-research/tab-ddpm",
            "numeric_process": "gaussian_diffusion",
            "categorical_process": "multinomial_diffusion",
            "timesteps": args.timesteps,
            "schedule": args.schedule,
            "parametrization": args.parametrization,
            "gaussian_parametrization": args.gaussian_parametrization,
            "gaussian_loss_type": args.gaussian_loss,
            "clip_x0": args.clip_x0,
            "conditional": bool(args.cond_dim > 0),
            "cat_input": args.cat_input,
        },
        "config": vars(args),
        "data": {
            "n_train": int(len(data.df_train)),
            "n_val": int(len(data.df_val)),
            "num_cols": data.num_cols,
            "cat_cols": data.cat_cols,
            "cat_dims": data.cat_dims,
            "data_dim": int(data.schema.data_dim),
            "held_out_val": True,                       # F2
            "n_val_unseen_cells": int(data.n_val_unseen_cells),
            "num_transform": args.num_transform,
        },
        "binning": {c: {"kind": data.spec[c].kind, "n_bins": data.spec[c].n_bins,
                        "support": list(data.spec.support(c))}
                    for c in data.num_cols},
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
        # The HPO objective reads this in preference to final_val_metrics; see
        # utils.eval_v2.tail_metrics for why a single last evaluation is the
        # noisier choice, and why every iterative trainer must emit it.
        "tail_val_metrics": ev.tail_metrics(history, k=int(args.tail_k)),
        "tail_k": int(args.tail_k),
        "final_val_metrics": final_m,
        "final_metric_backend": backend,
        "out_of_support_rate_before_clip": oos_pre,
        "out_of_support_rate_after_clip": oos_post,
        "numeric_chain_divergence_rate": diverged,
        "numeric_chain_divergence_z": args.diverged_z,
        "numeric_scaled_abs_max": {c: float(np.abs(scaled_num[:, i]).max())
                                   for i, c in enumerate(data.num_cols)},
        "history": history,
        "outputs": {"samples_csv": sample_csv, "best_ckpt": best_path,
                    "last_ckpt": last_path},
    }
    with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"[done] samples -> {sample_csv}")
    print(f"[done] metrics -> {os.path.join(save_dir, 'metrics.json')}")

    # ---- K=50 audit ensemble (v3 protocol, utils/ensemble_v2) -----------------
    # Finalized runs only (HPO trials never pass --ensemble).  Runs AFTER
    # metrics.json so a crash mid-ensemble cannot cost the training record.
    # sample_df is a deterministic function of its seed argument: every torch
    # draw inside GaussianMultinomialDiffusion.sample flows through the explicit
    # torch.Generator(device).manual_seed(seed) that sample_df builds, and a
    # conditioned run's draw_cond builds np.random.default_rng(seed) itself.
    if getattr(args, "ensemble", False):
        expected_cols = list(data.num_cols) + list(data.cat_cols)

        def ensemble_sample_fn(seed: int, n: int) -> pd.DataFrame:
            df_e = sample_df(n, seed)[0]
            if list(df_e.columns) != expected_cols:
                raise RuntimeError(
                    f"ensemble frame columns {list(df_e.columns)} != modelled "
                    f"columns (num then cat) {expected_cols}")
            return df_e

        n_ens = int(args.ensemble_n) if args.ensemble_n else _reference_len(args)
        write_ensemble(ensemble_sample_fn, n_samples=n_ens, save_dir=save_dir,
                       k=args.ensemble_k)
    return metrics


def _reference_len(args: argparse.Namespace) -> int:
    """N for the audit ensemble when --ensemble-n is 0: len(reference population).

    Only consulted at ensemble time (finalized runs) -- HPO trials never pass
    --ensemble, so the reference split stays unopened during a trial.  Row count
    via a one-column parse: the CSVs carry free-text persona/prompt columns with
    embedded newlines, so raw line counting would be wrong.
    """
    for fname in ("referencedata_pers.csv", "testdata_pers.csv"):
        path = os.path.join(args.data_dir, fname)
        if os.path.exists(path):
            n = int(len(pd.read_csv(path, usecols=[0])))
            print(f"[ensemble] N = len({fname}) = {n}")
            return n
    raise FileNotFoundError(
        f"--ensemble with --ensemble-n 0 needs referencedata_pers.csv or "
        f"testdata_pers.csv under {args.data_dir}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def resample_from_ckpt(ck, n: int, dev, batch: int = 8192, seed: int = 0):
    """Regenerate n rows from a saved checkpoint, without retraining.

    Needed because hpo/finalize.py emits |val| = 10,001 rows while the E1 headline
    table is specified at N = |reference|; block precision/recall and unique-tuple
    counts are not comparable across those two sizes, so the table has to be produced
    at the size it claims.  Everything decode() needs is already in the checkpoint.
    """
    import types
    c = ck["config"]
    sch = ck["schema"]
    schema = sch if hasattr(sch, "num_dim") else TabularSchema(sch["num_dim"], list(sch["cat_dims"]))
    denoiser = TabDenoiserV2(
        num_dim=schema.num_dim,
        cat_dims=ck["cat_dims"],
        hidden=tuple(c["hidden"]),
        dim_t=c["dim_t"],
        dropout=c["dropout"],
        cond_dim=c.get("cond_dim", 0),
        cat_input=c.get("cat_input", "onehot"),
    ).to(dev)
    diffusion = GaussianMultinomialDiffusion(
        denoise_fn=denoiser, num_numerical_features=schema.num_dim,
        num_classes=ck["cat_dims"], num_timesteps=c["timesteps"],
        scheduler=c["schedule"], gaussian_loss_type=c["gaussian_loss"],
        gaussian_parametrization=c["gaussian_parametrization"],
        parametrization=c["parametrization"], clip_x0=c.get("clip_x0", 5.0),
    ).to(dev)
    diffusion.load_state_dict(ck["model_state"]); diffusion.eval()
    bs = ck.get("binning_spec")
    spec = bs if hasattr(bs, "support") else BinningSpec.from_dict(bs)
    data = types.SimpleNamespace(
        num_cols=ck["num_cols"], cat_cols=ck["cat_cols"], cat_dims=ck["cat_dims"],
        schema=schema, num_transformer=ck["num_transformer"],
        label_encoders=ck["label_encoders"], spec=spec,
        integer_cols=ck["integer_cols"])
    g = torch.Generator(device=dev).manual_seed(int(seed))
    xn, xc = diffusion.sample_batched(n, cond=None, batch_size=batch, generator=g)
    df, _ = decode(xn.numpy(), xc.numpy(), data)
    return df


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TabDDPM v2 baseline trainer (faithful, D13)")
    # data
    p.add_argument("--data-dir", default=_PA.S("/data/pums_states/CA"))
    p.add_argument("--cols-config", default="cols_config.yaml")
    p.add_argument("--train-file", default="traindata_pers.csv")
    p.add_argument("--val-file", default="valdata_pers.csv")
    p.add_argument("--num-cols", nargs="*", default=None)
    p.add_argument("--cat-cols", nargs="*", default=None)
    p.add_argument("--subsample", type=int, default=0, help="subsample train (smoke runs)")
    p.add_argument("--subsample-val", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--num-transform", choices=["quantile", "standard"], default="quantile",
                   help="quantile = TabDDPM's default normalisation")
    p.add_argument("--bw-scale", type=float, default=0.35,
                   help="binning bandwidth scale (support/rounding spec only)")
    # diffusion
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--schedule", choices=["cosine", "linear"], default="cosine")
    p.add_argument("--parametrization", choices=["x0", "direct"], default="x0",
                   help="multinomial reverse parametrization")
    p.add_argument("--gaussian-parametrization", choices=["eps", "x0"], default="eps")
    p.add_argument("--gaussian-loss", choices=["mse", "kl"], default="mse")
    # D21 (2026-08-19): default 5.0, NOT the paper-faithful 0.0.  The strictly official
    # settings are numerically unstable on this data -- the numeric reverse chain diverges
    # (|z|>5 rate 0.41/0.44/0.71 for Age/HH_Size/Nchildren) and F6 clipping hides it because
    # the quantile transform runs before the clip.  Clipping the x0 prediction inside the
    # reverse chain is standard in TabDDPM reproductions and removes the divergence entirely
    # (SRMSE-1 0.662 -> 0.149, |z|>5 rate 0 / 0 / 0).  The strict-default row stays in the
    # appendix; pass --clip-x0 0 to reproduce it.
    p.add_argument("--clip-x0", type=float, default=5.0,
                   help="clamp the reconstructed x0 of the NUMERIC block to "
                        "[-v, v] in the scaled space during sampling. 0 = off = "
                        "faithful to the official TabDDPM; set e.g. 5 only if the "
                        "numeric reverse chain diverges (recorded in metrics.json)")
    p.add_argument("--time-sampler", choices=["uniform", "importance"], default="uniform")
    p.add_argument("--lambda-multi", type=float, default=1.0)
    p.add_argument("--lambda-gauss", type=float, default=1.0)
    # denoiser
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 1024, 1024, 512])
    p.add_argument("--dim-t", type=int, default=128)
    p.add_argument("--cat-input", choices=["logprob", "onehot"], default="logprob",
                   help="how the categorical part of x_t is fed to the MLP. "
                        "'logprob' = official TabDDPM (log one-hot, values in "
                        "{0, -69}); 'onehot' = exp of it. Same information, same "
                        "diffusion; 'onehot' is far better conditioned")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--cond-dim", type=int, default=0,
                   help="FiLM conditioning width (0 = unconditional TabDDPM baseline; "
                        ">0 reserved for SemaPop-DDPM)")
    # optimisation
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--anneal-lr", action="store_true",
                   help="linear lr decay to 0 over training (TabDDPM's Trainer)")
    # eval / selection
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-n", type=int, default=0, help="0 = use the whole val split")
    p.add_argument("--no-2way", action="store_true")
    p.add_argument("--eval-config", default=DEFAULT_EVAL_CONFIG,
                   help="frozen utils.eval_v2 EvalConfig yaml "
                        "(configs/eval_v2_pums20.yaml for paper runs)")
    p.add_argument("--f1-weight", type=float, default=1.0)
    p.add_argument("--load-final", action="store_true")
    p.add_argument("--select-on", default="val_srmse",
        choices=["val_srmse", "if1", "ml_eff"],
        help="hyperparameter-selection criterion (utils/criteria.py): each model is\n"
             "selected the way its own authors select, not by one shared metric")
    # sampling
    p.add_argument("--diverged-z", type=float, default=5.0,
                   help="|scaled value| above which a sampled numeric is counted as a "
                        "diverged reverse chain in metrics.json")
    p.add_argument("--n-samples", type=int, default=0, help="0 = size of the val split")
    p.add_argument("--sample-batch", type=int, default=4096)
    p.add_argument("--generate-only", action="store_true",
                   help="skip training; load the checkpoint already in the run dir "
                        "and only re-run sampling/metrics")
    add_ensemble_args(p)   # --ensemble / --ensemble-k / --ensemble-n (audit contract)
    # reproducibility / io
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="EMA decay for the diffusion weights used at eval and "
                        "checkpoint time; 0 disables.  Present so this baseline "
                        "gets the same generic practice SemaPop uses.")
    p.add_argument("--tail-k", type=int, default=5,
                   help="trailing in-training evaluations averaged into "
                        "`tail_val_metrics`, the quantity HPO tunes on.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-seed", type=int, default=1234)
    p.add_argument("--no-deterministic", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-root", default=_PA.S("/modelSave"))
    p.add_argument("--tag", default="tabddpm_v2")
    p.add_argument("--run-name", default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    run_name = args.run_name or (
        f"{args.tag}_T{args.timesteps}_s{args.seed}_"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    save_dir = os.path.join(args.out_root, run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[run] {save_dir}")

    set_torch_seed(args.seed, deterministic=not args.no_deterministic)
    data = prepare_data(args)
    return train(args, data, save_dir)


if __name__ == "__main__":
    main()
