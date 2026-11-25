#!/usr/bin/env python
"""
train_wgan_zcr_v2.py
====================
Clean v2 trainer for the **WGAN-GP-ZCR baseline** (Kim & Bansal 2023,
Transportation Research Part C 148, 104053).  A WGAN-GP with higher recall but
lower precision than WGAN-GP-ZCR, which contradicts the theoretical definition of
the regularisation term, is a symptom of the old implementation, not of the
method: in ``train_wgan.py:20-54`` the BD term is a **global** minimum over an
elementwise ``|difference|`` matrix whose categorical columns are integer
labels drawn with ``torch.multinomial``, so it is identically ``0.0`` with no
gradient, and the categorical logits receive no gradient at all.  ZCR was
therefore numerically almost the same model as plain WGAN-GP.

``model/zcr.py`` documents the defect and the fix in full; ``--audit-legacy``
recomputes the old terms on the same batch and writes both into
``metrics.json`` so the difference can be stated with numbers.

The trainer itself follows the conventions of ``train_semapop_gan_v2.py``
(which is the reference for CLI shape, seeding, checkpointing and
post-processing) with the SemaPop-specific parts removed -- this is an
**unconditional** baseline: no persona conditioning, no marginal regulariser:

F2  validation is held out.  Label encoders, the ``ColumnTransformer`` and the
    binning spec are fit on **train only**; val is used for model selection and
    diagnostics.
F4  the checkpoint that is loaded at the end is the one selected on the val
    split with the frozen ``utils.eval_v2`` protocol (SRMSE-1/2 + block P/R),
    never on the critic gap and never on the reference split.
F5  ``--seed`` seeds python/numpy/torch/cuda/cudnn, ``--sample-seed`` is
    separate, the full effective config + preprocessor + label encoders +
    binning spec go into every checkpoint, and every run writes
    ``metrics.json``.
F6  generated numeric values are inverse transformed, **clipped to the training
    support**, and only then rounded (integer columns only).  The
    out-of-support rate before and after clipping is reported.
F7  ZCR is a row-wise nearest-neighbour regulariser in a differentiable space
    (see ``model/zcr.py``).

Ablation switches for E1: ``--no-zcr`` gives the plain WGAN-GP row of Table 1
from the same code path (identical architecture, seeding and post-processing),
``--zcr-only-ad`` drops the BD term (the old ``if_zc_only_ad`` variant).

Example
-------
    python train_wgan_zcr_v2.py \
        --data-dir <semapop_repo>/data/pums_v2/CA \
        --eval-config configs/eval_v2_pums20.yaml \
        --tag wganzcr_ca --device cuda:0 --epochs 300 --seed 42
"""

from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[0])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.wgan import (  # noqa: E402
    TabularSchema,
    WGANCritic,
    WGANGenerator,
    gradient_penalty,
)
from model.zcr import (  # noqa: E402
    UncondSimplexGenerator,
    legacy_zcr_terms,
    sample_bank,
    zcr_terms,
)
from utils.ensemble_v2 import add_ensemble_args, write_ensemble  # noqa: E402
from utils.marg_loss_v2 import build_binning_spec  # noqa: E402

# The v2 SemaPop trainer is the reference implementation for the shared
# conventions (F5 seeding, F6 decoding, F4 selection metrics).  Importing keeps
# the two trainers bit-for-bit consistent instead of forking the code.
from train_semapop_gan_v2 import (  # noqa: E402

    DEFAULT_TAIL_MERGE,
    _build_preprocessor,
    _encode_cats,
    _fit_label_encoders,
    _try_eval_v2,
    decode_processed,
    make_eval_config,
    selection_score,
    set_torch_seed,
    val_metrics,
)

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
#: the frozen paper protocol; every paper run must use it.
DEFAULT_EVAL_CONFIG = os.path.join(_REPO_ROOT, "configs", "eval_v3_pums12.yaml")

#: ZCR weights.  Kim & Bansal (2023) report the regulariser weights as tuned
#: hyper-parameters; the published values could not be verified when this
#: trainer was written, so the defaults below were the repository's own
#: (bd, ad) = (1.5, 0.5) convention.  An earlier in-house implementation
#: (``train_wgan.py``, not shipped) used (bd 1.0, ad 0.2).  Both are only
#: starting points: the HPO tunes ``--bd-weight`` / ``--ad-weight`` on the val split.
# Kim & Bansal (2023) Fig. 9, DISCRETE space, both regularisers active:
# gamma_BD = 0.5, gamma_AD = 0.005.  Our previous defaults were 1.5 and 0.5, i.e.
# gamma_AD was 100x the published value.  AD pushes generated rows AWAY from the
# training data, so a 100x overweight trades feasibility for diversity -- which is
# the very symptom of ZCR failing to raise precision, compounding the
# degenerate BD term documented in model/zcr.py.
# The paper's own weights differ between the individual and the joint calibration
# (e.g. embedded gamma_BD is 1.0 alone but 0.75 when paired); we always run both, so
# the joint pair is the right one.  These are the paper's DATA, though, not ours --
# HPO recalibrates on PUMS, and its ranges are widened to contain these values so the
# published setting stays reachable.
DEFAULT_BD_WEIGHT = 0.5
DEFAULT_AD_WEIGHT = 0.005


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@dataclass
class PreparedData:
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    x_train: np.ndarray                 # [N, D] = [num z-scored | one-hot cats]
    x_val: np.ndarray
    legacy_bank: np.ndarray             # [N, num_dim + n_cat] audit only
    num_cols: List[str]
    cat_cols: List[str]
    cat_dims: List[int]
    schema: TabularSchema
    preprocessor: ColumnTransformer
    label_encoders: Dict[str, LabelEncoder]
    num_mean: np.ndarray
    num_std: np.ndarray
    n_val_unseen_cells: int


def prepare_data(args: argparse.Namespace) -> PreparedData:
    d = args.data_dir
    with open(os.path.join(d, getattr(args, "cols_config", "cols_config.yaml")), encoding="utf-8") as f:
        cols = yaml.safe_load(f)
    num_cols = list(args.num_cols) if args.num_cols else list(cols["num_cols"])
    cat_cols = list(args.cat_cols) if args.cat_cols else list(cols["cat_cols"])

    df_train = pd.read_csv(os.path.join(d, args.train_file), usecols=num_cols + cat_cols)
    df_val = pd.read_csv(os.path.join(d, args.val_file), usecols=num_cols + cat_cols)

    if args.subsample and args.subsample < len(df_train):
        rng = np.random.default_rng(args.data_seed)
        keep = np.sort(rng.choice(len(df_train), size=args.subsample, replace=False))
        df_train = df_train.iloc[keep].reset_index(drop=True)
    if args.subsample_val and args.subsample_val < len(df_val):
        rng = np.random.default_rng(args.data_seed + 1)
        keep = np.sort(rng.choice(len(df_val), size=args.subsample_val, replace=False))
        df_val = df_val.iloc[keep].reset_index(drop=True)

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

    # audit-only bank: [num z-scored | integer labels], i.e. the old train_set_x
    legacy_bank = np.concatenate(
        [x_train[:, : len(num_cols)],
         tr_enc[cat_cols].to_numpy().astype(np.float32)], axis=1
    )

    print(f"[data] train={x_train.shape} val={x_val.shape} cat_dims={cat_dims}")
    return PreparedData(df_train, df_val, x_train, x_val, legacy_bank,
                        num_cols, cat_cols, cat_dims, schema, pre, les,
                        num_mean, num_std, int(n_unseen))


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
class seeded_rng:
    """Sample with a fixed seed **without** perturbing the training RNG stream.

    ``F.gumbel_softmax`` / ``torch.multinomial`` draw from the global generator
    and take no ``generator=`` argument, so reproducible sampling has to reseed
    it; saving and restoring the state around the block keeps the number of
    evaluations from changing the training trajectory (F5).
    """

    def __init__(self, seed: int, device: torch.device) -> None:
        self.seed = int(seed)
        self.is_cuda = device.type == "cuda"

    def __enter__(self) -> "seeded_rng":
        self._cpu = torch.get_rng_state()
        self._cuda = torch.cuda.get_rng_state_all() if self.is_cuda else None
        torch.manual_seed(self.seed)
        if self.is_cuda:
            torch.cuda.manual_seed_all(self.seed)
        return self

    def __exit__(self, *exc) -> bool:
        torch.set_rng_state(self._cpu)
        if self._cuda is not None:
            torch.cuda.set_rng_state_all(self._cuda)
        return False


@torch.no_grad()
def sample_processed(
    gen: UncondSimplexGenerator,
    n: int,
    device: torch.device,
    batch_size: int = 4096,
    cat_mode: str = "argmax",
    temperature: float = 1.0,
) -> np.ndarray:
    was_training = gen.training
    gen.eval()
    outs = []
    remain = int(n)
    while remain > 0:
        b = min(batch_size, remain)
        z = torch.randn(b, gen.noise_dim, device=device)
        outs.append(gen(z, mode=cat_mode, temperature=temperature).detach().cpu().numpy())
        remain -= b
    if was_training:
        gen.train()
    return np.concatenate(outs, axis=0)


# --------------------------------------------------------------------------- #
# F7 audit: old vs new regulariser on one batch
# --------------------------------------------------------------------------- #
def _safe_grad(loss: torch.Tensor, inputs) -> Any:
    """``torch.autograd.grad`` that tolerates a loss detached from the graph.

    The legacy BD term is a constant, so asking autograd for its gradient
    raises ``RuntimeError: element 0 of tensors does not require grad``.  That
    *is* the finding, so it must be reported rather than crash the run.
    """
    if not loss.requires_grad:
        return None if isinstance(inputs, torch.Tensor) else [None] * len(list(inputs))
    single = isinstance(inputs, torch.Tensor)
    g = torch.autograd.grad(loss, [inputs] if single else list(inputs),
                            retain_graph=True, allow_unused=True)
    return g[0] if single else list(g)


def _param_grad_norm(grads) -> float:
    live = [g for g in (grads or []) if g is not None]
    if not live:
        return 0.0
    return float(torch.sqrt(sum((g ** 2).sum() for g in live)).item())


def zcr_audit(
    gen: UncondSimplexGenerator,
    bank: torch.Tensor,
    legacy_bank: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    """Compute both regularisers on the same noise batch and report gradients.

    The point of the audit is the *gradient*, not the value: the old BD term is
    a constant 0 with a zero (or absent) gradient, and the old AD term has a
    gradient only on the ``num_dim`` numeric columns because the categorical
    part went through ``torch.multinomial``.
    """
    b = int(min(args.audit_batch, args.batch_size))
    m = int(min(args.audit_neighbors, bank.shape[0]))
    z = torch.randn(b, gen.noise_dim, device=device)

    out: Dict[str, Any] = {
        "batch": b, "n_neighbours": m,
        "note": "grad_* are ||dL/dx_gen||_2 over the generator OUTPUT tensor; "
                "grad_*_cat_part is the norm restricted to the categorical columns",
    }

    # ---- new (row-wise, differentiable simplex space) -----------------------
    x_fake = gen(z)
    x_fake.retain_grad()
    sub = bank[:m]
    bd, ad = zcr_terms(x_fake, sub, num_dim=gen.num_dim,
                       cat_scale=args.zcr_cat_scale, chunk=args.zcr_chunk,
                       ad_clamp=args.ad_clamp)
    g_bd = _safe_grad(bd, x_fake)
    g_ad = _safe_grad(ad, x_fake)
    nd = gen.num_dim
    out["new"] = {
        "bd": float(bd.item()),
        "ad": float(ad.item()),
        "grad_bd_norm": float(g_bd.norm().item()) if g_bd is not None else 0.0,
        "grad_ad_norm": float(g_ad.norm().item()) if g_ad is not None else 0.0,
        "grad_bd_cat_part": float(g_bd[:, nd:].norm().item()) if g_bd is not None else 0.0,
        "grad_ad_cat_part": float(g_ad[:, nd:].norm().item()) if g_ad is not None else 0.0,
    }
    # gradient that actually reaches the generator parameters
    params = list(gen.parameters())
    out["new"]["grad_bd_param_norm"] = _param_grad_norm(_safe_grad(bd, params))
    out["new"]["grad_ad_param_norm"] = _param_grad_norm(_safe_grad(ad, params))

    # ---- old (train_wgan.py:20-54) -----------------------------------------
    raw = gen.raw(z)
    raw.retain_grad()
    lbd, lad = legacy_zcr_terms(raw, legacy_bank[:m], gen.num_dim, gen.cat_dims)
    lg_bd = _safe_grad(lbd, raw)
    lg_ad = _safe_grad(lad, raw)
    lparams = list(gen.parameters())
    out["legacy"] = {
        "bd": float(lbd.item()),
        "ad": float(lad.item()),
        "grad_bd_norm": float(lg_bd.norm().item()) if lg_bd is not None else 0.0,
        "grad_ad_norm": float(lg_ad.norm().item()) if lg_ad is not None else 0.0,
        "grad_bd_cat_part": float(lg_bd[:, nd:].norm().item()) if lg_bd is not None else 0.0,
        "grad_ad_cat_part": float(lg_ad[:, nd:].norm().item()) if lg_ad is not None else 0.0,
        "grad_bd_param_norm": _param_grad_norm(_safe_grad(lbd, lparams)),
        "grad_ad_param_norm": _param_grad_norm(_safe_grad(lad, lparams)),
        "bd_is_identically_zero": bool(abs(float(lbd.item())) < 1e-12),
        "bd_requires_grad": bool(lbd.requires_grad),
    }
    print("[audit] new    BD={:.6f} (|grad|={:.4e}, cat part {:.4e})  "
          "AD={:.6f} (|grad|={:.4e}, cat part {:.4e})".format(
              out["new"]["bd"], out["new"]["grad_bd_norm"], out["new"]["grad_bd_cat_part"],
              out["new"]["ad"], out["new"]["grad_ad_norm"], out["new"]["grad_ad_cat_part"]))
    print("[audit] legacy BD={:.6f} (|grad|={:.4e}, cat part {:.4e})  "
          "AD={:.6f} (|grad|={:.4e}, cat part {:.4e})".format(
              out["legacy"]["bd"], out["legacy"]["grad_bd_norm"],
              out["legacy"]["grad_bd_cat_part"],
              out["legacy"]["ad"], out["legacy"]["grad_ad_norm"],
              out["legacy"]["grad_ad_cat_part"]))
    return out


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train(args: argparse.Namespace, data: PreparedData, save_dir: str) -> Dict[str, Any]:
    dev = torch.device(args.device)
    set_torch_seed(args.seed, deterministic=not args.no_deterministic)

    schema = data.schema
    nd = schema.num_dim

    # ---- binning spec (train only): F6 support + selection bins ------------
    tail_merge = {k: v for k, v in DEFAULT_TAIL_MERGE.items() if k in data.num_cols}
    spec = build_binning_spec(
        data.df_train[data.num_cols].to_numpy(dtype=np.float64),
        data.num_cols,
        bw_scale=args.bw_scale,
        tail_merge=tail_merge,
        time_cols=(),          # v2 has no minute-of-day column; disable the name heuristic
    )
    print("[bins] " + ", ".join(f"{c}={spec[c].n_bins}({spec[c].kind})" for c in data.num_cols))

    integer_cols = {
        c: spec[c].kind in ("integer", "integer_tail", "time")
        or bool(np.allclose(data.df_train[c].values, np.rint(data.df_train[c].values)))
        for c in data.num_cols
    }

    # ---- models ------------------------------------------------------------
    base_gen = WGANGenerator(schema.data_dim, noise_dim=args.noise_dim,
                             hidden=list(args.g_hidden), dropout=args.dropout_g,
                             use_tanh=False).to(dev)
    gen = UncondSimplexGenerator(base_gen, nd, data.cat_dims,
                                 cat_mode=args.cat_mode,
                                 temperature=args.gumbel_tau).to(dev)
    critic = WGANCritic(schema.data_dim, hidden=list(args.d_hidden),
                        dropout=args.dropout_d).to(dev)

    if args.cat_mode.startswith("gumbel") and abs(args.sample_temperature - 1.0) > 1e-9:
        raise ValueError(
            f"--sample-temperature {args.sample_temperature} is incompatible with "
            f"--cat-mode {args.cat_mode}: straight-through Gumbel training is "
            "temperature-invariant, so sampling must use temperature 1.0."
        )
    if not getattr(args, "eval_config", None):
        print("[WARN] no --eval-config given: model selection uses an auto-derived "
              "matching space, NOT the frozen paper protocol "
              "(configs/eval_v2_pums20.yaml). Do not use such a run for paper tables.",
              flush=True)

    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(critic.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    ds = TensorDataset(torch.from_numpy(data.x_train))
    gen_cpu = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=0, generator=gen_cpu)

    # ---- ZCR banks ---------------------------------------------------------
    bank = torch.from_numpy(data.x_train).to(dev)                  # [N, D] one-hot space
    legacy_bank = torch.from_numpy(data.legacy_bank).to(dev)       # audit only
    use_zcr = (not args.no_zcr) and (args.bd_weight != 0.0 or args.ad_weight != 0.0)
    want_bd = use_zcr and not args.zcr_only_ad and args.bd_weight != 0.0
    want_ad = use_zcr and args.ad_weight != 0.0
    if use_zcr:
        n_nb = args.zcr_neighbors if args.zcr_neighbors > 0 else bank.shape[0]
        n_nb = min(n_nb, bank.shape[0])
        print(f"[zcr] bd_w={args.bd_weight if want_bd else 0.0} "
              f"ad_w={args.ad_weight if want_ad else 0.0} "
              f"neighbours={n_nb}/{bank.shape[0]} chunk={args.zcr_chunk} "
              f"every={args.zcr_every} cat_scale={args.zcr_cat_scale} "
              f"(bd={'on' if want_bd else 'off'}, ad={'on' if want_ad else 'off'}) "
              f"-> ~{args.batch_size * n_nb * schema.data_dim / 1e9:.2f} GFLOP/step, "
              f"{args.batch_size * min(args.zcr_chunk, n_nb) * 4 / 1e6:.1f} MB/chunk")
    else:
        print("[zcr] disabled -> plain WGAN-GP baseline")

    # ---- F4: evaluation backend -------------------------------------------
    ev = _try_eval_v2()
    eval_cfg = None
    if ev is not None:
        try:
            eval_cfg = (ev.EvalConfig.from_yaml(args.eval_config) if args.eval_config
                        else make_eval_config(data.num_cols, data.cat_cols, spec,
                                              data.df_val, ev))
            if args.eval_config:
                eval_cfg.fit(data.df_val)
            print(f"[eval] utils.eval_v2 config '{eval_cfg.name}' blocks={list(eval_cfg.blocks)}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not build an eval_v2 config ({type(exc).__name__}: {exc}); "
                  "using the local SRMSE fallback")
            eval_cfg = None

    audit: Optional[Dict[str, Any]] = None
    if args.audit_legacy:
        audit = zcr_audit(gen, bank, legacy_bank, args, dev)

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
            "cat_mode": args.cat_mode,
            "gumbel_tau": args.gumbel_tau,
            "epoch": epoch,
            "val_metrics": metrics,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        }

    # --- EMA of the generator weights ------------------------------------
    # Added for FAIRNESS, not as an improvement to this baseline.  SemaPop's arms
    # use EMA and it is worth 27.7% on SRMSE-1 here, which is larger than the
    # semantic conditioning's own contribution (~8%).  Running our arm with it and
    # the baseline without would amount to unfavourable preprocessing or
    # insufficient tuning of the benchmark.
    # EMA is generic optimisation practice for GANs, not a component of our method.
    gen_ema = None
    if getattr(args, "ema_decay", 0.0) > 0.0:
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
        return gen if gen_ema is None else gen_ema

    def evaluate(epoch: int) -> Dict[str, float]:
        nonlocal metric_backend
        n = min(len(data.df_val), args.eval_n) if args.eval_n > 0 else len(data.df_val)
        # F5: the sampling seed is separate from the training seed, and the
        # training RNG stream is restored afterwards.
        with seeded_rng(args.sample_seed + epoch, dev):
            xs = sample_processed(eval_gen(), n, dev, batch_size=args.sample_batch,
                                  cat_mode=gen.sample_mode(args.sample_cat_mode),
                                  temperature=args.sample_temperature)
        df_gen = decode_processed(xs, schema, data.num_cols, data.cat_cols,
                                  data.num_mean, data.num_std, spec, integer_cols,
                                  label_encoders=data.label_encoders)
        m, backend = val_metrics(data.df_val, df_gen, data.num_cols, data.cat_cols, spec,
                                 do_2way=not args.no_2way, ev=ev, eval_cfg=eval_cfg,
                                 with_pr=(args.select_on == "if1"))
        metric_backend = backend
        return m

    g_step = 0
    for ep in range(1, args.epochs + 1):
        gen.train()
        critic.train()
        d_ls, g_adv_ls, gp_ls, bd_ls, ad_ls = [], [], [], [], []

        for (x_real,) in loader:
            x_real = x_real.to(dev, non_blocking=True)
            B = x_real.size(0)

            for _ in range(args.n_critic):
                z = torch.randn(B, args.noise_dim, device=dev)
                with torch.no_grad():
                    x_fake = gen(z)
                d_real = critic(x_real).mean()
                d_fake = critic(x_fake).mean()
                gp = gradient_penalty(critic, x_real, x_fake, dev, lambda_gp=args.lambda_gp)
                loss_d = (d_fake - d_real) + gp
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()
                d_ls.append(float(loss_d.item()))
                gp_ls.append(float(gp.item()))

            z = torch.randn(B, args.noise_dim, device=dev)
            x_fake = gen(z)
            adv = -critic(x_fake).mean()
            loss_g = adv
            if use_zcr and (g_step % max(1, args.zcr_every) == 0):
                sub = sample_bank(bank, args.zcr_neighbors)
                bd, ad = zcr_terms(x_fake, sub, num_dim=nd,
                                   cat_scale=args.zcr_cat_scale, chunk=args.zcr_chunk,
                                   want_bd=want_bd, want_ad=want_ad,
                                   ad_clamp=args.ad_clamp)
                loss_g = loss_g + args.bd_weight * bd + args.ad_weight * ad
                bd_ls.append(float(bd.item()))
                ad_ls.append(float(ad.item()))
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            ema_update()
            g_step += 1
            g_adv_ls.append(float(adv.item()))

        rec: Dict[str, Any] = {
            "epoch": ep,
            "d_loss": float(np.mean(d_ls)),
            "gp": float(np.mean(gp_ls)),
            "g_adv": float(np.mean(g_adv_ls)),
            "zcr_bd": float(np.mean(bd_ls)) if bd_ls else 0.0,
            "zcr_ad": float(np.mean(ad_ls)) if ad_ls else 0.0,
            "elapsed_s": round(time.time() - t0, 2),
        }

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
                  f"G_adv={rec['g_adv']:+.4f} BD={rec['zcr_bd']:.4f} AD={rec['zcr_ad']:+.4f} "
                  f"| val_srmse1={m.get('srmse_1', float('nan')):.4f} "
                  f"val_srmse2={m.get('srmse_2', float('nan')):.4f} score={score:.4f}"
                  + ("  <- best" if rec.get("is_best") else ""))
        else:
            print(f"[ep {ep:04d}] D={rec['d_loss']:+.4f} G_adv={rec['g_adv']:+.4f} "
                  f"BD={rec['zcr_bd']:.4f} AD={rec['zcr_ad']:+.4f}")
        history.append(rec)
        torch.save(build_ckpt(ep, rec), last_path)

    # ---- F4: LOAD THE BEST CHECKPOINT --------------------------------------
    load_path = best_path if (os.path.exists(best_path) and not args.load_final) else last_path
    ck = torch.load(load_path, map_location=dev, weights_only=False)
    gen.load_state_dict(ck["gen_state"])
    critic.load_state_dict(ck["critic_state"])
    print(f"[select] loaded {os.path.basename(load_path)} from epoch {ck['epoch']} "
          f"(best score={best['score']:.4f} @ epoch {best['epoch']})")

    # ---- final sample ------------------------------------------------------
    n_out = args.n_samples if args.n_samples > 0 else len(data.df_val)
    with seeded_rng(args.sample_seed, dev):
        xs = sample_processed(eval_gen(), n_out, dev, batch_size=args.sample_batch,
                              cat_mode=gen.sample_mode(args.sample_cat_mode),
                              temperature=args.sample_temperature)
    df_out = decode_processed(xs, schema, data.num_cols, data.cat_cols,
                              data.num_mean, data.num_std, spec, integer_cols,
                              label_encoders=data.label_encoders)
    sample_csv = os.path.join(save_dir, "generated_samples.csv")
    df_out.to_csv(sample_csv, index=False)

    final_m, backend = val_metrics(data.df_val, df_out, data.num_cols, data.cat_cols,
                                   spec, do_2way=not args.no_2way, ev=ev, eval_cfg=eval_cfg,
                                   with_pr=(args.select_on == "if1"))

    # F6 evidence: out-of-support rate BEFORE clipping (what the generator
    # actually produced) and AFTER clipping (must be 0 -- invariant check).
    raw_num = (xs[:, :nd].astype(np.float64) * data.num_std[None, :] + data.num_mean[None, :])
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
        "model": "wgan_gp_zcr_v2" if use_zcr else "wgan_gp_v2",
        "config": vars(args),
        "data": {
            "n_train": int(len(data.df_train)),
            "n_val": int(len(data.df_val)),
            "num_cols": data.num_cols,
            "cat_cols": data.cat_cols,
            "cat_dims": data.cat_dims,
            "data_dim": int(schema.data_dim),
            "held_out_val": True,          # F2
            "n_val_unseen_cells": data.n_val_unseen_cells,
        },
        "zcr": {
            "enabled": bool(use_zcr),
            "bd_active": bool(want_bd),
            "ad_active": bool(want_ad),
            "bd_weight": args.bd_weight,
            "ad_weight": args.ad_weight,
            "neighbours": int(min(args.zcr_neighbors, bank.shape[0]))
            if args.zcr_neighbors > 0 else int(bank.shape[0]),
            "chunk": args.zcr_chunk,
            "cat_scale": args.zcr_cat_scale,
            "ad_clamp": args.ad_clamp,
            "space": "[z-scored numeric | per-block simplex]; real rows one-hot",
            "definition": "bd = mean_i min_j ||g_i - r_j||_2 (minimised); "
                          "ad = -mean_i mean_j ||g_i - r_j||_2 (i.e. the mean "
                          "distance is maximised); "
                          "loss_g = -E[C(G(z))] + bd_w*bd + ad_w*ad",
            "reference": "Kim & Bansal (2023), TR-C 148:104053",
            "audit": audit,
        },
        "binning": {c: {"kind": spec[c].kind, "n_bins": spec[c].n_bins,
                        "bw": spec[c].bw, "support": list(spec.support(c))}
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
        "history": history,
        "outputs": {"samples_csv": sample_csv, "best_ckpt": best_path, "last_ckpt": last_path},
    }

    # ---- audit-layer ensemble (v3 protocol) ----------------------------------
    # K seeded realizations of the SAME finalized generator/decoding path as
    # the final sample above; seeds come from audit_config.ENSEMBLE_SEEDS via
    # utils.ensemble_v2 (never re-derived locally).  Finalize-only: HPO trials
    # never pass --ensemble.
    if getattr(args, "ensemble", False):

        def ensemble_sample_fn(seed: int, n: int) -> pd.DataFrame:
            # Contract: the draw is a deterministic function of `seed` alone.
            # seeded_rng reseeds torch (CPU+CUDA) and restores the stream; the
            # numpy global stream is reseeded and restored the same way (the
            # sampling/decoding path is torch + deterministic numpy, so this is
            # belt-and-braces for the ensemble contract).
            np_state = np.random.get_state()
            np.random.seed(seed % (2**32))
            try:
                with seeded_rng(seed, dev):
                    xs_e = sample_processed(
                        eval_gen(), n, dev, batch_size=args.sample_batch,
                        cat_mode=gen.sample_mode(args.sample_cat_mode),
                        temperature=args.sample_temperature)
            finally:
                np.random.set_state(np_state)
            # decode_processed returns exactly num_cols + cat_cols in
            # cols-config order = the 12 modelled columns of cols_config_v3.
            return decode_processed(xs_e, schema, data.num_cols, data.cat_cols,
                                    data.num_mean, data.num_std, spec, integer_cols,
                                    label_encoders=data.label_encoders)

        if args.ensemble_n and args.ensemble_n > 0:
            n_ens = int(args.ensemble_n)
        else:
            ref_path = os.path.join(args.data_dir, args.ref_file or "referencedata_pers.csv")
            n_ens = int(len(pd.read_csv(ref_path, usecols=[data.num_cols[0]])))
            print(f"[ensemble] n per realization = len({args.ref_file}) = {n_ens}")
        ens_paths = write_ensemble(ensemble_sample_fn, n_samples=n_ens,
                                   save_dir=save_dir, k=args.ensemble_k)
        metrics["ensemble"] = {
            "n_samples": n_ens,
            "k_realizations": len(ens_paths),
            "seed_source": "audit_config.ENSEMBLE_SEEDS",
            "dir": os.path.join(save_dir, "generated_ensembles"),
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
    p = argparse.ArgumentParser(description="WGAN-GP-ZCR v2 baseline (Kim & Bansal 2023)")
    # data
    p.add_argument("--data-dir", default=_PA.S("/data/pums_states/CA"))
    p.add_argument("--train-file", default="traindata_pers.csv")
    p.add_argument("--val-file", default="valdata_pers.csv")
    # Default EMPTY so hpo/objective.py's fairness assertion sees that a trial
    # never even named the reference file; the ensemble block (finalize only)
    # falls back to referencedata_pers.csv when unset.
    p.add_argument("--ref-file", default="",
                   help="reference-population csv in --data-dir; opened ONLY to "
                        "size the audit ensemble when --ensemble is given and "
                        "--ensemble-n is 0 (never during HPO trials)")
    p.add_argument("--cols-config", default="cols_config.yaml")
    p.add_argument("--num-cols", nargs="*", default=None, help="override cols_config.yaml")
    p.add_argument("--cat-cols", nargs="*", default=None)
    p.add_argument("--subsample", type=int, default=0, help="subsample the train split (smoke runs)")
    p.add_argument("--subsample-val", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=42)
    # architecture
    p.add_argument("--noise-dim", type=int, default=128)
    p.add_argument("--g-hidden", type=int, nargs="+", default=[256, 512, 256])
    p.add_argument("--d-hidden", type=int, nargs="+", default=[256, 512, 256])
    p.add_argument("--dropout-g", type=float, default=0.0)
    p.add_argument("--dropout-d", type=float, default=0.0)
    p.add_argument("--cat-mode", choices=["gumbel_st", "gumbel_soft", "softmax"],
                   default="gumbel_st",
                   help="how categorical blocks become a differentiable simplex")
    p.add_argument("--gumbel-tau", type=float, default=0.667)
    # optimisation
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=256,
                   help="Kim & Bansal Fig. 9 report 256 for WGAN.")
    p.add_argument("--n-critic", type=int, default=5)
    p.add_argument("--lr-g", type=float, default=5e-3,
                   help="Kim & Bansal Fig. 9 report Adam at 0.005 for WGAN. "
                        "This is high for a WGAN-GP by convention (2e-4) but it "
                        "is what the paper reports; HPO recalibrates on PUMS.")
    p.add_argument("--lr-d", type=float, default=5e-3)
    p.add_argument("--beta1", type=float, default=0.0)
    p.add_argument("--beta2", type=float, default=0.9)
    p.add_argument("--lambda-gp", type=float, default=10.0)
    # ZCR
    p.add_argument("--bd-weight", type=float, default=DEFAULT_BD_WEIGHT,
                   help="weight of the structural-zero (nearest-neighbour) term")
    p.add_argument("--ad-weight", type=float, default=DEFAULT_AD_WEIGHT,
                   help="weight of the sampling-zero (diversity) term")
    p.add_argument("--no-zcr", action="store_true",
                   help="plain WGAN-GP baseline from the same code path (E1 control)")
    p.add_argument("--zcr-only-ad", action="store_true",
                   help="drop the BD term (old if_zc_only_ad variant)")
    p.add_argument("--zcr-neighbors", type=int, default=0,
                   help="rows resampled from the training bank per generator step; "
                        "0 = the WHOLE bank, which is what Kim & Bansal Eq.(9) specifies. "
                        "The paper singles this out as what distinguishes R_BD from the "
                        "Wasserstein term ('the R_BD measures the distance from the entire "
                        "training data'), so a subsample is a deviation from a defining "
                        "property, not an implementation detail: 4096 of the 40k bank "
                        "overestimates BD by 9.5%% (measured). Memory is O(batch*chunk).")
    p.add_argument("--zcr-chunk", type=int, default=20000,
                   help="bank rows per distance block. Purely a compute layout: "
                        "BD and AD are bit-identical across chunk sizes (verified). "
                        "On GPU the full 40k bank costs the same as a 4096-row "
                        "subsample at chunk=4096 (7.16 vs 7.09 ms/step) because the "
                        "matmul is trivial and the cost is kernel launches; at "
                        "chunk=20000 it is 2.46 ms/step, i.e. 3x FASTER than the old "
                        "subsampled default. An earlier 5.7x slowdown figure was "
                        "measured on CPU and does not carry over.")
    p.add_argument("--zcr-every", type=int, default=1,
                   help="apply ZCR every k generator steps (cost control)")
    p.add_argument("--zcr-cat-scale", type=float, default=1.0,
                   help="DEVIATION when != 1: rescales the categorical half of the "
                        "row before the distance")
    p.add_argument("--ad-clamp", type=float, default=0.0,
                   help="DEVIATION when > 0: hinge the AD term at this distance")
    p.add_argument("--audit-legacy", action="store_true",
                   help="recompute the old train_wgan.py:20-54 terms on one batch "
                        "and store values + gradient norms in metrics.json")
    p.add_argument("--audit-batch", type=int, default=64)
    p.add_argument("--audit-neighbors", type=int, default=512)
    # eval / selection
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-n", type=int, default=0, help="0 = use the whole val split")
    p.add_argument("--no-2way", action="store_true")
    p.add_argument("--eval-config", default=DEFAULT_EVAL_CONFIG,
                   help="frozen utils.eval_v2 EvalConfig yaml "
                        "(configs/eval_v2_pums20.yaml for paper runs)")
    p.add_argument("--f1-weight", type=float, default=1.0)
    p.add_argument("--bw-scale", type=float, default=0.35,
                   help="binning bandwidth scale (support/decoding only here)")
    p.add_argument("--select-on", default="val_srmse",
        choices=["val_srmse", "if1", "ml_eff"],
        help="hyperparameter-selection criterion (utils/criteria.py): each model is\n"
             "selected the way its own authors select, not by one shared metric")
    p.add_argument("--load-final", action="store_true",
                   help="load the final instead of the best checkpoint (F4 regression switch)")
    # sampling
    p.add_argument("--n-samples", type=int, default=0, help="0 = size of the val split")
    p.add_argument("--sample-batch", type=int, default=4096)
    p.add_argument("--sample-cat-mode",
                   choices=["auto", "argmax", "multinomial", "gumbel_st", "softmax"],
                   default="auto")
    p.add_argument("--sample-temperature", type=float, default=1.0)
    # reproducibility / io
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="EMA decay for the generator weights used at eval and "
                        "checkpoint time; 0 disables.  Present so this baseline "
                        "gets the same generic training practice SemaPop uses.")
    p.add_argument("--tail-k", type=int, default=5,
                   help="trailing in-training evaluations averaged into "
                        "`tail_val_metrics`, the quantity HPO tunes on.")
    p.add_argument("--seed", type=int, default=42, help="training seed")
    p.add_argument("--sample-seed", type=int, default=1234, help="sampling seed (separate, F5)")
    p.add_argument("--no-deterministic", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-root", default=_PA.S("/modelSave"))
    p.add_argument("--tag", default="wgan_zcr_v2")
    p.add_argument("--run-name", default=None)
    # audit-layer ensemble output (utils/ensemble_v2.py; finalize only)
    add_ensemble_args(p)
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    run_name = args.run_name or f"{args.tag}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    save_dir = os.path.join(args.out_root, run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[run] {save_dir}")

    set_torch_seed(args.seed, deterministic=not args.no_deterministic)
    data = prepare_data(args)
    return train(args, data, save_dir)


if __name__ == "__main__":
    main()
