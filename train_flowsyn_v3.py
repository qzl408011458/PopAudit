#!/usr/bin/env python
"""
train_flowsyn_v3.py
===================
FlowSyn under the **authors' official protocol**, on pop_audit's 12-attribute
PUMS build (``cols_config_v3.yaml``), emitting the K=50 audit ensemble.

Provenance
----------
This file grafts the official-protocol trainer of the sibling repo
(``<semapop_repo>/train_flowsyn_mixed.py``) onto pop_audit's
``pums_states`` data IO and the ``utils/ensemble_v2`` audit contract.
It replaces two earlier in-house versions (``train_flowsyn.py``, a "simple repro",
and ``train_flowsyn_v2.py``); neither is shipped.

WHAT IS THE AUTHORS' AND WHAT IS OURS
-------------------------------------
The model files are the faithful port of the official FlowSyn and are shared
byte-for-byte with the SemaPop repository (verified by ``diff``; no ``model/flowsyn_official.py``
copy was needed):

  ``model/flowsyn.py``      token-transformer VAE (Fourier continuous embed,
                            discrete embed, per-attribute latent [B, d, k_lat]),
                            ``vae_loss`` = MSE(num) + CE(cat) + beta*KL
  ``model/flowsyn_v2.py``   official gMLP velocity field with Conv1d token
                            mixing, Beta(5,5)->[-1,1] source, token-space CFM
                            loss and midpoint ``ode_sample_tokens``

Official (authors') settings, reproduced verbatim from the mixed trainer:
  * LRs           VAE 1e-2, flow 5e-4        (AdamW, weight_decay 1e-6)
  * epoch budgets VAE 900, flow 5000         -- as CAPS, not fixed lengths
  * batch sizes   VAE 4096, flow 1024
  * beta anneal   max 1e-2 -> min 1e-5, factor 0.7, patience 15 on TRAIN loss
  * masking       mask_ratio 0.4 denoising VAE objective
  * numerics      QuantileTransformer(output_distribution="normal")
  * latents       posterior mean, normalised as (z - z_mean) / 2.0
  * flow field    gMLP, 5 layers x 1280 units, emb 32, max_period 100
  * sampling      25-step midpoint ODE from the Beta(5,5) source

OURS -- the one addition, on held-out signals the authors' loop never computes:
  VAE   early stopping on validation RECONSTRUCTION loss (num MSE + cat CE with
        beta=0).  The beta term anneals during training, so the full ELBO is a
        moving target no stopping rule can watch; reconstruction is what stage 2
        consumes.  patience 40.
  flow  early stopping on the CFM loss over the VALIDATION split's latents
        (encoded by the trained VAE) -- the exact quantity the flow is fitted
        to, on rows it never saw.  patience 150.
  Both stages restore the best checkpoint at stop.

WHAT THE OFFICIAL PROTOCOL CHANGED vs pop_audit's ``train_flowsyn.py``
----------------------------------------------------------------------
 1. LR            1e-3 for BOTH stages  ->  official VAE 1e-2 / flow 5e-4.
 2. epochs        50 / 80 fixed         ->  official caps 900 / 5000 with early
                                            stopping (the old fixed 50/80 was an
                                            unregistered tuning decision).
 3. no stopping   -> val-recon ES (patience 40) + val-latent-CFM ES (patience
                     150), best checkpoint restored.
 4. beta          fixed 1e-3            ->  official anneal 1e-2 -> 1e-5.
 5. VAE objective plain reconstruction  ->  masked denoising (mask_ratio 0.4).
 6. numerics      StandardScaler        ->  QuantileTransformer(normal).
 7. velocity      ``model.flowsyn.VelocityField`` (flat [B, Z] gated MLP, N(0,I)
                  source, 15 ODE steps)  ->  ``model.flowsyn_v2.GatedMLPVelocity``
                  (token-mixing gMLP on [B, d, k_lat]), Beta(5,5) source,
                  25 ODE steps.
 8. LEAK FIX      encoders / numeric bounds were fitted on
                  ``concat(train, val, TEST)`` -- the reference population.  They
                  are now fitted on train+val only; the reference split is never
                  opened for anything except its row count (and not even that in
                  HPO mode).
 9. HPO LEAK FIX  the old quick mode scored the trial against ``testdata_pers``
                  (= the reference).  The objective is now computed on the val
                  split only, with the frozen ``configs/eval_v3_pums12.yaml``.
10. train rows    ``MAX_TRAIN=150_000`` subsample cap removed; production uses
                  the FULL train split (40,000).  ``--subsample`` remains for
                  smoke runs only.
11. columns       hardcoded ``cols_config.yaml`` (13 modelled cols incl. the
                  constant ``Region_Category``) -> ``--cols-config``, default
                  ``cols_config_v3.yaml`` (4 num + 8 cat = 12).  Splits are read
                  with ``usecols`` so the prompt/persona text columns are never
                  parsed.
12. eval          ``utils/evals.py`` (evaluate_marg / evaluate_bivar, whose
                  k-way tables and sqrt(N_j) factor are wrong) ->
                  ``utils.eval_v2`` + the frozen v3 config.
13. ensemble      hand-rolled 50-seed loop that wrote 14 columns
                  (12 + ``realization_id`` + ``sample_seed``) with no k limit ->
                  ``utils/ensemble_v2.write_ensemble``, which writes EXACTLY the
                  12 modelled columns in cols_config_v3 order, honours
                  ``--ensemble-k`` / ``--ensemble-n`` and also drops the
                  backward-compatible ``final_generated_samples.csv``.
14. CLI           env-var configuration (``FLOWSYN_*``) -> argparse, so a run's
                  effective config is recorded in ``metrics.json``.
                  ``FLOWSYN_HPO_OUT`` is still honoured as the default of
                  ``--hpo-out`` for the existing HPO driver.

RECORD REUSE
------------
Generation draws every row from noise: ``ode_sample_tokens`` starts at
z0 ~ 2*Beta(5,5)-1 and integrates the learned velocity field t: 0 -> 1, then the
VAE decoder maps the endpoint to attributes.  No training / val / reference row,
row index, or nearest-neighbour lookup enters the sampling path.  See the
``sample_df`` block below and ``metrics.json -> sampling.record_reuse``.

Example (production)
--------------------
    python train_flowsyn_v3.py --state CA --device cuda:1 \
        --out-root <repo>/modelSave \
        --ensemble
"""

from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[0])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)

import argparse
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)

from model.flowsyn import FlowSynVAE, vae_loss                        # noqa: E402
from model.flowsyn_v2 import (BetaSource, GatedMLPVelocity,           # noqa: E402
                              cfm_loss_tokens, ode_sample_tokens)
from utils.ensemble_v2 import add_ensemble_args, write_ensemble       # noqa: E402

DEFAULT_DATA_DIR = os.path.join(REPO_DIR, "data", "pums_states", "CA")
DEFAULT_EVAL_CONFIG = os.path.join(REPO_DIR, "configs", "eval_v3_pums12.yaml")
DEFAULT_OUT_ROOT = _PA.S("/modelSave")

# --------------------------------------------------------------------------- #
# official constants (train_flowsyn_mixed.py:41-45) -- CLI-overridable below
# --------------------------------------------------------------------------- #
OFFICIAL = {
    "k_emb": 32, "k_lat": 8,
    "vae_bs": 4096, "flow_bs": 1024, "sample_bs": 16384,
    "vae_lr": 1e-2, "flow_lr": 5e-4,
    "max_beta": 1e-2, "min_beta": 1e-5, "beta_lambda": 0.7, "beta_patience": 15,
    "mask_ratio": 0.4, "ode_steps": 25,
    "vae_epochs": 900, "flow_epochs": 5000,
    "flow_layers": 5, "flow_units": 1280, "flow_max_period": 100,
    "weight_decay": 1e-6, "grad_clip": 5.0,
}


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def set_torch_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python / numpy / torch / cuda and (optionally) pin cudnn."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def read_split(path: str, num_cols: Sequence[str], cat_cols: Sequence[str],
               subsample: int = 0, seed: int = 42) -> pd.DataFrame:
    """Read a split with ``usecols`` -- the CSVs also carry Pid / Region_Category /
    commute minutes / prompt / persona, which are NOT modelled and whose text
    columns make a naive ``read_csv`` both slow and memory-hungry."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    use = list(num_cols) + list(cat_cols)
    missing = [c for c in use if c not in header]
    if missing:
        raise KeyError(f"{path} is missing modelled columns {missing}")
    df = pd.read_csv(path, usecols=use, low_memory=False)[use]
    for c in cat_cols:
        df[c] = df[c].astype(str)
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="raise").astype(float)
    if subsample and subsample < len(df):
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(df), size=subsample, replace=False))
        df = df.iloc[keep].reset_index(drop=True)
    return df


def count_rows(path: str, probe_col: str) -> int:
    """Row count of a split without materialising it (one column only).

    Used for N = len(reference population).  This is the ONLY contact this
    trainer has with the reference split, and it is skipped entirely in HPO mode.
    """
    return int(len(pd.read_csv(path, usecols=[probe_col], low_memory=False)))


# --------------------------------------------------------------------------- #
# evaluation (frozen protocol: utils.eval_v2 + configs/eval_v3_pums12.yaml)
# --------------------------------------------------------------------------- #
def load_eval_module():
    import utils.eval_v2 as ev  # noqa: WPS433 -- deliberate local import
    return ev


def load_eval_config(path: str, ev, fit_df: Optional[pd.DataFrame] = None):
    if not path:
        raise ValueError("--eval-config is required: the paper's numbers must come from "
                         f"the frozen protocol ({DEFAULT_EVAL_CONFIG})")
    cfg = ev.EvalConfig.from_yaml(path)
    if not cfg.is_fitted:
        if fit_df is None:
            raise ValueError(f"eval config {path} is not fitted and no frame was given")
        cfg.fit(fit_df)
    return cfg


def val_metrics(df_ref: pd.DataFrame, df_gen: pd.DataFrame, cfg, ev,
                do_2way: bool = True) -> Dict[str, float]:
    """SRMSE-1 / SRMSE-2 / per-block P-R-F1 on the held-out val split."""
    m: Dict[str, float] = {}
    m["srmse_1"] = float(ev.srmse_kway(df_ref, df_gen, cfg, k=1)["mean"])
    m["srmse_2"] = (float(ev.srmse_kway(df_ref, df_gen, cfg, k=2)["mean"])
                    if do_2way else float("nan"))
    pr = ev.block_precision_recall(df_ref, df_gen, cfg)
    f1s: List[float] = []
    for b, v in pr.items():
        if not isinstance(v, dict) or "f1" not in v:
            continue
        m[f"block_f1_{b}"] = float(v["f1"])
        m[f"block_precision_{b}"] = float(v["precision"])
        m[f"block_recall_{b}"] = float(v["recall"])
        if b != "all":
            f1s.append(float(v["f1"]))
    if f1s:
        m["block_f1"] = float(np.mean(f1s))
    return m


def selection_score(m: Dict[str, float], f1_weight: float = 1.0) -> float:
    """HPO objective, LOWER is better: 0.5*(SRMSE-1 + SRMSE-2) - w * mean block F1.

    (``utils/criteria.py`` does not exist in pop_audit, so the formula is stated
    here rather than imported; it is the same rule the v2 trainers document.)
    """
    s1 = float(m.get("srmse_1", float("nan")))
    s2 = float(m.get("srmse_2", float("nan")))
    f1 = float(m.get("block_f1", 0.0))
    s2 = s1 if not np.isfinite(s2) else s2
    return 0.5 * (s1 + s2) - f1_weight * f1


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train(args: argparse.Namespace) -> Dict[str, Any]:
    ev = load_eval_module()
    t_start = time.time()

    # ---- columns (12 modelled attributes of cols_config_v3) ---------------- #
    cols_path = os.path.join(args.data_dir, args.cols_config)
    with open(cols_path, encoding="utf-8") as f:
        cols = yaml.safe_load(f)
    num_cols = list(args.num_cols) if args.num_cols else list(cols["num_cols"])
    cat_cols = list(args.cat_cols) if args.cat_cols else list(cols["cat_cols"])
    model_cols = num_cols + cat_cols

    cfg = load_eval_config(args.eval_config, ev)
    missing = [c for c in cfg.order if c not in model_cols]
    if missing:
        raise SystemExit(f"eval config {args.eval_config} scores columns the model does not "
                         f"produce: {missing}")

    hpo_mode = bool(args.hpo_quick or args.hpo_out)

    # ---- splits ------------------------------------------------------------ #
    tr = read_split(os.path.join(args.data_dir, args.train_file), num_cols, cat_cols,
                    args.subsample, args.data_seed)
    va = read_split(os.path.join(args.data_dir, args.val_file), num_cols, cat_cols,
                    args.subsample_val, args.data_seed + 1)
    ref_path = os.path.join(args.data_dir, args.ref_file)
    n_ref: Optional[int] = None
    if not hpo_mode:
        # row count only -- the reference is never read into the fit or the eval
        n_ref = count_rows(ref_path, num_cols[0])
    print(f"[data] train={len(tr)} val={len(va)} ref_rows={n_ref} "
          f"num={len(num_cols)} cat={len(cat_cols)} cols_config={args.cols_config}", flush=True)
    if args.subsample and not hpo_mode:
        print(f"[data] WARNING: --subsample {args.subsample} is a SMOKE setting; production "
              f"runs use the full train split", flush=True)

    dev = torch.device(args.device)
    set_torch_seed(args.seed, deterministic=not args.no_deterministic)
    sample_seed = args.sample_seed or (1234 + args.seed)

    run_name = args.run_name or f"{args.tag}_s{args.seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    save_dir = os.path.join(args.out_root, run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[run] {save_dir}", flush=True)

    # ---- preprocessing: fitted on train+val ONLY (the reference is never
    #      opened here -- the rule every trainer in this repo obeys) --------- #
    union = pd.concat([tr, va], ignore_index=True)
    les: Dict[str, LabelEncoder] = {}
    cat_dims: List[int] = []
    for c in cat_cols:
        le = LabelEncoder().fit(union[c].astype(str))
        les[c] = le
        cat_dims.append(len(le.classes_))
    qt = QuantileTransformer(output_distribution="normal",
                             n_quantiles=min(1000, len(tr)), subsample=10 ** 9,
                             random_state=args.seed).fit(tr[num_cols].values.astype(np.float32))
    num_bounds = {c: (float(union[c].min()), float(union[c].max())) for c in num_cols}

    def encode_frame(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        xn = torch.tensor(qt.transform(df[num_cols].values.astype(np.float32)),
                          dtype=torch.float32)
        xk = torch.tensor(np.stack([les[c].transform(df[c].astype(str)) for c in cat_cols], 1),
                          dtype=torch.long)
        return xn, xk

    xnum, xcat = encode_frame(tr)
    vnum, vcat = encode_frame(va)
    n = len(xnum)

    history: List[Dict[str, Any]] = []

    # ===================== stage 1: masked VAE (official loop + ES) ========= #
    vae = FlowSynVAE(n_cont=len(num_cols), cat_dims=cat_dims,
                     k=args.k_emb, k_lat=args.k_lat).to(dev)
    opt = torch.optim.AdamW(vae.parameters(), lr=args.vae_lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.95, patience=5)
    beta, best_tr, bad = args.max_beta, float("inf"), 0
    es_best, es_bad, es_state, es_ep = float("inf"), 0, None, 0
    t0 = time.time()

    @torch.no_grad()
    def val_recon() -> float:
        """OURS: validation reconstruction loss (beta=0), the ES signal for stage 1."""
        vae.eval()
        tot = 0.0
        for i in range(0, len(vnum), args.vae_bs):
            xc = vnum[i:i + args.vae_bs].to(dev)
            xk = vcat[i:i + args.vae_bs].to(dev)
            mu, lv = vae.encode(xc, xk)
            num_pred, cat_logits = vae.decode(mu)
            l, *_ = vae_loss(xc, xk, num_pred, cat_logits, mu, lv, beta=0.0)
            tot += float(l) * len(xc)
        return tot / len(vnum)

    ep = 0
    for ep in range(1, args.vae_epochs + 1):
        vae.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.vae_bs):
            idx = perm[i:i + args.vae_bs]
            xc = xnum[idx].to(dev)
            xk = xcat[idx].to(dev)
            # official masked-denoising objective (mask_ratio 0.4): the input is
            # corrupted, the target is the CLEAN row.
            mc = (torch.rand_like(xc) < args.mask_ratio)
            xc_in = xc.masked_fill(mc, 0.0)
            mk = (torch.rand(xk.shape, device=dev) < args.mask_ratio)
            xk_in = xk.masked_fill(mk, 0)
            mu, lv = vae.encode(xc_in, xk_in)
            z = mu + torch.exp(0.5 * lv) * torch.randn_like(lv)
            num_pred, cat_logits = vae.decode(z)
            loss, *_ = vae_loss(xc, xk, num_pred, cat_logits, mu, lv, beta=beta)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), args.grad_clip)
            opt.step()
            tot += loss.item() * len(idx)
        avg = tot / n
        sched.step(avg)
        # official beta anneal: driven by the TRAIN loss plateau
        if avg < best_tr - 1e-5:
            best_tr, bad = avg, 0
        else:
            bad += 1
            if bad >= args.beta_patience and beta > args.min_beta:
                beta = max(args.min_beta, beta * args.beta_lambda)
                bad = 0
        vr = val_recon()
        if vr < es_best - 1e-5:
            es_best, es_bad, es_ep = vr, 0, ep
            es_state = {k: v.detach().cpu().clone() for k, v in vae.state_dict().items()}
        else:
            es_bad += 1
        history.append({"stage": "vae", "epoch": ep, "train_loss": avg,
                        "val_recon": vr, "beta": beta, "es_bad": es_bad})
        if ep % args.log_every_vae == 0 or ep == 1:
            print(f"[vae ep {ep:4d}] train={avg:.4f} val_recon={vr:.4f} "
                  f"beta={beta:.1e} es_bad={es_bad}", flush=True)
        if es_bad >= args.vae_patience:
            print(f"[vae] EARLY STOP at ep {ep}; best val_recon={es_best:.4f} @ {es_ep}",
                  flush=True)
            break
    if es_state is not None:
        vae.load_state_dict(es_state)
    vae.eval()
    vae_stop = ep
    vae_seconds = time.time() - t0

    # ---- latents: posterior mean, no masking; official normalisation ------- #
    @torch.no_grad()
    def latents(xn: torch.Tensor, xk: torch.Tensor) -> torch.Tensor:
        zs = []
        for i in range(0, len(xn), args.flow_bs):
            mu, _ = vae.encode(xn[i:i + args.flow_bs].to(dev), xk[i:i + args.flow_bs].to(dev))
            zs.append(mu.cpu())
        return torch.cat(zs, 0)

    z1 = latents(xnum, xcat)
    zv = latents(vnum, vcat)
    z_mean = z1.mean(0, keepdim=True)
    z1n, zvn = (z1 - z_mean) / 2.0, (zv - z_mean) / 2.0
    d_tok = z1.shape[1]
    print(f"[latent] tokens d={d_tok} x k_lat={args.k_lat}", flush=True)

    # ===================== stage 2: gMLP flow (official loop + ES) ========== #
    vf = GatedMLPVelocity(n_tokens=d_tok, in_dim=args.k_lat, emb_dim=args.k_emb,
                          n_layers=args.flow_layers, n_units=args.flow_units,
                          max_period=args.flow_max_period).to(dev)
    optf = torch.optim.AdamW(vf.parameters(), lr=args.flow_lr, weight_decay=args.weight_decay)
    src = BetaSource(alpha=args.beta_source_alpha, device=dev)
    fes_best, fes_bad, fes_state, fes_ep = float("inf"), 0, None, 0
    t0 = time.time()
    ep = 0
    for ep in range(1, args.flow_epochs + 1):
        vf.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.flow_bs):
            zb = z1n[perm[i:i + args.flow_bs]].to(dev)
            loss = cfm_loss_tokens(vf, zb, src)
            optf.zero_grad()
            loss.backward()
            optf.step()
            tot += loss.item() * len(zb)
        # OURS: CFM loss on the VALIDATION latents.  The t / z0 draws are pinned
        # to a fixed seed so the ES signal is comparable across epochs, then the
        # training stream is re-seeded (official trainer's behaviour, kept).
        with torch.no_grad():
            vf.eval()
            vtot = 0.0
            torch.manual_seed(12345)
            for i in range(0, len(zvn), args.flow_bs):
                chunk = zvn[i:i + args.flow_bs]
                vtot += float(cfm_loss_tokens(vf, chunk.to(dev), src)) * len(chunk)
            vloss = vtot / len(zvn)
            torch.manual_seed(args.seed * 9973 + ep)
        if vloss < fes_best - 1e-5:
            fes_best, fes_bad, fes_ep = vloss, 0, ep
            fes_state = {k: v.detach().cpu().clone() for k, v in vf.state_dict().items()}
        else:
            fes_bad += 1
        history.append({"stage": "flow", "epoch": ep, "train_loss": tot / n,
                        "val_cfm": vloss, "es_bad": fes_bad})
        if ep % args.log_every_flow == 0 or ep == 1:
            print(f"[flow ep {ep:4d}] train={tot / n:.4f} val={vloss:.4f} "
                  f"es_bad={fes_bad}", flush=True)
        if fes_bad >= args.flow_patience:
            print(f"[flow] EARLY STOP at ep {ep}; best val={fes_best:.4f} @ {fes_ep}",
                  flush=True)
            break
    if fes_state is not None:
        vf.load_state_dict(fes_state)
    vf.eval()
    flow_stop = ep
    flow_seconds = time.time() - t0

    # ===================== sampling ========================================= #
    # RECORD-REUSE CHECK (spec item 4).  Every row of every realization is a
    # fresh noise draw: ode_sample_tokens (model/flowsyn_v2.py:150) starts at
    # z0 ~ 2*Beta(5,5)-1 and integrates the learned velocity field t: 0 -> 1;
    # decode_batch then runs the VAE decoder + the inverse QuantileTransformer +
    # the inverse LabelEncoders.  Nothing below this line touches xnum/xcat/
    # vnum/vcat, any dataframe of real rows, any row index, or any
    # nearest-neighbour lookup -- the only data-fitted objects involved are
    # PARAMETERS (decoder weights, quantile knots, label vocabularies, clip
    # bounds), never records.  Recorded in metrics.json -> sampling.
    zmean_dev = z_mean.to(dev)

    @torch.no_grad()
    def decode_batch(zt: torch.Tensor) -> pd.DataFrame:
        z = zt * 2.0 + zmean_dev
        num_pred, cat_logits = vae.decode(z)
        num_orig = qt.inverse_transform(num_pred.cpu().numpy())
        out: Dict[str, Any] = {}
        for j, c in enumerate(num_cols):
            lo, hi = num_bounds[c]
            out[c] = np.clip(np.round(num_orig[:, j]), lo, hi).astype(int)
        for j, c in enumerate(cat_cols):
            out[c] = les[c].inverse_transform(cat_logits[j].argmax(-1).cpu().numpy())
        return pd.DataFrame(out)

    @torch.no_grad()
    def sample_df(n_rows: int) -> pd.DataFrame:
        parts = []
        for i in range(0, n_rows, args.sample_bs):
            b = min(args.sample_bs, n_rows - i)
            zt = ode_sample_tokens(vf, b, d_tok, args.k_lat, src, dev, steps=args.ode_steps)
            parts.append(decode_batch(zt))
        return pd.concat(parts, ignore_index=True)[model_cols]

    sampling_record = {
        "draws_from": "Beta(5,5)->[-1,1] source noise, midpoint ODE t:0->1",
        "ode_steps": int(args.ode_steps),
        "record_reuse": False,
        "note": ("generation is unconditional noise -> ode_sample_tokens -> VAE decoder; "
                 "no training/val/reference row, row index or nearest-neighbour lookup "
                 "enters the sampling path (verified by inspection of sample_df/decode_batch)"),
    }

    # ---- HPO quick mode: one realization the size of val, scored on VAL ----- #
    # The reference split is NEVER opened in an HPO trial (spec, Eval
    # protocol).  The old train_flowsyn.py scored trials against testdata_pers,
    # i.e. against the reference population; that is fixed here.
    if hpo_mode:
        set_torch_seed(sample_seed, deterministic=not args.no_deterministic)
        gen = sample_df(len(va))
        m = val_metrics(va, gen, cfg, ev, do_2way=not args.no_2way)
        score = selection_score(m, args.f1_weight)
        rec = {
            "mode": "hpo_quick",
            "objective": "min 0.5*(val SRMSE-1 + SRMSE-2) - w * mean block F1  (val split only)",
            "score": score,
            "f1_weight": float(args.f1_weight),
            "eval_config": args.eval_config,
            "n_val": int(len(va)), "n_gen": int(len(gen)),
            "stopped": {"vae": vae_stop, "vae_best": es_ep,
                        "flow": flow_stop, "flow_best": fes_ep},
            "seconds": round(time.time() - t_start, 1),
            **m,
            # legacy aliases for the pre-v3 experiments/hpo driver (which
            # maximised F1/100 - 0.5*(SRMSE_M + SRMSE_B)); note these are now
            # computed on VAL with the v3 protocol, not on the reference.
            "SRMSE_M": m.get("srmse_1"), "SRMSE_B": m.get("srmse_2"),
            "F1": 100.0 * float(m.get("block_f1", 0.0)),
            "score_legacy_max": -score,
        }
        out_json = args.hpo_out or os.path.join(save_dir, "metrics.json")
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(ev._jsonable(rec), f, indent=2, default=str)
        with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(ev._jsonable({**rec, "sampling": sampling_record,
                                    "config": vars(args)}), f, indent=2, default=str)
        print(f"[hpo] srmse1={m.get('srmse_1', float('nan')):.4f} "
              f"srmse2={m.get('srmse_2', float('nan')):.4f} "
              f"blockF1={m.get('block_f1', float('nan')):.4f} score={score:.4f} -> {out_json}",
              flush=True)
        if args.hpo_cleanup and out_json != os.path.join(save_dir, "metrics.json"):
            shutil.rmtree(save_dir, ignore_errors=True)
        return rec

    # ---- final sample ------------------------------------------------------ #
    n_out = args.n_samples if args.n_samples > 0 else int(n_ref)
    set_torch_seed(sample_seed, deterministic=not args.no_deterministic)
    t_s = time.time()
    df_out = sample_df(n_out)
    sample_seconds = time.time() - t_s
    sample_csv = os.path.join(save_dir, "generated_samples.csv")
    df_out.to_csv(sample_csv, index=False)
    print(f"[sample] n={n_out} in {sample_seconds:.1f}s -> {sample_csv}", flush=True)

    # ---- post-training evaluation on the HELD-OUT VAL split ---------------- #
    n_eval = min(len(va), len(df_out))
    final_val = val_metrics(va.iloc[:n_eval], df_out.iloc[:n_eval], cfg, ev,
                            do_2way=not args.no_2way)
    print(f"[eval/val] srmse1={final_val.get('srmse_1', float('nan')):.4f} "
          f"srmse2={final_val.get('srmse_2', float('nan')):.4f} "
          f"blockF1={final_val.get('block_f1', float('nan')):.4f} "
          f"allF1={final_val.get('block_f1_all', float('nan')):.4f}", flush=True)

    # ---- checkpoint -------------------------------------------------------- #
    ckpt_path = None
    if not args.no_save_ckpt:
        ckpt_path = os.path.join(save_dir, "flowsyn_v3.pt")
        torch.save({
            "vae_state": vae.state_dict(), "flow_state": vf.state_dict(),
            "z_mean": z_mean, "d_tok": int(d_tok),
            "quantile_transformer": qt, "label_encoders": les,
            "num_bounds": num_bounds, "cat_dims": cat_dims,
            "num_cols": num_cols, "cat_cols": cat_cols,
            "config": vars(args),
            "stopped": {"vae": vae_stop, "vae_best": es_ep,
                        "flow": flow_stop, "flow_best": fes_ep},
            "torch_version": torch.__version__, "numpy_version": np.__version__,
        }, ckpt_path)

    metrics = {
        "run": {
            "save_dir": save_dir,
            "finished": datetime.now().isoformat(timespec="seconds"),
            "vae_wall_clock_s": round(vae_seconds, 2),
            "flow_wall_clock_s": round(flow_seconds, 2),
            "sample_wall_clock_s": round(sample_seconds, 2),
            "total_wall_clock_s": round(time.time() - t_start, 2),
            "device": args.device,
            "torch_version": torch.__version__,
            "python": sys.version.split()[0],
        },
        "implementation": {
            "impl": "official FlowSyn port (model/flowsyn.py + model/flowsyn_v2.py) "
                    "+ early stopping",
            "model_files_identical_to_llm_pop": True,
            "official": {
                "lrs": {"vae": args.vae_lr, "flow": args.flow_lr},
                "epoch_caps": {"vae": args.vae_epochs, "flow": args.flow_epochs},
                "batch_sizes": {"vae": args.vae_bs, "flow": args.flow_bs},
                "beta_anneal": {"max": args.max_beta, "min": args.min_beta,
                                "lambda": args.beta_lambda, "patience": args.beta_patience},
                "mask_ratio": args.mask_ratio,
                "numeric_transform": "QuantileTransformer(output_distribution='normal')",
                "source": f"Beta({args.beta_source_alpha},{args.beta_source_alpha}) -> [-1,1]",
                "velocity_field": "GatedMLPVelocity (Conv1d token mixing)",
            },
            "ours": {
                "early_stopping": True,
                "es_signal": {"vae": "val reconstruction (beta=0)",
                              "flow": "val-latent CFM loss"},
                "patience": {"vae": args.vae_patience, "flow": args.flow_patience},
                "rationale": "the official budget is 5000 flow epochs; cutting it by fiat "
                             "(the historical train_flowsyn.py ran 50/80) is an unregistered "
                             "tuning decision",
            },
            "stopped": {"vae": vae_stop, "vae_best_epoch": es_ep,
                        "vae_best_val_recon": es_best,
                        "flow": flow_stop, "flow_best_epoch": fes_ep,
                        "flow_best_val_cfm": fes_best},
        },
        "config": vars(args),
        "data": {
            "data_dir": args.data_dir,
            "cols_config": cols_path,
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "n_ref": (None if n_ref is None else int(n_ref)),
            "n_generated": int(len(df_out)),
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "subsample_used": int(args.subsample),
            "full_train_split_used": bool(not args.subsample),
            "held_out_val": True,
            "preprocessors_fitted_on": "train+val (the reference split is never opened; "
                                       "only its row count is read, for N)",
            "cat_vocab_sizes": {c: len(les[c].classes_) for c in cat_cols},
            "num_bounds_train_val": {c: list(v) for c, v in num_bounds.items()},
        },
        "sampling": sampling_record,
        "eval_config": cfg.to_dict(),
        "selection": {
            "rule": "official epoch caps with early stopping on held-out signals; "
                    "best checkpoint restored per stage",
            "metric_backend": "utils.eval_v2",
            "eval_config_path": args.eval_config,
        },
        # No in-training utils.eval_v2 evaluation exists by design: the official
        # loop selects on reconstruction / CFM loss, so there is no per-epoch
        # SRMSE history to average.  `tail_val_metrics` is therefore empty.
        "tail_val_metrics": {},
        "final_val_metrics": final_val,
        "history": history,
        "outputs": {"samples_csv": sample_csv, "ckpt": ckpt_path},
    }
    out_json = os.path.join(save_dir, "metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(ev._jsonable(metrics), f, indent=2, default=str)
    print(f"[done] metrics -> {out_json}", flush=True)

    # ---- K=50 audit ensemble (utils/ensemble_v2 contract) ------------------ #
    # Runs AFTER metrics.json so a mid-ensemble failure still leaves a complete
    # metrics record and the ensemble wall clock never leaks into the timings.
    if args.ensemble:
        ens_n = int(args.ensemble_n) if args.ensemble_n > 0 else int(n_ref)

        def ensemble_sample_fn(seed: int, n_rows: int) -> pd.DataFrame:
            # Reseed python/numpy/torch/cuda from `seed` (audit_config.ENSEMBLE_SEEDS
            # supplies them; never re-derived here), then run the SAME noise ->
            # ODE -> decoder path as the final sample.  Returns exactly the 12
            # modelled columns in cols_config_v3 order (num then cat).
            set_torch_seed(int(seed), deterministic=not args.no_deterministic)
            return sample_df(int(n_rows))[model_cols]

        paths = write_ensemble(ensemble_sample_fn, n_samples=ens_n, save_dir=save_dir,
                               k=args.ensemble_k)
        metrics["outputs"]["ensemble_dir"] = os.path.join(save_dir, "generated_ensembles")
        metrics["outputs"]["ensemble_k"] = len(paths)
        metrics["outputs"]["ensemble_n"] = ens_n
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(ev._jsonable(metrics), f, indent=2, default=str)

    return metrics


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FlowSyn (official protocol + early stopping) on pums_states / "
                    "cols_config_v3, with the K=50 audit-ensemble contract")
    # data
    p.add_argument("--state", default=None,
                   help="convenience: sets --data-dir to <repo>/data/pums_states/<STATE>")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--cols-config", default="cols_config_v3.yaml",
                   help="12 modelled columns (4 num + 8 cat); Region_Category excluded")
    p.add_argument("--train-file", default="traindata_pers.csv")
    p.add_argument("--val-file", default="valdata_pers.csv")
    p.add_argument("--ref-file", default="testdata_pers.csv",
                   help="reference population; opened ONLY for its row count (N)")
    p.add_argument("--num-cols", nargs="*", default=None)
    p.add_argument("--cat-cols", nargs="*", default=None)
    p.add_argument("--subsample", type=int, default=0,
                   help="SMOKE ONLY: cap training rows (0 = full 40k train split)")
    p.add_argument("--subsample-val", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=42)
    # official protocol -- epoch caps and LRs
    p.add_argument("--vae-epochs", type=int, default=OFFICIAL["vae_epochs"],
                   help="official cap (900); early stopping usually fires first")
    p.add_argument("--flow-epochs", type=int, default=OFFICIAL["flow_epochs"],
                   help="official cap (5000); early stopping usually fires first")
    p.add_argument("--vae-lr", type=float, default=OFFICIAL["vae_lr"])
    p.add_argument("--flow-lr", type=float, default=OFFICIAL["flow_lr"])
    p.add_argument("--vae-bs", type=int, default=OFFICIAL["vae_bs"])
    p.add_argument("--flow-bs", type=int, default=OFFICIAL["flow_bs"])
    p.add_argument("--sample-bs", type=int, default=OFFICIAL["sample_bs"])
    p.add_argument("--k-emb", type=int, default=OFFICIAL["k_emb"])
    p.add_argument("--k-lat", type=int, default=OFFICIAL["k_lat"])
    p.add_argument("--mask-ratio", type=float, default=OFFICIAL["mask_ratio"])
    p.add_argument("--ode-steps", type=int, default=OFFICIAL["ode_steps"])
    p.add_argument("--max-beta", type=float, default=OFFICIAL["max_beta"])
    p.add_argument("--min-beta", type=float, default=OFFICIAL["min_beta"])
    p.add_argument("--beta-lambda", type=float, default=OFFICIAL["beta_lambda"])
    p.add_argument("--beta-patience", type=int, default=OFFICIAL["beta_patience"])
    p.add_argument("--beta-source-alpha", type=float, default=5.0)
    p.add_argument("--flow-layers", type=int, default=OFFICIAL["flow_layers"])
    p.add_argument("--flow-units", type=int, default=OFFICIAL["flow_units"])
    p.add_argument("--flow-max-period", type=int, default=OFFICIAL["flow_max_period"])
    p.add_argument("--weight-decay", type=float, default=OFFICIAL["weight_decay"])
    p.add_argument("--grad-clip", type=float, default=OFFICIAL["grad_clip"])
    # ours: early stopping
    p.add_argument("--vae-patience", type=int, default=40,
                   help="epochs without val-reconstruction improvement")
    p.add_argument("--flow-patience", type=int, default=150,
                   help="epochs without val-latent CFM improvement")
    p.add_argument("--log-every-vae", type=int, default=25)
    p.add_argument("--log-every-flow", type=int, default=100)
    # eval
    p.add_argument("--eval-config", default=DEFAULT_EVAL_CONFIG,
                   help="frozen utils.eval_v2 EvalConfig yaml (v3 = 12 attributes)")
    p.add_argument("--no-2way", action="store_true")
    p.add_argument("--f1-weight", type=float, default=1.0)
    p.add_argument("--hpo-quick", action="store_true",
                   help="quick HPO trial: one val-sized realization, scored on VAL only, "
                        "no final sample / checkpoint / ensemble")
    p.add_argument("--hpo-out", default=os.environ.get("FLOWSYN_HPO_OUT", ""),
                   help="objective json path (defaults to $FLOWSYN_HPO_OUT); setting it "
                        "implies --hpo-quick")
    p.add_argument("--hpo-cleanup", action="store_true",
                   help="delete the run dir after writing --hpo-out (old driver behaviour)")
    # sampling / io
    p.add_argument("--n-samples", type=int, default=0, help="0 = len(reference population)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-seed", type=int, default=0, help="0 = 1234 + --seed")
    p.add_argument("--no-deterministic", action="store_true")
    p.add_argument("--no-save-ckpt", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--tag", default="flowsyn_v3")
    p.add_argument("--run-name", default=None)
    # audit-layer ensemble (utils/ensemble_v2): --ensemble / --ensemble-k / --ensemble-n.
    # Finalized runs only -- HPO trials never pass --ensemble.
    add_ensemble_args(p)
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.state:
        args.data_dir = os.path.join(REPO_DIR, "data", "pums_states", args.state)
        if args.tag == "flowsyn_v3":
            args.tag = f"flowsyn_v3_{args.state}"
    if args.k_emb % 4 != 0:
        raise SystemExit("--k-emb must be divisible by the transformer head count (4)")
    if (args.hpo_quick or args.hpo_out) and args.ensemble:
        raise SystemExit("--ensemble is for finalized runs only; HPO trials must not write "
                         "the audit ensemble")
    return train(args)


if __name__ == "__main__":
    main()
