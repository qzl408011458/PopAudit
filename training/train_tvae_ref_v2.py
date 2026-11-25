#!/usr/bin/env python
"""
train_tvae_ref_v2.py
====================
Faithful **TVAE baseline** using the reference implementation shipped with the
``ctgan`` package (``ctgan.TVAE``, the code released with the CTGAN/TVAE paper
and vendored by SDV).  It replaces pop_audit's old self-implemented ``tabvae``
(``train_tabvae*.py`` + ``model/`` VAE code, kept untouched as the historical
record of the earlier version) for the v3 protocol.

Structure, CLI, seeding, checkpointing and ``metrics.json`` layout mirror
``train_ctgan_v2.py`` (same package, same conventions); shared helpers are
imported from that file rather than duplicated:

* F2  the validation split is **held out** -- the synthesiser is fit on train
      only; val is used exclusively for model selection.
* F5  ``--seed`` seeds python/numpy/torch/cuda/cudnn, ``--sample-seed`` is
      separate, the checkpoint stores the full effective config *and* the
      fitted ``DataTransformer`` (VGM + one-hot preprocessor, inside the
      pickled TVAE), every run writes ``metrics.json``.
* F6  numeric outputs are **clipped to the training support, then rounded**
      (integer columns only), and categorical outputs are asserted to lie in
      the training vocabulary; the out-of-support rate before clipping is
      recorded.
* model selection on the **val** split with the frozen ``utils.eval_v2``
      protocol, never on the reference.

Model selection without forking the library
-------------------------------------------
Exactly as in ``train_ctgan_v2.py``: ``ctgan.TVAE.fit`` is a monolithic method
whose epoch loop is ``for i in tqdm(range(self.epochs), ...)``
(``ctgan/synthesizers/tvae.py``), so we temporarily replace the module-global
``tqdm`` in ``ctgan.synthesizers.tvae`` with the same ``_EpochProbe`` thin
generator.  The library's optimisation code runs **verbatim**; the callback
only samples, scores on val and snapshots the decoder state (the decoder +
transformer are everything TVAE sampling uses), with the RNG state
saved/restored around it so the training trajectory is identical to a plain
``fit`` regardless of ``--eval-every``.  No non-reference training features
are added (no EMA, no custom losses, no schedule changes).

Conditional sampling
--------------------
``ctgan.TVAE.sample(samples)`` takes only a row count -- the reference TVAE has
**no** conditional sampling at all (unlike ``ctgan.CTGAN.sample``'s one-column
condition vector).  This trainer therefore exposes no ``--cond-demo`` and
``metrics.json`` carries ``conditional_demo: null`` for layout compatibility.

Audit ensemble
--------------
``--ensemble`` wires ``utils/ensemble_v2.write_ensemble``: one realization per
seed in ``audit_config.ENSEMBLE_SEEDS``; ``sample_fn(seed, n)`` reseeds
python/numpy/torch(+cuda) globally (``TVAE.random_states`` is left ``None``,
the package default, so ``@random_state``-wrapped ``sample`` reads the global
streams) and returns exactly the 12 modelled columns of ``cols_config_v3`` in
num-then-cat order, post-processed by the same F6 clip-then-round + vocabulary
assertion as every other sampling path.

Example
-------
    python training/train_tvae_ref_v2.py \
        --data-dir data/pums_states/CA --cols-config cols_config_v3.yaml \
        --tag tvae_ref_ca --device cuda:2 --epochs 300 --seed 42 --ensemble
"""

from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); import popaudit_paths as _PA  # release bootstrap: repository root first on sys.path (see popaudit_paths.py)

import argparse
import copy
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, _PA.R(""))

# shared conventions: identical helpers, imported instead of copied ---------- #
from train_ctgan_v2 import (  # noqa: E402
    _EpochProbe,
    _read_split,
    cat_vocabulary,
    frozen_rng,
    integer_flags,
    load_eval_config,
    load_eval_module,
    numeric_support,
    postprocess,
    selection_score,
    set_torch_seed,
    val_metrics,
)
from utils.ensemble_v2 import add_ensemble_args, write_ensemble  # noqa: E402

import yaml  # noqa: E402

REPO_DIR = _PA.R("")
DEFAULT_EVAL_CONFIG = os.path.join(REPO_DIR, "configs", "eval_v3_pums12.yaml")


@contextmanager
def epoch_hook(callback):
    """Install :class:`_EpochProbe` as the TVAE fit loop's ``tqdm``.

    Copied in spirit from ``train_ctgan_v2.epoch_hook`` but pointed at the
    **tvae** module -- each synthesizer module imports its own ``tqdm`` symbol,
    so the CTGAN version would patch the wrong loop.
    """
    import ctgan.synthesizers.tvae as _mod

    original = _mod.tqdm
    _mod.tqdm = _EpochProbe(callback)
    try:
        yield
    finally:
        _mod.tqdm = original


def _reference_rows(data_dir: str, probe_col: str) -> int:
    """N for the audit ensemble = len(reference population) when --ensemble-n is 0.

    Copied from ``train_semapop_vae_v2.py`` (importing that module would drag
    in its model code).  Only the row count is needed, so a single modelled
    column is read (the reference split's content stays unopened, per the eval
    protocol).
    """
    for name in ("referencedata_pers.csv", "testdata_pers.csv"):
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            n = int(len(pd.read_csv(path, usecols=[probe_col])))
            print(f"[ensemble] n_samples = {n} (rows of {name})", flush=True)
            return n
    raise SystemExit("[fatal] --ensemble with --ensemble-n 0 requires "
                     "referencedata_pers.csv or testdata_pers.csv in --data-dir "
                     "(or pass --ensemble-n explicitly)")


# --------------------------------------------------------------------------- #
# the baseline itself
# --------------------------------------------------------------------------- #
class TVAEBaseline:
    """Thin wrapper around ``ctgan.TVAE`` holding everything needed to sample.

    Owns the post-processing contract (F6) so that every sampling path --
    in-training val evaluation, final sample, audit ensemble -- goes through
    the same clip-then-round + vocabulary assertion (mirrors
    ``train_ctgan_v2.CTGANBaseline``).
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

    @classmethod
    def from_train(cls, df_train: pd.DataFrame, num_cols, cat_cols, args) -> "TVAEBaseline":
        from ctgan import TVAE  # installed reference implementation

        dev = torch.device(args.device)
        model = TVAE(
            embedding_dim=args.embedding_dim,
            compress_dims=tuple(args.compress_dims),
            decompress_dims=tuple(args.decompress_dims),
            l2scale=args.l2scale,
            batch_size=args.batch_size,
            epochs=args.epochs,
            loss_factor=args.loss_factor,
            enable_gpu=dev.type != "cpu",
            verbose=False,           # we do our own per-epoch logging
        )
        # ctgan 0.11.1: TVAE.set_device() does ``self.decoder.to(...)`` and the
        # decoder only exists after fit(), so it cannot be called here (the
        # deprecated ``cuda=`` kwarg cannot select a GPU index either).
        # Assigning _device directly is exactly what set_device would do minus
        # moving the not-yet-existing decoder; fit() reads self._device for
        # every tensor placement.
        model._device = dev
        return cls(
            num_cols=num_cols,
            cat_cols=cat_cols,
            support=numeric_support(df_train, num_cols),
            integer_cols=integer_flags(df_train, num_cols),
            vocab=cat_vocabulary(df_train, cat_cols),
            train_median={c: float(df_train[c].median()) for c in num_cols},
            model=model,
        )

    def _sample_raw(self, n: int) -> pd.DataFrame:
        with torch.no_grad():
            # NOTE: TVAE.sample calls ``decoder.eval()`` and never restores
            # ``train()``.  The decoder is Linear/ReLU + a sigma Parameter (no
            # dropout / batchnorm), so train and eval mode are behaviourally
            # identical; mid-training evaluation therefore does not alter the
            # subsequent training trajectory (the RNG is handled by
            # ``frozen_rng`` in the caller).
            df = self.model.sample(n)
        return pd.DataFrame(df)[self.num_cols + self.cat_cols]

    def sample(self, n: int, strict_vocab: bool = True):
        """Unconditional sample, post-processed (F6). TVAE has no conditional API."""
        raw = self._sample_raw(n)
        return postprocess(raw, self.num_cols, self.cat_cols, self.support,
                           self.integer_cols, self.vocab, self.train_median,
                           strict_vocab=strict_vocab)


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def implementation_record(model) -> Dict[str, Any]:
    """What ``metrics.json`` states in the paper: which TVAE this really is.

    Adapted from ``train_ctgan_v2.implementation_record`` (that one hardcodes
    the CTGAN class path / conditional-sampling notes).
    """
    import ctgan as _ctgan

    rec: Dict[str, Any] = {
        "package": "ctgan",
        "version": getattr(_ctgan, "__version__", "unknown"),
        "class": "ctgan.synthesizers.tvae.TVAE",
        "module_file": getattr(_ctgan, "__file__", None),
        "custom_reimplementation_used": False,
        "note": ("reference TVAE (VGM mode-specific normalisation + one-hot, "
                 "tanh-Gaussian numeric decoder); replaces the old self-implemented "
                 "'tabvae' baseline"),
        "conditional_sampling": "not supported by ctgan.TVAE (sample(n) only)",
        "checkpoint_selection": (
            "per-epoch val evaluation via the tqdm epoch hook (library loop runs "
            "verbatim); best decoder state restored, same as train_ctgan_v2"
        ),
    }
    try:
        import sdv as _sdv

        rec["sdv_version"] = getattr(_sdv, "__version__", "unknown")
        rec["sdv_wrapper_used"] = False
        rec["sdv_note"] = ("sdv.single_table.TVAESynthesizer wraps this same ctgan.TVAE; "
                           "we call the raw synthesiser for parity with train_ctgan_v2.")
    except Exception:  # noqa: BLE001 -- sdv is optional here
        rec["sdv_version"] = None
    tr = getattr(model, "transformer", None)  # TVAE stores it as .transformer (no underscore)
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


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
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
    base = TVAEBaseline.from_train(df_fit, num_cols, cat_cols, args)

    n_eval = min(len(df_val), args.eval_n) if args.eval_n > 0 else len(df_val)
    df_val_sel = df_val.iloc[:n_eval].reset_index(drop=True)

    history: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {"score": float("inf"), "epoch": -1, "metrics": {}, "state": None}
    t0 = time.time()

    def on_epoch(ep: int) -> None:
        rec: Dict[str, Any] = {"epoch": ep, "elapsed_s": round(time.time() - t0, 2)}
        lv = getattr(base.model, "loss_values", None)
        if lv is not None and len(lv):
            # TVAE logs one row per BATCH (Epoch, Batch, Loss); mean over the
            # just-finished epoch (library epochs are 0-based).
            ep_rows = lv[lv["Epoch"] == ep - 1]
            if len(ep_rows):
                rec["loss"] = float(ep_rows["Loss"].mean())
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
                    # decoder (+ its sigma parameter) and the fitted transformer
                    # are everything TVAE sampling uses; the encoder is a local
                    # variable inside fit() and is not needed to sample.
                    "state": copy.deepcopy(
                        {k: v.detach().cpu().clone()
                         for k, v in base.model.decoder.state_dict().items()}
                    ),
                })
                rec["is_best"] = True
            print(f"[ep {ep:04d}] loss={rec.get('loss', float('nan')):+.4f} | "
                  f"val_srmse1={m.get('srmse_1', float('nan')):.4f} "
                  f"val_srmse2={m.get('srmse_2', float('nan')):.4f} "
                  f"val_blockF1={m.get('block_f1', float('nan')):.4f} score={score:.4f}"
                  + ("  <- best" if rec.get("is_best") else ""), flush=True)
        elif ep % args.log_every == 0:
            print(f"[ep {ep:04d}] loss={rec.get('loss', float('nan')):+.4f}", flush=True)
        history.append(rec)

    print(f"[fit] ctgan.TVAE epochs={args.epochs} batch={args.batch_size} "
          f"embed={args.embedding_dim} compress={tuple(args.compress_dims)} "
          f"decompress={tuple(args.decompress_dims)} l2scale={args.l2scale} "
          f"loss_factor={args.loss_factor} device={args.device}", flush=True)
    with epoch_hook(on_epoch):
        base.model.fit(df_fit, discrete_columns=cat_cols)
    fit_seconds = time.time() - t0

    impl = implementation_record(base.model)
    print("[impl] " + json.dumps({k: impl[k] for k in ("package", "version", "class")}))

    def save_ckpt(path: str, epoch: int, metrics: Dict[str, Any]) -> None:
        # F5: full effective config + the fitted preprocessor travel with the model
        torch.save(
            {
                "tvae_model": base.model,           # includes the fitted DataTransformer (VGM)
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

    last_path = os.path.join(save_dir, "tvae_last.pt")
    best_path = os.path.join(save_dir, "tvae_best.pt")
    save_ckpt(last_path, args.epochs, history[-1] if history else {})

    loaded = "tvae_last.pt"
    if best["state"] is not None and not args.load_final:
        base.model.decoder.load_state_dict(best["state"])
        base.model.decoder.to(base.model._device)
        save_ckpt(best_path, int(best["epoch"]), best["metrics"])
        loaded = "tvae_best.pt"
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
        # layout compatibility with train_ctgan_v2: always null here, because
        # ctgan.TVAE.sample(n) has no conditional interface at all.
        "conditional_demo": None,
        "history": history,
        "outputs": {
            "samples_csv": sample_csv,
            "last_ckpt": last_path,
            "best_ckpt": best_path if loaded == "tvae_best.pt" else None,
        },
    }
    out_json = os.path.join(save_dir, "metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(ev._jsonable(metrics), f, indent=2, default=str)
    print(f"[done] metrics -> {out_json}")

    # ---- K=50 audit ensemble (utils/ensemble_v2 contract; finalize only) ---- #
    if getattr(args, "ensemble", False):
        modelled_cols = num_cols + cat_cols     # cols_config_v3 order: num then cat

        def _ensemble_sample(e_seed: int, n: int) -> pd.DataFrame:
            # Each realization must be a deterministic function of its seed
            # alone: reseed python/numpy/torch(+cuda) globally, then follow
            # EXACTLY the final-sample path above (TVAE.random_states is None,
            # so the library's @random_state wrapper reads these global
            # streams).  base.sample applies the F6 clip-then-round policy, so
            # numeric columns come back as integers inside the training
            # support and categoricals inside the training vocabulary.
            set_torch_seed(e_seed, deterministic=not args.no_deterministic)
            df_e, _ = base.sample(n, strict_vocab=not args.no_strict_vocab)
            return df_e[modelled_cols]

        n_ens = int(args.ensemble_n) if args.ensemble_n else (
            len(df_ref) if df_ref is not None
            else _reference_rows(args.data_dir, modelled_cols[0]))
        ens_paths = write_ensemble(_ensemble_sample, n_samples=n_ens,
                                   save_dir=save_dir, k=args.ensemble_k)
        metrics["ensemble"] = {
            "n_realizations": len(ens_paths),
            "n_samples_each": int(n_ens),
            "dir": os.path.join(save_dir, "generated_ensembles"),
            "paths": ens_paths,
        }
        metrics["outputs"]["ensemble_dir"] = metrics["ensemble"]["dir"]
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(ev._jsonable(metrics), f, indent=2, default=str)
        print(f"[done] metrics updated with ensemble block -> {out_json}")
    return metrics


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TVAE baseline (reference ctgan.TVAE) for the v3 PUMS data")
    # data
    p.add_argument("--data-dir", default=os.path.join(REPO_DIR, "data", "pums_states", "CA"))
    p.add_argument("--cols-config", default="cols_config_v3.yaml")
    p.add_argument("--train-file", default="traindata_pers.csv")
    p.add_argument("--val-file", default="valdata_pers.csv")
    p.add_argument("--ref-file", default="referencedata_pers.csv")
    p.add_argument("--num-cols", nargs="*", default=None)
    p.add_argument("--cat-cols", nargs="*", default=None)
    p.add_argument("--subsample", type=int, default=0, help="subsample train (smoke runs)")
    p.add_argument("--subsample-val", type=int, default=0)
    p.add_argument("--subsample-ref", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=42)
    # TVAE hyper-parameters (ctgan 0.11.1 library defaults; the HPO search space)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--embedding-dim", type=int, default=128,
                   help="latent dimension (TVAE embedding_dim)")
    p.add_argument("--compress-dims", type=int, nargs="+", default=[128, 128],
                   help="encoder hidden layer sizes")
    p.add_argument("--decompress-dims", type=int, nargs="+", default=[128, 128],
                   help="decoder hidden layer sizes")
    p.add_argument("--l2scale", type=float, default=1e-5,
                   help="Adam weight_decay on encoder+decoder")
    p.add_argument("--loss-factor", type=float, default=2.0,
                   help="multiplier on the reconstruction term of the ELBO")
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
                   help="ignore the eval config's weight column")
    p.add_argument("--ref-eval", dest="ref_eval", action="store_true", default=True)
    p.add_argument("--no-ref-eval", dest="ref_eval", action="store_false")
    p.add_argument("--coverage-k", type=int, nargs="+", default=[2, 3])
    p.add_argument("--coverage-max-subsets", type=int, default=200)
    p.add_argument("--with-oracles", action="store_true")
    # sampling
    p.add_argument("--n-samples", type=int, default=0, help="0 = len(reference)")
    p.add_argument("--no-strict-vocab", action="store_true",
                   help="report instead of raising when a generated category is unseen in train")
    # reproducibility / io
    p.add_argument("--tail-k", type=int, default=5,
                   help="trailing in-training evaluations averaged into "
                        "`tail_val_metrics`, the quantity HPO tunes on.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-seed", type=int, default=1234)
    p.add_argument("--no-deterministic", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-root", default=_PA.S("/modelSave"))
    p.add_argument("--tag", default="tvae_ref_v2")
    p.add_argument("--run-name", default=None)
    add_ensemble_args(p)   # --ensemble / --ensemble-k / --ensemble-n (finalize only)
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    # (no batch-size/pac divisibility constraints: ctgan.TVAE has none)
    return train(args)


if __name__ == "__main__":
    main()
