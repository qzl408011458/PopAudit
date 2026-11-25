#!/usr/bin/env python
"""
DEPRECATED (2026-11-23): no longer the audited BN. The paper's BN row is produced by
train_bn_pgmpy.py (registry entry bn_pgmpy). Kept only to document the pre-swap runs
in hpo_v3/final/bn_{CA,TX,OH}. Note: this trainer uses its own numpy hill-climbing
backend and Laplace-smoothed CPTs, not pgmpy/MLE.

train_bn_v2.py
==============
Clean v2 Bayesian-network baseline.

This trainer replaces an earlier in-house implementation (``train_bn.py`` +
``model/bn_model.py``, not shipped).  It fixes three defects that made the
earlier BN row meaningless:

(a) **no structure learning.**  ``model/bn_model.py`` hard-coded a chain
    ``cols[0] -> cols[1] -> ... -> cols[n-1]`` in dataframe column order.  The
    chain asserts X_i _||_ X_j | X_{i-1} for every non-adjacent pair, so
    ancestral sampling freely emits combinations that cannot exist (a 4-year-old
    who is Married, an Under_16 person with a car commute, ...).  That is why the
    earlier BN scored 4.6% full-tuple precision *while every individual value
    was in support*.  Here the DAG is learned by greedy hill climbing over
    add/remove/reverse moves with a decomposable BIC/BDeu score, an acyclicity
    test and a max-parents cap (``model/bn_v2.py``).

(b) **no discretisation.**  The old code fed raw columns to the CPT
    estimator, so ``Age`` entered with 106 states and a single Age->X CPT had
    ~1400 free parameters estimated from 40k rows.  Here every column is
    discretised with the **frozen evaluation protocol**
    (``utils.eval_v2.EvalConfig.from_yaml(configs/eval_v2_pums20.yaml)``), so the
    BN's state space *is* the paper's matching space: Age -> 19 five-year bins,
    Household_Size -> 10 integer states with a >=10 tail, Number_of_children ->
    9 states with a >=8 tail, the 17 categoricals -> their fixed category lists.
    Nothing about the binning is chosen by this trainer; it is read from the
    frozen yaml and echoed verbatim into ``metrics.json``.

(c) **BN-Copula is dropped, not ported.**  ``train_bn.py:246-256`` fitted the
    *target* copula scaler on ``real_test_data`` -- i.e. the ECDF used to map
    generated values back to the target scale was estimated on the evaluation
    set.  That is test-set leakage, so the variant is deleted rather than
    rebuilt; there is no ``--copula`` flag here on purpose.

Conventions copied from ``train_semapop_gan_v2.py``
---------------------------------------------------
* F2  val is held out: the structure, the CPTs and the numeric decoder are fit
      on **train only**; val is used solely for model selection.
* F5  ``--seed`` seeds python/numpy (and torch when it is importable);
      ``--sample-seed`` is separate; the checkpoint stores the FULL effective
      config plus the fitted network and decoder; every run writes
      ``metrics.json``.
* F6  numeric outputs are decoded from their bin, clipped to the **training**
      support, and only then rounded (integer columns only).
* selection is on the val split with the frozen ``EvalConfig``
      (SRMSE-1/2 + block P/R), never on the reference.
* out-root defaults to ``<semapop_repo>/modelSave``; the run
      directory is ``<tag>_<timestamp>``.

Example
-------
    python training/train_bn_v2.py \
        --data-dir <semapop_repo>/data/pums_v2/CA \
        --eval-config configs/eval_v2_pums20.yaml \
        --max-parents 3 --score bic --n-samples 200000 --seed 42
"""

from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); import popaudit_paths as _PA  # release bootstrap: repository root first on sys.path (see popaudit_paths.py)

import argparse
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
import yaml

sys.path.insert(0, _PA.R(""))

import utils.eval_v2 as ev  # noqa: E402
from utils.ensemble_v2 import add_ensemble_args, write_ensemble  # noqa: E402
from model.bn_v2 import (  # noqa: E402
    DiscreteBN,
    NumericDecoder,
    build_numeric_decoder,
    learn_structure,
)

DEFAULT_EVAL_CONFIG = os.path.join(_PA.R(""),
                                   "configs", "eval_v3_pums12.yaml")


# --------------------------------------------------------------------------- #
# F5: determinism
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed python / numpy (and torch if importable -- the BN itself is CPU-only)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001  -- torch is not required by this trainer
        pass


def _save_ckpt(obj: Dict[str, Any], path: str) -> str:
    """torch.save when torch is available (same artefact shape as the GAN
    trainer), otherwise a plain pickle with a ``.pkl`` suffix."""
    try:
        import torch
        torch.save(obj, path)
        return path
    except Exception:  # noqa: BLE001
        alt = os.path.splitext(path)[0] + ".pkl"
        with open(alt, "wb") as f:
            pickle.dump(obj, f)
        return alt


# --------------------------------------------------------------------------- #
# F6: bin code -> value  (the only place where the BN leaves the discrete space)
#
# ``NumericDecoder`` / ``build_numeric_decoder`` live in ``model.bn_v2`` so that
# a checkpoint written by ``python training/train_bn_v2.py`` (module ``__main__``) can be
# unpickled from anywhere; the checkpoint additionally stores ``dec.to_dict()``
# so it can be read without importing this repo's classes at all.
# --------------------------------------------------------------------------- #
def decode_codes(codes: np.ndarray, cfg: ev.EvalConfig,
                 decoders: Dict[str, NumericDecoder],
                 rng: np.random.Generator) -> pd.DataFrame:
    """``(n, m)`` bin codes -> a dataframe in the ORIGINAL value space."""
    out: Dict[str, np.ndarray] = {}
    for j, c in enumerate(cfg.order):
        col = codes[:, j]
        if c in decoders:
            v = decoders[c].decode(col, rng)
            if decoders[c].is_integer:
                # nullable Int64 when the decoder can emit NaN (na_cell specs); a plain
                # int64 cast would turn a missing value into -9223372036854775808.
                out[c] = (pd.array(v, dtype="Int64") if np.isnan(v).any()
                          else v.astype(np.int64))
            else:
                out[c] = v
        else:
            cats = np.asarray(cfg.specs[c].categories, dtype=object)
            out[c] = cats[col]
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# evaluation (frozen protocol only)
# --------------------------------------------------------------------------- #
def eval_metrics(real_enc, gen_df: pd.DataFrame, cfg: ev.EvalConfig,
                 do_2way: bool = True, per_column: bool = False) -> Dict[str, Any]:
    """SRMSE-1/2 + block P/R/F1 under the frozen ``EvalConfig``."""
    gen_enc = cfg.encode(gen_df)
    m: Dict[str, Any] = {}
    r1 = ev.srmse_kway(real_enc, gen_enc, cfg, k=1)
    m["srmse_1"] = float(r1["mean"])
    m["srmse_1_max"] = float(r1["max"])
    if per_column:
        m["srmse_1_per_column"] = {row.attrs: float(row.srmse)
                                   for row in r1["per_j"].itertuples()}
    if do_2way:
        r2 = ev.srmse_kway(real_enc, gen_enc, cfg, k=2)
        m["srmse_2"] = float(r2["mean"])
        m["srmse_2_max"] = float(r2["max"])
    else:
        m["srmse_2"] = float("nan")
    pr = ev.block_precision_recall(real_enc, gen_enc, cfg)
    blocks = {}
    f1s = []
    for b, v in pr.items():
        blocks[b] = {k: (float(x) if isinstance(x, (int, float, np.floating)) else x)
                     for k, x in v.items()}
        if b != "all":
            f1s.append(float(v["f1"]))
    m["blocks"] = blocks
    m["block_f1"] = float(np.mean(f1s)) if f1s else float("nan")
    return m


def selection_score(m: Dict[str, Any], f1_weight: float = 1.0) -> float:
    """Lower is better: 0.5*(SRMSE-1 + SRMSE-2) - w*block-F1 (same rule as the
    GAN trainer, so baseline rows are selected under one policy)."""
    s1 = float(m.get("srmse_1", np.nan))
    s2 = float(m.get("srmse_2", np.nan))
    base = 0.5 * (s1 + s2) if np.isfinite(s2) else s1
    f1 = float(m.get("block_f1", np.nan))
    if np.isfinite(f1):
        base -= f1_weight * f1
    return float(base)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@dataclass
class PreparedData:
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    cfg: ev.EvalConfig
    cols: List[str]
    cards: List[int]
    codes_train: np.ndarray
    fit_weights: Optional[np.ndarray]
    val_enc: Any
    decoders: Dict[str, NumericDecoder]
    n_train_oos: int
    n_val_oos: int


def _usecols(cols: Sequence[str], extra: Optional[str]) -> List[str]:
    out = list(cols)
    if extra and extra not in out:
        out.append(extra)
    return out


def prepare_data(args: argparse.Namespace) -> PreparedData:
    cfg = ev.EvalConfig.from_yaml(args.eval_config)
    if not cfg.is_fitted:
        raise RuntimeError(
            f"{args.eval_config} has no stored category lists; the frozen protocol "
            "must ship fitted specs so the BN state space is reproducible")
    cols = list(cfg.order)

    d = args.data_dir
    cc = os.path.join(d, getattr(args, "cols_config", "cols_config.yaml"))
    if os.path.exists(cc):
        with open(cc, encoding="utf-8") as f:
            declared = yaml.safe_load(f)
        modelled = list(declared.get("num_cols", [])) + list(declared.get("cat_cols", []))
        missing = [c for c in modelled if c not in cols]
        extra = [c for c in cols if c not in modelled]
        if missing or extra:
            print(f"[warn] eval config columns differ from cols_config.yaml "
                  f"(missing={missing}, extra={extra})")

    df_train = pd.read_csv(os.path.join(d, args.train_file),
                           usecols=_usecols(cols, args.fit_weight_col or None))
    df_val = pd.read_csv(os.path.join(d, args.val_file),
                         usecols=_usecols(cols, args.val_weight_col or None))

    if args.subsample and args.subsample < len(df_train):
        rng = np.random.default_rng(args.data_seed)
        keep = np.sort(rng.choice(len(df_train), size=args.subsample, replace=False))
        df_train = df_train.iloc[keep].reset_index(drop=True)
    if args.subsample_val and args.subsample_val < len(df_val):
        rng = np.random.default_rng(args.data_seed + 1)
        keep = np.sort(rng.choice(len(df_val), size=args.subsample_val, replace=False))
        df_val = df_val.iloc[keep].reset_index(drop=True)

    # ---- discretise with the FROZEN protocol ---------------------------------
    enc_train = cfg.encode(df_train[cols])
    codes_train = enc_train.codes.astype(np.int32)
    ok = enc_train.valid.all(axis=1)
    n_train_oos = int((~ok).sum())
    if n_train_oos:
        print(f"[data] {n_train_oos} train rows had an out-of-support cell under the "
              f"frozen config -> dropped from the CPT/structure fit")
        codes_train = codes_train[ok]
        df_train = df_train.loc[ok].reset_index(drop=True)

    fit_w = None
    if args.fit_weight_col:
        fit_w = df_train[args.fit_weight_col].to_numpy(dtype=np.float64)

    val_enc = cfg.encode(df_val[_usecols(cols, args.val_weight_col or None)])
    n_val_oos = int((~val_enc.valid.all(axis=1)).sum())

    cards = [int(cfg.specs[c].n_categories) for c in cols]

    # ---- F6 decoder, fit on TRAIN only -------------------------------------
    decoders: Dict[str, NumericDecoder] = {}
    for j, c in enumerate(cols):
        if cfg.specs[c].kind == "categorical":
            continue
        vals = pd.to_numeric(df_train[c], errors="coerce").to_numpy(dtype=np.float64)
        decoders[c] = build_numeric_decoder(c, cfg.specs[c], vals, codes_train[:, j],
                                            args.numeric_decode)

    print(f"[data] train={codes_train.shape} val={val_enc.codes.shape} "
          f"cols={len(cols)} state-space={int(np.prod([float(x) for x in cards])):.3g} cells")
    print("[bins] " + ", ".join(f"{c}={k}" for c, k in zip(cols, cards)))
    return PreparedData(df_train, df_val, cfg, cols, cards, codes_train, fit_w,
                        val_enc, decoders, n_train_oos, n_val_oos)


# --------------------------------------------------------------------------- #
# fit one grid point
# --------------------------------------------------------------------------- #
def fit_one(data: PreparedData, max_parents: int, score: str, alpha: float,
            args: argparse.Namespace) -> Tuple[DiscreteBN, Any]:
    st = learn_structure(
        data.codes_train, data.cards, data.cols,
        backend=args.structure_backend,
        max_parents=max_parents, score=score, ess=args.ess,
        weights=data.fit_weights,
        fixed_edges=[tuple(e.split("->")) for e in (args.fixed_edge or [])],
        forbidden_edges=[tuple(e.split("->")) for e in (args.forbidden_edge or [])],
        max_iter=args.max_iter, tol=args.tol, verbose=not args.quiet,
    )
    bn = DiscreteBN(data.cols, data.cards, st.parents)
    bn.fit(data.codes_train, alpha=alpha, alpha_mode=args.alpha_mode,
           weights=data.fit_weights, ess=args.ess)
    return bn, st


def sample_df(bn: DiscreteBN, n: int, data: PreparedData, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    codes = bn.sample(n, rng)
    return decode_codes(codes, data.cfg, data.decoders, rng)


# --------------------------------------------------------------------------- #
# main run
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace, data: PreparedData, save_dir: str) -> Dict[str, Any]:
    t0 = time.time()
    cfg = data.cfg
    n_val_eval = args.eval_n if args.eval_n > 0 else len(data.df_val)

    grid = [(mp, sc, al) for mp in args.max_parents for sc in args.score
            for al in args.dirichlet_alpha]
    print(f"[grid] {len(grid)} configuration(s): {grid}")

    history: List[Dict[str, Any]] = []
    best = {"score": float("inf"), "point": None, "metrics": {},
            "bn": None, "structure": None}

    for (mp, sc, al) in grid:
        print(f"\n[fit] max_parents={mp} score={sc} alpha={al}")
        bn, st = fit_one(data, mp, sc, al, args)
        df_gen = sample_df(bn, n_val_eval, data, args.sample_seed)
        m = eval_metrics(data.val_enc, df_gen, cfg, do_2way=not args.no_2way)
        s = selection_score(m, args.f1_weight)
        rec = {
            "max_parents": mp, "score_fn": sc, "dirichlet_alpha": al,
            "n_edges": len(st.edges), "n_params": int(bn.n_params),
            "structure_score_start": st.score_start,
            "structure_score_final": st.score_final,
            "structure_score_improvement": st.score_final - st.score_start,
            "structure_n_iter": st.n_iter, "structure_converged": st.converged,
            "structure_wall_clock_s": st.wall_clock_s,
            "val_srmse_1": m["srmse_1"], "val_srmse_2": m["srmse_2"],
            "val_block_f1": m["block_f1"],
            "val_block_precision": {b: v["precision"] for b, v in m["blocks"].items()},
            "val_block_recall": {b: v["recall"] for b, v in m["blocks"].items()},
            "val_score": s,
        }
        print(f"[fit] edges={len(st.edges)} params={bn.n_params} | "
              f"val srmse1={m['srmse_1']:.4f} srmse2={m['srmse_2']:.4f} "
              f"blockF1={m['block_f1']:.4f} score={s:.4f}")
        if s < best["score"]:
            best = {"score": s, "point": (mp, sc, al), "metrics": m,
                    "bn": bn, "structure": st}
            rec["is_best"] = True
        history.append(rec)

    bn: DiscreteBN = best["bn"]
    st = best["structure"]
    print(f"\n[select] best = max_parents={best['point'][0]} score={best['point'][1]} "
          f"alpha={best['point'][2]} (val score {best['score']:.4f})")

    # ---- final sample -----------------------------------------------------
    n_out = args.n_samples if args.n_samples > 0 else len(data.df_val)
    df_out = sample_df(bn, n_out, data, args.sample_seed)
    sample_csv = os.path.join(save_dir, "generated_samples.csv")
    df_out.to_csv(sample_csv, index=False)

    # ---- final val metrics (selection split) ------------------------------
    final_val = eval_metrics(data.val_enc, df_out, cfg, do_2way=not args.no_2way,
                             per_column=True)

    # ---- reference metrics (never used for selection) ---------------------
    ref_block: Dict[str, Any] = {}
    if args.ref_file:
        ref_path = os.path.join(args.data_dir, args.ref_file)
        df_ref = pd.read_csv(ref_path, usecols=_usecols(data.cols,
                                                        args.ref_weight_col or None))
        ref_enc_w = cfg.encode(df_ref[_usecols(data.cols, args.ref_weight_col or None)])
        ref_enc_u = ev.Encoded(ref_enc_w.codes, ref_enc_w.valid, ref_enc_w.sizes,
                               list(ref_enc_w.cols), None)
        print(f"[ref] {ref_path} n={len(df_ref)} "
              f"weights={'PWGTP' if ref_enc_w.weights is not None else 'none'}")
        ref_block["n_ref"] = int(len(df_ref))
        ref_block["weight_col"] = (args.ref_weight_col
                                   if ref_enc_w.weights is not None else None)
        ref_block["weighted"] = eval_metrics(ref_enc_w, df_out, cfg,
                                             do_2way=not args.no_2way, per_column=True)
        ref_block["unweighted"] = eval_metrics(ref_enc_u, df_out, cfg,
                                               do_2way=not args.no_2way)
        if not args.no_coverage:
            cov = ev.support_coverage(ref_enc_u, cfg.encode(df_out), cfg, k=2,
                                      train_df=cfg.encode(data.df_train[data.cols]))
            cov.pop("per_j", None)          # 190 rows; keep metrics.json readable
            ref_block["support_coverage_k2"] = ev._jsonable(cov)

    # ---- record validity test ---------------------------------------------
    val_rules_gen = ev.record_validity_rate(df_out)
    val_rules_train = ev.record_validity_rate(data.df_train[data.cols])

    # F6 evidence
    oos_post = {}
    for c, dec in data.decoders.items():
        v = df_out[c].to_numpy(dtype=np.float64)
        oos_post[c] = float(np.mean((v < dec.lo - 1e-9) | (v > dec.hi + 1e-9)))

    ckpt = {
        "config": vars(args),
        "bn": bn.to_dict(),
        "structure": st.to_dict(),
        "eval_config": cfg.to_dict(),
        "cols": data.cols,
        "cards": data.cards,
        "decoders": {c: d.to_dict() for c, d in data.decoders.items()},
        "val_metrics": {k: v for k, v in final_val.items() if k != "srmse_1_per_column"},
        "numpy_version": np.__version__,
    }
    ckpt_path = _save_ckpt(ckpt, os.path.join(save_dir, "bn_best.pt"))

    # ---- K=50 audit ensemble (utils/ensemble_v2 contract; finalize only) ----
    # sample_df already returns label space via decode_codes: the 12 modelled
    # columns in cols_config_v3 order (cfg.order), numeric columns decoded from
    # their bin with the SAME train-fit decoders/policy as generated_samples.csv.
    ens_paths: List[str] = []
    if getattr(args, "ensemble", False):
        n_ens = args.ensemble_n if args.ensemble_n > 0 else int(ref_block.get("n_ref") or 0)
        if n_ens <= 0:
            raise RuntimeError("--ensemble needs --ensemble-n > 0 or a --ref-file so "
                               "N can default to the reference population size")
        ens_paths = write_ensemble(lambda seed, n: sample_df(bn, n, data, seed),
                                   n_samples=n_ens, save_dir=save_dir,
                                   k=args.ensemble_k)

    metrics = {
        "run": {
            "save_dir": save_dir,
            "started": datetime.now().isoformat(timespec="seconds"),
            "wall_clock_s": round(time.time() - t0, 2),
            "device": "cpu",
            "numpy_version": np.__version__,
        },
        "config": vars(args),
        "data": {
            "n_train": int(data.codes_train.shape[0]),
            "n_val": int(len(data.df_val)),
            "cols": data.cols,
            "cards": data.cards,
            "n_train_rows_dropped_oos": data.n_train_oos,
            "n_val_rows_with_oos_cell": data.n_val_oos,
            "held_out_val": True,             # F2
            "fit_weight_col": args.fit_weight_col or None,
        },
        # the binning is the frozen protocol's, echoed verbatim
        "binning": {c: cfg.specs[c].to_dict() for c in data.cols},
        "eval_config": cfg.to_dict(),
        "structure": st.to_dict(),
        "network": {
            "n_edges": len(bn.edges()),
            "edges": [list(e) for e in bn.edges()],
            "parents": {k: list(v) for k, v in bn.parents.items()},
            "topological_order": [bn.names[i] for i in bn.order],
            "n_free_params": int(bn.n_params),
            "dirichlet_alpha": bn.alpha,
            "empty_parent_config_share": bn.empty_config_share,
        },
        "selection": {
            "rule": "min 0.5*(val SRMSE-1 + SRMSE-2) - w*block_F1 "
                    "(block F1 excludes the all-tuple block); val split only",
            "metric_backend": "eval_v2",
            "eval_config_path": args.eval_config,
            "best_point": {"max_parents": best["point"][0], "score_fn": best["point"][1],
                           "dirichlet_alpha": best["point"][2]},
            "best_score": best["score"],
            "grid": history,
        },
        "final_val_metrics": final_val,
        "reference_metrics": ref_block,
        "record_validity": {
            "generated": ev._jsonable(val_rules_gen),
            "train_reference": ev._jsonable(val_rules_train),
        },
        "out_of_support_rate_after_clip": oos_post,
        "numeric_decode": args.numeric_decode,
        "outputs": {"samples_csv": sample_csv, "checkpoint": ckpt_path,
                    "ensemble_dir": (os.path.join(save_dir, "generated_ensembles")
                                     if ens_paths else None),
                    "ensemble_k": len(ens_paths)},
        "notes": {
            "bn_copula": "DROPPED (D14). train_bn.py:246-256 fitted the target "
                         "CopulaScaler on real_test_data, i.e. the inverse-ECDF used "
                         "to map generated values back to the target scale was "
                         "estimated on the evaluation set. Not ported.",
            "discretisation": "frozen protocol, see 'binning' (R3-5).",
        },
    }
    with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(ev._jsonable(metrics), f, indent=2, default=str)
    print(f"[done] samples  -> {sample_csv}")
    print(f"[done] ckpt     -> {ckpt_path}")
    print(f"[done] metrics  -> {os.path.join(save_dir, 'metrics.json')}")
    return metrics


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bayesian-network baseline v2 (structure learning + frozen "
                    "discretisation; R2-1.1, R3-5)")
    # data
    p.add_argument("--data-dir",
                   default=_PA.S("/data/pums_states/CA"))
    p.add_argument("--train-file", default="traindata_pers.csv")
    p.add_argument("--cols-config", default="cols_config.yaml")
    p.add_argument("--val-file", default="valdata_pers.csv")
    p.add_argument("--ref-file", default="referencedata_pers.csv",
                   help="reference split for the final (non-selection) numbers; "
                        "'' to skip")
    p.add_argument("--subsample", type=int, default=0)
    p.add_argument("--subsample-val", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=42)
    # discretisation / eval protocol
    p.add_argument("--eval-config", default=DEFAULT_EVAL_CONFIG,
                   help="FROZEN utils.eval_v2 EvalConfig yaml; its binning IS the "
                        "BN state space (R3-5)")
    p.add_argument("--fit-weight-col", default="",
                   help="column of survey weights used when fitting the BN "
                        "(default: unweighted, matching train_semapop_gan_v2.py)")
    p.add_argument("--val-weight-col", default="",
                   help="weights for the val split during selection (default: none, "
                        "matching train_semapop_gan_v2.py)")
    p.add_argument("--ref-weight-col", default="PWGTP",
                   help="weights for the reference split in the final metrics "
                        "(the frozen config declares PWGTP); '' to disable")
    # structure learning
    p.add_argument("--structure-backend", default="numpy_hc",
                   help="model.bn_v2.STRUCTURE_BACKENDS key (a pgmpy backend can be "
                        "registered there without touching this trainer)")
    p.add_argument("--max-parents", type=int, nargs="+", default=[3],
                   help="parent cap; several values = a selection grid")
    p.add_argument("--score", nargs="+", default=["bic"], choices=["bic", "bdeu"])
    p.add_argument("--ess", type=float, default=10.0,
                   help="BDeu equivalent sample size (also used by --alpha-mode bdeu)")
    p.add_argument("--max-iter", type=int, default=400)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--fixed-edge", nargs="*", default=[],
                   help="edges forced into the DAG, format 'Parent->Child'")
    p.add_argument("--forbidden-edge", nargs="*", default=[],
                   help="edges never added, format 'Parent->Child'")
    # CPTs
    p.add_argument("--dirichlet-alpha", type=float, nargs="+", default=[1.0],
                   help="Laplace/Dirichlet pseudo-count per cell (grid allowed)")
    p.add_argument("--alpha-mode", choices=["fixed", "bdeu"], default="fixed",
                   help="'bdeu' uses alpha = ess/(q*r) instead of the fixed value")
    # eval / selection
    p.add_argument("--eval-n", type=int, default=0,
                   help="rows sampled for val selection (0 = size of the val split)")
    p.add_argument("--no-2way", action="store_true")
    p.add_argument("--no-coverage", action="store_true")
    p.add_argument("--f1-weight", type=float, default=1.0)
    # sampling / io
    p.add_argument("--numeric-decode", choices=["empirical", "midpoint"],
                   default="empirical")
    p.add_argument("--n-samples", type=int, default=0,
                   help="0 = size of the val split")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-seed", type=int, default=1234)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--out-root", default=_PA.S("/modelSave"))
    p.add_argument("--tag", default="bn_v2")
    p.add_argument("--run-name", default=None)
    add_ensemble_args(p)
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    run_name = args.run_name or f"{args.tag}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    save_dir = os.path.join(args.out_root, run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[run] {save_dir}")

    set_seed(args.seed)
    data = prepare_data(args)
    return run(args, data, save_dir)


if __name__ == "__main__":
    main()
