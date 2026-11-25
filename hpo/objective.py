"""The ONE objective every model in the SemaPop v2 HPO is scored by.

Convention -- stated once here, used everywhere (run.py, finalize.py, README):

    score = -( 0.5 * (SRMSE_1 + SRMSE_2) ) + W_F1 * mean_block_F1        MAXIMISED

* ``SRMSE_1`` / ``SRMSE_2``  : ``utils.eval_v2.srmse_kway(..., k=1|2)["mean"]``
  under the FROZEN eval config (``configs/eval_v3_pums12.yaml`` in this release; unweighted).
* ``mean_block_F1``          : mean of ``utils.eval_v2.block_precision_recall``
  F1 over the three blocks (demographic / household / commute).  The ``all``
  block -- full-tuple exact match -- is EXCLUDED: on 20 de-duplicated PUMS
  attributes it is support-dominated (~80% singleton tuples) and is reported in
  the paper only as a floor diagnostic against oracle rows.
* ``W_F1 = 1.0``             : identical for every model and every state.
* Both terms are computed on the HELD-OUT VALIDATION SPLIT ONLY.  The reference
  population is never opened during a trial -- enforced by
  ``hpo/guard/sitecustomize.py``, not merely asserted in prose.

This is exactly ``-1 x`` the trainers' internal ``selection_score`` at
``--f1-weight 1.0``, so per-epoch checkpoint selection inside a trial and
between-trial selection by Optuna optimise the same quantity (higher = better
here, lower = better there).
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, Optional, Tuple

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.criteria as CR                                        # noqa: E402

#: Weight on mean block F1.  ZERO since 2026-08-25.
#:
#: Four of the five population-synthesis papers we benchmark against select
#: hyperparameters on validation SRMSE and nothing else (Borysov/Garrido 2020,
#: Jutras-Dube 2024, and the VAE/CVAE study which ranks on SRMSE-1 then SRMSE-2).  The
#: fifth, Kim & Bansal 2023, selects on F1 alone -- but their F1 is whole-row exact
#: matching, which at 50 attributes scores 0.00024 for the best of 108 trials because
#: 99.6% of reference rows are unique, so it cannot be a criterion here.
#:
#: The composite it replaces was also indefensible on its own terms: the weight of 1.0
#: had no source in the literature, and the block F1 it weighted was computed over
#: 20- and 26-column blocks whose F1 sat at 0.017 for every model -- two thirds of the
#: objective was a constant.  Fidelity and diversity are now carried by NIP and NIR --
#: reported SEPARATELY, with no IF1 harmonic mean, because IF1 ranks models identically to
#: NIP alone while hiding the one thing the pair diagnoses (low NIP with high NIR is a
#: diffuse generator; high NIP with low NIR is mode collapse) -- and by record validity.
#: All three are reported but not optimised; a second selection under NIP from the same
#: trial pool goes in the appendix.
W_F1 = 0.0


WORST = -1e6          # value handed back for a crashed/degenerate trial


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
DEFAULT_CRITERION = CR.DEFAULT


def score_from_metrics(m: Dict[str, Any], w_f1: float = W_F1,
                       criterion: str = DEFAULT_CRITERION) -> float:
    """Apply this model's criterion to one ``final_val_metrics``-shaped dict.

    The formula lives in utils.criteria so the TRAINER's per-epoch checkpoint pick and
    Optuna's between-trial pick read the same definition; a criterion that differed between
    the two would select a checkpoint on one quantity and a configuration on another.
    """
    return CR.score(criterion, m, w_f1)


def criterion_doc(criterion: str = DEFAULT_CRITERION) -> str:
    """Provenance string recorded on the study, and quoted in the paper's protocol table."""
    return (f"{CR.doc(criterion)}; computed on the held-out val split with the condition's "
            f"frozen eval config; fidelity/diversity (NIP and NIR, reported separately) and "
            f"record validity are reported but not optimised; the reference split is never "
            f"opened during a trial.")


#: kept as a module-level name: five call sites read it for the study's user_attrs
OBJECTIVE_DOC = criterion_doc(CR.DEFAULT)


def selection_score_to_objective(s: float) -> float:
    """Trainer ``selection_score`` (lower better) -> HPO objective (higher better)."""
    return float(-s)


# --------------------------------------------------------------------------- #
# reading a finished trial
# --------------------------------------------------------------------------- #
def read_metrics_json(run_dir: str) -> Dict[str, Any]:
    p = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"trainer produced no metrics.json in {run_dir}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def assert_no_reference(metrics: Dict[str, Any]) -> None:
    """Second line of defence behind the sitecustomize guard.

    The guard makes a reference read impossible; this checks the *intent*
    recorded in the trainer's own config, so a future trainer that silently
    grows a reference path still fails the trial loudly.
    """
    cfg = metrics.get("config") or {}
    if "ref_eval" in cfg:
        # ctgan-style: --ref-file is always populated, --no-ref-eval is the switch
        if cfg["ref_eval"]:
            raise AssertionError("trial ran with --ref-eval: the objective would see "
                                 "the reference split")
    elif str(cfg.get("ref_file", "") or "").strip():
        # bn-style: an empty --ref-file is the switch
        raise AssertionError(f"trial ran with --ref-file={cfg['ref_file']!r}")
    if metrics.get("reference_metrics"):
        raise AssertionError("trial metrics.json contains reference_metrics")


def objective_from_run(run_dir: str, w_f1: float = W_F1,
                       criterion: str = DEFAULT_CRITERION,
                       strict: bool = True) -> Tuple[float, Dict[str, Any]]:
    """Score a finished trial directory.

    Primary source is ``tail_val_metrics`` -- the MEAN of the last ``tail_k``
    in-training evaluations.  Measured run-to-run sigma on val SRMSE-1:

        last single evaluation   9.3%     <- what this used to read
        mean of the last five    5.2%     <- what it reads now

    TPE fits its surrogate on these values *during* the search, so objective noise
    corrupts the sampling path, not merely the final ranking -- a top-k rerun cannot
    repair a search that concentrated on lucky regions.  Tail averaging costs nothing
    and beats a 2-seed mean (6.6%) that would cost twice the GPU budget.

    Falls back to ``final_val_metrics`` then ``selection.best_metrics`` for runs
    produced before the trainer emitted the tail block.  Note that
    ``selection.best_metrics`` is a best-of-~30 selection on a metric with ~9%
    noise and therefore carries a ~2-sigma optimistic bias; it is a last resort.
    """
    m = read_metrics_json(run_dir)
    if strict:
        assert_no_reference(m)
    score, vals, src = None, None, None
    for cand, getter in (("tail_val_metrics", lambda: m.get("tail_val_metrics") or {}),
                         ("final_val_metrics", lambda: m.get("final_val_metrics") or {}),
                         ("selection.best_metrics",
                          lambda: (m.get("selection") or {}).get("best_metrics") or {})):
        v = getter()
        try:
            score = score_from_metrics(v, w_f1, criterion)
        except ValueError:
            continue
        vals, src = v, cand
        break
    if score is None:
        raise ValueError(f"no usable metric block in {run_dir}")
    info = {
        "objective": score,
        "objective_source": src,
        "criterion": criterion,
        # every criterion computable on this trial, so a second selection under a different
        # one can be run from the same pool without retraining -- the appendix comparison
        "all_criteria": CR.score_all(vals, w_f1),
        "NIP": vals.get("NIP"),
        "NIR": vals.get("NIR"),
        "ml_eff": vals.get("ml_eff"),
        "srmse_1": vals.get("srmse_1"),
        "srmse_2": vals.get("srmse_2"),
        "block_f1": vals.get("block_f1"),
        "block_f1_demographic": vals.get("block_f1_demographic"),
        "block_f1_household": vals.get("block_f1_household"),
        "block_f1_commute": vals.get("block_f1_commute"),
        "best_epoch": (m.get("selection") or {}).get("best_epoch"),
        "trainer_wall_clock_s": (m.get("run") or {}).get("wall_clock_s"),
        "metric_backend": m.get("final_metric_backend")
                          or (m.get("selection") or {}).get("metric_backend"),
        "n_train": (m.get("data") or {}).get("n_train"),
        "n_val": (m.get("data") or {}).get("n_val"),
    }
    return score, info


# --------------------------------------------------------------------------- #
# intermediate values for pruning
# --------------------------------------------------------------------------- #
# Every v2 trainer prints one evaluation line per eval epoch that ends with
# "score=<trainer selection_score>" (lower = better).  We parse it rather than
# patching five trainers, so pruning needs no trainer change at all.
_EP_RE = re.compile(r"\[ep\s+(\d+)\]")
_SCORE_RE = re.compile(r"score=([-+0-9.eE]+)")
# Gate on the token whose VALUE is parsed, plus the epoch marker.  The old gate was
# "val_srmse1=", which since the per-model criteria has nothing to do with what
# score= carries: trimming the srmse print for an if1 model would kill pruning
# silently, with no error anywhere.
_MARK_RE = re.compile(r"score=[-+0-9.eE]+")


def parse_intermediate(line: str) -> Optional[Tuple[int, float]]:
    """-> (step, objective) for a trainer progress line, else None.

    ``step`` is the epoch number when the trainer prints one (GAN/DDPM), and is
    left to the caller's running counter for the grid-style trainers (BN).
    """
    # BOTH markers are required.  score= alone would swallow the trainer's one-off
    # "[select] loaded ck_last.pt from epoch 700 (best score=...)" line and report an
    # end-of-run value as an intermediate one; the epoch bracket alone would not tell us
    # which number on the line to read.
    me = _EP_RE.search(line)
    if me is None or not _MARK_RE.search(line):
        return None
    ms = _SCORE_RE.search(line)
    if not ms:
        return None
    try:
        sel = float(ms.group(1))
    except ValueError:
        return None
    if not math.isfinite(sel):
        return None
    return int(me.group(1)), selection_score_to_objective(sel)
