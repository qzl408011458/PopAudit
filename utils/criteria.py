"""The hyperparameter-selection criterion, one per model, defined once.

WHY THIS EXISTS.  Until 2026-08-25 every model in the HPO was ranked by ONE objective of
ours -- validation SRMSE.  Baseline implementations must be trustworthy, and choosing a
baseline's hyperparameters by OUR metric rather than the one its own authors used undermines
that.  So each model is now selected the way its own paper selects, and where the paper's
criterion cannot survive 50 attributes the substitution is named in ``source`` rather than
left implicit.

WHAT EACH MODEL GETS, and why (evidence gathered 2026-08-25 from the papers and their
reference implementations):

  BN, CTGAN     no criterion at all -- both are reproduced at their published defaults and
                are NOT searched.  Jutras-Dube et al. (2024 s3.1) call pomegranate's
                from_samples(algorithm='greedy') with every default; Xu et al. (2019 s4.4,
                s5.3) fix one configuration across all fifteen benchmark datasets, and
                `ctgan` 0.11.1's fit() has no validation pass, no early stopping and no
                best-epoch tracking.  See hpo/spaces.py, no_hpo=True.
  wgan_zcr      "if1".  Kim & Bansal select on their Eq. 16 F1, the harmonic mean of
                whole-row exact-tuple precision and recall.  That estimator is dead at 50
                attributes (0.000031 for our BN, 0.00024 for the best of 108 trials, because
                99.6% of reference rows are unique), so it is instantiated by the manifold
                pair: the harmonic mean of NIP and NIR from utils.coverage_pr.  The four
                structural commitments of the original survive -- precision as feasibility,
                recall as diversity, a harmonic mean because gamma_BD and gamma_AD trade
                them off, and both measured against real data.
                NOTE the apparent contradiction with the reporting change made the same day:
                we stopped REPORTING IF1 because it ranks models identically to NIP alone
                while hiding the diffuse/collapsed diagnosis.  Selecting on it is a
                different act -- it reproduces an author's criterion -- and is documented
                as such.
  tabddpm       "ml_eff".  Kotelnikov et al. tune by MAXIMISING downstream machine-learning
                efficiency: train CatBoost on the synthetic sample, score it on a held-out
                real split, average over 5 sampling seeds.  We have no downstream task, so
                the substitution is a pre-registered panel of PUMS attributes used as
                surrogate targets (utils/ml_efficacy.py).  Their second commitment matters
                as much as the metric: TabDDPM does NO in-training model selection -- a
                fixed number of steps, final weights, with `steps` itself searched -- so its
                trials must run with --load-final.
  semapop       "val_srmse".  The SemaPop authors report one hand-chosen configuration
                per instantiation and no search.  The criterion is therefore adopted, not
                reproduced, and its provenance is the lineage SemaPop belongs to --
                Borysov & Garrido (2020) and Jutras-Dube et al. (2024) select on
                validation SRMSE.  The hand-chosen configuration is kept as a documented
                extra row so that the headline is not a no-search row while baselines
                get a search.

CONVENTION.  Every criterion here returns HIGHER IS BETTER, so it can be handed to Optuna
directly.  ``selection_score`` is its negation, which is the trainers' existing convention
(lower is better), so per-epoch checkpoint selection inside a trial and between-trial
selection by Optuna optimise the same quantity.

NO I/O AND NO TORCH in this module: the trainers import it, and a heavyweight import here
would be paid by every trial.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple


def _f(m: Mapping[str, Any], k: str) -> float:
    try:
        return float(m.get(k, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def _val_srmse(m: Mapping[str, Any], f1_weight: float = 0.0) -> float:
    s1, s2 = _f(m, "srmse_1"), _f(m, "srmse_2")
    if not math.isfinite(s1):
        raise ValueError("srmse_1 missing/non-finite: cannot score this trial")
    base = 0.5 * (s1 + s2) if math.isfinite(s2) else s1
    f1 = _f(m, "block_f1")
    if f1_weight and math.isfinite(f1):
        base -= f1_weight * f1
    return float(-base)


def _if1(m: Mapping[str, Any], f1_weight: float = 0.0) -> float:
    nip, nir = _f(m, "NIP"), _f(m, "NIR")
    if not (math.isfinite(nip) and math.isfinite(nir)):
        raise ValueError("NIP/NIR missing/non-finite: cannot score this trial")
    if nip + nir <= 0:
        return 0.0
    return float(2.0 * nip * nir / (nip + nir))


def _ml_eff(m: Mapping[str, Any], f1_weight: float = 0.0) -> float:
    v = _f(m, "ml_eff")
    if not math.isfinite(v):
        raise ValueError("ml_eff missing/non-finite: cannot score this trial")
    return float(v)


@dataclass(frozen=True)
class Criterion:
    name: str
    keys: Tuple[str, ...]
    fn: Callable[..., float]
    doc: str
    source: str


CRITERIA: Dict[str, Criterion] = {
    "val_srmse": Criterion(
        "val_srmse", ("srmse_1", "srmse_2"), _val_srmse,
        "score = -(0.5*(SRMSE_1 + SRMSE_2)); MAXIMISED; held-out val split, the "
        "condition's frozen eval config, unweighted (D19)",
        "adopted, not reproduced: Borysov & Garrido (2020) and Jutras-Dube et al. (2024) "
        "select on validation SRMSE; SemaPop's own submission reports no search"),
    "if1": Criterion(
        "if1", ("NIP", "NIR"), _if1,
        "score = harmonic mean of NIP and NIR (utils.coverage_pr); MAXIMISED; held-out "
        "val split",
        "substituted for Kim & Bansal (2023) Eq. 16, the harmonic mean of whole-row "
        "exact-tuple precision and recall, which is degenerate at 50 attributes "
        "(best-of-108-trials F1 = 0.00024; 99.6% of reference rows are unique)"),
    "ml_eff": Criterion(
        "ml_eff", ("ml_eff",), _ml_eff,
        "score = mean macro-F1 over the pre-registered surrogate target panel, classifier "
        "trained on the synthetic sample and scored on the held-out val split; MAXIMISED",
        "substituted for Kotelnikov et al. (2023), who maximise downstream ML efficiency "
        "with CatBoost on a real downstream task; we have no downstream task, so the "
        "targets are PUMS attributes predicted from the remaining ones "
        "(utils/ml_efficacy.py)"),
}

DEFAULT = "val_srmse"


def get(name: str) -> Criterion:
    if name not in CRITERIA:
        raise ValueError(f"unknown criterion {name!r}; available: {sorted(CRITERIA)}")
    return CRITERIA[name]


def score(name: str, m: Mapping[str, Any], f1_weight: float = 0.0) -> float:
    """HIGHER is better.  Raises ValueError when a required key is missing/non-finite."""
    return get(name).fn(m, f1_weight)


def selection_score(name: str, m: Mapping[str, Any], f1_weight: float = 0.0) -> float:
    """LOWER is better -- the trainers' convention for per-epoch checkpoint selection."""
    return -score(name, m, f1_weight)


def score_all(m: Mapping[str, Any], f1_weight: float = 0.0) -> Dict[str, Optional[float]]:
    """Every criterion computable on this metrics dict; None where its keys are absent.

    Recorded on every trial so a second selection under a different criterion can be run
    from the same trial pool without retraining -- which is what the appendix comparison
    needs.
    """
    out: Dict[str, Optional[float]] = {}
    for k in CRITERIA:
        try:
            out[k] = score(k, m, f1_weight)
        except ValueError:
            out[k] = None
    return out


def doc(name: str) -> str:
    c = get(name)
    return f"{c.doc} [{c.source}]"
