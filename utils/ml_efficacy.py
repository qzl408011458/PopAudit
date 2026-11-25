"""TabDDPM's selection criterion, instantiated for a dataset with no downstream task.

Kotelnikov et al. (2023) tune TabDDPM by maximising downstream machine-learning efficiency:
each Optuna trial trains the model, draws a synthetic sample, fits CatBoost ON THE SYNTHETIC
SAMPLE, and scores it against a held-out REAL split -- macro-averaged F1 for classification,
R^2 for regression -- averaged over five sampling seeds.  Their benchmark datasets ship with
a designated target column; PUMS does not.

THE SUBSTITUTION, stated so the paper can state it.  We pre-register a panel of four PUMS
attributes as surrogate targets, each predicted from the remaining forty-nine, and report
the mean of their macro-F1 scores.  The train-on-synthetic / test-on-real protocol, the
classifier being fixed rather than tuned, and the five-seed average are all kept.  The panel
is fixed before any trial runs and never changed:

    Commute_Mode         travel behaviour, the modelling target of this literature
    Employment_status    the labour-market gate that most other attributes hang off
    Tenure               the housing gate, and the driver of five universe rules
    Household_Income     a many-level ordinal, so the panel is not all low-cardinality

They span the four semantic blocks that carry the most structure and none is a near-duplicate
of another.  Picking targets AFTER seeing results would make this a tuned metric, which is
the failure mode the substitution exists to avoid.

MIXED-TIER NOTE, and why the panel did NOT change.  On that tier Household_Income is a
CONTINUOUS attribute rather than an income bracket: 2,421 distinct values in a 20,002-row
validation split.  The first version of this file coerced every column with .astype(str)
and handed all four targets to a classifier, which meant Household_Income became a
~2,400-class problem -- macro-F1 near zero for every model alike, i.e. a criterion with no
dynamic range, and slow enough that a two-trial CPU dry run had not finished a single trial
in three minutes.  Kotelnikov et al. already cover this case: their protocol is macro-F1
for classification AND R^2 for regression, so a continuous target is scored by R^2 here and
the pre-registered panel is left exactly as it was.  R^2 is floored at 0 so that one
diverged trial cannot dominate the mean of four with a large negative number; F1 has the
same floor, which keeps the panel on one scale.

Continuous columns are also passed to the classifier AS NUMBERS rather than ordinal codes.
Ordinal-encoding 5,059 income values destroys their ordering, so a generator that puts a
household within a few hundred dollars got no more credit than one that was off by
$300,000.  HistGradientBoosting takes numerics (and NaN) natively.

EARLY STOPPING IS OFF, DELIBERATELY.  scikit-learn defaults both estimators to
``early_stopping="auto"``, which switches ON above 10,000 samples and then carves out an
internal validation split -- STRATIFIED, for a classifier.  A synthetic sample that emits
exactly one row of some rare category makes that stratification impossible, and the fit
raises "The least populated classes in y have only 1 member".  That killed tabddpm's CA
study twice on 2026-08-26, ~650 epochs into a trial each time.

The preflight could not have caught it: a dry run generates 1,000 rows, below the 10,000
threshold, so early stopping was off there and the failing branch never executed.  Sample
size selecting a different code path is exactly the failure mode a smoke test is blind to.

Turning it off is also the right thing on its own terms.  This criterion requires a FIXED
classifier, not a good one; an internal split that changes with the data adds variance to
the number every trial is ranked by, and max_iter is already pinned.

CLASSIFIER.  HistGradientBoostingClassifier at fixed hyperparameters, not CatBoost: it is
already a dependency, it handles the integer-coded categoricals natively, and the criterion
requires a FIXED classifier rather than a good one -- tuning it would confound the model
comparison.  Cost is ~7 s for the four targets at n=20,000 WITH BLAS THREADS CAPPED -- uncapped on a
56-core box it measured 100-293 s, and hpo/common.trial_env now pins OMP_NUM_THREADS
because three trials run concurrently.  TabDDPM trials compute it BOTH per evaluation and
as a five-seed block at the end: per evaluation because pruning has to see the trajectory
of the criterion being optimised, and at the end because that is the authors' protocol and
because it closes the middle rung of hpo/objective.objective_from_run's fallback chain.
The checkpoint that is kept is still the final one (--load-final), since Kotelnikov et al.
do no in-training selection.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

#: pre-registered, fixed before any trial ran (2026-08-25)
TARGETS: List[str] = ["Commute_Mode", "Employment_status", "Tenure", "Household_Income"]


#: A classification target with more levels than this is a misconfigured continuous column,
#: not a hard problem.  Raising beats hanging: the mixed tier's Household_Income would have
#: fitted a ~2,400-class booster on every evaluation of every trial.
MAX_CLASSES = 200


def _numeric_columns(df: pd.DataFrame, cols: Sequence[str],
                     numeric: Optional[Sequence[str]] = None) -> List[str]:
    """Which columns are numbers, taken from the REAL frame.

    The real frame decides, never the synthetic one: a generator that emits a string where
    a number belongs would otherwise silently move a column from one branch to the other
    and change what the criterion measures.
    """
    if numeric is not None:
        return [c for c in numeric if c in cols]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def ml_efficiency(df_syn: pd.DataFrame, df_val: pd.DataFrame,
                  targets: Optional[Sequence[str]] = None,
                  seed: int = 0, max_iter: int = 100,
                  numeric: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Train on the synthetic sample, score on the held-out val split.

    Returns ``{"ml_eff": mean score, "ml_f1_<target>": per-target score}`` -- macro-F1 for
    a categorical target, R^2 floored at 0 for a continuous one, both in [0, 1] so the
    panel mean is on one scale.  ``numeric`` names the continuous columns; when omitted
    they are read off ``df_val``'s dtypes.  Targets absent from either frame are skipped
    and named in ``ml_eff_targets``.
    """
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)
    from sklearn.metrics import f1_score, r2_score
    from sklearn.preprocessing import OrdinalEncoder

    targets = list(targets or TARGETS)
    cols = [c for c in df_val.columns if c in df_syn.columns]
    use = [t for t in targets if t in cols]
    if not use:
        raise ValueError(f"none of {targets} present in both frames")

    num = _numeric_columns(df_val, cols, numeric)
    cat = [c for c in cols if c not in num]

    S = pd.DataFrame(index=range(len(df_syn)))
    V = pd.DataFrame(index=range(len(df_val)))
    if cat:
        # one encoder fitted on BOTH sides, so a category the synthetic data invented does
        # not shift the coding of everything after it.  unknown_value=-1 keeps it total.
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(pd.concat([df_syn[cat].astype(str), df_val[cat].astype(str)]))
        S[cat] = enc.transform(df_syn[cat].astype(str))
        V[cat] = enc.transform(df_val[cat].astype(str))
    for c in num:
        S[c] = pd.to_numeric(df_syn[c], errors="coerce").to_numpy(dtype=float)
        V[c] = pd.to_numeric(df_val[c], errors="coerce").to_numpy(dtype=float)
    S, V = S[cols], V[cols]

    out: Dict[str, float] = {}
    for t in use:
        X = [c for c in cols if c != t]
        ys = S[t].to_numpy()
        finite = ys[np.isfinite(ys)] if t in num else ys
        if finite.size == 0 or len(np.unique(finite)) < 2:
            # a collapsed generator can emit one value for a whole column; that is a real
            # failure and scores 0 rather than crashing the trial
            out[f"ml_f1_{t}"] = 0.0
            continue
        if t in num:
            keep = np.isfinite(ys)
            reg = HistGradientBoostingRegressor(max_iter=max_iter, random_state=seed,
                                                early_stopping=False)
            reg.fit(S.loc[keep, X], ys[keep])
            yv = V[t].to_numpy()
            kv = np.isfinite(yv)
            out[f"ml_f1_{t}"] = float(max(0.0, r2_score(yv[kv], reg.predict(V.loc[kv, X]))))
        else:
            n_cls = int(len(np.unique(ys)))
            if n_cls > MAX_CLASSES:
                raise ValueError(
                    f"{t}: {n_cls} classes -- this is a continuous column being scored as "
                    f"classification.  Name it in `numeric`, or drop it from TARGETS.")
            clf = HistGradientBoostingClassifier(max_iter=max_iter, random_state=seed,
                                                 early_stopping=False)
            clf.fit(S[X], ys)
            out[f"ml_f1_{t}"] = float(f1_score(V[t], clf.predict(V[X]),
                                               average="macro", zero_division=0))
    out["ml_eff"] = float(np.mean([out[f"ml_f1_{t}"] for t in use]))
    out["ml_eff_targets"] = ",".join(use)          # type: ignore[assignment]
    return out


def ml_efficiency_multiseed(sample_fn, df_val: pd.DataFrame, n_seeds: int = 5,
                            targets: Optional[Sequence[str]] = None,
                            numeric: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Kotelnikov et al.'s five-seed average.  ``sample_fn(seed) -> DataFrame``.

    The average is over SAMPLING seeds at one trained checkpoint, not over training runs:
    it removes the generator's draw noise from the criterion, not its training noise.
    """
    vals, per = [], {}
    for s in range(n_seeds):
        r = ml_efficiency(sample_fn(s), df_val, targets, seed=s, numeric=numeric)
        vals.append(r["ml_eff"])
        for k, v in r.items():
            if k.startswith("ml_f1_"):
                per.setdefault(k, []).append(v)
    out = {"ml_eff": float(np.mean(vals)), "ml_eff_sd": float(np.std(vals, ddof=1)),
           "ml_eff_n_seeds": int(n_seeds)}
    out.update({k: float(np.mean(v)) for k, v in per.items()})
    return out
