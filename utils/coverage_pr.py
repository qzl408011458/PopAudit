#!/usr/bin/env python
"""alpha-precision / beta-recall for population synthesis (Lyu's NIP / NIR / IF1).

WHY THIS EXISTS.  Exact-tuple matching -- Kim & Bansal's Eqs. (14)-(15), and the block F1
we were using -- dies at 50 attributes.  Measured on our data: the best of 108 trials
attains a whole-row F1 of 0.00024, because 99.6% of reference rows are unique over 50
columns, so no generated row can match one exactly.  Lyu's chapter 6 reports IF1 between
0.54 and 0.98 at 47 attributes for the same kind of models, because his metric is not
exact matching: it is an alpha-precision / beta-recall pair in the sense of Alaa et al.
(2022), which asks whether a generated record falls in a DENSE REGION of the real support
rather than whether it is a verbatim copy.  (In the sense of, not the estimator of -- see
the note on the two classes below.)

PORTED, WITH TWO DEVIATIONS THAT MOVE THE NUMBERS.  The scoring rule is a line-by-line
port of ``flowsyn/metrics/coverage.py::IsolationForestPrecisionRecall`` from the FlowSyn
release (c-lyu/flowsyn): same alpha grid, same quantile direction, same comparison, same
NIP/NIR/IF1 formulas, and ISOTREE_PARAMS is their default dict verbatim.  What differs:

  1. ``subsample`` (below) has no counterpart there.  Every call site of ours sets 8000,
     which deflates both scores by roughly 1.35/sqrt(n) -- the real-vs-real ceiling sits
     near 0.984 at n=8000 against ~0.998 on full frames.  It is a common shift, so the
     oracle anchors absorb it and the ordering is stable (measured: 8k vs 32k moves any
     cell <=0.015, ordering identical), but absolute levels are NOT theirs.
  2. This passes ``random_seed`` to isotree; their ``build_isotree`` forwards only
     n_jobs, so theirs runs at isotree's built-in seed 1 while ours runs at the caller's.
     Worth ~2e-4 on IF1 (measured), i.e. reproducibility, not a level shift.

Also fixed here: their ``_prepare_data`` builds the category list from ``astype(str)`` but
casts the ORIGINAL column, which yields all-NaN on a non-str categorical; this casts to
str first.  Harmless in their pipeline, load-bearing in ours.

DO NOT claim these numbers are comparable to that paper's Table 6.2.  Two reasons.  First
the subsample above.  Second, and worse, the release ships TWO registered metrics that
return the same key set -- this one, and ``RobustPrecisionRecall`` (the actual Alaa et al.
estimator: one-class SVDD embedding, k-NN beta-recall with k-tuning), which we did not
port -- and the release contains no config or driver naming either one, while the CLI's
surviving ``--oc-model-path`` / ``--oc-config-path`` flags exist only to serve the SVDD
one.  Which of the two produced Table 6.2 cannot be established from the release.  Read
our values against our own oracle anchors, never against his.

Resolution is a property of the DATA, and MEASURED, not guessed.  On our mixed tier the
usable range (real-vs-real ceiling minus a dependence-free frame) is 0.267 over all 20
columns, 0.264 over the 12 categorical columns alone, and 0.022 over the 8 continuous
columns alone -- on the continuous block a frame with every dependence destroyed scores
0.957 against a ceiling of 0.979, i.e. the forest cannot see dependence there at all.  So
the discrimination lives entirely in the CATEGORICAL block, and a higher continuous share
would make this metric less sensitive, not more.  (An earlier version of this comment
claimed the opposite; the column-subset experiment refuted it.)  Neither the forest
parameters nor an attribute count of 20 -> 50 opens the spread either.  Absolute levels are
therefore not comparable across datasets with different column mixes.

Packaging: the original depends on the FlowSyn dataset/config classes, this takes two
DataFrames.

  alpha_precision(a) = fraction of GENERATED rows inside the densest a-quantile of the
                       REAL support, where support is an isolation forest fitted on real
  beta_recall(a)     = the same with the roles swapped
  NIP = 1 - 2*mean|alpha_precision(a) - a|   (a perfect model traces the diagonal)
  NIR = 1 - 2*mean|beta_recall(a) - a|
  IF1 = harmonic mean of NIP and NIR

FAIRNESS.  The support model is fitted on the REAL data (and, for recall, on each model's
own output); it never sees any generator's internal representation.  In particular it is
NOT the persona embedding space -- scoring SemaPop in its own encoder's geometry would
flatter it.  Every model is scored in the shared attribute space.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

#: the FlowSyn defaults, verbatim
ISOTREE_PARAMS = dict(ndim=2, ntrees=1000, penalize_range=False, prob_pick_pooled_gain=0)


def _prepare(real: pd.DataFrame, gen: pd.DataFrame, cat: Sequence[str],
             cont: Sequence[str], backend: str):
    real, gen = real.copy(), gen.copy()
    for df in (real, gen):
        if df.isna().any().any():
            raise ValueError("NaNs in the input; out-of-universe must already be a value")
    if backend == "isotree":
        for c in cat:                       # unify categories so the codes align
            allc = pd.Index(sorted(set(real[c].astype(str)) | set(gen[c].astype(str))))
            dt = pd.api.types.CategoricalDtype(categories=allc)
            real[c] = real[c].astype(str).astype(dt)
            gen[c] = gen[c].astype(str).astype(dt)
        for c in cont:
            real[c] = pd.to_numeric(real[c]); gen[c] = pd.to_numeric(gen[c])
        cols = list(cat) + list(cont)
        return real[cols], gen[cols]
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    rc = ohe.fit_transform(real[list(cat)]) if len(cat) else np.empty((len(real), 0))
    gc = ohe.transform(gen[list(cat)]) if len(cat) else np.empty((len(gen), 0))
    rn = real[list(cont)].to_numpy(float) if len(cont) else np.empty((len(real), 0))
    gn = gen[list(cont)].to_numpy(float) if len(cont) else np.empty((len(gen), 0))
    return np.concatenate([rn, rc], 1), np.concatenate([gn, gc], 1)


def alpha_precision_beta_recall(
    real: pd.DataFrame, gen: pd.DataFrame, cat_features: Sequence[str],
    cont_features: Sequence[str] = (), n_steps: int = 20, backend: str = "isotree",
    random_state: int = 42, n_jobs: int = -1, subsample: Optional[int] = None,
) -> Dict[str, object]:
    """NIP / NIR / IF1 plus the two curves.

    ``subsample`` caps both sides at n rows before fitting; an isolation forest over
    321,402 x 50 is slow enough that the full reference would dominate an evaluation that
    runs once per trial.  Report whatever value is used -- it moves the numbers.
    """
    rng = np.random.default_rng(random_state)
    if subsample and len(real) > subsample:
        real = real.iloc[rng.choice(len(real), subsample, replace=False)]
    if subsample and len(gen) > subsample:
        gen = gen.iloc[rng.choice(len(gen), subsample, replace=False)]

    rx, gx = _prepare(real, gen, cat_features, cont_features, backend)
    if backend == "isotree":
        import isotree
        f_real = isotree.IsolationForest(n_jobs=n_jobs, random_seed=random_state,
                                         **ISOTREE_PARAMS).fit(rx)
        f_gen = isotree.IsolationForest(n_jobs=n_jobs, random_seed=random_state,
                                        **ISOTREE_PARAMS).fit(gx)
        s_rr, s_rg = f_real.decision_function(rx), f_real.decision_function(gx)
        s_gg, s_gr = f_gen.decision_function(gx), f_gen.decision_function(rx)
        denser = np.less_equal          # isotree: lower score = denser
    else:
        from sklearn.ensemble import IsolationForest
        f_real = IsolationForest(random_state=random_state, n_jobs=n_jobs).fit(rx)
        f_gen = IsolationForest(random_state=random_state, n_jobs=n_jobs).fit(gx)
        s_rr, s_rg = f_real.decision_function(rx), f_real.decision_function(gx)
        s_gg, s_gr = f_gen.decision_function(gx), f_gen.decision_function(rx)
        denser = np.greater_equal       # sklearn: higher score = denser

    alphas = np.linspace(0, 1, n_steps)
    prec, rec = [], []
    for a in alphas:
        q = 1 - a if denser is np.greater_equal else a
        prec.append(float(np.mean(denser(s_rg, np.quantile(s_rr, q)))))
        rec.append(float(np.mean(denser(s_gr, np.quantile(s_gg, q)))))
    prec, rec = np.asarray(prec), np.asarray(rec)

    nip = 1 - 2 * float(np.mean(np.abs(prec - alphas)))
    nir = 1 - 2 * float(np.mean(np.abs(rec - alphas)))
    h = lambda a, b: (2 * a * b / (a + b)) if (a + b) > 0 else 0.0
    return {"NIP": nip, "NIR": nir, "IF1": h(nip, nir),
            "precision_1": float(prec[-1]), "recall_1": float(rec[-1]),
            "F1_at_1": h(float(prec[-1]), float(rec[-1])),
            "alpha_precision_curve": prec.tolist(),
            "beta_recall_curve": rec.tolist(),
            "backend": backend, "n_real": int(len(real)), "n_gen": int(len(gen)),
            "subsample": subsample}
