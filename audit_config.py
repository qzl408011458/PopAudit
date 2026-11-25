"""
Shared K = 50 ensemble configuration, imported by utils/ensemble_v2.py (the trainers' ensemble
writer) and train_ttvae_CA.py.

The same K=50 ensemble seeds are used by every model so that Layer-2
metrics (per-realization mean, 95% CI, model ranking, paired tests)
are computed on identical sampling randomness sequences across models.

DO NOT change these mid-experiment. If K must grow, append seeds to
the end of the list; never modify existing entries, since that would
invalidate prior model runs.

Pattern: extends the original 5-seed convention
    [42, 142, 242, 342, 442]
to 50 seeds by spacing 100 apart, yielding deterministic, reproducible,
collision-free values.
"""

# K = 50 fixed ensemble seeds for Layer-2 audit.
ENSEMBLE_SEEDS = [42 + 100 * i for i in range(50)]
assert len(ENSEMBLE_SEEDS) == 50
assert len(set(ENSEMBLE_SEEDS)) == 50, "duplicate seeds detected"

# Convenience: K and the per-realization sample size policy.
K_REALIZATIONS = len(ENSEMBLE_SEEDS)

# Per-realization N policy: match held-out test set size (scheme B).
# In train_*.py we keep `n_samples = len(real_test_data)`, which for the
# CA pilot resolves to ~330k → 16.5M total agents per model at K=50.
