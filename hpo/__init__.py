"""Fair-HPO harness for the candidate priors (Supplementary S4.1).

Modules
-------
common        paths, budgets, trial context, provenance helpers
spaces        Optuna search spaces + subprocess argv builders, one per model
objective     the single objective definition and the metrics.json reader
run           subprocess-per-trial Optuna driver (resumable, pruning, timeout)
finalize      3-seed retrain of the winner + the budget table
budget_probe  (not shipped) CPU trial-equivalent measurement
"""
