"""K=50 ensemble writer — shared by every v2 trainer (v3 protocol, 2026-08-29).

pop_audit's audit layers consume, per finalized model run:

    <save_dir>/generated_ensembles/final_generated_samples_{rid}.csv   rid = 0..K-1

one realization per seed in ``audit_config.ENSEMBLE_SEEDS`` (K=50, fixed across
models so Layer-2 statistics share sampling randomness), each with
N = len(reference population) rows in label space (the 12 modelled columns).

Usage inside a trainer, after the model is finalized::

    from utils.ensemble_v2 import write_ensemble
    write_ensemble(lambda seed, n: my_sample_df(n, seed),
                   n_samples=len(df_ref), save_dir=save_dir,
                   k=args.ensemble_k)          # k=0 -> all 50

The trainer's own ``sample_fn(seed, n) -> pd.DataFrame`` must make the draw a
deterministic function of ``seed`` (set torch/numpy RNGs inside, as the
existing final-sample blocks already do with ``args.sample_seed``).
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable, List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from audit_config import ENSEMBLE_SEEDS  # noqa: E402  (K=50, fixed)


def write_ensemble(sample_fn: Callable[[int, int], pd.DataFrame],
                   n_samples: int,
                   save_dir: str,
                   k: int = 0,
                   seeds: Optional[List[int]] = None,
                   log_every: int = 5) -> List[str]:
    """Draw one realization per seed and write the audit-layer ensemble files.

    k=0 uses all seeds (production, K=50); a small k (e.g. 2) is the smoke mode.
    Returns the list of file paths written. Realization 0 is also copied to
    ``final_generated_samples.csv`` for backward compatibility with older
    audit drivers.
    """
    use = list(seeds if seeds is not None else ENSEMBLE_SEEDS)
    if k and k > 0:
        use = use[:k]
    ens_dir = os.path.join(save_dir, "generated_ensembles")
    os.makedirs(ens_dir, exist_ok=True)
    paths: List[str] = []
    t0 = time.time()
    ref_cols = None
    for rid, seed in enumerate(use):
        df = sample_fn(int(seed), int(n_samples))
        if len(df) != n_samples:
            raise RuntimeError(f"ensemble rid={rid}: sample_fn returned {len(df)} rows, "
                               f"expected {n_samples}")
        if ref_cols is None:
            ref_cols = list(df.columns)
        elif list(df.columns) != ref_cols:
            raise RuntimeError(f"ensemble rid={rid}: column set drifted "
                               f"({list(df.columns)} != {ref_cols})")
        p = os.path.join(ens_dir, f"final_generated_samples_{rid}.csv")
        df.to_csv(p, index=False)
        paths.append(p)
        if rid == 0:
            df.to_csv(os.path.join(save_dir, "final_generated_samples.csv"), index=False)
        if log_every and (rid + 1) % log_every == 0:
            el = time.time() - t0
            print(f"[ensemble] {rid + 1}/{len(use)} realizations "
                  f"({el:.0f}s elapsed, ~{el / (rid + 1):.0f}s each)", flush=True)
    print(f"[ensemble] wrote {len(paths)} realizations x {n_samples} rows -> {ens_dir}",
          flush=True)
    return paths


def add_ensemble_args(parser) -> None:
    """Uniform CLI: --ensemble turns the writer on (finalize only, never HPO trials);
    --ensemble-k limits realizations (smoke); --ensemble-n overrides N (0 = len(reference))."""
    parser.add_argument("--ensemble", action="store_true",
                        help="after finalizing, write the K=50 audit ensemble")
    parser.add_argument("--ensemble-k", type=int, default=0,
                        help="limit realizations (0 = all 50; use 2 for smoke tests)")
    parser.add_argument("--ensemble-n", type=int, default=0,
                        help="rows per realization (0 = len(reference population))")
