"""
CA pilot dataset prep.

Design rationale (revised):
  - LLM-grounded generative models (SemaPop-VAE+, TTVAE, WGAN-LLM, ...) need
    persona text + persona embeddings for every train/val row. Persona
    generation is the bottleneck (~1.5 s/row on A100), so we cap train+val.
  - Test set acts as the "real" reference for Layer-1 structural audit. The
    larger the test set, the lower the sampling variance on marginal /
    dependency / support metrics. We therefore assign ALL remaining CA rows
    to the test split.

Split (positional indices into person_level_2013_CA.csv, 380,109 rows):
  train:  40,000 random  (persona will be generated)
  val:    10,000 random  (persona will be generated; disjoint from train)
  test:  330,109 rest    (no persona; serves as L1 real reference)

Output:
  data/pums_states/CA/dataset_ids.pkl
  data/pums_states/CA/cols_config.yaml  (copied from data/pums/)
"""
from __future__ import annotations
import pickle
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STATE_DIR = ROOT / "data/pums_states/CA"
STATE_FULL_CSV = STATE_DIR / "person_level_2013_CA.csv"

OUT_IDS = STATE_DIR / "dataset_ids.pkl"
OUT_COLS_YAML = STATE_DIR / "cols_config.yaml"
SRC_COLS_YAML = ROOT / "data/pums/cols_config.yaml"

N_TRAIN = 40_000
N_VAL = 10_000
SEED = 42


def main():
    print(f"Reading {STATE_FULL_CSV} ...")
    df = pd.read_csv(STATE_FULL_CSV)
    N_total = len(df)
    print(f"  CA full rows: {N_total:,}")
    if N_total < N_TRAIN + N_VAL + 1:
        raise ValueError(f"CA full slice has {N_total} < {N_TRAIN + N_VAL + 1} required.")

    rng = np.random.default_rng(SEED)
    all_pos = np.arange(N_total)
    rng.shuffle(all_pos)

    train_idx = sorted(all_pos[:N_TRAIN].tolist())
    val_idx = sorted(all_pos[N_TRAIN:N_TRAIN + N_VAL].tolist())
    test_idx = sorted(all_pos[N_TRAIN + N_VAL:].tolist())

    assert len(train_idx) == N_TRAIN
    assert len(val_idx) == N_VAL
    assert len(test_idx) == N_total - N_TRAIN - N_VAL
    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx).isdisjoint(test_idx)
    assert set(val_idx).isdisjoint(test_idx)

    print(f"  split: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}")

    # Clean up legacy 50K subset file if present (no longer needed; we index into full CA CSV).
    legacy = STATE_DIR / "person_level_50k_CA.csv"
    if legacy.exists():
        legacy.unlink()
        print(f"  removed legacy {legacy.name}")

    ids = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "meta": {
            "state": "CA",
            "state_fips": 6,
            "n_train": N_TRAIN,
            "n_val": N_VAL,
            "n_test": len(test_idx),
            "n_state_total": N_total,
            "seed": SEED,
            "source_csv": STATE_FULL_CSV.name,
            "note": (
                "positional indices into person_level_2013_CA.csv (rows 0..N-1). "
                "Train/val are disjoint random samples; test = full CA minus train ∪ val "
                "(used as Layer-1 real reference; no persona generated)."
            ),
        },
    }
    with OUT_IDS.open("wb") as f:
        pickle.dump(ids, f)
    print(f"  wrote {OUT_IDS.name}")

    if SRC_COLS_YAML.exists():
        shutil.copy2(SRC_COLS_YAML, OUT_COLS_YAML)
        print(f"  copied cols_config.yaml from {SRC_COLS_YAML}")

    # sanity
    print("\n=== sanity ===")
    print(f"  L1 test reference size: {len(test_idx):,}  (sampling SE @ p=0.1 ≈ "
          f"{((0.1*0.9/len(test_idx))**0.5)*100:.3f}%)")
    print(f"  per-target reference sample count for rare p=0.001: ≈ {int(len(test_idx)*0.001):,}")


if __name__ == "__main__":
    main()
