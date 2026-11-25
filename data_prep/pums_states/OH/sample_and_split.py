"""
OH dataset split, mirroring CA pilot.

Splits person_level_2013_OH.csv (114,112 rows) into:
  train:  40,000  (persona to be generated)
  val:    10,000  (persona to be generated)
  test:   64,112  (no persona; serves as L1 in-domain reference)

OVERWRITES the prior 80/10/10 split produced by build_state_datasets.

Output:
  data/pums_states/OH/dataset_ids.pkl       (positional row indices into person_level_2013_OH.csv)
  data/pums_states/OH/cols_config.yaml      (copied from data/pums/)
"""
from __future__ import annotations
import pickle
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STATE_DIR = ROOT / "data/pums_states/OH"
STATE_FULL_CSV = STATE_DIR / "person_level_2013_OH.csv"

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
    print(f"  OH full rows: {N_total:,}")
    if N_total < N_TRAIN + N_VAL + 1:
        raise ValueError(f"OH full slice has {N_total} < {N_TRAIN + N_VAL + 1} required.")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N_total)
    train_pos = perm[:N_TRAIN]
    val_pos = perm[N_TRAIN:N_TRAIN + N_VAL]
    test_pos = perm[N_TRAIN + N_VAL:]
    print(f"  split: train={len(train_pos):,}, val={len(val_pos):,}, test={len(test_pos):,}")

    # IMPORTANT: store POSITIONAL indices (0..N-1 into person_level_2013_OH.csv),
    # NOT Sampled_Person_ID Pids — to match CA pipeline convention so that
    # _gen_pub_personas_traintest.generate_split_csv can use df.iloc[indices].
    ids = {
        "train_idx": train_pos.tolist(),
        "val_idx":   val_pos.tolist(),
        "test_idx":  test_pos.tolist(),
        "meta": {
            "state": "OH",
            "state_fips": 39,
            "n_train": N_TRAIN, "n_val": N_VAL, "n_test": len(test_pos),
            "n_total_rows": int(N_total),
            "seed": SEED,
            "note": "positional indices into person_level_2013_OH.csv (rows 0..N-1); matches CA convention",
        },
    }
    with OUT_IDS.open("wb") as f:
        pickle.dump(ids, f)
    print(f"  wrote {OUT_IDS.name}")

    if not OUT_COLS_YAML.exists():
        shutil.copy(SRC_COLS_YAML, OUT_COLS_YAML)
        print(f"  copied cols_config.yaml from data/pums/")

    # Write a thin testdata_pers.csv (columns only — no persona text). Note this
    # OVERWRITES the prior 80/10/10 testdata.
    test_df = df.iloc[test_pos].reset_index(drop=True).copy()
    if "Sampled_Person_ID" in test_df.columns and "Pid" not in test_df.columns:
        test_df = test_df.rename(columns={"Sampled_Person_ID": "Pid"})
    test_path = STATE_DIR / "testdata_pers.csv"
    test_df.to_csv(test_path, index=False)
    print(f"  wrote {test_path.name}: {len(test_df):,} rows (no persona col)")

    print("\nNext step: run data/pums_states/OH/run_persona_OH.py to generate train+val personas.")


if __name__ == "__main__":
    main()
