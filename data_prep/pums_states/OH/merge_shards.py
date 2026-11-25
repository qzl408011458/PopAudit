"""
Merge sharded persona-generation outputs (after multi-GPU parallel run) back
into traindata_pers.csv and valdata_pers.csv.

Expects per-shard files in data/pums_states/OH/ (saved by run_persona_CA.py with
type_name=f"train_shard{K}" or "val_shard{K}", producing filenames like
"train_shard0data_pers.csv"):
  train_shard{K}data_pers.csv  for K in 0..num_shards-1
  val_shard{K}data_pers.csv    for K in 0..num_shards-1

Output:
  traindata_pers.csv  (concatenated, sorted by Pid)
  valdata_pers.csv    (same)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data/pums_states/OH"


def merge_one(name: str, num_shards: int):
    dfs = []
    for k in range(num_shards):
        # File naming follows generate_split_csv: type_name=f"{name}_shard{k}",
        # _save_path_for_type appends "data_pers.csv" → "{name}_shard{k}data_pers.csv"
        p = DATA_DIR / f"{name}_shard{k}data_pers.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing shard: {p}")
        df = pd.read_csv(p)
        df["__shard__"] = k
        dfs.append(df)
        print(f"  loaded {p.name} | rows={len(df)}")
    merged = pd.concat(dfs, ignore_index=True)
    if "Pid" in merged.columns:
        merged = merged.sort_values("Pid").reset_index(drop=True)
    out_path = DATA_DIR / f"{name}data_pers.csv"
    # __shard__ kept for diagnostics; drop on save
    merged_save = merged.drop(columns="__shard__")
    merged_save.to_csv(out_path, index=False)
    print(f"  ==> wrote {out_path.name} | total rows={len(merged_save)}")
    return merged_save


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_shards", type=int, required=True,
                   help="Number of shards (must match what was used in run_persona_CA.py).")
    args = p.parse_args()

    print(f"Merging {args.num_shards} shards in {DATA_DIR} ...")
    for name in ("train", "val"):
        print(f"\n=== {name} ===")
        merge_one(name, args.num_shards)
    print("\n✅ merge complete.")
    print("   testdata_pers.csv was written by shard 0 alone (no merge needed).")
    print("   Next: run run_emb_CA.py")


if __name__ == "__main__":
    main()
