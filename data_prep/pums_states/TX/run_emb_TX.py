"""
Wrapper to embed persona text for the TX state dataset, mirroring 0embPersona.py
but with CA-specific paths.

Run AFTER run_persona_CA.py finishes (needs traindata_pers.csv + valdata_pers.csv
with persona column populated).

Usage:
  python data/pums_states/TX/run_emb_CA.py --device cuda:0
"""
from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from model.EmbeddingLLM import QwenPersonaEmbeddingB

DATA_DIR = ROOT / "data/pums_states/TX"
TRAIN_CSV = DATA_DIR / "traindata_pers.csv"
VAL_CSV = DATA_DIR / "valdata_pers.csv"
OUT_TRAIN_PKL = DATA_DIR / "embPers.pkl"
OUT_VAL_PKL = DATA_DIR / "embPers_val.pkl"


def build_embeddings(texts, model, batch_size=4):
    from tqdm import tqdm
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="embed"):
        emb = model(texts[i:i + batch_size])  # [B, cond_dim]
        out.append(emb.cpu())
    return torch.cat(out, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=4)
    args = p.parse_args()

    print(f"TX state persona embedding")
    print(f"  train csv: {TRAIN_CSV}")
    print(f"  val   csv: {VAL_CSV}")
    print(f"  device: {args.device}, batch_size: {args.batch_size}")

    if not TRAIN_CSV.exists() or not VAL_CSV.exists():
        raise FileNotFoundError(
            f"Persona CSVs missing. Run run_persona_CA.py first.\n"
            f"  expected: {TRAIN_CSV} and {VAL_CSV}"
        )

    print(f"  loading QwenPersonaEmbeddingB on {args.device} ...")
    emb_model = QwenPersonaEmbeddingB(
        model_name="Qwen/Qwen3-8B",
        device=args.device,
        max_length=512,
        last_n_layers=4,
        cond_dim=512,
        use_layernorm=True,
    ).to(args.device)

    for csv_path, out_pkl, name in [
        (TRAIN_CSV, OUT_TRAIN_PKL, "TRAIN"),
        (VAL_CSV, OUT_VAL_PKL, "VAL"),
    ]:
        print(f"\n  >>> {name} embedding ({csv_path.name}) <<<")
        df = pd.read_csv(csv_path)
        if "persona" not in df.columns:
            raise ValueError(f"{csv_path.name} missing 'persona' column.")
        n_missing = df["persona"].isna().sum()
        if n_missing > 0:
            print(f"    WARNING: {n_missing} rows have no persona text; using empty string.")
            df["persona"] = df["persona"].fillna("")
        texts = df["persona"].astype(str).tolist()
        emb = build_embeddings(texts, emb_model, batch_size=args.batch_size)
        save = emb.numpy().astype(np.float32)
        with out_pkl.open("wb") as f:
            pickle.dump(save, f)
        print(f"    wrote {out_pkl.name} | shape={save.shape}")

    print("\n✅ embedding complete. Data ready for training:")
    print(f"  data_name = 'pums_states/TX'  (use in train_*.py scripts)")


if __name__ == "__main__":
    main()
