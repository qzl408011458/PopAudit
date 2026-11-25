"""TTVAE generation-only (shadow): K=50 ensemble from the STORED finalize checkpoint
(hpo_v3/final/ttvae_<ST>/ttvae_<ST>_final_s<seed>/best.pt + train_processed.npy, copied),
reconstructing exactly the label encoders / numeric bounds of train_ttvae_CA.py's main
(fit on traindata_pers.csv; bounds over train+val), and calling its own
sample_ttvae_ensemble with the published triangular sampler.  Retraining is avoided because
TTVAE training is not bitwise deterministic on GPU (max |dw| ~ 3e-3 to 8e-3 on a retrain).
usage (cwd = repo root): ttvae_gen_only.py ST SEED OUT_ROOT [--batch 2048] [--k 0]"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[2])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import os, sys, shutil, argparse, yaml
sys.path.insert(0, _PA.R(""))
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import train_ttvae_CA as T
from audit_config import ENSEMBLE_SEEDS
F = _PA.S("/hpo_v3/final")
N = {"CA": 330109, "TX": 212756, "OH": 64112}
ap = argparse.ArgumentParser(); ap.add_argument("st"); ap.add_argument("seed", type=int); ap.add_argument("out_root")
ap.add_argument("--batch", type=int, default=2048); ap.add_argument("--k", type=int, default=0)
ap.add_argument("--rids", default="", help="comma list of realization ids (shard); each realization depends only on its seed")
a = ap.parse_args()
run = f"ttvae_{a.st}_final_s{a.seed}"; src = f"{F}/ttvae_{a.st}/{run}"; dst = f"{a.out_root}/ttvae_{a.st}/{run}"
os.makedirs(dst, exist_ok=True)
for f in ("best.pt", "train_processed.npy"):
    if not os.path.exists(f"{dst}/{f}"): shutil.copy2(f"{src}/{f}", f"{dst}/{f}")
data_name = f"pums_states/{a.st}"
cols = yaml.safe_load(open(f"data/{data_name}/cols_config_v3.yaml"))
cat_cols, num_cols = cols["cat_cols"], cols["num_cols"]; mc = num_cols + cat_cols
df_train = pd.read_csv(f"data/{data_name}/traindata_pers.csv", usecols=mc)[mc]
df_val = pd.read_csv(f"data/{data_name}/valdata_pers.csv", usecols=mc)[mc]
dic_les = {}
for c in cat_cols:
    le = LabelEncoder(); df_train[c] = le.fit_transform(df_train[c]); dic_les[c] = le
df_all = pd.concat([df_train, df_val], ignore_index=True)
num_bounds = {c: (df_all[c].min(), df_all[c].max()) for c in num_cols}
seeds = list(ENSEMBLE_SEEDS)[: (a.k or None)]
if a.rids:
    # shard mode: the loop body of T.sample_ttvae_ensemble, verbatim, for the given rids
    ed = os.path.join(dst, "generated_ensembles"); os.makedirs(ed, exist_ok=True)
    for rid in [int(x) for x in a.rids.split(",")]:
        seed = int(ENSEMBLE_SEEDS[rid]); T.set_sampling_seed(seed)
        df_k = T.sample_ttvae(f"{dst}/best.pt", n=N[a.st], device="cuda:0", method="triangular", batch_size=a.batch)
        df_k = T.postprocess_generated_population(df_k, cat_cols=cat_cols, num_cols=num_cols, dic_les=dic_les, num_bounds=num_bounds)
        df_k = df_k[num_cols + cat_cols]
        tmp = os.path.join(ed, f".tmp_{rid}.part"); df_k.to_csv(tmp, index=False)
        os.replace(tmp, os.path.join(ed, f"final_generated_samples_{rid}.csv"))
        print(f"[shard] rid={rid} seed={seed} written", flush=True)
    sys.exit(0)
T.sample_ttvae_ensemble(best_path=f"{dst}/best.pt", n_samples=N[a.st], device="cuda:0", cat_cols=cat_cols,
                        num_cols=num_cols, dic_les=dic_les, num_bounds=num_bounds, save_path=dst,
                        ensemble_seeds=seeds, method="triangular", batch_size=a.batch)
print("[ttvae_gen_only] done", dst, flush=True)
