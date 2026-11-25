"""tab:fidelity-v3 (base-year one-way SRMSE, 3 seeds) against the FRESH iid reference.
= supplementary/l1_disjoint/code/fidelity_disjoint.py logic (hpo/finalize.reference_metrics_for body:
SRMSE-1 over the frozen EvalConfig attrs of run_dir/generated_samples.csv vs the reference) with the
reference table changed to {orig = referencedata_pers.csv (symlink to testdata_pers.csv), fresh}.
'orig' must reproduce seed_runs.json exactly.
Yardsticks (10k rows each, 3 draws): 'oracle_prodtest' = sample of the production test split,
'oracle_iid' = fresh iid PWGTP draw from the state PUMS, 'train_copy' = sample of the TRAIN rows
(a pure memoriser) -> shows whether train copies gain from the production reference's overlap.
Output: ../results/fidelity_per_seed.csv, fidelity_compare.csv, fidelity_yardsticks.csv, fidelity_rank.csv"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[5])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import os, sys, json, pickle
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau
ROOT = _PA.R(""); sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from utils import eval_v2 as ev
import hpo.common as C
F = _PA.S("/hpo_v3/final")
ROWS = [("SemaPop-VAE+", "semapop_vae", "semapop_vae_{st}_train"), ("TTVAE", "ttvae", "ttvae_{st}"),
        ("BN", "bn", "bn_pgmpy_{st}"), ("WGAN-GP-ZCR", "wgan_zcr", "wgan_zcr_{st}"), ("CTGAN tuned", "ctgan", "ctgan_{st}"),
        ("CTGAN as shipped", "ctgan_defaults", "ctgan_defaults_{st}"), ("FlowSyn", "flowsyn", "flowsyn_{st}"),
        ("Borysov-VAE tuned", "borysov", "borysov_{st}"), ("Borysov-VAE as shipped", "borysov_defaults", "borysov_defaults_{st}"),
        ("TabDDPM val-SRMSE", "tabddpm", "tabddpm_{st}_valsrmse"), ("TabDDPM ML-eff", "tabddpm_mleff", "tabddpm_{st}"),
        ("TVAE tuned", "tvae", "tvae_ref_{st}"), ("TVAE as shipped", "tvae_defaults", "tvae_ref_defaults_{st}")]
REFS = {"orig": f"{C.DATA_ROOT}/{{st}}/referencedata_pers.csv", "fresh": f"{HERE}/refs/{{st}}/testdata_fresh.csv"}
STATES = ["CA", "TX", "OH"]
cache = {}
def ref(v, st, cols):
    if (v, st) not in cache: cache[(v, st)] = pd.read_csv(REFS[v].format(st=st))
    return cache[(v, st)][cols]
out = []
for name, tag, pat in ROWS:
    for st in STATES:
        study = pat.format(st=st); sr = f"{F}/{study}/seed_runs.json"
        if not os.path.exists(sr): print("missing", sr); continue
        for r in json.load(open(sr)):
            if "reference_metrics" not in r: continue
            run_dir = f"{F}/{study}/{os.path.basename(r['run_dir'])}"
            assert os.path.isdir(run_dir), run_dir
            m = json.load(open(f"{run_dir}/metrics.json"))
            ecfg = (m.get("config") or {}).get("eval_config") or C.EVAL_CONFIG
            cfg = ev.EvalConfig.from_yaml(ecfg); cols = list(cfg.order)
            gen = pd.read_csv(f"{run_dir}/generated_samples.csv", usecols=cols)
            row = dict(row=name, tag=tag, state=st, seed=r["seed"], n_generated=len(gen),
                       eval_config=os.path.basename(ecfg), srmse_1_stored=r["reference_metrics"]["srmse_1"])
            for v in REFS:
                row[f"srmse_1_{v}"] = float(ev.srmse_kway(ref(v, st, cols), gen, cfg, k=1)["mean"])
            out.append(row)
            print(name, st, r["seed"], {k: round(x, 4) for k, x in row.items() if k.startswith("srmse")}, flush=True)
df = pd.DataFrame(out); os.makedirs(f"{HERE}/results", exist_ok=True)
df["absdiff_orig_vs_stored"] = (df.srmse_1_orig - df.srmse_1_stored).abs()
df["delta"] = df.srmse_1_fresh - df.srmse_1_orig
df.to_csv(f"{HERE}/results/fidelity_per_seed.csv", index=False)
agg = df.groupby(["row", "tag", "state"], sort=False).agg(
    n_seeds=("seed", "size"), stored_mean=("srmse_1_stored", "mean"), stored_sd=("srmse_1_stored", "std"),
    orig_mean=("srmse_1_orig", "mean"), orig_sd=("srmse_1_orig", "std"),
    fresh_mean=("srmse_1_fresh", "mean"), fresh_sd=("srmse_1_fresh", "std")).reset_index()
agg["delta_mean"] = agg.fresh_mean - agg.orig_mean; agg["rel_delta_mean"] = agg.delta_mean / agg.orig_mean
agg["rank_orig"] = agg.groupby("state").orig_mean.rank(method="min").astype(int)
agg["rank_fresh"] = agg.groupby("state").fresh_mean.rank(method="min").astype(int)
agg.to_csv(f"{HERE}/results/fidelity_compare.csv", index=False)
rk = []
for st in STATES:
    a = agg[agg.state == st]
    rk.append(dict(state=st, n_rows=len(a), spearman=spearmanr(a.orig_mean, a.fresh_mean)[0],
                   kendall=kendalltau(a.orig_mean, a.fresh_mean)[0], n_rank_changes=int((a.rank_orig != a.rank_fresh).sum()),
                   best_orig=a.loc[a.orig_mean.idxmin(), "row"], best_fresh=a.loc[a.fresh_mean.idxmin(), "row"],
                   max_abs_delta=a.delta_mean.abs().max(), max_rel_delta=a.rel_delta_mean.abs().max()))
pd.DataFrame(rk).to_csv(f"{HERE}/results/fidelity_rank.csv", index=False)
# yardsticks
cfg = ev.EvalConfig.from_yaml(C.EVAL_CONFIG); cols = list(cfg.order); ys = []
for st in STATES:
    d = f"{ROOT}/data/pums_states/{st}"
    ids = pickle.load(open(f"{d}/dataset_ids.pkl", "rb"))
    pl = pd.read_csv(f"{d}/person_level_2013_{st}.csv")
    train = pl.iloc[np.asarray(ids["train_idx"])][cols]
    U = pd.read_csv(f"{d}/historical/person_level_{st}_2013.csv")
    w = U.PWGTP.to_numpy(float)
    for s in range(3):
        rg = np.random.default_rng([99, s])
        smp = {"oracle_prodtest": ref("orig", st, cols).sample(n=10000, random_state=s),
               "oracle_iid": U.iloc[rg.choice(len(U), 10000, replace=True, p=w / w.sum())][cols],
               "train_copy": train.sample(n=10000, random_state=s)}
        for k, g in smp.items():
            ys.append(dict(state=st, draw=s, sample=k, **{f"srmse_1_{v}": float(ev.srmse_kway(ref(v, st, cols), g, cfg, k=1)["mean"]) for v in REFS}))
Y = pd.DataFrame(ys); Y.to_csv(f"{HERE}/results/fidelity_yardsticks.csv", index=False)
print("max |orig - stored| =", df.absdiff_orig_vs_stored.max())
print(pd.DataFrame(rk).to_string(index=False)); print(Y.groupby(["state", "sample"])[["srmse_1_orig", "srmse_1_fresh"]].mean())
