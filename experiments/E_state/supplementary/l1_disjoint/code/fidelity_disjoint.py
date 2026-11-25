"""Step 6: base-year fidelity (tab:fidelity-v3) against household-disjoint references.
Production: hpo/finalize.py::reference_metrics_for(run_dir, state, eval_config) = SRMSE-1 (mean over the
12 eval_v3_pums12 attrs, unweighted, frozen EvalConfig) of run_dir/generated_samples.csv (10k rows)
vs data/pums_states/<ST>/referencedata_pers.csv (a symlink to testdata_pers.csv); stored in
final/<study>/seed_runs.json and aggregated (mean/sd over seeds 42/142/242) in aggregate.json, which
hpo/fidelity_table_tex.py prints.  Here the same function body is re-evaluated with ONLY the reference
file swapped; the 'orig' column must reproduce seed_runs.json exactly (checked)."""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[5])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import os, sys, json, glob
import numpy as np, pandas as pd
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
REFS = {"orig": f"{C.DATA_ROOT}/{{st}}/referencedata_pers.csv",
        "hhdisjoint_trainval": f"{HERE}/refs/{{st}}/testdata_hhdisjoint_trainval.csv",
        "hhdisjoint_trainval_ipw": f"{HERE}/refs/{{st}}/testdata_hhdisjoint_trainval_ipw.csv",
        "placebo_s1": f"{HERE}/refs/{{st}}/testdata_placebo_s1.csv"}
cache = {}
def ref(v, st, cols):
    k = (v, st)
    if k not in cache: cache[k] = pd.read_csv(REFS[v].format(st=st))
    return cache[k][cols]
out = []
for name, tag, pat in ROWS:
    for st in ["CA", "TX", "OH"]:
        study = pat.format(st=st); sr = f"{F}/{study}/seed_runs.json"
        if not os.path.exists(sr): print("missing", sr); continue
        for r in json.load(open(sr)):
            if "reference_metrics" not in r: continue
            # renamed finalize dirs (ctgan_defaults_<ST>, tvae_ref_defaults_<ST>) keep the original run
            # name inside, and seed_runs.json's run_dir still points at the tuned study's folder
            run_dir = f"{F}/{study}/{os.path.basename(r['run_dir'])}"
            assert os.path.isdir(run_dir), run_dir
            m = json.load(open(f"{run_dir}/metrics.json"))
            ecfg = (m.get("config") or {}).get("eval_config") or C.EVAL_CONFIG
            cfg = ev.EvalConfig.from_yaml(ecfg); cols = list(cfg.order)
            gen = pd.read_csv(f"{run_dir}/generated_samples.csv", usecols=cols)
            row = dict(row=name, tag=tag, state=st, seed=r["seed"], n_generated=len(gen),
                       eval_config=os.path.basename(ecfg), srmse_1_stored=r["reference_metrics"]["srmse_1"])
            for v in REFS:
                R = ref(v, st, cols)
                row[f"srmse_1_{v}"] = float(ev.srmse_kway(R, gen, cfg, k=1)["mean"])
                row[f"n_ref_{v}"] = len(R)
            out.append(row)
            print(name, st, r["seed"], {k: round(v, 4) for k, v in row.items() if k.startswith("srmse")}, flush=True)
df = pd.DataFrame(out); os.makedirs(f"{HERE}/results", exist_ok=True)
df["absdiff_orig_vs_stored"] = (df.srmse_1_orig - df.srmse_1_stored).abs()
df.to_csv(f"{HERE}/results/fidelity_per_seed.csv", index=False)
# oracle floor analogue: a 10k real sample drawn from the ORIGINAL test split, scored against each reference
cfg = ev.EvalConfig.from_yaml(C.EVAL_CONFIG); cols = list(cfg.order); orc = []
for st in ["CA", "TX", "OH"]:
    full = ref("orig", st, cols)
    for s in range(3):
        smp = full.sample(n=10000, random_state=s)
        orc.append(dict(state=st, draw=s, **{f"srmse_1_{v}": float(ev.srmse_kway(ref(v, st, cols), smp, cfg, k=1)["mean"]) for v in REFS}))
pd.DataFrame(orc).to_csv(f"{HERE}/results/fidelity_oracle_floor.csv", index=False)
agg = df.groupby(["row", "tag", "state"], sort=False).agg(
    n_seeds=("seed", "size"), **{f"{v}_mean": (f"srmse_1_{v}", "mean") for v in REFS},
    **{f"{v}_sd": (f"srmse_1_{v}", "std") for v in REFS}, s42_orig=("srmse_1_orig", "first")).reset_index()
for v in REFS: agg[f"s42_{v}"] = df[df.seed == 42].set_index(["row", "state"]).loc[list(zip(agg.row, agg.state)), f"srmse_1_{v}"].to_numpy()
agg.to_csv(f"{HERE}/results/fidelity_disjoint_compare.csv", index=False)
print("max |orig - stored| =", df.absdiff_orig_vs_stored.max())
print(pd.DataFrame(orc).groupby("state").mean())
