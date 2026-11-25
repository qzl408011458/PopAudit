"""TabDDPM generation-only, sharded over realization ids (2026-11-24, g1fix seed rerun).

Identical to tabddpm_gen_wrapper.py (train_tabddpm_v2.main(argv) with --load-final --ensemble
--generate-only; the CPU-only ml_eff multiseed diagnostic stubbed out) except that
utils.ensemble_v2.write_ensemble is replaced by a writer that draws ONLY the realization ids
given in --rids and writes them into --ens-dir.  Each realization is a deterministic function of
its seed (train_tabddpm_v2.sample_df builds torch.Generator(dev).manual_seed(seed) per call),
so a shard reproduces exactly the files a full sequential run would write; this is checked
against the stored seed-42 CA ensemble byte for byte (see launch_ca_gen.sh).

usage: tabddpm_gen_shard.py --rids 0,4,8 --ens-dir DIR -- <train_tabddpm_v2 argv...>
The argv's --out-root/--run-name must hold tabddpm_last.pt (the stored checkpoint, copied).
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[2])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import os, sys, time, builtins, functools, argparse
sys.path.insert(0, _PA.R(""))
builtins.print = functools.partial(builtins.print, flush=True)
i = sys.argv.index("--")
ap = argparse.ArgumentParser(); ap.add_argument("--rids", required=True); ap.add_argument("--ens-dir", required=True)
a = ap.parse_args(sys.argv[1:i]); argv = sys.argv[i + 1:]
RIDS = [int(x) for x in a.rids.split(",") if x != ""]
import utils.ml_efficacy as M
M.ml_efficiency_multiseed = lambda *x, **k: {"ml_eff_skipped": 1.0}
import train_tabddpm_v2 as T
from audit_config import ENSEMBLE_SEEDS


def write_ensemble_shard(sample_fn, n_samples, save_dir, k=0, seeds=None, log_every=5):
    use = list(seeds if seeds is not None else ENSEMBLE_SEEDS)
    os.makedirs(a.ens_dir, exist_ok=True)
    t0 = time.time(); paths = []
    for j, rid in enumerate(RIDS):
        p = os.path.join(a.ens_dir, f"final_generated_samples_{rid}.csv")
        if os.path.exists(p):
            print(f"[shard] rid={rid} exists, skipped"); continue
        df = sample_fn(int(use[rid]), int(n_samples))
        if len(df) != n_samples:
            raise RuntimeError(f"rid={rid}: {len(df)} rows, expected {n_samples}")
        tmp = os.path.join(a.ens_dir, f".tmp_{rid}.part"); df.to_csv(tmp, index=False); os.replace(tmp, p)
        paths.append(p)
        print(f"[shard] rid={rid} seed={use[rid]} written ({time.time() - t0:.0f}s, {j + 1}/{len(RIDS)})")
    return paths


T.write_ensemble = write_ensemble_shard
T.main(argv)
print("[shard] done", RIDS)
