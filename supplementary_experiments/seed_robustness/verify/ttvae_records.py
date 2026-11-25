"""TTVAE record reuse across training seeds (Supplementary S3.8), all three states.

For realization files rid 0 and 17 of the seed-42, seed-142 and seed-242 TTVAE models, prints
  - the share of generated records whose 12-column tuple coincides with a record of the
    40,000-record training split (data/pums_states/<ST>/traindata_pers.csv; the file holds
    40,000 records, its line count is larger because the persona texts contain newlines),
  - the share that coincides with a record of the test split, and
  - for each pair of seeds, the share of rows that are identical position by position.
Generalises ttvae_ca_records.py (California only) to Ohio and Texas. Read-only.

Inputs (available on request): the seed-42 ensembles under $POPAUDIT_ENSEMBLES and the
seed-142/242 ensembles under supplementary_experiments/seed_robustness/final/ttvae_<ST>/.
Usage: python ttvae_records.py [CA OH TX]   (output: ttvae_records.log)
"""
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))
import popaudit_paths as _PA
import pandas as pd

SEEDS = (42, 142, 242)
RIDS = (0, 17)


def ens_dir(st, s):
    if s == 42:
        return f"{_PA.ENSEMBLES}/ttvae_{st}/ttvae_{st}_final_s42"
    return _PA.R(f"/supplementary_experiments/seed_robustness/final/ttvae_{st}/ttvae_{st}_final_s{s}")


def main(states):
    for st in states:
        tr = pd.read_csv(f"{_PA.DATA}/pums_states/{st}/traindata_pers.csv", low_memory=False)
        te = pd.read_csv(f"{_PA.DATA}/pums_states/{st}/testdata_pers.csv", low_memory=False)
        print(f"== {st}: training split {len(tr)} records, test split {len(te)} records", flush=True)
        for rid in RIDS:
            G = {s: pd.read_csv(f"{ens_dir(st, s)}/generated_ensembles/final_generated_samples_{rid}.csv")
                 for s in SEEDS}
            cols = list(G[42].columns)
            key = lambda d: d[cols].astype(str).agg("|".join, axis=1)
            ktr, kte = set(key(tr[cols])), set(key(te[cols]))
            K = {s: key(g) for s, g in G.items()}
            for s in SEEDS:
                print(f"{st} rid={rid} seed={s} n={len(K[s])} in_train={K[s].isin(ktr).mean():.4f} "
                      f"in_test={K[s].isin(kte).mean():.4f}", flush=True)
            for a, b in ((42, 142), (42, 242), (142, 242)):
                print(f"{st} rid={rid} seeds {a}-{b} rowwise_identical={(K[a] == K[b]).mean():.4f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["CA", "OH", "TX"])
