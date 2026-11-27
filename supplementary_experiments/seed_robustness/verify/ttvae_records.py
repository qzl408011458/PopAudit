"""TTVAE record reuse across training seeds (main Sec. 5.3, Supplementary S3.8), all three states.

For all 50 realization files of the seed-42, seed-142 and seed-242 TTVAE models, prints
  - the share of generated records whose 12-column tuple coincides with a record of the
    40,000-record training split (data/pums_states/<ST>/traindata_pers.csv; the file holds
    40,000 records, its line count is larger because the persona texts contain newlines),
  - the share that coincides with a record of the test split, and
  - for each pair of seeds and for all three seeds, the share of rows that are identical
    position by position,
and ends with a per-state summary (range over all realizations and seeds) that gives the
figures reported in the paper. Read-only.

Inputs (available on request): the seed-42 ensembles under $POPAUDIT_ENSEMBLES and the
seed-142/242 ensembles under supplementary_experiments/seed_robustness/final/ttvae_<ST>/.
Usage: python ttvae_records.py [CA OH TX]   (output: ttvae_records.log)
"""
import sys, pathlib
from concurrent.futures import ProcessPoolExecutor
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))
import popaudit_paths as _PA
import pandas as pd

SEEDS = (42, 142, 242)
RIDS = tuple(range(50))


def ens_dir(st, s):
    if s == 42:
        return f"{_PA.ENSEMBLES}/ttvae_{st}/ttvae_{st}_final_s42"
    return _PA.R(f"/supplementary_experiments/seed_robustness/final/ttvae_{st}/ttvae_{st}_final_s{s}")


def one(args):
    st, rid = args
    tr = pd.read_csv(f"{_PA.DATA}/pums_states/{st}/traindata_pers.csv", low_memory=False)
    te = pd.read_csv(f"{_PA.DATA}/pums_states/{st}/testdata_pers.csv", low_memory=False)
    G = {s: pd.read_csv(f"{ens_dir(st, s)}/generated_ensembles/final_generated_samples_{rid}.csv") for s in SEEDS}
    cols = list(G[42].columns)
    key = lambda d: d[cols].astype(str).agg("|".join, axis=1)
    ktr, kte = set(key(tr[cols])), set(key(te[cols]))
    K = {s: key(g) for s, g in G.items()}
    lines, row = [], {"st": st, "rid": rid}
    for s in SEEDS:
        row[f"in_train_{s}"] = K[s].isin(ktr).mean()
        lines.append(f"{st} rid={rid} seed={s} n={len(K[s])} in_train={row[f'in_train_{s}']:.5f} "
                     f"in_test={K[s].isin(kte).mean():.5f}")
    for a, b in ((42, 142), (42, 242), (142, 242)):
        lines.append(f"{st} rid={rid} seeds {a}-{b} rowwise_identical={(K[a] == K[b]).mean():.5f}")
    row["ident3"] = ((K[42] == K[142]) & (K[42] == K[242])).mean()
    lines.append(f"{st} rid={rid} seeds 42-142-242 rowwise_identical={row['ident3']:.5f}")
    return lines, row


def main(states):
    jobs = [(st, rid) for st in states for rid in RIDS]
    with ProcessPoolExecutor(max_workers=16) as ex:
        res = list(ex.map(one, jobs))
    rows = []
    for lines, row in res:
        print("\n".join(lines), flush=True)
        rows.append(row)
    df = pd.DataFrame(rows)
    print("\n== summary over all realizations and seeds")
    for st, g in df.groupby("st", sort=False):
        it = pd.concat([g[f"in_train_{s}"] for s in SEEDS])
        print(f"{st}: in_train {it.min():.5f} to {it.max():.5f}; "
              f"identical across all three seeds {g.ident3.min():.5f} to {g.ident3.max():.5f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["CA", "OH", "TX"])
