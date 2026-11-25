#!/usr/bin/env python
"""Exact-tuple IF1 (Kim & Bansal 2023, Eqs. 14-16) on the 12-attribute 2013 build.

Question: is the supplementary's sentence "On this build that quantity is degenerate,
because almost every reference record is unique" true on the CURRENT 12-attribute build
(it was written for the earlier 50-attribute setting)?

Kim & Bansal (2023), Eqs. 14-16, row-based:
    Precision = (1/M) sum_j 1[ Xhat_j in X ]     (share of generated rows whose tuple exists in the reference)
    Recall    = (1/N) sum_i 1[ X_i in Xhat ]     (share of reference rows whose tuple exists in the generated set)
    F1        = harmonic mean.
Their reference X is the h-population (all 1.07M records, of which the 5% training
h-sample is a subset) and every attribute is categorical.  Here we use, as the HPO did,
the held-out val split (10k) as the reference, and generated samples of 10k rows.

Variants:
  ref=val      : val (10k) reference, 10k generated rows  (the HPO setting)
  ref=test_pop : test split (|test| rows) as an h-population analogue, full realization
                 (|test| rows) as generated; memorizer = train resampled to |test| rows.

Tuple encoding: the 12 modelled attributes as stored (Age, Household_Size,
Number_of_children, Number_of_cars_of_household as integers; generated numerics rounded
to integers); a sensitivity with Age in 5-year bands (85+ top-coded) is also reported.

Outputs (this directory): if1_results.csv (per state x generator x replicate),
if1_summary.csv (means), split_stats.csv (task item 1).
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[4])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import glob
import os
import sys
import time
import pickle

import numpy as np
import pandas as pd

ROOT = _PA.R("")
sys.path.insert(0, ROOT)
from utils.coverage_pr import alpha_precision_beta_recall  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
FINAL = _PA.S("/hpo_v3/final")
STATES = ["CA", "TX", "OH"]
NUM = ["Age", "Household_Size", "Number_of_children", "Number_of_cars_of_household"]
CAT = ["Gender", "Marital_status", "Employment_status", "Studenthood_status",
       "Household_Type", "Commute_Mode", "Personal_Income_Class", "Household_Income_Class"]
COLS = NUM + CAT
# audited primaries (experiments/E_state/run_audit_v3.py PRIMARY)
MODELS = {
    "WGAN-GP-ZCR": "wgan_zcr_{st}",
    "SemaPop-VAE+": "semapop_vae_{st}_train",
    "TTVAE": "ttvae_{st}",
    "BN": "bn_pgmpy_{st}",
    "TabDDPM": "tabddpm_{st}_valsrmse",
    "CTGAN": "ctgan_{st}",
}
N_REAL = 3        # realizations per model
N_SUB = 10_000
DO_NIP = True


def norm(df):
    df = df[COLS].copy()
    for c in NUM:
        df[c] = np.rint(pd.to_numeric(df[c], errors="coerce")).astype("int64")
    for c in CAT:
        df[c] = df[c].astype(str)
    return df


def keys(df, age5=False):
    d = df
    if age5:
        d = df.copy()
        d["Age"] = np.minimum(d["Age"] // 5, 17)
    return pd.util.hash_pandas_object(d[COLS], index=False).to_numpy()


def exact_pr(kref, kgen):
    prec = float(np.isin(kgen, kref).mean())
    rec = float(np.isin(kref, kgen).mean())
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    # set-based (unique-combination) variant, for reference
    ur, ug = np.unique(kref), np.unique(kgen)
    inter = np.intersect1d(ur, ug).size
    return dict(P=prec, R=rec, IF1=f1, P_set=inter / ug.size, R_set=inter / ur.size,
                n_unique_gen=int(ug.size))


def unique_share(k):
    _, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
    return float((cnt[inv] == 1).mean())


def ens_dir(study):
    d = f"{FINAL}/{study}/{study}_final_s42"
    if os.path.isdir(d):
        return d
    c = glob.glob(f"{FINAL}/{study}/*_final_s42")
    return c[0]


def main():
    rows, split_rows = [], []
    for st in STATES:
        t0 = time.time()
        raw = pd.read_csv(f"{ROOT}/data/pums_states/{st}/person_level_2013_{st}.csv",
                          usecols=COLS + ["Source_Person_ID"])
        ids = pickle.load(open(f"{ROOT}/data/pums_states/{st}/dataset_ids.pkl", "rb"))
        full = norm(raw)
        src = raw["Source_Person_ID"].astype(str).to_numpy()
        tr, va, te = (np.asarray(ids[k]) for k in ("train_idx", "val_idx", "test_idx"))
        D = {n: full.iloc[ix].reset_index(drop=True) for n, ix in
             (("train", tr), ("val", va), ("test", te))}
        K = {n: keys(d) for n, d in D.items()}
        K5 = {n: keys(d, age5=True) for n, d in D.items()}

        # ---------- item 1: split structure ----------
        kv, kt = K["val"], K["train"]
        u, inv, cnt = np.unique(kv, return_inverse=True, return_counts=True)
        val_unique_share = float((cnt[inv] == 1).mean())
        in_train = np.isin(kv, kt)
        sv, stt = src[va], src[tr]
        src_in_train = np.isin(sv, stt)
        # within-val source duplication
        _, sinv, scnt = np.unique(sv, return_inverse=True, return_counts=True)
        split_rows.append(dict(
            state=st, n_train=len(tr), n_val=len(va), n_test=len(te),
            val_tuple_unique_within_val=val_unique_share,
            val_distinct_tuples_per_row=u.size / kv.size,
            train_distinct_tuples_per_row=np.unique(kt).size / kt.size,
            val_exact_match_in_train=float(in_train.mean()),
            val_source_person_in_train=float(src_in_train.mean()),
            val_source_person_dup_within_val=float((scnt[sinv] > 1).mean()),
            match_explained_by_shared_source=float((in_train & src_in_train).sum() / in_train.sum()),
            match_coincidental=float((in_train & ~src_in_train).sum() / in_train.sum()),
            val_exact_match_in_train_excl_shared_source=float((in_train & ~src_in_train).mean()),
            val_exact_match_in_test=float(np.isin(kv, K["test"]).mean()),
            val_source_person_in_test=float(np.isin(sv, src[te]).mean()),
            val_tuple_unique_within_val_age5=unique_share(K5["val"]),
            val_exact_match_in_train_age5=float(np.isin(K5["val"], K5["train"]).mean()),
        ))
        print(f"[{st}] split stats: {split_rows[-1]}", flush=True)

        # ---------- item 2: generators ----------
        rng = np.random.default_rng(20261124)
        gens = []   # (name, rep, df10k, df_full_or_None)
        for r in range(N_REAL):
            gens.append(("Memorizer (train subsample)", r,
                         D["train"].iloc[rng.choice(len(tr), N_SUB, replace=False)], None))
            gens.append(("Real (test subsample)", r,
                         D["test"].iloc[rng.choice(len(te), N_SUB, replace=False)], None))
        for name, tpl in MODELS.items():
            d = ens_dir(tpl.format(st=st))
            for r in range(N_REAL):
                g = pd.read_csv(f"{d}/generated_ensembles/final_generated_samples_{r}.csv",
                                usecols=COLS)
                nonint = float((pd.to_numeric(g["Age"]) % 1 != 0).mean())
                g = norm(g)
                sub = g.iloc[rng.choice(len(g), N_SUB, replace=False)].reset_index(drop=True)
                gens.append((name, r, sub, (g, nonint)))

        for name, r, g10, extra in gens:
            kg, kg5 = keys(g10), keys(g10, age5=True)
            e = exact_pr(K["val"], kg)
            e5 = exact_pr(K5["val"], kg5)
            row = dict(state=st, generator=name, rep=r,
                       P=e["P"], R=e["R"], IF1=e["IF1"], P_set=e["P_set"], R_set=e["R_set"],
                       n_unique_gen=e["n_unique_gen"],
                       P_age5=e5["P"], R_age5=e5["R"], IF1_age5=e5["IF1"],
                       gen_in_train=float(np.isin(kg, K["train"]).mean()),
                       gen_in_val_not_train=float((np.isin(kg, K["val"]) & ~np.isin(kg, K["train"])).mean()))
            if extra is not None:
                gfull, nonint = extra
                row["age_nonint_share_raw"] = nonint
                # h-population analogue: ref = test split, gen = full realization (|test| rows)
                ep = exact_pr(K["test"], keys(gfull))
                row.update(P_pop=ep["P"], R_pop=ep["R"], IF1_pop=ep["IF1"])
            elif name.startswith("Memorizer"):
                big = D["train"].iloc[rng.choice(len(tr), len(te), replace=True)]
                ep = exact_pr(K["test"], keys(big))
                row.update(P_pop=ep["P"], R_pop=ep["R"], IF1_pop=ep["IF1"])
            if DO_NIP:
                pr = alpha_precision_beta_recall(D["val"], g10.reset_index(drop=True), CAT, NUM,
                                                 subsample=8000, random_state=0)
                row.update(NIP=pr["NIP"], NIR=pr["NIR"], NIP_NIR_HM=pr["IF1"])
            rows.append(row)
            print(f"  {st} {name:30s} r{r} P={row['P']:.3f} R={row['R']:.3f} IF1={row['IF1']:.3f} "
                  f"inTrain={row['gen_in_train']:.3f} HM={row.get('NIP_NIR_HM', float('nan')):.3f} "
                  f"IF1pop={row.get('IF1_pop', float('nan')):.3f}", flush=True)
        print(f"[{st}] done in {time.time() - t0:.0f}s", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/if1_results.csv", index=False)
    pd.DataFrame(split_rows).to_csv(f"{OUT}/split_stats.csv", index=False)
    num = [c for c in res.columns if c not in ("state", "generator", "rep")]
    summ = res.groupby(["state", "generator"], sort=False)[num].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ["rank_IF1"] = summ.groupby("state")["IF1_mean"].rank(ascending=False)
    summ["rank_HM"] = summ.groupby("state")["NIP_NIR_HM_mean"].rank(ascending=False)
    summ.to_csv(f"{OUT}/if1_summary.csv", index=False)
    print(summ[["state", "generator", "P_mean", "R_mean", "IF1_mean", "IF1_std", "IF1_age5_mean",
                "IF1_pop_mean", "gen_in_train_mean", "NIP_NIR_HM_mean", "rank_IF1", "rank_HM"]]
          .to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
