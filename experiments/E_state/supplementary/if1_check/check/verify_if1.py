#!/usr/bin/env python
"""Independent re-computation of the IF1 check (independent verification).

Own implementation: tuples are '|'-joined strings, membership via Python sets
(no hashing, no np.isin), different RNG seed from if1_check.py.
Kim & Bansal (2023) Eqs. 14-16, row-based:
  P = (1/M) sum_j 1[Xhat_j in set(X)],  R = (1/N) sum_i 1[X_i in set(Xhat)],  F1 = HM.
Outputs: verify_split.csv, verify_if1.csv (per rep), verify_if1_summary.csv
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[5])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import os, sys, pickle, time
import numpy as np
import pandas as pd

ROOT = _PA.R("")
sys.path.insert(0, ROOT)
from utils.coverage_pr import alpha_precision_beta_recall  # noqa

OUT = os.path.dirname(os.path.abspath(__file__))
FINAL = _PA.S("/hpo_v3/final")
STATES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["CA", "TX"]
NUM = ["Age", "Household_Size", "Number_of_children", "Number_of_cars_of_household"]
CAT = ["Gender", "Marital_status", "Employment_status", "Studenthood_status",
       "Household_Type", "Commute_Mode", "Personal_Income_Class", "Household_Income_Class"]
COLS = NUM + CAT
MODELS = {"WGAN-GP-ZCR": "wgan_zcr_{st}", "SemaPop-VAE+": "semapop_vae_{st}_train",
          "TTVAE": "ttvae_{st}", "BN": "bn_pgmpy_{st}",
          "TabDDPM": "tabddpm_{st}_valsrmse", "CTGAN": "ctgan_{st}"}
NREP, NSUB = 3, 10_000


def clean(df):
    df = df[COLS].copy()
    for c in NUM:
        df[c] = np.rint(pd.to_numeric(df[c])).astype(int)
    for c in CAT:
        df[c] = df[c].astype(str).str.strip()
    return df


def tup(df, age5=False):
    d = df.copy()
    if age5:
        d["Age"] = np.minimum(d["Age"] // 5, 17)
    return d.astype(str).agg("|".join, axis=1).tolist()


def pr(ref, gen):
    sr, sg = set(ref), set(gen)
    p = sum(g in sr for g in gen) / len(gen)
    r = sum(x in sg for x in ref) / len(ref)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


split_rows, rows = [], []
for st in STATES:
    t0 = time.time()
    raw = pd.read_csv(f"{ROOT}/data/pums_states/{st}/person_level_2013_{st}.csv")
    ids = pickle.load(open(f"{ROOT}/data/pums_states/{st}/dataset_ids.pkl", "rb"))
    tr, va, te = (np.array(ids[k]) for k in ("train_idx", "val_idx", "test_idx"))
    assert len(set(tr) & set(va)) == 0 and len(set(te) & (set(tr) | set(va))) == 0
    full = clean(raw)
    T = tup(full)
    T5 = tup(full, age5=True)
    src = raw["Source_Person_ID"].astype(str).to_numpy()
    Ttr, Tva, Tte = [T[i] for i in tr], [T[i] for i in va], [T[i] for i in te]
    Str, Sva, Ste = set(src[tr]), src[va], set(src[te])
    setTtr = set(Ttr)
    m_tr = np.array([t in setTtr for t in Tva])
    s_tr = np.array([s in Str for s in Sva])
    vc = pd.Series(Tva).value_counts()
    sc = pd.Series(Sva).value_counts()
    setT5tr = set(T5[i] for i in tr)
    v5 = [T5[i] for i in va]
    split_rows.append(dict(
        state=st,
        val_unique_within_val=float((pd.Series(Tva).map(vc) == 1).mean()),
        val_unique_within_val_age5=float((pd.Series(v5).map(pd.Series(v5).value_counts()) == 1).mean()),
        val_distinct_per_row=len(set(Tva)) / len(Tva),
        train_distinct_per_row=len(setTtr) / len(Ttr),
        val_match_in_train=m_tr.mean(),
        val_match_in_train_age5=np.mean([t in setT5tr for t in v5]),
        val_src_in_train=s_tr.mean(),
        val_src_dup_within_val=float((pd.Series(Sva).map(sc) > 1).mean()),
        match_share_shared_src=(m_tr & s_tr).sum() / m_tr.sum(),
        match_share_coincidental=(m_tr & ~s_tr).sum() / m_tr.sum(),
        val_coincidental_match=(m_tr & ~s_tr).mean(),
        val_match_in_test=np.mean([t in set(Tte) for t in Tva]),
        val_src_in_test=np.mean([s in Ste for s in Sva]),
        n_leakfree_val=int((~s_tr).sum()),
    ))
    print(split_rows[-1], flush=True)

    Dval = full.iloc[va].reset_index(drop=True)
    lf = ~s_tr
    Tva_lf = [t for t, k in zip(Tva, lf) if k]
    rng = np.random.default_rng(777)
    gens = []
    for r in range(NREP):
        ix = rng.choice(len(tr), NSUB, replace=False)
        gens.append(("Memorizer", r, full.iloc[tr[ix]].reset_index(drop=True)))
        ix = rng.choice(len(te), NSUB, replace=False)
        gens.append(("Real(test)", r, full.iloc[te[ix]].reset_index(drop=True)))
    for name, tpl in MODELS.items():
        s = tpl.format(st=st)
        for r in range(NREP):
            g = clean(pd.read_csv(f"{FINAL}/{s}/{s}_final_s42/generated_ensembles/"
                                  f"final_generated_samples_{r}.csv"))
            g = g.iloc[rng.choice(len(g), NSUB, replace=False)].reset_index(drop=True)
            gens.append((name, r, g))
    for name, r, g in gens:
        tg = tup(g)
        p, rc, f = pr(Tva, tg)
        p5, r5, f5 = pr(v5, tup(g, age5=True))
        _, _, flf = pr(Tva_lf, tg)
        apr = alpha_precision_beta_recall(Dval, g, CAT, NUM, subsample=8000, random_state=0)
        hm = 2 * apr["NIP"] * apr["NIR"] / (apr["NIP"] + apr["NIR"])
        rows.append(dict(state=st, gen=name, rep=r, P=p, R=rc, IF1=f, IF1_age5=f5,
                         IF1_leakfree=flf, in_train=np.mean([t in setTtr for t in tg]),
                         NIP=apr["NIP"], NIR=apr["NIR"], HM=hm))
        print(f"{st} {name:14s} r{r} P={p:.3f} R={rc:.3f} IF1={f:.3f} age5={f5:.3f} "
              f"lf={flf:.3f} inTr={rows[-1]['in_train']:.3f} HM={hm:.3f}", flush=True)
    print(f"[{st}] {time.time()-t0:.0f}s", flush=True)

pd.DataFrame(split_rows).to_csv(f"{OUT}/verify_split.csv", index=False)
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/verify_if1.csv", index=False)
summ = res.groupby(["state", "gen"], sort=False).agg(["mean", "std"]).drop(columns="rep")
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ = summ.reset_index()
summ.to_csv(f"{OUT}/verify_if1_summary.csv", index=False)
pri = summ[summ.gen.isin(MODELS)]
for st, d in pri.groupby("state"):
    print(st, "Spearman IF1 vs HM:", round(d["IF1_mean"].corr(d["HM_mean"], method="spearman"), 3),
          "| IF1 vs IF1_leakfree:", round(d["IF1_mean"].corr(d["IF1_leakfree_mean"], method="spearman"), 3))
print(summ[["state", "gen", "IF1_mean", "IF1_std", "IF1_age5_mean", "IF1_leakfree_mean",
            "in_train_mean", "HM_mean"]].round(3).to_string())
