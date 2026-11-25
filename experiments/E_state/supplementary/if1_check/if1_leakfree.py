#!/usr/bin/env python
"""Companion to if1_check.py: how much of the exact-tuple IF1 against val comes from
copies of the same PUMS source person (PWGTP resample with replacement, split by row)?

Reference = val records whose Source_Person_ID does NOT occur in train ("leak-free val").
Generators as in if1_check.py (same RNG seed / realizations), 10k rows each, exact P/R/IF1.
Also reports Spearman rank correlation, over the six priors, between exact IF1 (full val)
and the NIP/NIR harmonic mean, read from if1_summary.csv.
Output: if1_leakfree.csv, rank_agreement.csv
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import if1_check as C  # noqa: E402

OUT = C.OUT


def main():
    rows = []
    for st in C.STATES:
        raw = pd.read_csv(f"{C.ROOT}/data/pums_states/{st}/person_level_2013_{st}.csv",
                          usecols=C.COLS + ["Source_Person_ID"])
        ids = pickle.load(open(f"{C.ROOT}/data/pums_states/{st}/dataset_ids.pkl", "rb"))
        full = C.norm(raw)
        src = pd.factorize(raw["Source_Person_ID"].astype(str))[0]  # integer codes: fast isin
        tr, va, te = (np.asarray(ids[k]) for k in ("train_idx", "val_idx", "test_idx"))
        clean = va[~np.isin(src[va], src[tr])]
        kref = C.keys(full.iloc[clean])
        ktr = C.keys(full.iloc[tr])
        rng = np.random.default_rng(20261124)
        gens = []
        te_ok = te[~np.isin(src[te], np.concatenate([src[tr], src[clean]]))]
        for r in range(C.N_REAL):
            gens.append(("Memorizer (train subsample)", r, full.iloc[tr].iloc[rng.choice(len(tr), C.N_SUB, replace=False)]))
            # real comparator with no source person shared with train or with the clean val
            gens.append(("Real (test subsample, no shared source)", r,
                         full.iloc[te_ok].iloc[rng.choice(len(te_ok), min(C.N_SUB, len(te_ok)), replace=False)]))
        for name, tpl in C.MODELS.items():
            d = C.ens_dir(tpl.format(st=st))
            for r in range(C.N_REAL):
                g = C.norm(pd.read_csv(f"{d}/generated_ensembles/final_generated_samples_{r}.csv", usecols=C.COLS))
                gens.append((name, r, g.iloc[rng.choice(len(g), C.N_SUB, replace=False)]))
        for name, r, g in gens:
            kg = C.keys(g)
            e = C.exact_pr(kref, kg)
            rows.append(dict(state=st, generator=name, rep=r, n_ref=len(clean), P=e["P"], R=e["R"],
                             IF1=e["IF1"], gen_in_train=float(np.isin(kg, ktr).mean())))
        print(st, len(clean), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/if1_leakfree.csv", index=False)
    s = res.groupby(["state", "generator"], sort=False)[["n_ref", "P", "R", "IF1", "gen_in_train"]].mean()
    print(s.to_string(float_format=lambda x: f"{x:.4f}"))

    summ = pd.read_csv(f"{OUT}/if1_summary.csv")
    pri = summ[summ.generator.isin(C.MODELS)]
    ra = []
    for st in C.STATES:
        q = pri[pri.state == st]
        rho = spearmanr(q.IF1_mean, q.NIP_NIR_HM_mean).statistic
        lf = s.loc[st].reindex(q.generator)["IF1"].to_numpy()
        ra.append(dict(state=st, spearman_IF1_vs_HM=rho,
                       spearman_IF1_vs_IF1leakfree=spearmanr(q.IF1_mean, lf).statistic,
                       order_IF1=" > ".join(q.sort_values("IF1_mean", ascending=False).generator),
                       order_HM=" > ".join(q.sort_values("NIP_NIR_HM_mean", ascending=False).generator),
                       order_IF1_leakfree=" > ".join(q.generator.to_numpy()[np.argsort(-lf)])))
    ra = pd.DataFrame(ra)
    ra.to_csv(f"{OUT}/rank_agreement.csv", index=False)
    print(ra.to_string())


if __name__ == "__main__":
    main()
