"""Step 1e (added diagnostic): matched-selection PLACEBO for reference (a).
(a) keeps a test row iff its source household has no row in train U val (M = 50,000 random positions).
The placebo applies the SAME rule to an independent uniform random M-subset S of all N positions
(seed s): keep test rows whose household has no row in S.  Household-level keep probability is
C(N-n_h, M)/C(N, M) and kept households contribute t_h test rows (E t_h = n_h (N-M)/N), so the
expected row composition is proportional to (a)'s, but households are kept regardless of whether
they also have rows in the REAL train split.  (a) vs placebo therefore isolates the train-overlap
(memorisation) effect at matched composition; production vs placebo shows the composition effect.
Output: refs/<ST>/testdata_placebo_s<seed>.csv + refs/placebo_summary.csv"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[5])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import pickle, sys, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
DATA = Path(_PA.R("/data/pums_states"))
SEEDS = [int(x) for x in (sys.argv[1:] or ["1", "2"])]
AGE_BINS = [0, 18, 35, 50, 65, 200]; AGE_LABELS = ["0-17", "18-34", "35-49", "50-64", "65+"]
ATTRS = ["Age_Group", "Household_Size", "Number_of_children", "Number_of_cars_of_household",
         "Employment_status", "Household_Income_Class", "Commute_Mode"]
def ag(df):
    df = df.copy(); df["Age_Group"] = pd.cut(pd.to_numeric(df["Age"], errors="coerce"), bins=AGE_BINS,
                                             labels=AGE_LABELS, right=False, include_lowest=True).astype(object); return df
def tv(a, b, col):
    pa = a[col].value_counts(normalize=True); pb = b[col].value_counts(normalize=True); i = pa.index.union(pb.index)
    return 0.5 * float((pa.reindex(i, fill_value=0) - pb.reindex(i, fill_value=0)).abs().sum())
rows = []
for st in ["CA", "OH", "TX"]:
    d = DATA / st; ids = pickle.load(open(d / "dataset_ids.pkl", "rb"))
    pl = pd.read_csv(d / f"person_level_2013_{st}.csv", usecols=["Source_Person_ID"])
    hh = pl["Source_Person_ID"].astype(str).str.rsplit("-", n=1).str[0].to_numpy()
    N = len(pl); M = len(ids["train_idx"]) + len(ids["val_idx"])
    hh_tr = set(hh[np.asarray(ids["train_idx"])]); hh_trva = hh_tr | set(hh[np.asarray(ids["val_idx"])])
    te = pd.read_csv(d / "testdata_pers.csv"); A = pd.read_csv(HERE / "refs" / st / "testdata_hhdisjoint_trainval.csv")
    hh_te = pd.Series(hh[te["Pid"].to_numpy()])
    tg, Ag = ag(te), ag(A)
    for s in SEEDS:
        S = np.random.default_rng(s).choice(N, size=M, replace=False)
        keep = ~hh_te.isin(set(hh[S])).to_numpy()
        P = te.loc[keep]; P.to_csv(HERE / "refs" / st / f"testdata_placebo_s{s}.csv", index=False)
        Pg = ag(P)
        r = dict(state=st, seed=s, n_rows=len(P), n_a=len(A),
                 share_rows_hh_in_train=round(float(hh_te[keep].isin(hh_tr).mean()), 6),
                 share_rows_hh_in_trainval=round(float(hh_te[keep].isin(hh_trva).mean()), 6),
                 mean_tv_vs_test=round(np.mean([tv(tg, Pg, c) for c in ATTRS]), 6),
                 mean_tv_a_vs_test=round(np.mean([tv(tg, Ag, c) for c in ATTRS]), 6),
                 mean_tv_vs_a=round(np.mean([tv(Ag, Pg, c) for c in ATTRS]), 6))
        r.update({f"tv_vs_a_{c}": round(tv(Ag, Pg, c), 6) for c in ATTRS})
        rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(HERE / "refs" / "placebo_summary.csv", index=False)
