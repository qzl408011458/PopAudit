"""Overlap of the fresh reference with train (and train U val), and marginal TV of the 7 audited attributes
(Age -> Age_Group) fresh vs production testdata_pers.csv.
Overlap: share of reference rows whose source person / source household (SERIALNO) has >=1 row in train.
  * production test vs train (observed), fresh vs train (observed)
  * conditional iid expectation given the actual train: E[share | train] = PWGTP share of the state PUMS
    persons that are in train (person) / whose household is in train (household); binomial sd sqrt(p(1-p)/n)
TV: 0.5*sum|p-q| per attribute; yardsticks: TV of each sample vs the PWGTP-weighted state PUMS population,
and the Monte Carlo distribution of TV between two independent iid draws of size n_test (200 reps, multinomial
on the attribute's weighted population distribution).
Output: ../results/overlap.csv, ../results/marginal_tv_fresh_vs_prod.csv"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[3])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import pickle, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
DATA = Path(_PA.R("/data/pums_states"))
AGE_BINS = [0, 18, 35, 50, 65, 200]; AGE_LABELS = ["0-17", "18-34", "35-49", "50-64", "65+"]
ATTRS = ["Age_Group", "Household_Size", "Number_of_children", "Number_of_cars_of_household",
         "Employment_status", "Household_Income_Class", "Commute_Mode"]
def ag(df):
    df = df.copy(); df["Age_Group"] = pd.cut(pd.to_numeric(df["Age"], errors="coerce"), bins=AGE_BINS, labels=AGE_LABELS,
                                             right=False, include_lowest=True).astype(str); return df
def dist(df, col, w=None):
    p = df.groupby(col).size() if w is None else df.groupby(col)[w].sum(); return p / p.sum()
def tv(p, q):
    idx = p.index.union(q.index); return 0.5 * float((p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).abs().sum())
ov, tvr = [], []
rng = np.random.default_rng(7)
for st in ["CA", "OH", "TX"]:
    d = DATA / st
    ids = pickle.load(open(d / "dataset_ids.pkl", "rb"))
    pl = pd.read_csv(d / f"person_level_2013_{st}.csv", dtype={"Source_Person_ID": str})
    tr, va, ti = (np.asarray(ids[k]) for k in ("train_idx", "val_idx", "test_idx"))
    sp = pl.Source_Person_ID; hh = sp.str.rsplit("-", n=1).str[0]
    U = pd.read_csv(d / "historical" / f"person_level_{st}_2013.csv", dtype={"Person_ID": str})
    U["hh"] = U.Person_ID.str.rsplit("-", n=1).str[0]; W = U.PWGTP.sum()
    side = pd.read_csv(HERE / f"refs/{st}/testdata_fresh_source_ids.csv", dtype={"Source_Person_ID": str})
    f_sp = side.Source_Person_ID; f_hh = f_sp.str.rsplit("-", n=1).str[0]
    p_sp = sp.iloc[ti].reset_index(drop=True); p_hh = hh.iloc[ti].reset_index(drop=True)
    for excl_name, pos in [("train", tr), ("train_val", np.concatenate([tr, va]))]:
        S_sp, S_hh = set(sp.iloc[pos]), set(hh.iloc[pos])
        e_sp = U.PWGTP[U.Person_ID.isin(S_sp)].sum() / W; e_hh = U.PWGTP[U.hh.isin(S_hh)].sum() / W
        n = len(side)
        ov.append(dict(state=st, vs=excl_name, n_ref=n,
                       person_prod=p_sp.isin(S_sp).mean(), person_fresh=f_sp.isin(S_sp).mean(),
                       person_iid_expect=e_sp, person_iid_sd=np.sqrt(e_sp * (1 - e_sp) / n),
                       hh_prod=p_hh.isin(S_hh).mean(), hh_fresh=f_hh.isin(S_hh).mean(),
                       hh_iid_expect=e_hh, hh_iid_sd=np.sqrt(e_hh * (1 - e_hh) / n)))
    # fresh vs prod overlap with each other (both are iid from the same population)
    ov[-1]["fresh_rows_sharing_person_with_prod_test"] = f_sp.isin(set(p_sp)).mean()
    ov[-1]["fresh_rows_sharing_hh_with_prod_test"] = f_hh.isin(set(p_hh)).mean()
    prod = ag(pd.read_csv(d / "testdata_pers.csv")); fresh = ag(pd.read_csv(HERE / f"refs/{st}/testdata_fresh.csv"))
    pop = ag(U); n = len(prod)
    for a in ATTRS:
        pp, pf, pw = dist(prod, a), dist(fresh, a), dist(pop, a, "PWGTP")
        mc = []
        for _ in range(200):
            x = rng.multinomial(n, pw.to_numpy()) / n; y = rng.multinomial(n, pw.to_numpy()) / n
            mc.append(0.5 * np.abs(x - y).sum())
        tvr.append(dict(state=st, attr=a, n=n, n_cat=len(pw), tv_fresh_vs_prod=tv(pf, pp), tv_prod_vs_pop=tv(pp, pw),
                        tv_fresh_vs_pop=tv(pf, pw), mc_tv_two_iid_mean=np.mean(mc), mc_tv_two_iid_p95=np.quantile(mc, 0.95),
                        max_abs_cell_diff=float((pf.reindex(pw.index, fill_value=0) - pp.reindex(pw.index, fill_value=0)).abs().max())))
O = pd.DataFrame(ov); T = pd.DataFrame(tvr)
m = T.groupby("state")[["tv_fresh_vs_prod", "tv_prod_vs_pop", "tv_fresh_vs_pop", "mc_tv_two_iid_mean", "mc_tv_two_iid_p95"]].mean().reset_index()
m["attr"] = "MEAN_7"; T = pd.concat([T, m], ignore_index=True)
(HERE / "results").mkdir(exist_ok=True)
O.to_csv(HERE / "results/overlap.csv", index=False); T.to_csv(HERE / "results/marginal_tv_fresh_vs_prod.csv", index=False)
pd.set_option("display.width", 250); print(O.round(4).to_string(index=False)); print(T.round(5).to_string(index=False))
