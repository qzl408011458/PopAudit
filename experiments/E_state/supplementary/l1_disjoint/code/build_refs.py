"""Step 1: household-disjoint Layer-1 references (shadow; production data untouched).
testdata_pers.csv row i == person_level_2013_<ST>.csv row test_idx[i] (checked: Pid == test_idx[i],
all other columns equal). Source household = Source_Person_ID.rsplit('-',1)[0] (= PUMS SERIALNO).
 (a) testdata_hhdisjoint_trainval.csv : test rows whose source household has no member in train_idx U val_idx  [PRIMARY]
 (b) testdata_hhdisjoint_train.csv    : test rows whose source household has no member in train_idx
Also: marginal TV (7 audited attrs, Age->Age_Group) orig test vs (a)/(b), PWGTP-weighted TV of the
source-person attributes, and val-vs-train household overlap (HPO validation split)."""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[5])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import pickle, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
DATA = Path(_PA.R("/data/pums_states"))
AGE_BINS = [0, 18, 35, 50, 65, 200]; AGE_LABELS = ["0-17", "18-34", "35-49", "50-64", "65+"]
ATTRS = ["Age_Group", "Household_Size", "Number_of_children", "Number_of_cars_of_household",
         "Employment_status", "Household_Income_Class", "Commute_Mode"]

def ag(df):
    df = df.copy()
    df["Age_Group"] = pd.cut(pd.to_numeric(df["Age"], errors="coerce"), bins=AGE_BINS, labels=AGE_LABELS,
                             right=False, include_lowest=True).astype(object)
    return df

def tv(a, b, col, wa=None, wb=None):
    pa = (a.groupby(col).size() if wa is None else a.groupby(col)[wa].sum()); pa = pa / pa.sum()
    pb = (b.groupby(col).size() if wb is None else b.groupby(col)[wb].sum()); pb = pb / pb.sum()
    idx = pa.index.union(pb.index)
    return 0.5 * float((pa.reindex(idx, fill_value=0) - pb.reindex(idx, fill_value=0)).abs().sum())

rows, shift = [], []
for st in ["CA", "OH", "TX"]:
    d = DATA / st
    ids = pickle.load(open(d / "dataset_ids.pkl", "rb"))
    pl = pd.read_csv(d / f"person_level_2013_{st}.csv")
    te = pd.read_csv(d / "testdata_pers.csv")
    tr, va, ti = (np.asarray(ids[k]) for k in ("train_idx", "val_idx", "test_idx"))
    # mapping check
    assert len(te) == len(ti) and (te["Pid"].to_numpy() == ti).all()
    sub = pl.iloc[ti].reset_index(drop=True)
    for c in te.columns:
        if c != "Pid": assert (sub[c].to_numpy() == te[c].to_numpy()).all(), c
    assert pl["Source_Person_ID"].notna().all()
    sp = pl["Source_Person_ID"].astype(str); hh = sp.str.rsplit("-", n=1).str[0]
    hh_tr, hh_va = set(hh.iloc[tr]), set(hh.iloc[va]); sp_tr, sp_va = set(sp.iloc[tr]), set(sp.iloc[va])
    hh_te = hh.iloc[ti].to_numpy(); sp_te = sp.iloc[ti].to_numpy()
    in_tr = pd.Series(hh_te).isin(hh_tr).to_numpy(); in_va = pd.Series(hh_te).isin(hh_va).to_numpy()
    m_a = ~(in_tr | in_va); m_b = ~in_tr
    out = HERE / "refs" / st; out.mkdir(parents=True, exist_ok=True)
    A, B = te.loc[m_a], te.loc[m_b]
    A.to_csv(out / "testdata_hhdisjoint_trainval.csv", index=False)
    B.to_csv(out / "testdata_hhdisjoint_train.csv", index=False)
    # programmatic disjointness checks (re-read from disk)
    chk = {}
    for name, m, excl_hh, excl_sp in [("a", m_a, hh_tr | hh_va, sp_tr | sp_va), ("b", m_b, hh_tr, sp_tr)]:
        f = pd.read_csv(out / ("testdata_hhdisjoint_trainval.csv" if name == "a" else "testdata_hhdisjoint_train.csv"))
        fsp = sp.iloc[f["Pid"].to_numpy()]; fhh = hh.iloc[f["Pid"].to_numpy()]
        chk[f"{name}_shared_hh"] = len(set(fhh) & excl_hh); chk[f"{name}_shared_persons"] = len(set(fsp) & excl_sp)
        assert chk[f"{name}_shared_hh"] == 0 and chk[f"{name}_shared_persons"] == 0
        assert list(f.columns) == list(te.columns) and len(f) == int(m.sum())
    # val-vs-train overlap (HPO validation split)
    hh_v = hh.iloc[va]; sp_v = sp.iloc[va]
    rows.append(dict(
        state=st, n_total=len(pl), n_train=len(tr), n_val=len(va), n_test=len(te),
        n_disjoint_trainval=int(m_a.sum()), n_disjoint_train=int(m_b.sum()),
        drop_share_trainval=round(1 - m_a.mean(), 6), drop_share_train=round(1 - m_b.mean(), 6),
        test_rows_sharing_person_with_train=round(float(pd.Series(sp_te).isin(sp_tr).mean()), 6),
        test_rows_sharing_hh_with_train=round(float(in_tr.mean()), 6),
        test_rows_sharing_hh_with_trainval=round(float((in_tr | in_va).mean()), 6),
        n_src_persons_test=len(set(sp_te)), n_src_hh_test=len(set(hh_te)),
        n_src_hh_disjoint_trainval=len(set(hh_te[m_a])), n_src_hh_disjoint_train=len(set(hh_te[m_b])),
        n_src_persons_disjoint_trainval=len(set(sp_te[m_a])),
        n_src_hh_train=len(hh_tr), n_src_hh_val=len(hh_va), n_src_hh_total=hh.nunique(),
        val_rows_sharing_hh_with_train=round(float(hh_v.isin(hh_tr).mean()), 6),
        val_rows_sharing_person_with_train=round(float(sp_v.isin(sp_tr).mean()), 6),
        val_distinct_hh_shared_with_train_share=round(len(hh_va & hh_tr) / len(hh_va), 6),
        **chk))
    # representativeness: unweighted TV on the 7 audit attrs (L1 representation) + PWGTP-weighted
    teg, Ag, Bg = ag(te), ag(A), ag(B)
    w = pl["PWGTP"].to_numpy()
    T = ag(sub.assign(PWGTP=w[ti])); TA = T.loc[m_a]
    full = ag(pl)
    for c in ATTRS:
        shift.append(dict(state=st, attr=c, tv_test_vs_a=round(tv(teg, Ag, c), 6), tv_test_vs_b=round(tv(teg, Bg, c), 6),
                          tv_pwgtp_test_vs_a=round(tv(T, TA, c, "PWGTP", "PWGTP"), 6),
                          tv_full_vs_test=round(tv(full, teg, c), 6), tv_full_vs_a=round(tv(full, Ag, c), 6),
                          n_cat_test=int(teg[c].nunique()), n_cat_a=int(Ag[c].nunique())))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(HERE / "refs" / "summary.csv", index=False)
s = pd.DataFrame(shift); s.to_csv(HERE / "refs" / "marginal_shift.csv", index=False)
print(s.to_string(index=False))
