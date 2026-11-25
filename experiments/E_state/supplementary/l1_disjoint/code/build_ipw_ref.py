"""Step 1c (added diagnostic): representative household-disjoint reference.
Reference (a) is a SELECTION-BIASED subset of the test split: a test row survives iff none of the
n_h rows that its source household contributed to person_level_2013 landed in train U val, so
P(survive | n_h) = prod_{j=0}^{n_h-1} (N-M-j)/(N-j) / ((N-M)/N) -- large / high-PWGTP households are
strongly under-represented (checked: empirical survival rates match this exactly). Because the
split is a uniform random permutation, selection depends ONLY on n_h, so post-stratifying (a) on n_h
back to the test split's n_h distribution removes the bias in expectation. n_h is binned so every bin
keeps >= MIN_HH distinct (a)-households (the open top bin absorbs the tail; OH has no (a) rows with
n_h >= 12, so this is approximate there). Weights are materialised as integer row replication
(stochastic rounding, seed 0, total ~= |test|) so the unweighted Layer-1 driver can read it.
Output: refs/<ST>/testdata_hhdisjoint_trainval_ipw.csv  + refs/ipw_summary.csv, refs/ipw_marginal_shift.csv"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[5])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import pickle, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
DATA = Path(_PA.R("/data/pums_states"))
MIN_HH = 30
AGE_BINS = [0, 18, 35, 50, 65, 200]; AGE_LABELS = ["0-17", "18-34", "35-49", "50-64", "65+"]
ATTRS = ["Age_Group", "Household_Size", "Number_of_children", "Number_of_cars_of_household",
         "Employment_status", "Household_Income_Class", "Commute_Mode"]
def ag(df):
    df = df.copy(); df["Age_Group"] = pd.cut(pd.to_numeric(df["Age"], errors="coerce"), bins=AGE_BINS,
                                             labels=AGE_LABELS, right=False, include_lowest=True).astype(object); return df
def tv(a, b, col):
    pa = a[col].value_counts(normalize=True); pb = b[col].value_counts(normalize=True); i = pa.index.union(pb.index)
    return 0.5 * float((pa.reindex(i, fill_value=0) - pb.reindex(i, fill_value=0)).abs().sum())
rows, shift = [], []
for st in ["CA", "OH", "TX"]:
    d = DATA / st; ids = pickle.load(open(d / "dataset_ids.pkl", "rb"))
    pl = pd.read_csv(d / f"person_level_2013_{st}.csv", usecols=["Source_Person_ID"])
    hh = pl["Source_Person_ID"].astype(str).str.rsplit("-", n=1).str[0]; nh = hh.map(hh.value_counts()).to_numpy()
    te = pd.read_csv(d / "testdata_pers.csv"); A = pd.read_csv(HERE / "refs" / st / "testdata_hhdisjoint_trainval.csv")
    nh_te = nh[te["Pid"].to_numpy()]; nh_a = nh[A["Pid"].to_numpy()]; hh_a = hh.iloc[A["Pid"].to_numpy()].to_numpy()
    # bins: ascending n_h, close a bin once it holds >= MIN_HH distinct (a)-households; remainder -> last bin
    vals = np.sort(np.unique(nh_te)); edges = [vals[0]]; cur = set()
    for v in vals:
        cur |= set(hh_a[nh_a == v])
        if len(cur) >= MIN_HH: edges.append(v + 1); cur = set()
    edges[-1] = vals[-1] + 1 if cur == set() else edges[-1]
    edges = np.array(sorted(set(edges)) + ([vals[-1] + 1] if edges[-1] <= vals[-1] else []))
    b_te = np.clip(np.searchsorted(edges, nh_te, side="right") - 1, 0, len(edges) - 2)
    b_a = np.clip(np.searchsorted(edges, nh_a, side="right") - 1, 0, len(edges) - 2)
    ct_te = np.bincount(b_te, minlength=len(edges) - 1); ct_a = np.bincount(b_a, minlength=len(edges) - 1)
    lost = ct_te[ct_a == 0].sum() / len(te)
    w = ct_te[b_a] / ct_a[b_a]; w = w * len(te) / w.sum()
    rng = np.random.default_rng(0); reps = np.floor(w).astype(int) + (rng.random(len(w)) < (w - np.floor(w)))
    C = A.loc[A.index.repeat(reps)].reset_index(drop=True)
    (HERE / "refs" / st).mkdir(parents=True, exist_ok=True)
    C.to_csv(HERE / "refs" / st / "testdata_hhdisjoint_trainval_ipw.csv", index=False)
    ess = w.sum() ** 2 / (w ** 2).sum()
    rows.append(dict(state=st, n_test=len(te), n_a=len(A), n_ipw_rows=len(C), n_bins=len(edges) - 1,
                     top_bin_from_nh=int(edges[-2]), max_weight=round(float(w.max()), 3), kish_ess_rows=round(float(ess), 1),
                     test_mass_in_empty_bins=round(float(lost), 6)))
    tg, Ag, Cg = ag(te), ag(A), ag(C)
    for c in ATTRS:
        shift.append(dict(state=st, attr=c, tv_test_vs_a=round(tv(tg, Ag, c), 6), tv_test_vs_ipw=round(tv(tg, Cg, c), 6)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(HERE / "refs" / "ipw_summary.csv", index=False)
s = pd.DataFrame(shift); s.to_csv(HERE / "refs" / "ipw_marginal_shift.csv", index=False); print(s.to_string(index=False))
print(s.groupby("state")[["tv_test_vs_a", "tv_test_vs_ipw"]].mean())
