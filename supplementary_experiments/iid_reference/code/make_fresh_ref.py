"""Fresh, independent base-year Layer-1 reference of the SAME design as production (seed 20261124).

Production design (reproduced bit-for-bit in reproduce_resample.py): person-unit, PWGTP-proportional,
with-replacement iid draws from the full 2013 PUMS person sample (households not drawn jointly), then a
random positional train/val/test split.  Conditional on the state row count, the state slice is an iid
PWGTP-proportional sample of the state's PUMS persons, and the test split is an iid sample independent
of train.  Equivalent fresh test:

    pos = rng.choice(N_state_PUMS, size=n_test, replace=True, p=PWGTP / sum PWGTP)

drawn from the FULL state PUMS person sample (training persons NOT excluded), rng independent of 42.

Source table = data/pums_states/<ST>/historical/person_level_<ST>_2013.csv (processed, non-resampled,
same _pums_proc code path).  Checked here:
  * its Person_ID set == raw PUMS SERIALNO-SPORDER set of data/pums13/person_2013_<ST>_full.csv
    (i.e. it IS the full state PUMS person sample, 1 row per person);
  * its PWGTP == raw PWGTP;
  * attribute values == production resampled file for the same source person;
  * the unique source persons of person_level_2013_<ST>.csv cover only ~55-57% of the state PUMS
    persons, so resampling from them would NOT reproduce the design (persons never drawn in the
    production resample would have probability 0) -> not used.
Output: ../refs/<ST>/testdata_fresh.csv (columns/order/dtypes == testdata_pers.csv; Pid = 0..n-1, the
fresh per-draw Sampled_Person_ID renamed Pid) + ../refs/<ST>/testdata_fresh_source_ids.csv
(Pid, Source_Person_ID, PWGTP) + ../refs/<ST>/fresh_info.json.  Read-only on data/.
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[3])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(_PA.R(""))
HERE = Path(__file__).resolve().parents[1]
FIPS = {"CA": 6, "OH": 39, "TX": 48}
SEED = 20261124


def make(st):
    sd = ROOT / f"data/pums_states/{st}"
    prod = pd.read_csv(sd / "testdata_pers.csv")
    cols = list(prod.columns); n_test = len(prod)
    U = pd.read_csv(sd / "historical" / f"person_level_{st}_2013.csv", dtype={"Person_ID": str})
    info = {"state": st, "seed_sequence": [SEED, FIPS[st]], "n_test": n_test, "n_state_pums_persons": len(U)}
    # (1) source table == full raw state PUMS person sample
    raw = pd.read_csv(ROOT / f"data/pums13/person_2013_{st}_full.csv", usecols=["SERIALNO", "SPORDER", "PWGTP"],
                      dtype={"SERIALNO": str, "SPORDER": str})
    raw_id = raw.SERIALNO.str.strip() + "-" + raw.SPORDER.astype(int).astype(str)
    info["raw_pums_rows"] = len(raw)
    info["source_ids_unique"] = bool(U.Person_ID.is_unique)
    info["source_id_set_equals_raw"] = bool(set(U.Person_ID) == set(raw_id))
    rw = pd.Series(raw.PWGTP.to_numpy(), index=raw_id)
    info["pwgtp_equals_raw_n_mismatch"] = int((rw.loc[U.Person_ID].to_numpy() != U.PWGTP.to_numpy()).sum())
    # (2) coverage of the production resample's unique source persons
    R = pd.read_csv(sd / f"person_level_2013_{st}.csv", dtype={"Source_Person_ID": str})
    uniq = R.Source_Person_ID.unique()
    info["prod_resample_rows"] = len(R)
    info["prod_resample_unique_source_persons"] = len(uniq)
    info["prod_resample_coverage_of_pums_persons"] = len(uniq) / len(U)
    info["prod_resample_coverage_of_pums_weight"] = float(U.set_index("Person_ID").PWGTP.loc[uniq].sum() / U.PWGTP.sum())
    Rd = R.drop_duplicates("Source_Person_ID").set_index("Source_Person_ID")
    Ui = U.set_index("Person_ID").loc[Rd.index]
    common = [c for c in cols if c != "Pid"]
    info["attr_mismatch_vs_prod_resample"] = int(sum((Rd[c].astype(str).to_numpy() != Ui[c].astype(str).to_numpy()).sum()
                                                    for c in common))
    # (3) fresh draw
    w = U.PWGTP.to_numpy(float); assert (w > 0).all()
    rng = np.random.default_rng(np.random.SeedSequence([SEED, FIPS[st]]))
    pos = rng.choice(len(U), size=n_test, replace=True, p=w / w.sum())
    fresh = U.iloc[pos].reset_index(drop=True)
    side = pd.DataFrame({"Pid": np.arange(n_test), "Source_Person_ID": fresh.Person_ID.to_numpy(),
                         "PWGTP": fresh.PWGTP.to_numpy()})
    fresh.insert(0, "Pid", np.arange(n_test))
    fresh = fresh[cols]
    for c in cols:
        if str(fresh[c].dtype) != str(prod[c].dtype): fresh[c] = fresh[c].astype(prod[c].dtype)
    info["dtype_equal"] = bool((fresh.dtypes.astype(str).values == prod.dtypes.astype(str).values).all())
    info["new_categories_vs_prod_test"] = {c: sorted(set(fresh[c].astype(str)) - set(prod[c].astype(str)))
                                           for c in common if prod[c].dtype == object}
    info["missing_categories_vs_prod_test"] = {c: sorted(set(prod[c].astype(str)) - set(fresh[c].astype(str)))
                                               for c in common if prod[c].dtype == object}
    info["n_unique_source_persons_fresh"] = int(side.Source_Person_ID.nunique())
    od = HERE / "refs" / st; od.mkdir(parents=True, exist_ok=True)
    fresh.to_csv(od / "testdata_fresh.csv", index=False)
    side.to_csv(od / "testdata_fresh_source_ids.csv", index=False)
    chk = pd.read_csv(od / "testdata_fresh.csv")
    info["reread_columns_equal"] = list(chk.columns) == cols
    info["reread_dtype_equal"] = bool((chk.dtypes.astype(str).values == prod.dtypes.astype(str).values).all())
    (od / "fresh_info.json").write_text(json.dumps(info, indent=1))
    return info


if __name__ == "__main__":
    for st in FIPS:
        print(json.dumps(make(st), indent=1), flush=True)
