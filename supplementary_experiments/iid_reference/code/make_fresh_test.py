"""Generate a fresh, independent base-year test reference of the SAME design as the production one.

Production design (reproduced bit-for-bit, see reproduce_resample.py):
  national 2013 ACS PUMS persons (ss13pusa+pusb merged with ss13husa+husb, processed by
  _pums_proc.build_processed_person_level) -> rng=np.random.default_rng(42);
  rng.choice(arange(N_nat), size=N_nat, replace=True, p=PWGTP/sum(PWGTP))  (person unit,
  independent draws, households NOT drawn jointly) -> slice State_Code -> random positional split
  train 40k / val 10k / test rest.

Conditional on the state's draw count n_s, the national multinomial restricted to the state is an
iid PWGTP-proportional sample of size n_s from the state's PUMS persons; a random positional split of
an iid sample yields independent iid sub-samples. Hence an equivalent fresh test is

  rng.choice(state_persons, size=n_test, replace=True, p=PWGTP_i / sum_state PWGTP)

with an RNG stream independent of seed 42. The train/val sets (and all trained models) stay unchanged.

Source table: data/pums_states/<ST>/historical/person_level_<ST>_2013.csv (== national processed
non-resampled 2013 table filtered on State_Code; verified in verify_design.py).

Output (under ../out/fresh_test/<ST>/): testdata_pers_fresh_s<seed>.csv with exactly the columns and
column order of the production testdata_pers.csv (Pid = 0..n_test-1), plus a sidecar
source_ids_fresh_s<seed>.csv (Pid, Source_Person_ID, PWGTP) for overlap diagnostics.

Usage: python make_fresh_test.py [--seeds 1 2 ...] [--states CA OH TX]
Read-only on data/.
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[3])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(_PA.R(""))
OUT = Path(_PA.R("/supplementary_experiments/iid_reference/out/fresh_test"))
FIPS = {"CA": 6, "OH": 39, "TX": 48}
SEED_BASE = 2013_000_000  # fresh seed = SEED_BASE + 1000*fips + replicate  (never 42)


def make(st: str, rep: int, check: bool = True) -> dict:
    sd = ROOT / f"data/pums_states/{st}"
    prod_test = pd.read_csv(sd / "testdata_pers.csv")
    cols = list(prod_test.columns)  # ['Pid', ...]
    n_test = len(prod_test)
    U = pd.read_csv(sd / "historical" / f"person_level_{st}_2013.csv", dtype={"Person_ID": str})
    w = U["PWGTP"].to_numpy(float)
    assert (w > 0).all()
    seed = SEED_BASE + 1000 * FIPS[st] + rep
    rng = np.random.default_rng(seed)
    pos = rng.choice(len(U), size=n_test, replace=True, p=w / w.sum())
    fresh = U.iloc[pos].reset_index(drop=True)
    side = pd.DataFrame({"Pid": np.arange(n_test), "Source_Person_ID": fresh["Person_ID"].to_numpy(),
                         "PWGTP": fresh["PWGTP"].to_numpy()})
    fresh.insert(0, "Pid", np.arange(n_test))
    fresh = fresh[cols]
    info = {"state": st, "replicate": rep, "seed": seed, "n_test": n_test, "n_source_persons": len(U)}
    if check:
        # attribute values of the processed source table must equal those in the production
        # resampled file for the same source person (same processing code path)
        R = pd.read_csv(sd / f"person_level_2013_{st}.csv", dtype={"Source_Person_ID": str})
        Rd = R.drop_duplicates("Source_Person_ID").set_index("Source_Person_ID")
        Ui = U.set_index("Person_ID").loc[Rd.index]
        common = [c for c in cols if c != "Pid"]
        mism = {c: int((Rd[c].astype(str).to_numpy() != Ui[c].astype(str).to_numpy()).sum()) for c in common}
        info["attr_mismatch_vs_production_resample"] = mism
        # dtypes / categories identical to production test
        info["dtype_equal"] = bool((fresh.dtypes.astype(str).values == prod_test.dtypes.astype(str).values).all())
        info["new_categories_vs_prod_test"] = {
            c: sorted(set(fresh[c].astype(str)) - set(prod_test[c].astype(str)))[:5]
            for c in common if fresh[c].dtype == object}
    od = OUT / st
    od.mkdir(parents=True, exist_ok=True)
    fresh.to_csv(od / f"testdata_pers_fresh_s{rep}.csv", index=False)
    side.to_csv(od / f"source_ids_fresh_s{rep}.csv", index=False)
    (od / f"fresh_s{rep}_info.json").write_text(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--states", nargs="+", default=list(FIPS))
    a = ap.parse_args()
    for st in a.states:
        for r in a.seeds:
            print(json.dumps(make(st, r, check=(r == a.seeds[0])), indent=1))
