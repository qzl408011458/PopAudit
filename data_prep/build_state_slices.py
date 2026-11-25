"""State slices of the national compact PUMS tables (Section 5.2, Appendix B).

Recipe (identical to the authors' build_state_datasets.py / slice_states.py):
    df = pd.read_csv(national); sub = df[df["State_Code"] == fips].reset_index(drop=True); sub.to_csv(out, index=False)

  2013 base slice  data/pums/person_level_merged_compact_2013.csv (PWGTP resample, seed 42)
                   -> data/pums_states/<ST>/person_level_2013_<ST>.csv
  history          data/pums_historical/person_level_merged_compact_pwgtp_<Y>.csv, Y = 2009..2013
                   -> data/pums_states/<ST>/historical/person_level_<ST>_<Y>.csv
  references       data/pums{18,21,24}/person_level_merged_compact_pwgtp_<Y>.csv
                   -> data/pums_states/<ST>/reference_<Y>_<ST>.csv

usage:
    python data_prep/build_state_slices.py --write [--years base 2009 ... 2024]
    python data_prep/build_state_slices.py --check   # compare bytes with existing state files
"""
import argparse, hashlib, sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import popaudit_paths as _PA  # noqa: E402

DATA = Path(_PA.DATA)
STATES = {"CA": 6, "OH": 39, "TX": 48}
NATIONAL = {"base": DATA / "pums/person_level_merged_compact_2013.csv",
            "2018": DATA / "pums18/person_level_merged_compact_pwgtp_2018.csv",
            "2021": DATA / "pums21/person_level_merged_compact_pwgtp_2021.csv",
            "2024": DATA / "pums24/person_level_merged_compact_pwgtp_2024.csv"}
for y in (2009, 2010, 2011, 2012, 2013):
    NATIONAL[str(y)] = DATA / f"pums_historical/person_level_merged_compact_pwgtp_{y}.csv"


def out_path(st, key):
    d = DATA / "pums_states" / st
    if key == "base":
        return d / f"person_level_2013_{st}.csv"
    y = int(key)
    return d / f"reference_{y}_{st}.csv" if y >= 2014 else d / "historical" / f"person_level_{st}_{y}.csv"


def main():
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true"); g.add_argument("--write", action="store_true")
    ap.add_argument("--years", nargs="+", default=list(NATIONAL))
    a = ap.parse_args()
    for key in a.years:
        df = pd.read_csv(NATIONAL[key])
        for st, fips in STATES.items():
            sub = df[df["State_Code"] == fips].reset_index(drop=True)
            o = out_path(st, key)
            if a.check:
                buf = sub.to_csv(index=False).encode()
                same = o.is_file() and hashlib.md5(buf).hexdigest() == hashlib.md5(o.read_bytes()).hexdigest()
                print(f"CHECK {st} {key} rows={len(sub)} bytes_identical={same}", flush=True)
            else:
                o.parent.mkdir(parents=True, exist_ok=True)
                sub.to_csv(o, index=False)
                print(f"WROTE {o} rows={len(sub)}", flush=True)


if __name__ == "__main__":
    main()
