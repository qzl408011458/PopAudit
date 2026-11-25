"""Supplementary S4.1 (household / group-quarters paragraph): PWGTP-weighted share of persons living in
group quarters (Household_Type == Group_Quarter) and the GQ fraction of the zero-car person share, for the
nine scenario references.  Release helper derived from the authors' verification script (which also
compared against the pre-fix references); prints one line per state and year:
    <ST> <year> [('new', gq_share, gq_fraction_of_zero_car)]
Needs data/pums_states/<ST>/reference_<year>_<ST>.csv (processed PUMS, not redistributed).
usage: python experiments/E_state/supplementary/household/household_gq_share.py
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[4]))
import popaudit_paths as _PA  # noqa: E402

for st in ["CA", "TX", "OH"]:
    for y in (2018, 2021, 2024):
        out = []
        for tag, f in [("new", f"{_PA.DATA}/pums_states/{st}/reference_{y}_{st}.csv")]:
            d = pd.read_csv(f, usecols=["Household_Type", "Number_of_cars_of_household", "PWGTP"])
            w = d.PWGTP; gq = d.Household_Type.eq("Group_Quarter"); z = d.Number_of_cars_of_household.astype(int).eq(0)
            out.append((tag, round(w[gq].sum() / w.sum(), 4), round(w[gq & z].sum() / w[z].sum(), 4)))
        print(st, y, out)
