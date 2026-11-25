"""Relabel Household_Income_Class under alternative dollar definitions and write
shadow historical/reference CSVs (trimmed to the audited source columns + PWGTP).

Definitions (multiplier applied to raw HINCP before the _pums_proc.categorize_income bins
0 / (0,25k] / (25k,50k] / (50k,100k] / >100k, NaN->0):
  nom     : 1                                   (production)
  adj     : ADJINC_Y/1e6                        (survey-year dollars)
  real13  : ADJINC_Y/1e6 * CPI13/CPI_Y          (constant 2013 dollars, strict; changes 2013 too)
  real13n : ADJINC_Y/ADJINC_13 * CPI13/CPI_Y    (constant dollars on the 2013-file scale; 2013 unchanged)
CPI-U, U.S. city average, all items, annual average (BLS series CUUR0000SA0, 1982-84=100).
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[5])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import sys
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(_PA.R("")); HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); from _pums_proc import categorize_income
CPI = {2009: 214.537, 2010: 218.056, 2011: 224.939, 2012: 229.594, 2013: 232.957,
       2018: 251.107, 2021: 270.970, 2024: 313.689}
ADJ = {2009: 999480, 2010: 1007624, 2011: 1018237, 2012: 1010207, 2013: 1007549,
       2018: 1013097, 2021: 1029928, 2024: 1015250}
def factor(defn, y):
    return {"nom": 1.0, "adj": ADJ[y] / 1e6, "real13": ADJ[y] / 1e6 * CPI[2013] / CPI[y],
            "real13n": ADJ[y] / ADJ[2013] * CPI[2013] / CPI[y]}[defn]
DEFS = ["nom", "adj", "real13", "real13n"]
CLS = ["Loss_or_Negative", "No_Income", "Low", "Lower_Middle", "Upper_Middle", "High"]
KEEP = ["Person_ID", "Age", "Household_Size", "Number_of_children", "Number_of_cars_of_household",
        "Employment_status", "Commute_Mode", "PWGTP", "Household_Income_Class"]
def src(st, y):
    sd = ROOT / f"data/pums_states/{st}"
    return sd / (f"historical/person_level_{st}_{y}.csv" if y <= 2013 else f"reference_{y}_{st}.csv")
def main(write=True):
    rows, chk = [], []
    for st in ["CA", "OH", "TX"]:
        for y in [2009, 2010, 2011, 2012, 2013, 2018, 2021, 2024]:
            df = pd.read_csv(src(st, y), low_memory=False)
            hh = pd.read_parquet(HERE / f"raw/hh_{st}_{y}.parquet")
            ser = df.Person_ID.astype(str).str.rsplit("-", n=1).str[0]
            hinc = ser.map(hh.drop_duplicates("SERIALNO").set_index("SERIALNO").HINCP)
            unmatched = int(ser.isin(hh.SERIALNO).eq(False).sum())
            lab = {d: categorize_income(hinc * factor(d, y)) for d in DEFS}
            agree = float((lab["nom"].values == df.Household_Income_Class.values).mean())
            chk.append(dict(state=st, year=y, n=len(df), unmatched=unmatched, nominal_label_agreement=agree))
            w = df.PWGTP.to_numpy(float)
            for d in DEFS:
                s = pd.Series(w).groupby(lab[d].values).sum() / w.sum()
                rows.append(dict(state=st, year=y, defn=d, factor=factor(d, y), **{c: float(s.get(c, 0.0)) for c in CLS}))
            if write:
                for d in ["nom", "real13n", "real13"]:
                    o = df[KEEP].copy(); o["Household_Income_Class"] = lab[d].values
                    p = HERE / f"data/{d}/{st}" / src(st, y).relative_to(ROOT / f"data/pums_states/{st}")
                    p.parent.mkdir(parents=True, exist_ok=True); o.to_csv(p, index=False)
            print(chk[-1], flush=True)
    pd.DataFrame(chk).to_csv(HERE / "results/label_check.csv", index=False)
    pd.DataFrame(rows).to_csv(HERE / "results/income_shares_by_definition.csv", index=False)
if __name__ == "__main__":
    main()
