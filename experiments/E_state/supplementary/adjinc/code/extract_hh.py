"""Extract household (SERIALNO, ADJINC, HINCP) per state-year from raw PUMS into
raw/hh_<ST>_<Y>.parquet.  2009-2012: Census per-state 1-year housing files
(downloaded to raw/).  2013: data/pums13/housing_2013_<ST>_full.csv.
2018/2021/2024: national psam_hus{a,b}.csv in data/pums<yy>/ (read-only)."""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[5])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import zipfile, io, sys
import pandas as pd
from pathlib import Path
ROOT = Path(_PA.R("")); HERE = Path(__file__).resolve().parents[1]; RAW = HERE / "raw"
FIPS = {"CA": 6, "OH": 39, "TX": 48}
cols = ["SERIALNO", "ADJINC", "HINCP"]
def save(df, st, y):
    df = df[cols].copy(); df["SERIALNO"] = df["SERIALNO"].astype(str)
    df.to_parquet(RAW / f"hh_{st}_{y}.parquet", index=False)
    print(st, y, len(df), "ADJINC unique:", sorted(df.ADJINC.dropna().unique())[:5], flush=True)
for y in [2009, 2010, 2011, 2012]:
    for st in FIPS:
        z = zipfile.ZipFile(RAW / f"h{st.lower()}_{y}.zip")
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        save(pd.read_csv(z.open(name), usecols=cols), st, y)
for st in FIPS:
    save(pd.read_csv(ROOT / f"data/pums13/housing_2013_{st}_full.csv", usecols=cols), st, 2013)
for y in [2018, 2021, 2024]:
    stcol = "STATE" if y >= 2023 else "ST"
    parts = []
    for f in ["psam_husa.csv", "psam_husb.csv"]:
        for ch in pd.read_csv(ROOT / f"data/pums{str(y)[2:]}/{f}", usecols=cols + [stcol], chunksize=500000):
            parts.append(ch[ch[stcol].isin(FIPS.values())])
    d = pd.concat(parts)
    for st, f in FIPS.items():
        save(d[d[stcol] == f], st, y)
