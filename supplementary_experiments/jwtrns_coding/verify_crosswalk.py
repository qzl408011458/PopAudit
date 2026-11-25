"""Read-only check that the CURRENT production references carry the Census crosswalk on JWTRNS
(2021/2024) and JWTR (2018), joined on SERIALNO-SPORDER with the raw psam person files; also the
PWGTP footprint of the relabeling (persons in JWTRNS codes 03-05, whose JWTR reading would put them
in another rail category) and the 2024 taxicab share (code 07, includes ride-hailing in 2024)."""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[2])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import pandas as pd
from pathlib import Path

ROOT = Path(_PA.R(""))
OUT = Path(__file__).resolve().parent / "results"
FIPS = {"CA": 6, "OH": 39, "TX": 48}
XWALK = {1: "Car_truck_or_van", 2: "Bus_or_trolley_bus", 3: "Subway_or_elevated", 4: "Railroad",
         5: "Streetcar_or_trolley_car", 6: "Ferryboat", 7: "Taxicab", 8: "Motorcycle", 9: "Bicycle",
         10: "Walked", 11: "Worked_at_home", 12: "Other_method", 0: "No_Commute"}
JWTR = dict(XWALK); JWTR.update({3: "Streetcar_or_trolley_car", 4: "Subway_or_elevated", 5: "Railroad"})
rows, fp = [], []
for y in (2018, 2021, 2024):
    var = "JWTR" if y == 2018 else "JWTRNS"
    stc = "STATE" if y >= 2023 else "ST"
    p = pd.concat([pd.read_csv(ROOT / f"data/pums{str(y)[2:]}/{f}", usecols=["SERIALNO", "SPORDER", stc, var],
                               dtype={"SERIALNO": str}) for f in ("psam_pusa.csv", "psam_pusb.csv")])
    p = p[p[stc].isin(FIPS.values())]
    p["Person_ID"] = p.SERIALNO + "-" + p.SPORDER.astype(str)
    for st, f in FIPS.items():
        ref = pd.read_csv(ROOT / f"data/pums_states/{st}/reference_{y}_{st}.csv",
                          usecols=["Person_ID", "Commute_Mode", "PWGTP"])
        m = ref.merge(p.loc[p[stc] == f, ["Person_ID", var]], on="Person_ID", how="left", validate="1:1")
        code = m[var].fillna(0).astype(int)
        mp = JWTR if y == 2018 else XWALK
        ok = (code.map(mp) == m.Commute_Mode)
        W = m.PWGTP.sum()
        rows.append(dict(state=st, year=y, n_ref=len(m), n_unmatched=int(m[var].isna().sum() - (m.Commute_Mode == "No_Commute").sum() * 0),
                         frac_records_on_map=float(ok.mean()), w_off_map=float(m.PWGTP[~ok].sum() / W)))
        fp.append(dict(state=st, year=y, var=var,
                       share_code03_05=float(m.PWGTP[code.isin([3, 4, 5])].sum() / W),
                       share_code05=float(m.PWGTP[code == 5].sum() / W),
                       share_taxi=float(m.PWGTP[code == 7].sum() / W)))
        print(y, st, rows[-1], fp[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT / "verify_crosswalk.csv", index=False)
pd.DataFrame(fp).to_csv(OUT / "footprint_taxi.csv", index=False)
print(pd.DataFrame(fp).assign(**{c: lambda d, c=c: (100 * d[c]).round(4) for c in ["share_code03_05", "share_code05", "share_taxi"]}).to_string())
