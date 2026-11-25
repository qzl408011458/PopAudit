"""JWTRNS light-rail assignment sensitivity on the CURRENT production setup (2026-11-24, after the
group-1 fix: constant-2013-dollar income, JWTRNS Census crosswalk, D on the category union).

Production references (data/pums_states/<st>/reference_<y>_<st>.csv, read-only) carry the Census
crosswalk for 2021/2024:  JWTRNS 02 -> Bus_or_trolley_bus, 03 -> Subway_or_elevated,
04 -> Railroad, 05 -> Streetcar_or_trolley_car, 11 -> Worked_at_home.  Verified 1:1 against the raw
psam files in verify_crosswalk.py.  Alternatives relabel only the code-05 records:
  lr2rail   : Streetcar_or_trolley_car -> Railroad
  lr2subway : Streetcar_or_trolley_car -> Subway_or_elevated
Under both alternatives the 2013 category Streetcar_or_trolley_car has no 2021/2024 reference mass.
Two conventions for that empty category:
  <v>      : as the production code treats an unobserved category: Layers 2/3 standardize only the
             reference-observed categories (commute |C_a| = 12 in L2/L3), the category enters D through
             its projected share (category union, mu_out).
  <v>_k13  : the category keeps a zero target in Layers 2/3 (|C_a| = 13, as in the 2013 coding and in
             the census mapping).  Implemented as in the pre-fix shadow: one copy of an existing record
             relabeled Streetcar_or_trolley_car with PWGTP = 1e-6 (target share ~1e-14).
Only copies under this directory are written; data/ is never modified.
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[4])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import pandas as pd
from pathlib import Path

ROOT = Path(_PA.R(""))
HERE = Path(__file__).resolve().parent
L05 = {"lr2rail": "Railroad", "lr2subway": "Subway_or_elevated"}
RAIL = ["Bus_or_trolley_bus", "Streetcar_or_trolley_car", "Subway_or_elevated", "Railroad"]
rows = []
for st in ("CA", "OH", "TX"):
    for y in (2021, 2024):
        df = pd.read_csv(ROOT / f"data/pums_states/{st}/reference_{y}_{st}.csv", low_memory=False)
        W = df.PWGTP.sum()
        for v, lab in L05.items():
            new = df.Commute_Mode.where(df.Commute_Mode != "Streetcar_or_trolley_car", lab)
            d1 = df.assign(Commute_Mode=new)
            for vv, dd in ((v, d1), (f"{v}_k13", None)):
                out = HERE / "refs" / vv / st
                out.mkdir(parents=True, exist_ok=True)
                if dd is None:
                    dummy = d1.iloc[[0]].copy()
                    dummy["Commute_Mode"] = "Streetcar_or_trolley_car"
                    dummy["PWGTP"] = 1e-6
                    dummy["Person_ID"] = "ZERO_TARGET_DUMMY"
                    dd = pd.concat([d1, dummy], ignore_index=True)
                dd.to_csv(out / f"reference_{y}_{st}.csv", index=False)
            for l in RAIL:
                rows.append(dict(variant=v, state=st, year=y, label=l,
                                 share_prod=df.PWGTP[df.Commute_Mode == l].sum() / W,
                                 share_alt=d1.PWGTP[d1.Commute_Mode == l].sum() / W))
        print(st, y, "done", flush=True)
T = pd.DataFrame(rows)
(HERE / "results").mkdir(exist_ok=True)
T.to_csv(HERE / "results/ref_target_shares.csv", index=False)
pd.set_option("display.width", 200)
print((T.pivot_table(index=["state", "year", "label"], columns="variant", values=["share_prod", "share_alt"]) * 100)
      .round(4).to_string())
