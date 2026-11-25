import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# =============================================================================
# Income in constant reference-year dollars (changed 2026-11-24, FIX 2 "real13n")
# -----------------------------------------------------------------------------
# HINCP / PINCP are reported in survey-year PUMS dollars that must be scaled by the
# PUMS adjustment factor ADJINC (stored as an integer; ADJINC/1e6 is the factor, see the
# PUMS data dictionary). Before binning we convert income of survey year Y to constant
# dollars of the reference year 2013 (the base year of the generative priors):
#
#     x_real = x_nominal * (ADJINC_Y / ADJINC_2013) * (CPI_2013 / CPI_Y)
#
# where ADJINC_Y is the ADJINC value of the survey-year file (read from the data),
# ADJINC_2013 = 1007549 is the ADJINC value of the ACS 2013 1-year PUMS file, and CPI_Y is
# the CPI-U annual average (U.S. city average, all items, BLS series CUUR0000SA0,
# 1982-84=100). Both 1e6 divisors cancel in the ratio. Because the reference factor is the
# 2013 file's own ADJINC, the factor for the 2013 file is exactly 1.0 and the 2013 labels
# (training data and priors) are unchanged by construction. The bin edges are fixed in
# these constant dollars (categorize_income). The bins are therefore on the scale of the
# 2013 PUMS file, which is within 0.75% of 2013 calendar-year dollars.
# Previous behavior (binning nominal income) is reproducible with --nominal-income.
# =============================================================================
INCOME_REFERENCE_YEAR = 2013
ADJINC_REFERENCE = 1007549  # ADJINC of the ACS 2013 1-year PUMS file (ss13pus*/ss13hus*)
CPI_U_ANNUAL = {
    2009: 214.537,
    2010: 218.056,
    2011: 224.939,
    2012: 229.594,
    2013: 232.957,
    2018: 251.107,
    2021: 270.970,
    2024: 313.689,
}
# ADJINC values of the 1-year PUMS files used in this project (documentation and check only;
# the conversion reads ADJINC from the data).
ADJINC_KNOWN = {
    2009: 999480, 2010: 1007624, 2011: 1018237, 2012: 1010207, 2013: 1007549,
    2018: 1013097, 2021: 1029928, 2024: 1015250,
}


def income_to_reference_dollars(income: pd.Series, adjinc: pd.Series, survey_year: int) -> pd.Series:
    """Convert nominal PUMS income of `survey_year` to constant INCOME_REFERENCE_YEAR dollars.

    x_real = x_nominal * (ADJINC_Y / ADJINC_REFERENCE) * (CPI_ref / CPI_Y). NaN income stays NaN
    (categorize_income maps it to No_Income, as before).
    """
    if survey_year not in CPI_U_ANNUAL:
        raise ValueError(f"No CPI-U annual average for survey year {survey_year}; add it to CPI_U_ANNUAL.")
    adj = pd.to_numeric(adjinc, errors="coerce")
    if adj.isna().any():
        raise ValueError("ADJINC missing for some records; cannot convert income to constant dollars.")
    known = ADJINC_KNOWN.get(survey_year)
    uniq = sorted(adj.unique())
    if known is not None and uniq != [known]:
        print(f"Warning: ADJINC values {uniq[:5]} differ from the documented {known} for {survey_year}.")
    # Same operation order as the verified shadow relabel (relabel.py, 'real13n').
    factor = adj / ADJINC_REFERENCE * CPI_U_ANNUAL[INCOME_REFERENCE_YEAR] / CPI_U_ANNUAL[survey_year]
    return pd.to_numeric(income, errors="coerce") * factor


def categorize_income(series: pd.Series) -> pd.Series:
    """Bin income (already in the intended dollar unit) into six classes. NaN -> 0 (No_Income)."""
    x = pd.to_numeric(series, errors="coerce").fillna(0)

    out = pd.Series("Unknown", index=series.index)

    out[x < 0] = "Loss_or_Negative"
    out[x == 0] = "No_Income"
    out[(x > 0) & (x <= 25000)] = "Low"
    out[(x > 25000) & (x <= 50000)] = "Lower_Middle"
    out[(x > 50000) & (x <= 100000)] = "Upper_Middle"
    out[x > 100000] = "High"

    return out


def add_income_categories(df: pd.DataFrame, survey_year: int | None = None) -> pd.DataFrame:
    """Add Personal_Income_Class / Household_Income_Class.

    survey_year=None bins nominal income (pre-2026-11-24 behavior, --nominal-income).
    Otherwise income is converted to constant INCOME_REFERENCE_YEAR dollars with the
    ADJINC column before binning (see the constants block above).
    """
    df = df.copy()

    def _prep(col):
        if survey_year is None:
            return df[col]
        if "ADJINC" not in df.columns:
            raise ValueError("ADJINC column required for constant-dollar income classes.")
        return income_to_reference_dollars(df[col], df["ADJINC"], survey_year)

    if "Personal_Income" in df.columns:
        df["Personal_Income_Class"] = categorize_income(_prep("Personal_Income"))

    if "Household_Income" in df.columns:
        df["Household_Income_Class"] = categorize_income(_prep("Household_Income"))

    return df


def infer_survey_year(df_person: pd.DataFrame) -> int | None:
    """Infer the ACS survey year from SERIALNO (format 'YYYYGQ...'/'YYYYHU...' since ACS 2017).
    Returns None for older files whose SERIALNO is purely numeric."""
    if "SERIALNO" not in df_person.columns:
        return None
    pref = df_person["SERIALNO"].astype(str).str[:4]
    if not pref.str.fullmatch(r"\d{4}").all() or pref.nunique() != 1:
        return None
    s = df_person["SERIALNO"].astype(str).str[4:6]
    if not s.isin(["GQ", "HU"]).all():
        return None
    return int(pref.iloc[0])

def check_nan_columns(df: pd.DataFrame, sort_by_missing: bool = True):
    """
    Print columns containing NaN values with counts and percentages.
    """

    total_rows = len(df)

    nan_counts = df.isna().sum()
    nan_percent = (nan_counts / total_rows) * 100

    summary = pd.DataFrame({
        "missing_count": nan_counts,
        "missing_percent": nan_percent
    })

    # keep only the columns that have missing values
    summary = summary[summary["missing_count"] > 0]

    if sort_by_missing:
        summary = summary.sort_values(by="missing_percent", ascending=False)

    print(f"\nTotal rows: {total_rows}")
    print(f"Columns with NaN: {len(summary)}\n")

    return summary

def drop_low_semantic_commute_vars(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    drop_cols = [
        "Commute_Travel_Time_Minutes",
        "Commute_Arrival_Time_Bin"
    ]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df

def merge_pums_household_person(
    df_housing: pd.DataFrame,
    df_person: pd.DataFrame,
    *,
    region_category_name: str = "Region_Category",
    keep_ids: bool = True,
) -> pd.DataFrame:
    """
    Merge already-concatenated PUMS housing df (e.g., [ss13husa + ss13husb]) and
    person df (e.g., [ss13pusa + ss13pusb]) into a person-level table (one row per person).

    Join keys (from PUMS):
      - SERIALNO: household/housing unit identifier (exists in both)
      - SPORDER: person order within household (exists in person)

    Geography requirement:
      - Keep ONLY a 4-class region category derived from REGION.
      - Drop the original REGION (and any other detailed geo fields are simply not selected).

    No value recoding is performed (we keep original codes), except creating the
    readable 4-class Region_Category label.
    """

    # ---- 1) Minimal column selection ----
    # Household-level columns we want to bring to each person row
    # NOTE: ACS renamed ST -> STATE in PUMS releases >=2023; include both so the
    # script works for ACS <=2022 (ST) and ACS >=2023 (STATE).
    h_keep = [
        c for c in ["SERIALNO", "REGION", "ST", "STATE", "HHT", "NP", "NOC", "VEH", "HINCP", "WGTP"]
        if c in df_housing.columns
    ]
    # Person-level columns we want
    # NOTE: ACS replaced JWTR by JWTRNS starting in 2019 (codes 03-05 changed meaning; see
    # map_commute_mode). Include both so the script works for ACS <=2018 and >=2019.
    # ADJINC (income adjustment factor) is kept for the constant-dollar income classes and
    # dropped before output.
    if "JWTR" in df_person.columns and "JWTRNS" in df_person.columns:
        raise ValueError("Person file has both JWTR and JWTRNS; expected exactly one.")
    p_keep = [
        c for c in ["SERIALNO", "SPORDER", "SEX", "AGEP", "MAR", "ESR", "SCH", "SCHL",
                    "JWTR", "JWTRNS", "JWRIP", "JWMNP", "JWAP", "PINCP", "PWGTP", "ADJINC"]
        if c in df_person.columns
    ]

    h = df_housing[h_keep].copy()
    p = df_person[p_keep].copy()

    # ---- 2) Build 4-class Region_Category from REGION, then drop REGION ----
    region_map = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}
    if "REGION" in h.columns:
        h[region_category_name] = pd.to_numeric(h["REGION"], errors="coerce").map(region_map).astype("string")
        h = h.drop(columns=["REGION"])
    else:
        h[region_category_name] = pd.Series([pd.NA] * len(h), index=h.index, dtype="string")

    # ---- 3) Rename to readable style (no transformations) ----
    rename_h = {
        "ST":    "State_Code",   # ACS <=2022
        "STATE": "State_Code",   # ACS >=2023 (Census renamed the column; same FIPS codes)
        "HHT": "Household_Type",
        "NP": "Household_Size",
        "NOC": "Number_of_children",
        "VEH": "Number_of_cars_of_household",
        "HINCP": "Household_Income",

    }
    h = h.rename(columns={k: v for k, v in rename_h.items() if k in h.columns})

    # Prefer SCH over SCHL for "Studenthood_status" if both exist
    if "SCH" in p.columns:
        p["Studenthood_status"] = p["SCH"]
    elif "SCHL" in p.columns:
        p["Studenthood_status"] = p["SCHL"]
    else:
        p["Studenthood_status"] = pd.NA

    rename_p = {
        "SEX": "Gender",
        "AGEP": "Age",
        "MAR": "Marital_status",
        "ESR": "Employment_status",
        "JWTR": "Commute_Mode",     # ACS <=2018
        "JWTRNS": "Commute_Mode",   # ACS >=2019: DIFFERENT codes 03-05; map_commute_mode(source_var="JWTRNS")
        "JWRIP": "Commute_Vehicle_Occupancy",
        "JWMNP": "Commute_Travel_Time_Minutes",
        "JWAP": "Commute_Arrival_Time_Bin",
        "PINCP": "Personal_Income",
    }
    p = p.rename(columns={k: v for k, v in rename_p.items() if k in p.columns})

    # Drop raw SCH/SCHL after consolidating (optional tidy)
    for c in ["SCH", "SCHL"]:
        if c in p.columns:
            p = p.drop(columns=[c])

    # ---- 4) Merge: person m:1 household ----
    # validate="m:1" ensures many persons map to one household SERIALNO
    df = p.merge(h, on="SERIALNO", how="left", validate="m:1")

    # ---- 5) Create person-level identifier ----
    if "SPORDER" in df.columns:
        df["Person_ID"] = df["SERIALNO"].astype("string") + "-" + df["SPORDER"].astype("string")
    else:
        df["Person_ID"] = df["SERIALNO"].astype("string")

    # ---- 6) Optional: drop ids ----
    if not keep_ids:
        drop_cols = [c for c in ["SERIALNO", "SPORDER"] if c in df.columns]
        df = df.drop(columns=drop_cols)

    # ---- 7) Column order ----
    preferred = [
        "Person_ID",
        region_category_name,
        "Gender", "Age", "Marital_status", "Employment_status", "Studenthood_status",
        "Household_Type", "Household_Size", "Number_of_children", "Number_of_cars_of_household", "Household_Income",
        "Personal_Income",
        "Commute_Mode", "Commute_Vehicle_Occupancy", "Commute_Travel_Time_Minutes", "Commute_Arrival_Time_Bin", "Departure_Travel_Time_Bin",
        "PWGTP", "WGTP",
        "SERIALNO", "SPORDER",
    ]
    preferred = [c for c in preferred if c in df.columns]
    df = df[preferred + [c for c in df.columns if c not in preferred]]

    return df

def map_jwap_continuous(
    df: pd.DataFrame,
    src_col: str = "JWAP",
    out_col: str = "Arrival_Time_Minutes",
    fill_na_to_zero: bool = True,
    as_int: bool = True
) -> pd.DataFrame:
    """
    Map JWAP (arrival time code) to continuous minutes-from-midnight.

    Spec:
      - 'bbb' or NaN -> 0
      - 001..285 -> (code - 1) * 5 minutes
      - Output range: 0–1435
    """
    df = df.copy()
    if src_col not in df.columns:
        return df

    s = df[src_col].astype(str).str.strip()

    # normalize N/A tokens
    s = s.replace({"bbb": np.nan, "nan": np.nan, "<NA>": np.nan, "None": np.nan, "": np.nan})

    # convert to numeric
    code = pd.to_numeric(s, errors="coerce")

    if fill_na_to_zero:
        code = code.fillna(0)

    # convert to minutes
    minutes = (code - 1) * 5

    # for code == 0 (originally NA), force to 0
    minutes = minutes.where(code > 0, 0)

    if as_int:
        minutes = minutes.round().astype(int)

    df[out_col] = minutes

    return df

def semantic_fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- Commute ----
    # if "Commute_Mode" in df.columns:
    #     df["Commute_Mode"] = df["Commute_Mode"].fillna("No_Commute")

    # if "Commute_Travel_Time_Minutes" in df.columns:
    #     df["Commute_Travel_Time_Minutes"] = df["Commute_Travel_Time_Minutes"].fillna(0)

    # if "Commute_Arrival_Time_Bin" in df.columns:
    #     df["Commute_Arrival_Time_Bin"] = df["Commute_Arrival_Time_Bin"].fillna(-1)

    if "Commute_Vehicle_Occupancy" in df.columns:
        df["Commute_Vehicle_Occupancy"] = df["Commute_Vehicle_Occupancy"].fillna(0)

    # ---- Employment ----
    # if "Employment_status" in df.columns:
    #     df["Employment_status"] = df["Employment_status"].fillna("Not_in_universe")

    # if "Personal_Income" in df.columns:
    #     df["Personal_Income"] = df["Personal_Income"].fillna(0)

    # ---- Household ----
    # if "Household_Income" in df.columns:
    #     df["Household_Income"] = df["Household_Income"].fillna(0)

    # if "Household_Type" in df.columns:
    #     df["Household_Type"] = df["Household_Type"].fillna("Group_Quarter")

    if "Number_of_cars_of_household" in df.columns:
        df["Number_of_cars_of_household"] = df["Number_of_cars_of_household"].fillna(0)

    if "Number_of_children" in df.columns:
        df["Number_of_children"] = df["Number_of_children"].fillna(0)

    # ---- Student ----
    # if "Studenthood_status" in df.columns:
    #     df["Studenthood_status"] = df["Studenthood_status"].fillna("Not_in_universe")

    return df

# JWTR (ACS <=2018) code -> label. These labels define the 13 Commute_Mode categories.
JWTR_LABELS = {
    "bb": "No_Commute",
    "01": "Car_truck_or_van",
    "02": "Bus_or_trolley_bus",
    "03": "Streetcar_or_trolley_car",
    "04": "Subway_or_elevated",
    "05": "Railroad",
    "06": "Ferryboat",
    "07": "Taxicab",
    "08": "Motorcycle",
    "09": "Bicycle",
    "10": "Walked",
    "11": "Worked_at_home",
    "12": "Other_method",
}
# JWTRNS (ACS >=2019) code -> JWTR label, by the Census comparability crosswalk
# (ACS user note 2020-03, "Changes to the Means of Transportation Question"):
#   02 Bus                                      -> Bus_or_trolley_bus
#   03 Subway or elevated rail                  -> Subway_or_elevated
#   04 Long-distance train or commuter rail     -> Railroad
#   05 Light rail, streetcar, or trolley        -> Streetcar_or_trolley_car
#   11 Worked from home                         -> Worked_at_home
# Other codes have the same meaning as in JWTR. (Before 2026-11-24 JWTRNS codes were read
# with the JWTR labels, which mislabeled codes 03-05.)
JWTRNS_LABELS = dict(JWTR_LABELS)
JWTRNS_LABELS.update({
    "02": "Bus_or_trolley_bus",
    "03": "Subway_or_elevated",
    "04": "Railroad",
    "05": "Streetcar_or_trolley_car",
    "11": "Worked_at_home",
})


def map_commute_mode(df: pd.DataFrame, source_var: str = "JWTR") -> pd.DataFrame:
    """Map raw commute codes in Commute_Mode to labels. source_var is the PUMS variable the
    codes come from: 'JWTR' (ACS <=2018) or 'JWTRNS' (ACS >=2019)."""
    df = df.copy()

    if "Commute_Mode" not in df.columns:
        return df

    if source_var not in ("JWTR", "JWTRNS"):
        raise ValueError(f"Unknown commute source variable {source_var!r}.")
    mode_map = JWTRNS_LABELS if source_var == "JWTRNS" else JWTR_LABELS

    mode = df["Commute_Mode"].astype("Int64").astype(str).str.zfill(2)

    df["Commute_Mode"] = mode.map(mode_map)

    # safety net in case the column was already filled with No_Commute earlier
    df["Commute_Mode"] = df["Commute_Mode"].fillna("No_Commute")

    return df

def map_employment_status(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Employment_status" not in df.columns:
        return df

    status = df["Employment_status"].astype("Int64").astype(str)

    status_map = {
        "b": "Under_16",
        "1": "Civilian_Employed_At_Work",
        "2": "Civilian_Employed_Not_At_Work",
        "3": "Unemployed",
        "4": "Armed_Forces_At_Work",
        "5": "Armed_Forces_Not_At_Work",
        "6": "Not_In_Labor_Force",
    }

    df["Employment_status"] = status.map(status_map)

    # guard against NaN
    df["Employment_status"] = df["Employment_status"].fillna("Under_16")

    return df


def map_studenthood_status_full(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Studenthood_status" not in df.columns:
        return df

    status = df["Studenthood_status"].astype("Int64").astype(str)

    status_map = {
        "b": "Under_3",
        "1": "Not_Enrolled",
        "2": "Public_School_or_College",
        "3": "Private_or_Home_School",
    }

    df["Studenthood_status"] = status.map(status_map)
    df["Studenthood_status"] = df["Studenthood_status"].fillna("Under_3")

    return df

def map_household_type_full(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    hht_map = {
        "b": "Group_Quarter",
        "1": "Married_Couple",
        "2": "Male_Single_Parent",
        "3": "Female_Single_Parent",
        "4": "Male_Living_Alone",
        "5": "Male_Nonfamily_Shared",
        "6": "Female_Living_Alone",
        "7": "Female_Nonfamily_Shared",
    }

    df["Household_Type"] = df["Household_Type"].astype("Int64").astype(str).map(hht_map)
    df["Household_Type"] = df["Household_Type"].fillna("Group_Quarter")

    return df

def map_marital_status(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Marital_status" not in df.columns:
        return df

    mar = df["Marital_status"].astype("Int64").astype(str)

    mar_map = {
        "1": "Married",
        "2": "Widowed",
        "3": "Divorced",
        "4": "Separated",
        "5": "Never_Married",
    }

    df["Marital_status"] = mar.map(mar_map)

    df["Marital_status"] = df["Marital_status"].fillna("Never_Married")

    return df

def map_jwmnp_continuous(
    df: pd.DataFrame,
    src_col: str = "Commute_Travel_Time_Minutes",
    out_col: str = "Commute_Travel_Time_Minutes",
    fill_na_to_zero: bool = True,
    clip_max: int = 200,
    as_int: bool = True,
) -> pd.DataFrame:
    """
    Map JWMNP (travel time to work) to a continuous numeric minutes variable.

    Spec:
      - 'bbb' (N/A) -> 0
      - NaN -> 0 (if fill_na_to_zero=True)
      - 001..200 -> numeric minutes (top-coded at 200)
      - optionally clip to [0, 200]
      - optionally cast to int (default) for clean minutes

    Works whether the source is float (e.g., 23.0) or string (e.g., '001', 'bbb').
    """
    df = df.copy()
    if src_col not in df.columns:
        return df

    s = df[src_col]

    # 1) Turn into string for safe token handling (bbb, etc.)
    s_str = s.astype(str).str.strip()

    # normalize common missing tokens
    # (pandas may stringify NaN as 'nan', and nullable ints as '<NA>')
    s_str = s_str.replace({"bbb": np.nan, "nan": np.nan, "<NA>": np.nan, "None": np.nan, "": np.nan})

    # 2) Convert to numeric (float), then fill/clip
    x = pd.to_numeric(s_str, errors="coerce")

    if fill_na_to_zero:
        x = x.fillna(0)

    # 3) clip range: minutes should be within [0, 200]
    if clip_max is not None:
        x = x.clip(lower=0, upper=clip_max)

    # 4) optionally cast to int minutes
    if as_int:
        # if you want to preserve NA instead of 0, set fill_na_to_zero=False and use Int64
        x = x.round().astype(int)

    df[out_col] = x
    return df

def map_sex(df: pd.DataFrame,
            src_col: str = "SEX",
            out_col: str = "Gender") -> pd.DataFrame:
    df = df.copy()

    if src_col not in df.columns:
        return df

    s = pd.to_numeric(df[src_col], errors="coerce")

    sex_map = {
        1: "Male",
        2: "Female"
    }

    df[out_col] = s.map(sex_map)

    # in case of outliers or missing values
    df[out_col] = df[out_col].fillna("Unknown")

    return df

def ensure_integer_columns(df: pd.DataFrame,
                           cols: list[str],
                           fill_na: int | None = None,
                           clip_min: int | None = 0) -> pd.DataFrame:
    """
    Ensure selected columns are clean integer type.

    Parameters
    ----------
    cols : list of column names
    fill_na : value to fill NaN (if None, keep nullable Int64)
    clip_min : lower bound (default 0)

    Returns
    -------
    DataFrame with cleaned integer columns.
    """
    df = df.copy()

    for col in cols:
        if col not in df.columns:
            continue

        # force numeric
        s = pd.to_numeric(df[col], errors="coerce")

        # optional: drop negative values
        if clip_min is not None:
            s = s.clip(lower=clip_min)

        # NaN handling
        if fill_na is not None:
            s = s.fillna(fill_na).astype(int)
        else:
            s = s.astype("Int64")  # pandas nullable int

        df[col] = s

    return df



def weighted_resample_by_pwgtp(
    df: pd.DataFrame,
    *,
    weight_col: str = "PWGTP",
    n_samples: int | None = None,
    random_state: int = 42,
    replace: bool = True,
    drop_weight_col: bool = False,
) -> pd.DataFrame:
    """
    Resample a person-level PUMS dataframe with probability proportional to PWGTP.

    Use this after person + housing merge and attribute recoding, because the row unit
    remains one person. Household attributes are attached to each person row, so PWGTP
    is the correct weight for person-level training data and person-level scenario
    marginals. WGTP is kept only for diagnostics and should not be used for this
    person-level resampling.
    """
    if weight_col not in df.columns:
        raise ValueError(
            f"{weight_col!r} not found. Make sure PWGTP is included when reading the person file."
        )

    work = df.copy()
    work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce")
    work = work[work[weight_col].notna() & (work[weight_col] > 0)].copy()

    if len(work) == 0:
        raise ValueError(f"No valid positive values found in {weight_col!r}.")

    if n_samples is None:
        n_samples = len(work)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    probs = work[weight_col].to_numpy(dtype=float)
    probs = probs / probs.sum()

    rng = np.random.default_rng(random_state)
    sampled_pos = rng.choice(
        np.arange(len(work)),
        size=n_samples,
        replace=replace,
        p=probs,
    )

    out = work.iloc[sampled_pos].reset_index(drop=True)

    # Person_ID is no longer unique after sampling with replacement. Keep the original
    # PUMS person ID for traceability and create a unique row id for the sampled file.
    if "Person_ID" in out.columns:
        out = out.rename(columns={"Person_ID": "Source_Person_ID"})
    out.insert(0, "Sampled_Person_ID", np.arange(len(out), dtype=np.int64))

    if drop_weight_col:
        out = out.drop(columns=[weight_col], errors="ignore")

    return out


def read_concat_csv(paths: list[str]) -> pd.DataFrame:
    """Read and concatenate multiple CSV files."""
    frames = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        frames.append(pd.read_csv(p, low_memory=False))
    return pd.concat(frames, ignore_index=True)


def build_processed_person_level(
    housing_paths: list[str],
    person_paths: list[str],
    *,
    keep_ids: bool = True,
    survey_year: int | None = None,
    nominal_income: bool = False,
) -> pd.DataFrame:
    """Run the original PUMS processing pipeline and keep PWGTP for resampling.

    survey_year: ACS survey year of the input files (needed for constant-dollar income;
    inferred from SERIALNO for ACS >=2017 if None). nominal_income=True reproduces the
    pre-2026-11-24 nominal income classes.
    """
    housing = read_concat_csv(housing_paths)
    person = read_concat_csv(person_paths)

    inferred = infer_survey_year(person)
    if survey_year is None:
        survey_year = inferred
    elif inferred is not None and inferred != survey_year:
        raise ValueError(f"--survey-year {survey_year} contradicts SERIALNO year {inferred}.")
    if not nominal_income and survey_year is None:
        raise ValueError("Survey year unknown: pass --survey-year (or --nominal-income).")
    commute_var = "JWTRNS" if "JWTRNS" in person.columns else "JWTR"
    print(f"Survey year: {survey_year}; commute variable: {commute_var}; income: "
          f"{'nominal' if nominal_income else f'constant {INCOME_REFERENCE_YEAR} dollars'}")

    df_person_level = merge_pums_household_person(housing, person, keep_ids=keep_ids)
    del housing, person
    df_person_level = map_commute_mode(df_person_level, source_var=commute_var)
    df_person_level = map_employment_status(df_person_level)
    df_person_level = map_studenthood_status_full(df_person_level)
    df_person_level = map_household_type_full(df_person_level)
    df_person_level = map_marital_status(df_person_level)
    df_person_level = map_jwmnp_continuous(df_person_level)
    df_person_level = map_jwap_continuous(
        df_person_level,
        src_col="Commute_Arrival_Time_Bin",
        out_col="Commute_Arrival_Time_Minutes",
    )
    df_person_level = map_sex(df_person_level, src_col="Gender", out_col="Gender")
    df_person_level = add_income_categories(
        df_person_level, survey_year=None if nominal_income else survey_year
    )

    int_cols = [
        "Age",
        "Number_of_children",
        "Number_of_cars_of_household",
        "Household_Size",
        "State_Code",
    ]
    df_person_level = ensure_integer_columns(
        df_person_level,
        cols=int_cols,
        fill_na=0,
        clip_min=0,
    )

    # Drop raw variables not used by the compact training table. Keep PWGTP so the
    # resampling step can construct a population-weighted person-level dataset.
    df_person_level = df_person_level.drop(
        columns=[
            "Personal_Income",
            "Household_Income",
            "SERIALNO",
            "SPORDER",
            "Commute_Vehicle_Occupancy",
            "Commute_Arrival_Time_Bin",
            "ADJINC",
        ],
        errors="ignore",
    )

    df_person_level = semantic_fill_missing(df_person_level)
    return df_person_level


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Process ACS PUMS person + housing files into a compact person-level table. "
            "By default, the script resamples rows with probability proportional to PWGTP. "
            "Use --unweighted to only merge/process the raw PUMS files and keep PWGTP."
        )
    )
    parser.add_argument(
        "--housing-files",
        nargs="+",
        default=["data/hist_ref_data/pums2010/ss10husa.csv", "data/hist_ref_data/pums2010/ss10husb.csv"],
        help="Housing CSV files, e.g. psam_husa.csv psam_husb.csv.",
    )
    parser.add_argument(
        "--person-files",
        nargs="+",
        default=["data/hist_ref_data/pums2010/ss10pusa.csv", "data/hist_ref_data/pums2010/ss10pusb.csv"],
        help="Person CSV files, e.g. psam_pusa.csv psam_pusb.csv.",
    )
    parser.add_argument(
        "--output",
        default="data/hist_ref_data/pums2010/person_level_merged_compact_pwgtp_resampled_2010.csv",
        help=(
            "Output path. In default mode, this is the PWGTP-resampled compact CSV. "
            "With --unweighted, this is the processed non-resampled compact CSV that keeps PWGTP."
        ),
    )
    parser.add_argument(
        "--unweighted",
        action="store_true",
        help=(
            "Only merge/process the PUMS files and save the compact person-level table to --output. "
            "No PWGTP resampling is performed, and PWGTP is kept in the output."
        ),
    )
    parser.add_argument(
        "--save-unweighted",
        default=None,
        help=(
            "Optional path to save the processed but non-resampled person-level CSV before "
            "the default resampling step. Not needed when using --unweighted."
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of rows after resampling. Default: same as processed person-level row count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for weighted resampling.",
    )
    parser.add_argument(
        "--drop-pwgtp-after-resampling",
        action="store_true",
        help="Drop PWGTP from the final output after resampling. Default keeps it for diagnostics.",
    )
    parser.add_argument(
        "--drop-wgtp",
        action="store_true",
        help="Drop WGTP from output. Recommended for model training because the row unit is person.",
    )

    parser.add_argument(
        "--survey-year",
        type=int,
        default=None,
        help=(
            "ACS survey year of the input files, used to convert income to constant "
            f"{INCOME_REFERENCE_YEAR} dollars (CPI_U_ANNUAL). Inferred from SERIALNO for ACS >=2017; "
            "required for older files unless --nominal-income."
        ),
    )
    parser.add_argument(
        "--nominal-income",
        action="store_true",
        help="Bin nominal HINCP/PINCP without ADJINC/CPI conversion (behavior before 2026-11-24).",
    )

    args = parser.parse_args()

    df_person_level = build_processed_person_level(
        housing_paths=args.housing_files,
        person_paths=args.person_files,
        keep_ids=True,
        survey_year=args.survey_year,
        nominal_income=args.nominal_income,
    )

    print("Processed person-level table before resampling:")
    print(df_person_level.shape)
    print(check_nan_columns(df_person_level))

    if args.drop_wgtp:
        df_person_level = df_person_level.drop(columns=["WGTP"], errors="ignore")

    if args.unweighted:
        if args.drop_pwgtp_after_resampling:
            print("Warning: --drop-pwgtp-after-resampling is ignored in --unweighted mode; PWGTP is kept.")

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_person_level.to_csv(out_path, index=False)

        print("Unweighted processed person-level table:")
        print(df_person_level.shape)
        print(f"Saved non-resampled compact file with PWGTP to: {out_path}")
        return

    if args.save_unweighted is not None:
        unweighted_path = Path(args.save_unweighted)
        unweighted_path.parent.mkdir(parents=True, exist_ok=True)
        df_person_level.to_csv(unweighted_path, index=False)
        print(f"Saved unweighted processed file to: {unweighted_path}")

    df_resampled = weighted_resample_by_pwgtp(
        df_person_level,
        weight_col="PWGTP",
        n_samples=args.n_samples,
        random_state=args.seed,
        replace=True,
        drop_weight_col=args.drop_pwgtp_after_resampling,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_resampled.to_csv(out_path, index=False)

    print("Resampled person-level table:")
    print(df_resampled.shape)
    print(f"Saved PWGTP-resampled file to: {out_path}")



if __name__ == '__main__':
    main()
