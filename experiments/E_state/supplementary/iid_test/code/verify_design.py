"""Design forensics step 2: empirical verification of the iid PWGTP-proportional
with-replacement person-level design, per state, plus overlap expectation.

Read-only on data/. Writes ../out/verify_design_<ST>.json and ../out/verify_design_summary.csv
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[5])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(_PA.R(""))
OUT = Path(_PA.R("/experiments/E_state/supplementary/iid_test/out"))
STATES = {"CA": 6, "OH": 39, "TX": 48}
N_MC = 20

nat = pd.read_csv(ROOT / "data/pums_historical/person_level_merged_compact_pwgtp_2013.csv",
                  usecols=["Person_ID", "PWGTP", "State_Code"], dtype={"Person_ID": str})
res = pd.read_csv(ROOT / "data/pums/person_level_merged_compact_2013.csv",
                  usecols=["Source_Person_ID", "State_Code"], dtype={"Source_Person_ID": str})
N_NAT = len(nat)
W_NAT = float(nat["PWGTP"].sum())

rows = []
for st, fips in STATES.items():
    out = {"state": st}
    sd = ROOT / f"data/pums_states/{st}"
    R = pd.read_csv(sd / f"person_level_2013_{st}.csv", dtype={"Source_Person_ID": str})
    # 1. state slice == national resample filtered on State_Code (order preserved)
    rs = res[res["State_Code"] == fips]["Source_Person_ID"].to_numpy()
    out["slice_equals_national_resample_filtered"] = bool(len(rs) == len(R) and (rs == R["Source_Person_ID"].to_numpy()).all())
    # historical per-state 2013 == national unweighted filtered
    H = pd.read_csv(sd / "historical" / f"person_level_{st}_2013.csv", usecols=["Person_ID", "PWGTP"], dtype={"Person_ID": str})
    U = nat[nat["State_Code"] == fips].reset_index(drop=True)
    out["historical_2013_equals_national_unweighted_filtered"] = bool(len(H) == len(U) and (H["Person_ID"].to_numpy() == U["Person_ID"].to_numpy()).all())
    # raw state PUMS person file count (extracted from ss13pus*)
    try:
        rawp = pd.read_csv(ROOT / f"data/pums13/person_2013_{st}_full.csv", usecols=["SERIALNO", "SPORDER", "PWGTP"])
        out["raw_person_rows"] = int(len(rawp)); out["raw_pwgtp_sum"] = float(rawp["PWGTP"].sum())
    except Exception as e:
        out["raw_person_rows"] = f"err {e}"

    w = U["PWGTP"].to_numpy(float)
    W_s = w.sum()
    n_s = len(R)
    out["n_raw_state_persons"] = int(len(U)); out["n_resampled_state_rows"] = int(n_s)
    out["state_total_expected"] = N_NAT * W_s / W_NAT
    out["state_total_sd"] = float(np.sqrt(N_NAT * (W_s / W_NAT) * (1 - W_s / W_NAT)))
    out["state_total_z"] = (n_s - out["state_total_expected"]) / out["state_total_sd"]

    # 2. copies per source person vs Binomial(N_nat, w_i/W_nat)
    cnt = R["Source_Person_ID"].value_counts()
    c = U["Person_ID"].map(cnt).fillna(0).to_numpy(float)
    p_i = w / W_NAT
    lam = N_NAT * p_i
    out["sum_copies_check"] = int(c.sum())
    shares_obs, shares_exp = {}, {}
    for k in [0, 1, 2, 3]:
        shares_obs[str(k)] = float((c == k).mean())
        shares_exp[str(k)] = float(stats.binom.pmf(k, N_NAT, p_i).mean())
    shares_obs["4+"] = float((c >= 4).mean()); shares_exp["4+"] = float(stats.binom.sf(3, N_NAT, p_i).mean())
    out["share_persons_with_k_copies_obs"] = shares_obs
    out["share_persons_with_k_copies_exp"] = shares_exp
    # chi-square goodness of fit on the pooled count table
    obs_ct = np.array([(c == k).sum() for k in range(4)] + [(c >= 4).sum()], float)
    exp_ct = np.array([stats.binom.pmf(k, N_NAT, p_i).sum() for k in range(4)] + [stats.binom.sf(3, N_NAT, p_i).sum()])
    chi2 = float(((obs_ct - exp_ct) ** 2 / exp_ct).sum())
    out["gof_chi2_df4"] = chi2; out["gof_p"] = float(stats.chi2.sf(chi2, 4))
    # regression c ~ a + b*lam
    b, a = np.polyfit(lam, c, 1)
    out["ols_copies_on_lambda"] = {"slope": float(b), "intercept": float(a)}
    out["dispersion_index"] = float(((c - lam) ** 2).sum() / (lam * (1 - p_i)).sum())
    # by PWGTP decile
    dec = pd.qcut(w, 10, labels=False, duplicates="drop")
    tab = pd.DataFrame({"dec": dec, "w": w, "c": c, "lam": lam}).groupby("dec").agg(
        w_mean=("w", "mean"), c_mean=("c", "mean"), lam_mean=("lam", "mean"), n=("c", "size"))
    out["by_pwgtp_decile"] = tab.round(4).reset_index().to_dict(orient="records")

    # 3. household jointness: within-household correlation of standardized residuals
    hh = U["Person_ID"].str.rsplit("-", n=1).str[0].to_numpy()
    z = (c - lam) / np.sqrt(lam)
    dfz = pd.DataFrame({"hh": hh, "z": z, "c": c})
    g = dfz.groupby("hh")
    first = g.nth(0).set_index("hh"); second = g.nth(1).set_index("hh")
    pair = first.join(second, lsuffix="_1", rsuffix="_2", how="inner")
    out["n_multiperson_households"] = int(len(pair))
    out["within_hh_corr_resid_person1_person2"] = float(np.corrcoef(pair["z_1"], pair["z_2"])[0, 1])
    out["within_hh_share_equal_copies_person1_person2_obs"] = float((pair["c_1"] == pair["c_2"]).mean())
    # expected share equal under independence: simulate one fresh national draw
    rng = np.random.default_rng(20261124)
    cs = rng.binomial(N_NAT, p_i)  # marginal of multinomial, adequate for independence benchmark
    dfs = pd.DataFrame({"hh": hh, "c": cs}).groupby("hh")
    f1 = dfs.nth(0).set_index("hh"); f2 = dfs.nth(1).set_index("hh")
    ps = f1.join(f2, lsuffix="_1", rsuffix="_2", how="inner")
    out["within_hh_share_equal_copies_indep_sim"] = float((ps["c_1"] == ps["c_2"]).mean())
    # share of households in which ALL members have identical copies (joint draw => 1.0)
    allsame = g["c"].agg(lambda s: s.nunique() == 1)
    sizes = g.size()
    out["share_multiperson_hh_all_members_same_copies_obs"] = float(allsame[sizes >= 2].mean())
    allsame_s = pd.DataFrame({"hh": hh, "c": cs}).groupby("hh")["c"].agg(lambda s: s.nunique() == 1)
    out["share_multiperson_hh_all_members_same_copies_indep_sim"] = float(allsame_s[sizes >= 2].mean())

    # 4. overlap: observed (actual split) vs iid expectation
    ids = pickle.load(open(sd / "dataset_ids.pkl", "rb"))
    tr = np.asarray(ids["train_idx"]); va = np.asarray(ids["val_idx"]); te = np.asarray(ids["test_idx"])
    src = R["Source_Person_ID"].to_numpy()
    hhR = R["Source_Person_ID"].str.rsplit("-", n=1).str[0].to_numpy()
    tr_set = set(src[tr]); trva_set = set(src[np.r_[tr, va]])
    tr_hh = set(hhR[tr]); trva_hh = set(hhR[np.r_[tr, va]])
    s_te = pd.Series(src[te]); h_te = pd.Series(hhR[te])
    obs = {
        "person_vs_train": float(s_te.isin(tr_set).mean()),
        "person_vs_trainval": float(s_te.isin(trva_set).mean()),
        "hh_vs_train": float(h_te.isin(tr_hh).mean()),
        "hh_vs_trainval": float(h_te.isin(trva_hh).mean()),
    }
    out["overlap_observed"] = obs
    # analytic iid expectation (fresh test row drawn ∝ w; train = n_tr iid draws)
    pi = w / W_s
    hh_series = pd.Series(pi).groupby(hh).transform("sum").to_numpy()
    n_tr, n_trva = len(tr), len(tr) + len(va)
    out["overlap_iid_analytic"] = {
        "person_vs_train": float((pi * (1 - (1 - pi) ** n_tr)).sum()),
        "person_vs_trainval": float((pi * (1 - (1 - pi) ** n_trva)).sum()),
        "hh_vs_train": float((pi * (1 - (1 - hh_series) ** n_tr)).sum()),
        "hh_vs_trainval": float((pi * (1 - (1 - hh_series) ** n_trva)).sum()),
    }
    # Monte Carlo: fresh independent test of the same size, drawn iid ∝ PWGTP within state,
    # compared against the ACTUAL train / train∪val
    mc = {k: [] for k in obs}
    Uid = U["Person_ID"].to_numpy()
    for s in range(N_MC):
        r = np.random.default_rng(1000 + s)
        pos = r.choice(len(U), size=len(te), replace=True, p=pi)
        fs = pd.Series(Uid[pos]); fh = pd.Series(hh[pos])
        mc["person_vs_train"].append(fs.isin(tr_set).mean())
        mc["person_vs_trainval"].append(fs.isin(trva_set).mean())
        mc["hh_vs_train"].append(fh.isin(tr_hh).mean())
        mc["hh_vs_trainval"].append(fh.isin(trva_hh).mean())
    out["overlap_fresh_draw_mc"] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                                        "min": float(np.min(v)), "max": float(np.max(v))} for k, v in mc.items()}
    rows.append({"state": st, "n_raw": len(U), "n_resampled": n_s, "state_total_z": out["state_total_z"],
                 "slope": b, "intercept": a, "dispersion": out["dispersion_index"], "gof_p": out["gof_p"],
                 "share0_obs": shares_obs["0"], "share0_exp": shares_exp["0"],
                 "share1_obs": shares_obs["1"], "share1_exp": shares_exp["1"],
                 "share2p_obs": 1 - shares_obs["0"] - shares_obs["1"], "share2p_exp": 1 - shares_exp["0"] - shares_exp["1"],
                 "hh_resid_corr": out["within_hh_corr_resid_person1_person2"],
                 "hh_allsame_obs": out["share_multiperson_hh_all_members_same_copies_obs"],
                 "hh_allsame_indep": out["share_multiperson_hh_all_members_same_copies_indep_sim"],
                 **{f"obs_{k}": v for k, v in obs.items()},
                 **{f"iid_{k}": v for k, v in out["overlap_iid_analytic"].items()},
                 **{f"mc_{k}": v["mean"] for k, v in out["overlap_fresh_draw_mc"].items()},
                 **{f"mcsd_{k}": v["sd"] for k, v in out["overlap_fresh_draw_mc"].items()}})
    (OUT / f"verify_design_{st}.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "by_pwgtp_decile"}, indent=1, default=float))

pd.DataFrame(rows).to_csv(OUT / "verify_design_summary.csv", index=False)
