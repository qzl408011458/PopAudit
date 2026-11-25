"""Independence test: is the production 2013 train/val/test split statistically
equivalent to INDEPENDENT draws of the established design
(person-level, PWGTP-proportional, with-replacement iid; national N_nat draws, seed 42)?

Compares, per state (CA/OH/TX), the observed overlap shares
  - test rows whose source person / source household also appears in train
  - val  rows ... in train
  - test rows ... in train U val
  - test rows ... in val
against
  (1) analytic iid expectation   E = sum_u P_u [1-(1-P_u)^n_ref]
      (P_u = within-state PWGTP share of person u, or summed share of household u)
  (2) MC-A: >= 200 fresh, independent (train, val, test) triples drawn iid ∝ PWGTP
      within state with production sizes (40,000 / 10,000 / |test|)
  (3) MC-B: >= 50 replays of the WHOLE production chain with new seeds
      (national choice over 3,132,795 persons -> state slice -> random permutation split),
      i.e. state size random too
  and contrasts with three alternative designs that would change the overlap:
  (a) without-replacement person sample + random split (SRS-WOR)
  (b) fixed (deterministic) expansion: copies_i = integerized n_s * pi_i, random split
  (c) household-joint PPS with replacement (households ∝ WGTP, all members kept), row split
Also a per-person residual-correlation test between train and test copy counts.

Read-only on data/. Writes ../out/.
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[6])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(_PA.R(""))
OUT = Path(__file__).resolve().parents[1] / "out"
OUT.mkdir(parents=True, exist_ok=True)
STATES = {"CA": 6, "OH": 39, "TX": 48}
N_MC_A, N_MC_B, N_ALT = 200, 50, 20
N_TR, N_VA = 40_000, 10_000
METRICS = ["person_test_vs_train", "hh_test_vs_train", "person_val_vs_train", "hh_val_vs_train",
           "person_test_vs_trainval", "hh_test_vs_trainval", "person_test_vs_val", "hh_test_vs_val"]


def overlaps(pid, hid, tr, va, te, n_p, n_h):
    """pid/hid: integer person/household codes of rows; tr/va/te: row index arrays."""
    mp_tr = np.zeros(n_p, bool); mp_tr[pid[tr]] = True
    mh_tr = np.zeros(n_h, bool); mh_tr[hid[tr]] = True
    mp_va = np.zeros(n_p, bool); mp_va[pid[va]] = True
    mh_va = np.zeros(n_h, bool); mh_va[hid[va]] = True
    pt, ht, pv, hv = pid[te], hid[te], pid[va], hid[va]
    return {
        "person_test_vs_train": mp_tr[pt].mean(), "hh_test_vs_train": mh_tr[ht].mean(),
        "person_val_vs_train": mp_tr[pv].mean(), "hh_val_vs_train": mh_tr[hv].mean(),
        "person_test_vs_trainval": (mp_tr | mp_va)[pt].mean(), "hh_test_vs_trainval": (mh_tr | mh_va)[ht].mean(),
        "person_test_vs_val": mp_va[pt].mean(), "hh_test_vs_val": mh_va[ht].mean(),
    }


def resid_corr(pid, tr, te, pi, n_p):
    """corr over persons of standardized copy-count residuals in train vs test.
    Independent iid draws => ~0 (SE ~ 1/sqrt(N)); fixed-total designs => negative."""
    ctr = np.bincount(pid[tr], minlength=n_p).astype(float)
    cte = np.bincount(pid[te], minlength=n_p).astype(float)
    ztr = (ctr - len(tr) * pi) / np.sqrt(len(tr) * pi)
    zte = (cte - len(te) * pi) / np.sqrt(len(te) * pi)
    return float(np.corrcoef(ztr, zte)[0, 1])


def summarize(lst):
    a = np.asarray(lst, float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "min": float(a.min()),
            "max": float(a.max()), "q025": float(np.quantile(a, .025)), "q975": float(np.quantile(a, .975)), "n": int(len(a))}


t0 = time.time()
# national unweighted (non-resampled) 2013 processed table: the population the production chain resampled
nat = pd.read_csv(ROOT / "data/pums_historical/person_level_merged_compact_pwgtp_2013.csv",
                  usecols=["Person_ID", "PWGTP", "State_Code"], dtype={"Person_ID": str})
N_NAT = len(nat); p_nat = nat["PWGTP"].to_numpy(float); p_nat /= p_nat.sum()
nat_state = nat["State_Code"].to_numpy()
print(f"national N={N_NAT} loaded {time.time()-t0:.1f}s", flush=True)

ctx = {}
results = {}
for st, fips in STATES.items():
    sd = ROOT / f"data/pums_states/{st}"
    # ---- raw PUMS person + housing files (population U) ----
    raw = pd.read_csv(ROOT / f"data/pums13/person_2013_{st}_full.csv", usecols=["SERIALNO", "SPORDER", "PWGTP"])
    hous = pd.read_csv(ROOT / f"data/pums13/housing_2013_{st}_full.csv", usecols=["SERIALNO", "WGTP"])
    raw = raw.merge(hous, on="SERIALNO", how="left", validate="m:1")
    raw["Person_ID"] = raw["SERIALNO"].astype(str) + "-" + raw["SPORDER"].astype(str)
    Unat = nat[nat["State_Code"] == fips]
    chk = {"raw_rows": int(len(raw)), "nat_slice_rows": int(len(Unat)),
           "raw_set_equals_nat_slice": bool(set(raw["Person_ID"]) == set(Unat["Person_ID"])),
           "raw_pwgtp_equals_nat_slice": bool((raw.set_index("Person_ID")["PWGTP"].reindex(Unat["Person_ID"]).to_numpy()
                                               == Unat["PWGTP"].to_numpy()).all()),
           "raw_pwgtp_min": int(raw["PWGTP"].min())}
    raw = raw.reset_index(drop=True)
    n_p = len(raw)
    pid_of = pd.Series(np.arange(n_p), index=raw["Person_ID"].to_numpy())
    hcode, hlab = pd.factorize(raw["SERIALNO"])
    n_h = len(hlab)
    w = raw["PWGTP"].to_numpy(float); pi = w / w.sum()
    Ph = np.bincount(hcode, weights=pi, minlength=n_h)

    # ---- production split (observed) ----
    R = pd.read_csv(sd / f"person_level_2013_{st}.csv", usecols=["Source_Person_ID"], dtype={"Source_Person_ID": str})
    ids = pickle.load(open(sd / "dataset_ids.pkl", "rb"))
    tr = np.asarray(ids["train_idx"]); va = np.asarray(ids["val_idx"]); te = np.asarray(ids["test_idx"])
    n_s, n_te = len(R), len(te)
    assert len(tr) == N_TR and len(va) == N_VA and len(tr) + len(va) + len(te) == n_s
    rpid = pid_of.reindex(R["Source_Person_ID"].to_numpy()).to_numpy()
    chk["all_resampled_rows_map_to_raw"] = bool(not np.isnan(rpid).any())
    rpid = rpid.astype(int); rhid = hcode[rpid]
    obs = overlaps(rpid, rhid, tr, va, te, n_p, n_h)
    obs = {k: float(v) for k, v in obs.items()}
    obs["resid_corr_train_test"] = resid_corr(rpid, tr, te, pi, n_p)

    # ---- (1) analytic iid expectation ----
    Ep = lambda n: float((pi * (1 - (1 - pi) ** n)).sum())
    Eh = lambda n: float((Ph * (1 - (1 - Ph) ** n)).sum())
    ana = {"person_test_vs_train": Ep(N_TR), "hh_test_vs_train": Eh(N_TR),
           "person_val_vs_train": Ep(N_TR), "hh_val_vs_train": Eh(N_TR),
           "person_test_vs_trainval": Ep(N_TR + N_VA), "hh_test_vs_trainval": Eh(N_TR + N_VA),
           "person_test_vs_val": Ep(N_VA), "hh_test_vs_val": Eh(N_VA), "resid_corr_train_test": 0.0}

    # ---- (2) MC-A: fresh independent triples within state ----
    cdf = np.cumsum(pi); cdf /= cdf[-1]
    mcA = {k: [] for k in METRICS + ["resid_corr_train_test"]}
    for r in range(N_MC_A):
        rng = np.random.default_rng(7_000_000 + 1000 * fips + r)
        draw = np.searchsorted(cdf, rng.random(N_TR + N_VA + n_te), side="right")  # iid ∝ PWGTP
        a_tr = np.arange(N_TR); a_va = np.arange(N_TR, N_TR + N_VA); a_te = np.arange(N_TR + N_VA, len(draw))
        o = overlaps(draw, hcode[draw], a_tr, a_va, a_te, n_p, n_h)
        for k, v in o.items(): mcA[k].append(v)
        mcA["resid_corr_train_test"].append(resid_corr(draw, a_tr, a_te, pi, n_p))

    ctx[st] = dict(raw=raw, pid_of=pid_of, hcode=hcode, n_p=n_p, n_h=n_h, pi=pi, n_s=n_s, n_te=n_te,
                   fips=fips, R=R, tr=tr, va=va, te=te)
    results[st] = {"checks": chk, "n_raw": n_p, "n_households": n_h, "n_resampled": n_s, "n_test": n_te,
                   "observed": obs, "analytic_iid": ana, "mcA_fresh_independent": {k: summarize(v) for k, v in mcA.items()}}
    print(st, "obs/analytic/MC-A done", f"{time.time()-t0:.1f}s", json.dumps(chk), flush=True)

# ---- (3) MC-B: replay the whole production chain with new seeds ----
mcB = {st: {k: [] for k in METRICS + ["n_s", "resid_corr_train_test"]} for st in STATES}
nat_ids = nat["Person_ID"].to_numpy()
for r in range(N_MC_B):
    rng = np.random.default_rng(8_000_000 + r)
    pos = rng.choice(N_NAT, size=N_NAT, replace=True, p=p_nat)   # identical call to weighted_resample_by_pwgtp
    for st, c in ctx.items():
        sel = pos[nat_state[pos] == c["fips"]]                     # state slice, order preserved
        spid = c["pid_of"].reindex(nat_ids[sel]).to_numpy().astype(int)
        perm = np.random.default_rng(9_000_000 + 1000 * c["fips"] + r).permutation(len(spid))
        b_tr, b_va, b_te = perm[:N_TR], perm[N_TR:N_TR + N_VA], perm[N_TR + N_VA:]
        o = overlaps(spid, c["hcode"][spid], b_tr, b_va, b_te, c["n_p"], c["n_h"])
        for k, v in o.items(): mcB[st][k].append(v)
        mcB[st]["n_s"].append(len(spid))
        mcB[st]["resid_corr_train_test"].append(resid_corr(spid, b_tr, b_te, c["pi"], c["n_p"]))
    if r % 10 == 0: print("MC-B rep", r, f"{time.time()-t0:.1f}s", flush=True)
for st in STATES:
    results[st]["mcB_full_chain_replay"] = {k: summarize(v) for k, v in mcB[st].items()}

# ---- alternative designs ----
for st, c in ctx.items():
    raw, pi, hcode, n_p, n_h, n_s = c["raw"], c["pi"], c["hcode"], c["n_p"], c["n_h"], c["n_s"]
    alt = {}
    def split_rows(n, rng):
        perm = rng.permutation(n); return perm[:N_TR], perm[N_TR:N_TR + N_VA], perm[N_TR + N_VA:]
    # (a) SRS without replacement of min(n_s, N) persons, random split
    acc = {k: [] for k in METRICS}
    for r in range(N_ALT):
        rng = np.random.default_rng(10_000 + r)
        m = min(n_s, n_p); draw = rng.choice(n_p, size=m, replace=False)
        o = overlaps(draw, hcode[draw], *split_rows(m, rng), n_p, n_h)
        for k, v in o.items(): acc[k].append(v)
    alt["a_without_replacement_person"] = {k: summarize(v) for k, v in acc.items()}
    alt["a_without_replacement_person"]["note"] = (f"sample size min(n_s={n_s}, N={n_p}); "
        "any WOR person design (SRS or PPS) gives person overlap identically 0")
    # (b) fixed expansion: integerized n_s*pi (floor + largest remainder), random split
    lam = n_s * pi; cps = np.floor(lam).astype(int)
    rem = n_s - cps.sum(); cps[np.argsort(-(lam - cps))[:rem]] += 1
    pool = np.repeat(np.arange(n_p), cps)
    acc = {k: [] for k in METRICS + ["resid_corr_train_test"]}
    for r in range(N_ALT):
        rng = np.random.default_rng(20_000 + r)
        s_tr, s_va, s_te = split_rows(len(pool), rng)
        o = overlaps(pool, hcode[pool], s_tr, s_va, s_te, n_p, n_h)
        for k, v in o.items(): acc[k].append(v)
        acc["resid_corr_train_test"].append(resid_corr(pool, s_tr, s_te, pi, n_p))
    alt["b_fixed_expansion"] = {k: summarize(v) for k, v in acc.items()}
    alt["b_fixed_expansion"]["share_persons_zero_copies"] = float((cps == 0).mean())
    # (c) household-joint PPS with replacement ∝ WGTP (GQ persons: own PWGTP), all members, row split
    hw = raw.groupby(hcode)["WGTP"].first().to_numpy(float)
    gq = hw == 0
    hw[gq] = np.bincount(hcode, weights=raw["PWGTP"].to_numpy(float), minlength=n_h)[gq]
    hsize = np.bincount(hcode, minlength=n_h)
    members = np.argsort(hcode, kind="stable"); starts = np.r_[0, np.cumsum(hsize)[:-1]]
    ph = hw / hw.sum(); m_hh = int(round(n_s / (ph * hsize).sum()))
    acc = {k: [] for k in METRICS}
    for r in range(N_ALT):
        rng = np.random.default_rng(30_000 + r)
        hd = rng.choice(n_h, size=m_hh, replace=True, p=ph)
        rows = np.concatenate([members[starts[h]:starts[h] + hsize[h]] for h in hd])
        o = overlaps(rows, hcode[rows], *split_rows(len(rows), rng), n_p, n_h)
        for k, v in o.items(): acc[k].append(v)
    alt["c_household_joint_pps"] = {k: summarize(v) for k, v in acc.items()}
    alt["c_household_joint_pps"]["n_households_drawn"] = m_hh
    results[st]["alternative_designs"] = alt
    print(st, "alternatives done", f"{time.time()-t0:.1f}s", flush=True)

# ---- tests + summary table ----
rows = []
for st, res in results.items():
    tests = {}
    for k in METRICS + ["resid_corr_train_test"]:
        o = res["observed"][k]; A = res["mcA_fresh_independent"][k]; B = res["mcB_full_chain_replay"][k]
        zA = (o - A["mean"]) / A["sd"]; zB = (o - B["mean"]) / B["sd"]
        tests[k] = {"z_vs_mcA": float(zA), "z_vs_mcB": float(zB)}
        row = {"state": st, "metric": k, "observed": o, "analytic_iid": res["analytic_iid"][k],
               "mcA_mean": A["mean"], "mcA_sd": A["sd"], "mcA_q025": A["q025"], "mcA_q975": A["q975"], "z_vs_mcA": zA,
               "mcB_mean": B["mean"], "mcB_sd": B["sd"], "z_vs_mcB": zB}
        for an, ad in res["alternative_designs"].items():
            if k in ad: row[f"alt_{an}_mean"] = ad[k]["mean"]
        rows.append(row)
    res["tests"] = tests
    (OUT / f"independence_{st}.json").write_text(json.dumps(res, indent=2))
summ = pd.DataFrame(rows)
summ.to_csv(OUT / "independence_summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(summ.round(4).to_string(index=False))
print(f"total {time.time()-t0:.1f}s")
