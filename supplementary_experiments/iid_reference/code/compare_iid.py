"""SUPERSEDED for D and the 81-case consequences: code/compare_iid_g1fix.py is the same
script with D computed on the category union (dA_union.D_union) and writes
corrected/results/; use that for the certificate, valid count and selected priors.
Production vs fresh-reference Layer-1 statistics and gate per (prior, state); consequences for the 81-case
certificate (L1 pass cases, VALID count, selected priors).
Gate = decision_certificate_xstate_v3.layer1_test VERBATIM (imported), as in household_disjoint/code/compare.py:
invalid > 0.005 -> reject; else marg <= 0.05-eta, dep <= 0.10-eta, 1-cov <= 0.30-eta_supp,
eta = DKW sqrt(log(2u/a')/2n), a'=0.05/3, u=9 (n=K*n_agents) / u=1 (n=n_agents), n_agents = STATE_N_AGENTS
(= |testdata| = |fresh|, so the cutoffs are identical for both references).
Selected prior per scenario = argmin post-projection d_A(Q*) over VALID priors (make_figE_postproj.py rule;
d_A from audits_v3/layer3_paperW_<tag>_<ST>_<year>/layer3_target_summary.csv, read-only).
Outputs: ../results/l1_compare.csv, l1_headline.csv, certificate_L1_fresh.csv, selected_priors.csv,
         l1_marginal_attr_compare.csv"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[3])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import os, sys, json
import numpy as np, pandas as pd
ROOT = _PA.R(""); E = _PA.R("/main_experiments")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, E); sys.argv = sys.argv[:1]
import decision_certificate_xstate_v3 as DC
PROD = f"{E}/audits_v3"
TAGS = DC.PRIMARY_MODELS; STATES = ["CA", "TX", "OH"]
NAME = {"semapop_vae": "SemaPop-VAE+", "ttvae": "TTVAE", "bn": "BN", "wgan_zcr": "WGAN-GP-ZCR", "ctgan": "CTGAN",
        "flowsyn": "FlowSyn", "borysov": "Borysov-VAE", "tabddpm": "TabDDPM", "tvae": "TVAE"}
V = "fresh"
def load(path):
    if not os.path.exists(path): return None
    j = json.load(open(path))
    if "invalid_rules_note" not in j: return None
    return {"marg_TVD": float(j["marginal_total_discrepancy_mean"]), "dep_TVD": float(j["dependency_total_discrepancy_mean"]),
            "support_cov": float(j["ensemble_support_coverage"]), "invalid_mass": float(j.get("invalid_mass_mean", 0.0)),
            "n_realizations": int(j["n_realizations"])}
def gate(l1, n):
    a = DC.ALPHA[0] / 3; K = l1["n_realizations"]
    em = DC.dkw_critical(a, n=K * n, n_categories=10); es = DC.dkw_critical(a, n=n, n_categories=1)
    cut = {"marg": DC.KAPPA_1_MARG - em, "dep": DC.KAPPA_1_DEP - em, "supp": DC.KAPPA_1_SUPPORT - es, "inv": DC.INVALID_MASS_TOL}
    val = {"marg": l1["marg_TVD"], "dep": l1["dep_TVD"], "supp": 1 - l1["support_cov"], "inv": l1["invalid_mass"]}
    t = DC.layer1_test(l1, DC.ALPHA[0], n)
    return t, cut, val
rows = []
for tag in TAGS:
    for st in STATES:
        p = load(f"{PROD}/layer1_paper_{tag}_{st}/layer1_ensemble_summary.json")
        s = load(f"{HERE}/runs/{V}/layer1_paper_{tag}_{st}/layer1_ensemble_summary.json")
        if s is None: print("missing", tag, st); continue
        n = DC.STATE_N_AGENTS[st]
        tp, cp, vp = gate(p, n); ts, cs, vs = gate(s, n)
        r = dict(prior=NAME[tag], tag=tag, state=st, n_agents=n, K=s["n_realizations"])
        for g, key in [("marg", "marg"), ("dep", "dep"), ("supp", "one_minus_cov"), ("inv", "invalid")]:
            r[f"{key}_prod"] = vp[g]; r[f"{key}_fresh"] = vs[g]; r[f"{key}_delta"] = vs[g] - vp[g]
        r.update(cut_marg_prod=cp["marg"], cut_marg_fresh=cs["marg"], cut_dep_prod=cp["dep"], cut_dep_fresh=cs["dep"],
                 cut_supp_prod=cp["supp"], cut_supp_fresh=cs["supp"], cut_inv=cs["inv"])
        r.update(accept_prod=tp["accept"], failed_prod=",".join(tp["failed"]),
                 accept_fresh=ts["accept"], failed_fresh=",".join(ts["failed"]), flip=bool(tp["accept"] != ts["accept"]))
        r["failed_prod_all_gates"] = ",".join(g for g in ["inv", "marg", "dep", "supp"] if vp[g] > cp[g])
        r["failed_fresh_all_gates"] = ",".join(g for g in ["inv", "marg", "dep", "supp"] if vs[g] > cs[g])
        # slack to the tightest non-invalid cutoff (positive = room), production and fresh
        r["min_slack_prod"] = min(cp[g] - vp[g] for g in ["marg", "dep", "supp"])
        r["min_slack_fresh"] = min(cs[g] - vs[g] for g in ["marg", "dep", "supp"])
        rows.append(r)
df = pd.DataFrame(rows); os.makedirs(f"{HERE}/results", exist_ok=True)
df.to_csv(f"{HERE}/results/l1_compare.csv", index=False)
# per-attribute marginal TV
pa = []
for tag in TAGS:
    for st in STATES:
        f = f"{HERE}/runs/{V}/layer1_paper_{tag}_{st}/layer1_marginal_attr_summary.csv"
        if not os.path.exists(f): continue
        a = pd.read_csv(f); b = pd.read_csv(f"{PROD}/layer1_paper_{tag}_{st}/layer1_marginal_attr_summary.csv")
        a["tag"], a["state"] = tag, st
        a = a.merge(b, on=[c for c in a.columns if c in b.columns and not pd.api.types.is_float_dtype(b[c])], suffixes=("", "_prod"), how="left")
        pa.append(a)
if pa: pd.concat(pa).to_csv(f"{HERE}/results/l1_marginal_attr_compare.csv", index=False)
# certificate consequences
cert = pd.read_csv(f"{PROD}/PAPER_xstate_v3/xstate_certificate_all.csv")
acc = dict(zip(zip(df.tag, df.state), df.accept_fresh))
c = cert.copy(); c["L1_fresh"] = [acc.get((m, s), np.nan) for m, s in zip(c.model, c.state)]
c["VALID_fresh"] = c.L1_fresh.astype(bool) & c.L2_accept.astype(bool) & c.L3_accept.astype(bool)
c.to_csv(f"{HERE}/results/certificate_L1_fresh.csv", index=False)
def dA(st, m, y):
    p = f"{PROD}/layer3_paperW_{m}_{st}_{int(y)}/layer3_target_summary.csv"
    return 0.5 * float(pd.read_csv(p)["projected_abs_err_mean"].sum()) if os.path.exists(p) else np.nan
sel = []
for y in (2018, 2021, 2024):
    for st in ("CA", "OH", "TX"):
        row = {"state": st, "year": y}
        for lab, col in [("prod", "VALID"), ("fresh", "VALID_fresh")]:
            g = c[(c.state == st) & (c.year == y) & (c[col].astype(bool))].copy()
            g["d_A"] = [dA(st, m, y) for m in g.model]
            row[f"n_adm_{lab}"] = len(g)
            row[f"admitted_{lab}"] = ";".join(sorted(g.model))
            if len(g):
                b = g.loc[g.d_A.idxmin()]; row[f"theta_dagger_{lab}"] = b.model; row[f"D_dagger_{lab}"] = b.d_A
                row[f"flat_{lab}"] = "Deploy" if b.d_A <= 0.30 else "Defer"
            else:
                row[f"theta_dagger_{lab}"] = None; row[f"D_dagger_{lab}"] = np.nan; row[f"flat_{lab}"] = "Defer"
        row["same_selection"] = (row["theta_dagger_prod"] == row["theta_dagger_fresh"]) and (row["flat_prod"] == row["flat_fresh"])
        sel.append(row)
S = pd.DataFrame(sel); S.to_csv(f"{HERE}/results/selected_priors.csv", index=False)
pass_prod = set(zip(df[df.accept_prod].tag, df[df.accept_prod].state))
pass_fresh = set(zip(df[df.accept_fresh].tag, df[df.accept_fresh].state))
H = dict(pairs=len(df), l1_pass_pairs_prod=len(pass_prod), l1_pass_pairs_fresh=len(pass_fresh),
         l1_fail_pairs_prod=len(df) - len(pass_prod), l1_fail_pairs_fresh=len(df) - len(pass_fresh),
         same_pass_set=pass_prod == pass_fresh, flips=";".join(f"{a}_{b}" for a, b in zip(df[df.flip].tag, df[df.flip].state)),
         l1_pass_cases_of_81_prod=int(c.L1_accept.astype(bool).sum()), l1_pass_cases_of_81_fresh=int(c.L1_fresh.astype(bool).sum()),
         valid_of_81_prod=int(c.VALID.astype(bool).sum()), valid_of_81_fresh=int(c.VALID_fresh.sum()),
         same_valid_set=bool((c.VALID.astype(bool) == c.VALID_fresh).all()),
         selected_unchanged=bool(S.same_selection.all()),
         max_abs_dmarg=df.marg_delta.abs().max(), max_abs_ddep=df.dep_delta.abs().max(),
         max_abs_d1mcov=df.one_minus_cov_delta.abs().max(), max_abs_dinv=df.invalid_delta.abs().max(),
         max_abs_dcut=max((df.cut_marg_fresh - df.cut_marg_prod).abs().max(), (df.cut_dep_fresh - df.cut_dep_prod).abs().max(),
                          (df.cut_supp_fresh - df.cut_supp_prod).abs().max()),
         min_slack_prod_among_passers=df[df.accept_prod].min_slack_prod.min(),
         min_slack_fresh_among_passers=df[df.accept_prod].min_slack_fresh.min(),
         mean_dmarg=df.marg_delta.mean(), mean_ddep=df.dep_delta.mean(), mean_d1mcov=df.one_minus_cov_delta.mean())
pd.DataFrame([H]).to_csv(f"{HERE}/results/l1_headline.csv", index=False)
pd.set_option("display.width", 250)
print(pd.Series(H).to_string())
cols = ["tag", "state", "marg_prod", "marg_fresh", "dep_prod", "dep_fresh", "one_minus_cov_prod", "one_minus_cov_fresh",
        "invalid_fresh", "cut_marg_fresh", "accept_prod", "accept_fresh", "failed_fresh_all_gates", "min_slack_prod", "min_slack_fresh"]
print(df[cols].round(5).to_string(index=False))
print(S.to_string(index=False))
