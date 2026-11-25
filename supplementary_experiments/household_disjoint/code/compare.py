"""Step 5: production vs shadow Layer-1 statistics and gate per (prior, state, reference variant).
Gate = decision_certificate_xstate_v3.layer1_test VERBATIM (imported): invalid > 0.005 -> reject;
else marg <= 0.05 - eta, dep <= 0.10 - eta, 1-supp <= 0.30 - eta with eta = DKW sqrt(log(2u/a')/2n),
a' = 0.05/3, u = 9 (marg/dep, n = K*n_agents) or 1 (supp, n = n_agents), n_agents = STATE_N_AGENTS
(= |testdata| = ensemble N).  'prod_n' keeps the production n_agents (only the reference changed);
'refn' sensitivity sets n_agents = |reference variant| (larger eta, tighter cutoff)."""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[3])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import os, sys, json, glob
import numpy as np, pandas as pd
ROOT = _PA.R(""); E = _PA.R("/main_experiments")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, E); sys.argv = sys.argv[:1]
import decision_certificate_xstate_v3 as DC
PROD = f"{E}/audits_v3"
TAGS = DC.PRIMARY_MODELS; STATES = ["CA", "TX", "OH"]
NAME = {"semapop_vae": "SemaPop-VAE+", "ttvae": "TTVAE", "bn": "BN", "wgan_zcr": "WGAN-GP-ZCR", "ctgan": "CTGAN",
        "flowsyn": "FlowSyn", "borysov": "Borysov-VAE", "tabddpm": "TabDDPM", "tvae": "TVAE"}
REFSIZE = {}
def nref(v, st):
    if (v, st) not in REFSIZE:
        sys.path.insert(0, f"{HERE}/code"); import run_l1_disjoint as W
        p = W.ref_for(v).format(state=st)
        REFSIZE[(v, st)] = sum(1 for _ in open(p)) - 1 if p else None
    return REFSIZE[(v, st)]
def load(path):
    if not os.path.exists(path): return None
    j = json.load(open(path))
    if "invalid_rules_note" not in j: return None          # unpatched / in-progress cell
    return {"marg_TVD": float(j["marginal_total_discrepancy_mean"]), "dep_TVD": float(j["dependency_total_discrepancy_mean"]),
            "support_cov": float(j["ensemble_support_coverage"]), "invalid_mass": float(j.get("invalid_mass_mean", 0.0)),
            "n_realizations": int(j["n_realizations"])}
def gate(l1, n):
    a = DC.ALPHA[0] / 3; K = l1["n_realizations"]
    em = DC.dkw_critical(a, n=K * n, n_categories=10); es = DC.dkw_critical(a, n=n, n_categories=1)
    cut = {"marg": DC.KAPPA_1_MARG - em, "dep": DC.KAPPA_1_DEP - em, "supp": DC.KAPPA_1_SUPPORT - es, "inv": DC.INVALID_MASS_TOL}
    val = {"marg": l1["marg_TVD"], "dep": l1["dep_TVD"], "supp": 1 - l1["support_cov"], "inv": l1["invalid_mass"]}
    t = DC.layer1_test(l1, DC.ALPHA[0], n)                     # verbatim production gate
    ratio = {g: val[g] / cut[g] for g in cut}
    bind = max(ratio, key=ratio.get)
    return t, cut, val, ratio, bind, em, es
rows = []
variants = sorted(d for d in os.listdir(f"{HERE}/runs") if os.path.isdir(f"{HERE}/runs/{d}"))
for v in variants:
    for tag in TAGS:
        for st in STATES:
            p = load(f"{PROD}/layer1_paper_{tag}_{st}/layer1_ensemble_summary.json")
            s = load(f"{HERE}/runs/{v}/layer1_paper_{tag}_{st}/layer1_ensemble_summary.json")
            if s is None: continue
            n_prod = DC.STATE_N_AGENTS[st]
            tp, cp, vp, rp, bp, _, _ = gate(p, n_prod)
            ts, cs, vs, rs, bs, em, es = gate(s, n_prod)
            tr, cr, _, _, br, emr, esr = gate(s, nref(v, st))
            r = dict(variant=v, prior=NAME[tag], tag=tag, state=st, n_ref=nref(v, st), n_agents_prod=n_prod, K=s["n_realizations"])
            for g, key in [("marg", "marg"), ("dep", "dep"), ("supp", "one_minus_cov"), ("inv", "invalid")]:
                r[f"{key}_prod"] = vp[g]; r[f"{key}_shadow"] = vs[g]; r[f"{key}_delta"] = vs[g] - vp[g]
            r["support_cov_prod"] = p["support_cov"]; r["support_cov_shadow"] = s["support_cov"]
            r.update(eta_margdep=em, eta_supp=es, cut_marg=cs["marg"], cut_dep=cs["dep"], cut_supp=cs["supp"], cut_inv=cs["inv"],
                     eta_margdep_refn=emr, eta_supp_refn=esr, cut_marg_refn=cr["marg"], cut_dep_refn=cr["dep"], cut_supp_refn=cr["supp"])
            r.update(accept_prod=tp["accept"], failed_prod=",".join(tp["failed"]), binding_prod=bp,
                     accept_shadow=ts["accept"], failed_shadow=",".join(ts["failed"]), binding_shadow=bs,
                     ratio_marg_shadow=rs["marg"], ratio_dep_shadow=rs["dep"], ratio_supp_shadow=rs["supp"], ratio_inv_shadow=rs["inv"],
                     accept_shadow_refn=tr["accept"], failed_shadow_refn=",".join(tr["failed"]),
                     flip=bool(tp["accept"] != ts["accept"]), flip_refn=bool(tp["accept"] != tr["accept"]))
            # all gates evaluated without the invalid-mass short-circuit (to show what else would fail)
            r["failed_shadow_all_gates"] = ",".join(g for g in ["inv", "marg", "dep", "supp"] if vs[g] > cs[g])
            r["failed_prod_all_gates"] = ",".join(g for g in ["inv", "marg", "dep", "supp"] if vp[g] > cp[g])
            rows.append(r)
df = pd.DataFrame(rows); os.makedirs(f"{HERE}/results", exist_ok=True)
df.to_csv(f"{HERE}/results/l1_disjoint_compare.csv", index=False)
# per-attribute marginal TV (mean over K) to show what drives changes
pa = []
for v in variants:
    for tag in TAGS:
        for st in STATES:
            f = f"{HERE}/runs/{v}/layer1_paper_{tag}_{st}/layer1_marginal_attr_summary.csv"
            if not os.path.exists(f): continue
            a = pd.read_csv(f); b = pd.read_csv(f"{PROD}/layer1_paper_{tag}_{st}/layer1_marginal_attr_summary.csv")
            a["variant"], a["tag"], a["state"] = v, tag, st
            a = a.merge(b, on=[c for c in a.columns if c in b.columns and not pd.api.types.is_float_dtype(b[c])], suffixes=("", "_prod"), how="left")
            pa.append(a)
if pa: pd.concat(pa).to_csv(f"{HERE}/results/l1_marginal_attr_compare.csv", index=False)
# headline
cert = pd.read_csv(f"{PROD}/PAPER_xstate_v3/xstate_certificate_all.csv")
summ = []
for v in variants:
    d = df[df.variant == v]
    if len(d) == 0: continue
    acc = dict(zip(zip(d.tag, d.state), d.accept_shadow)); accr = dict(zip(zip(d.tag, d.state), d.accept_shadow_refn))
    c = cert.copy(); c["L1_new"] = [acc.get((m, s), np.nan) for m, s in zip(c.model, c.state)]
    c["L1_new_refn"] = [accr.get((m, s), np.nan) for m, s in zip(c.model, c.state)]
    c["VALID_new"] = c.L1_new.astype(bool) & c.L2_accept.astype(bool) & c.L3_accept.astype(bool)
    c["VALID_new_refn"] = c.L1_new_refn.astype(bool) & c.L2_accept.astype(bool) & c.L3_accept.astype(bool)
    summ.append(dict(variant=v, pairs=len(d), l1_pass_pairs_prod=int(d.accept_prod.sum()), l1_pass_pairs_shadow=int(d.accept_shadow.sum()),
                     l1_pass_pairs_shadow_refn=int(d.accept_shadow_refn.sum()),
                     flips=";".join(f"{a}_{b}" for a, b in zip(d[d.flip].tag, d[d.flip].state)),
                     flips_refn=";".join(f"{a}_{b}" for a, b in zip(d[d.flip_refn].tag, d[d.flip_refn].state)),
                     l1_pass_cases_of_81=int(c.L1_new.fillna(False).astype(bool).sum()) if len(d) == 27 else None,
                     valid_cases_of_81=int(c.VALID_new.sum()) if len(d) == 27 else None,
                     valid_cases_of_81_refn=int(c.VALID_new_refn.sum()) if len(d) == 27 else None,
                     max_abs_dmarg=d.marg_delta.abs().max(), max_abs_ddep=d.dep_delta.abs().max(),
                     max_abs_dsupp=d.one_minus_cov_delta.abs().max(), max_abs_dinv=d.invalid_delta.abs().max(),
                     max_ratio_marg_among_passers=d[d.accept_prod].ratio_marg_shadow.max(),
                     max_ratio_dep_among_passers=d[d.accept_prod].ratio_dep_shadow.max()))
    c.to_csv(f"{HERE}/results/certificate_L1_{v}.csv", index=False)
S = pd.DataFrame(summ); S.to_csv(f"{HERE}/results/l1_disjoint_headline.csv", index=False)
pd.set_option("display.width", 250); print(S.to_string(index=False))
cols = ["variant", "tag", "state", "marg_prod", "marg_shadow", "dep_prod", "dep_shadow", "one_minus_cov_prod", "one_minus_cov_shadow",
        "invalid_shadow", "cut_marg", "accept_prod", "accept_shadow", "binding_shadow", "failed_shadow_all_gates"]
print(df[cols].round(4).to_string(index=False))
