"""Compare the JWTRNS light-rail alternatives with production (census crosswalk) on the current setup.
Gates as in decision_certificate_xstate_v3 (kappa_2 = 5, alpha = 0.05, tau = 1 + 1e-3, feasibility
>= 0.95); D = d_A(Q*, S_H) on the category union (PAPER_xstate_v3/dA_union.py) against the reference
the case was audited with; selection = argmin D among VALID; Deploy iff D-dagger <= B0 = 0.30.
2018 rows and Layer 1 are production (no mapping touches them)."""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[2])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import t as student_t

ROOT = Path(_PA.R(""))
PROD = Path(_PA.R("/main_experiments/audits_v3"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(PROD / "PAPER_xstate_v3"))
import dA_union as U

KAPPA_2, ALPHA, TAU, B0 = 5.0, 0.05, 1.0, 0.30
VARIANTS = ["lr2rail", "lr2subway", "lr2rail_k13", "lr2subway_k13"]
cert = pd.read_csv(PROD / "PAPER_xstate_v3/xstate_certificate_all.csv")
vH = pd.read_csv(PROD / "PAPER_xstate_v3/exp_ii_vH.csv").set_index("year")["v_H"]


def l2(d):
    j = json.loads((d / "layer2_summary.json").read_text())
    w = pd.read_csv(d / "layer2_per_realization_weights.csv")
    loss = w.scenario_loss.to_numpy(float); K = len(loss)
    cut = KAPPA_2 + float(student_t.ppf(1 - ALPHA, K - 1)) * float(np.std(loss, ddof=1) / np.sqrt(K))
    chi = float(j["prior_mean_scenario_loss"])
    assert abs(chi - loss.mean()) < 1e-9
    sig = pd.read_csv(d / "layer2_historical_sigma_summary.csv")
    cm = sig[sig.attr == "Commute_Mode"]
    att = {}
    for a, g in sig.groupby("attr"):
        z2 = sum(((w[r.target].to_numpy(float) - float(r.target_m)) / float(r.sigma)) ** 2 for r in g.itertuples())
        att[a] = float(np.mean(z2) / (2 * len(g)))
    assert abs(sum(att.values()) - chi) < 1e-8, (d, sum(att.values()), chi)
    return dict(chi=chi, cutoff=cut, L2=chi <= cut, rstar=int(w.loc[w.scenario_loss.idxmin(), "realization_id"]),
                n_commute=len(cm), chi_commute=att["Commute_Mode"])


def l3(d, ref_csv):
    if not (d / "layer3_summary.json").exists():
        return dict(L3=False, feas=np.nan, KL=np.nan, nESS=np.nan, D=np.nan, D_ref=np.nan, mu_out=np.nan)
    j = json.loads((d / "layer3_summary.json").read_text())
    pr = pd.read_csv(d / "layer3_per_realization_diagnostics.csv")
    raw = pr.iloc[0].get("attribute_projected_losses_json", None)
    ap = {k: float(v) for k, v in json.loads(raw).items()} if isinstance(raw, str) and raw not in ("", "nan") else {}
    feas = float(j["feasibility_rate"])
    ok = bool(ap) and feas >= 1 - ALPHA and all(c <= TAU + 1e-3 for c in ap.values())
    out = dict(L3=ok, feas=feas, KL=j.get("mean_distortion"), nESS=j.get("mean_agent_nESS", j.get("mean_nESS")),
               D=np.nan, D_ref=np.nan, mu_out=np.nan)
    if feas > 0 and (d / "projected_weights").exists():
        r = U.dA_union_l3(d, ref_csv=ref_csv)
        out.update(D=r["d_A"], D_ref=r["d_A_ref"], mu_out=r["mu_out"])
    return out


def case_rows(v, cases):
    rows = []
    for c in cases.itertuples():
        m, st, y = c.model, c.state, int(c.year)
        if v == "prod":
            base, ref = PROD, U.reference_path(st, y)
        else:
            base, ref = HERE / "runs" / v, HERE / "refs" / v / st / f"reference_{y}_{st}.csv"
        a = l2(base / f"layer2_{m}_{st}_{y}")
        b = l3(base / f"layer3_paperW_{m}_{st}_{y}", ref)
        rows.append(dict(variant=v, model=m, state=st, year=y, **a, **b, VALID=bool(a["L2"] and b["L3"])))
    return pd.DataFrame(rows)


def scenarios(df):
    out = []
    for (st, y), g in df.groupby(["state", "year"]):
        g = g[g.VALID].sort_values("D")
        r = dict(state=st, year=y, n_valid=len(g))
        if len(g):
            Dd = float(g.D.iloc[0])
            r.update(theta=g.model.iloc[0], D_dagger=Dd, decision="Deploy" if Dd <= B0 else "Defer",
                     gamma_min=0.0 if Dd <= B0 else (Dd / B0 - 1) / float(vH[y]),
                     runner_up=g.model.iloc[1] if len(g) > 1 else None,
                     gap=float(g.D.iloc[1] - Dd) if len(g) > 1 else np.nan,
                     D_range=(round(float(g.D.min()), 4), round(float(g.D.max()), 4)))
        else:
            r.update(min_chi_L1=float(df[(df.state == st) & (df.year == y)].chi.min()))
        out.append(r)
    return pd.DataFrame(out)


if __name__ == "__main__":
    cases = cert[(cert.L1_accept == True) & cert.year.isin([2021, 2024])]
    assert len(cases) == 36 and int(cert.L1_accept.sum()) == 54
    P = case_rows("prod", cases)
    # production check: chi and VALID must match the certificate
    pc = P.merge(cert[["model", "state", "year", "L2_chi_H", "VALID"]], on=["model", "state", "year"], suffixes=("", "_cert"))
    print("prod: max|chi - cert|", float((pc.chi - pc.L2_chi_H).abs().max()), " VALID match", int((pc.VALID == pc.VALID_cert).sum()), "/36")
    ALL = [P]; SC = [scenarios(P).assign(variant="prod")]
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
    summ = []
    for v in VARIANTS:
        A = case_rows(v, cases); ALL.append(A); S = scenarios(A).assign(variant=v); SC.append(S)
        M = A.merge(P, on=["model", "state", "year"], suffixes=("", "_p"))
        M["dchi"] = M.chi - M.chi_p; M["dD"] = M.D - M.D_p
        both = M[M.VALID & M.VALID_p]
        summ.append(dict(variant=v, n=len(M), n_commute=sorted(A.n_commute.unique()),
                         VALID=int(A.VALID.sum()), VALID_2021=int(A[A.year == 2021].VALID.sum()), VALID_2024=int(A[A.year == 2024].VALID.sum()),
                         L2_flips=int((M.L2 != M.L2_p).sum()), L3_flips=int((M.L3 != M.L3_p).sum()), VALID_flips=int((M.VALID != M.VALID_p).sum()),
                         rstar_changes=int((M.rstar != M.rstar_p).sum()),
                         dchi_min=M.dchi.min(), dchi_max=M.dchi.max(), absdchi_max=M.dchi.abs().max(),
                         absdD_max_all=M.dD.abs().max(), absdD_max_valid=both.dD.abs().max(),
                         absdD_max_OHTX=M[M.state != "CA"].dD.abs().max(),
                         mu_out_max_valid=A[A.VALID].mu_out.max(),
                         absdKL_max=(M.KL - M.KL_p).abs().max(), absdnESS_max=(M.nESS - M.nESS_p).abs().max(),
                         min_margin_to_cutoff=float((M.cutoff - M.chi).abs().min())))
        print(f"\n== {v}")
        print(M.loc[M.dchi.abs().sort_values(ascending=False).index[:6], ["model", "state", "year", "chi_p", "chi", "dchi", "cutoff", "D_p", "D", "dD", "VALID_p", "VALID"]].to_string(index=False))
    ALL = pd.concat(ALL); SC = pd.concat(SC); SUM = pd.DataFrame(summ)
    ALL.to_csv(HERE / "results/jw_cases.csv", index=False); SC.to_csv(HERE / "results/jw_scenarios.csv", index=False)
    SUM.to_csv(HERE / "results/jw_summary.csv", index=False)
    print("\nSUMMARY\n", SUM.T.to_string())
    print("\nSCENARIOS\n", SC.to_string(index=False))
    # pairwise spread across the three assignments (prod = census, lr2rail, lr2subway), per convention
    for conv, vs in (("drop", ["prod", "lr2rail", "lr2subway"]), ("k13", ["prod", "lr2rail_k13", "lr2subway_k13"])):
        W = ALL[ALL.variant.isin(vs)]
        chi_sp = W.pivot_table(index=["model", "state", "year"], columns="variant", values="chi")
        D_sp = W.pivot_table(index=["model", "state", "year"], columns="variant", values="D")
        V = W.pivot_table(index=["model", "state", "year"], columns="variant", values="VALID", aggfunc="first")
        allvalid = V.all(axis=1)
        print(f"\n[{conv}] spread across the three assignments over 36 cases (2018: 0):  max chi spread "
              f"{(chi_sp.max(1) - chi_sp.min(1)).max():.4f}  max D spread (all feasible) {(D_sp.max(1) - D_sp.min(1)).max():.5f}"
              f"  max D spread (valid in all) {(D_sp[allvalid].max(1) - D_sp[allvalid].min(1)).max():.5f}"
              f"  argmax chi spread {(chi_sp.max(1) - chi_sp.min(1)).idxmax()}")
