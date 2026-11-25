"""Certificate + selected-prior budget per sigma variant, with the paper's gates.
Layer-1 from audits_v3 (sigma-independent); Layer-2 gate (kappa_2 + eta_2, t-quantile
empirical SE), Layer-3 gate (feasibility, chi_{H,a} <= tau + 1e-3) and the D / v(H) /
gamma_min formulas are imported from the shadow copies of decision_certificate_xstate_v3.py
and make_figE_postproj.py logic.  Outputs -> results/.

g1fix (2026-11-24): D = d_A(Q*, S_H) is now computed on the UNION of the categories of the
projected population Q* (layer3_paperW_*/projected_weights) and of the scenario reference
(zero target outside the reference) with the shared production helper
audits_v3/PAPER_xstate_v3/dA_union.py (check=True: its reference-category part must equal
0.5*sum projected_abs_err_mean of layer3_target_summary.csv).  The pre-fix quantity is kept
as D_ref.  v(H) is rounded to 4 decimals before gamma_min, as in production
(rebuild_lever_data.py -> exp_ii_vH.csv -> make_figE_postproj.py).  The base variant is
checked against the production certificate and tableE_decision_dA.csv.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code")); sys.path.insert(0, str(HERE))
import decision_certificate_xstate_v3 as DC
from sigma_sens import VARIANTS, SRC, cases, vdir
sys.path.insert(0, str(Path(SRC) / "PAPER_xstate_v3"))
from dA_union import dA_union_l3

B0 = DC.B0
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
cert0 = pd.read_csv(Path(SRC) / "PAPER_xstate_v3/xstate_certificate_all.csv")
tabE0 = pd.read_csv(Path(SRC) / "PAPER_xstate_v3/figures/tables/tableE_decision_dA.csv")
WFH = "Commute_Mode==Worked_at_home"


def run_variant(v):
    DC.BASE = vdir(v)
    rows = []
    for m, st, y in cases():
        l2 = DC.load_layer2(st, m, y); t2 = DC.layer2_test(l2, DC.ALPHA[1])
        l3 = DC.load_layer3(st, m, y)
        t3 = DC.layer3_test(l3, DC.ALPHA[2]) if l3 else {"accept": False, "failed_attrs": ["missing"]}
        sig = pd.read_csv(vdir(v) / f"layer2_{m}_{st}_{y}" / "layer2_historical_sigma_summary.csv")
        D = D_ref = mu_out = rstar_pre = wfh_proj = wfh_tgt = None
        l3d = vdir(v) / f"layer3_paperW_{m}_{st}_{y}"
        if (l3d / "layer3_target_summary.csv").exists():
            u = dA_union_l3(l3d, st, y, check=True)
            D, D_ref, mu_out = u["d_A"], u["d_A_ref"], u["mu_out"]
            t = pd.read_csv(l3d / "layer3_target_summary.csv").set_index("target")
            rstar_pre = 0.5 * t.prior_abs_err_mean.sum()
            wfh_proj, wfh_tgt = float(t.loc[WFH, "projected_mean"]), float(t.loc[WFH, "target_m"])
        sw = sig.set_index("target").sigma
        rows.append(dict(variant=v, state=st, model=m, year=int(y), chi_bar=l2["chi_H"], cutoff=t2["cutoff"],
                         L2=t2["accept"], L3=t3["accept"], L3_failed=",".join(t3["failed_attrs"]),
                         KL=(l3 or {}).get("KL"), nESS=(l3 or {}).get("nESS"),
                         VALID=bool(t2["accept"] and t3["accept"]), D=D, D_ref=D_ref, mu_out=mu_out,
                         dA_rstar_pre=rstar_pre, wfh_proj=wfh_proj, wfh_target=wfh_tgt,
                         sigma_wfh=float(sw.get(WFH, np.nan)),
                         sigma_norm=float(np.sqrt((sig.sigma ** 2).sum())),
                         at_floor=int((sig.sigma <= VARIANTS[v]["floor"] + 1e-15).sum()), n_comp=len(sig)))
    df = pd.DataFrame(rows)
    msn = df.drop_duplicates(["state", "model", "year"]).groupby("year").sigma_norm.mean()
    vH = (msn / msn[2018] - 1.0).round(4).to_dict()          # production rounding (exp_ii_vH.csv)
    sc = []
    for y in (2018, 2021, 2024):
        for st in ("CA", "OH", "TX"):
            g = df[(df.state == st) & (df.year == y) & df.VALID].sort_values("D")
            r = dict(variant=v, state=st, year=y, n_adm=len(g), v_H=vH[y],
                     admitted=";".join(g.model))
            if len(g):
                Dd = float(g.D.iloc[0])
                r.update(wfh_target=float(g.wfh_target.iloc[0]), wfh_proj=float(g.wfh_proj.iloc[0]))
                r.update(theta_dagger=g.model.iloc[0], D_dagger=Dd, D_dagger_ref=float(g.D_ref.iloc[0]),
                         decision="Deploy" if Dd <= B0 else "Defer",
                         gamma_min=0.0 if Dd <= B0 else ((Dd / B0 - 1) / vH[y] if vH[y] > 0 else np.inf),
                         theta_runner_up=g.model.iloc[1] if len(g) > 1 else None,
                         runner_up_gap=float(g.D.iloc[1] - Dd) if len(g) > 1 else np.nan,
                         max_D_admitted=float(g.D.max()))
            else:
                r.update(theta_dagger=None, D_dagger=None, decision="Defer", gamma_min=None)
            sc.append(r)
    return df, pd.DataFrame(sc)


def main():
    vs = [v for v in VARIANTS if (vdir(v)).exists()]
    allc, alls = [], []
    for v in vs:
        c, s = run_variant(v); allc.append(c); alls.append(s)
    C = pd.concat(allc); S = pd.concat(alls)
    C.to_csv(OUT / "sigma_sens_cases.csv", index=False); S.to_csv(OUT / "sigma_sens_scenarios.csv", index=False)
    summ = []
    for v in vs:
        c = C[C.variant == v]; s = S[S.variant == v]
        dep = s[s.decision == "Deploy"]
        def sv(st, y, col):
            r = s[(s.state == st) & (s.year == y)].iloc[0]
            return None if pd.isna(r[col]) else (round(float(r[col]), 4) if col != "theta_dagger" else r[col])
        row = dict(variant=v, VALID=int(c.VALID.sum()),
                   V2018=int(c[c.year == 2018].VALID.sum()), V2021=int(c[c.year == 2021].VALID.sum()),
                   V2024=int(c[c.year == 2024].VALID.sum()), L2_pass=int(c.L2.sum()),
                   L3_fail_after_L2=int((c.L2 & ~c.L3).sum()),
                   deploy=",".join(f"{r.state}{r.year}" for r in dep.itertuples()),
                   defer_after_admission=",".join(f"{r.state}{r.year}" for r in s[(s.decision == "Defer") & (s.n_adm > 0)].itertuples()))
        for st, y in [(a, b) for b in (2018, 2021, 2024) for a in ("CA", "OH", "TX")]:
            row[f"D_{st}{str(y)[2:]}"] = sv(st, y, "D_dagger")
        for st in ("CA", "OH", "TX"):
            row[f"gmin_{st}24"] = sv(st, 2024, "gamma_min")
        row.update(vH21=s[s.year == 2021].v_H.iloc[0], vH24=s[s.year == 2024].v_H.iloc[0],
                   min_chi_2021=round(c[c.year == 2021].chi_bar.min(), 3),
                   max_mu_out_valid=float(c[c.VALID].mu_out.max()) if c.VALID.any() else None)
        summ.append(row)
    T = pd.DataFrame(summ); T.to_csv(OUT / "sigma_sens_summary.csv", index=False)
    pd.set_option("display.width", 320); pd.set_option("display.max_columns", 50)
    print(T.to_string(index=False))
    # baseline reproduction check against the production certificate and tableE (D on the category union)
    if "base" in vs:
        b = C[C.variant == "base"].merge(cert0.assign(year=lambda d: d.year.astype(int))
                                         [["state", "model", "year", "L2_chi_H", "L2_cutoff", "L2_accept", "VALID"]],
                                         on=["state", "model", "year"])
        print("\nbase vs production certificate: VALID match", int((b.VALID_x == b.VALID_y).sum()), "/", len(b),
              "| L2 flags match", int((b.L2 == b.L2_accept).sum()), "| VALID", int(b.VALID_x.sum()),
              "| max |chi diff|", float((b.chi_bar - b.L2_chi_H).abs().max()),
              "| max |cutoff diff|", float((b.cutoff - b.L2_cutoff).abs().max()))
        e = C[(C.variant == "base") & C.VALID].merge(tabE0[["state", "model", "year", "d_A", "d_A_ref", "mu_out"]],
                                                     on=["state", "model", "year"], how="outer", indicator=True)
        print("base valid vs tableE rows:", e._merge.value_counts().to_dict(),
              "| max |D - tableE d_A|", float((e.D - e.d_A).abs().max()),
              "| max |D_ref - d_A_ref|", float((e.D_ref - e.d_A_ref).abs().max()))
        bs = S[S.variant == "base"].merge(pd.read_csv(Path(SRC) / "PAPER_xstate_v3/exp_ii_scenario_budget_post.csv"),
                                          on=["state", "year"], suffixes=("", "_prod"))
        print("base scenarios vs production budget: theta match",
              int((bs.theta_dagger.fillna("-") == bs.theta_dagger_prod.fillna("-")).sum()), "/ 9",
              "| decision match", int(((bs.decision == bs.flat) | (bs.n_adm == 0)).sum()), "/ 9",
              "| max |D diff|", float((bs.D_dagger - bs.D_dagger_prod).abs().max()),
              "| max |gamma_min diff|", float((bs.gamma_min - bs.gamma_min_prod).abs().max()),
              "| v_H", bs.v_H.unique().tolist(), "vs", bs.v_H_prod.unique().tolist())


if __name__ == "__main__":
    main()
