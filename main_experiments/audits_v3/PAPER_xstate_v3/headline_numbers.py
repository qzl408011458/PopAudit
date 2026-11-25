"""Headline numbers of Sections 6.1-6.3 that no other script prints (release helper).

Derived from the authors' one-off dump script (headline_g1fix.py) with the comparison against the
pre-fix backup removed; it prints, from the released outputs only: valid counts and first-failed
gates, Layer-3 KL / nESS / displacement coefficient sqrt(1/nESS - 1) ranges of the 23 valid cases
per horizon, the number of cases where the tau constraint binds, the post-projection scenario
budget table, the decision-relevant d_A table, the chi_H decomposition, the kappa_2 curve and B0
ladder, the transfer statistics, and the single-share example (top income class and work from
home of each selected prior).  The authors' run of the original script is
main_experiments/audits_v3/regen_logs_g1fix/24_headline_dump.log (NEW sections).

usage: python main_experiments/audits_v3/PAPER_xstate_v3/headline_numbers.py
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[3]))
import popaudit_paths as _PA  # noqa: E402

ROOT = Path(_PA.ROOT)
NEW = Path(_PA.R("/main_experiments/audits_v3"))
DISP = {"semapop_vae": "SemaPop-VAE+", "ttvae": "TTVAE", "tabddpm": "TabDDPM", "bn": "BN", "wgan_zcr": "WGAN-GP-ZCR",
        "ctgan": "CTGAN", "flowsyn": "FlowSyn", "borysov": "Borysov-VAE", "tvae": "TVAE"}
ORDER = list(DISP)


def cert(base, extras=False):
    c = pd.read_csv(base / ("PAPER_xstate_v3_extras" if extras else "PAPER_xstate_v3") / "xstate_certificate_all.csv")
    c["year"] = c.year.astype(int)
    return c


def p(*a):
    print(*a)


for tag, base in (("NEW", NEW),):
    c = cert(base)
    p(f"\n===== {tag} certificate (primary) =====")
    p("VALID", int(c.VALID.sum()), "/", len(c), "by year", c.groupby("year").VALID.sum().to_dict())
    p("by prior", {DISP[m]: int(c[c.model == m].VALID.sum()) for m in ORDER})
    p("by state", c.groupby("state").VALID.sum().to_dict())
    p("first_fail", c.first_fail.fillna("VALID").value_counts().to_dict())
    p("L1 pass & L2 fail", int((c.L1_accept & ~c.L2_accept).sum()))
    p("L2 pass & L3 fail", int((c.L1_accept & c.L2_accept & ~c.L3_accept).sum()))
    l1p = c[c.L1_accept]
    for y in (2018, 2021, 2024):
        g = l1p[l1p.year == y]
        p(f"  {y}: L1-pass {len(g)}, min chi_H {g.L2_chi_H.min():.3f} ({g.loc[g.L2_chi_H.idxmin(), ['state','model']].tolist()}), "
          f"min margin chi-cutoff {(g.L2_chi_H-g.L2_cutoff).min():.3f}")
    p("VALID cells:", sorted(f"{r.state}-{r.model}-{r.year}" for r in c[c.VALID].itertuples()))
    try:
        ce = cert(base, extras=True)
        p("EXTRAS VALID", int(ce.VALID.sum()), "/", len(ce), {m: int(ce[ce.model == m].VALID.sum()) for m in ce.model.unique()},
          "first_fail", ce.first_fail.fillna("VALID").value_counts().to_dict())
    except Exception as e:
        p("extras:", e)

for tag, base in (("NEW", NEW),):
    p(f"\n===== {tag} scenario budget (post-projection) =====")
    b = pd.read_csv(base / "PAPER_xstate_v3/exp_ii_scenario_budget_post.csv")
    p(b.round(5).to_string(index=False))
    p(pd.read_csv(base / "PAPER_xstate_v3/exp_ii_vH.csv").to_string(index=False))

p("\n===== tableE NEW (sorted by scenario) =====")
e = pd.read_csv(NEW / "PAPER_xstate_v3/figures/tables/tableE_decision_dA.csv")
p(e.sort_values(["year", "state", "d_A"])[["state", "year", "model", "d_A", "d_A_ref", "mu_out", "d_A_pre", "R_2_MA"]].round(5).to_string(index=False))

for tag, base in (("NEW", NEW),):
    t = (base / "PAPER_xstate_v3/figures/tables/tab_chiH_decomposition.tex").read_text()
    rows = re.findall(r"^(.+?) & ([\d.]+) & ([\d.]+) \\\\", t, re.M)
    p(f"\n{tag} chi decomposition (all / L1-pass-L2-fail):", [(a.strip(), x, y) for a, x, y in rows])

# Layer-3 diagnostics over VALID cells
p("\n===== Layer-3 diagnostics over VALID cells =====")
for tag, base in (("NEW", NEW),):
    c = cert(base); v = c[c.VALID].copy()
    v["disp"] = np.sqrt(1 / v.L3_nESS - 1)
    binding = []
    for r in v.itertuples():
        d = pd.read_csv(base / f"layer3_paperW_{r.model}_{r.state}_{r.year}/layer3_per_realization_diagnostics.csv")
        pl = json.loads(d.attribute_projected_losses_json.iloc[0])
        binding.append(max(pl.values()) >= 0.999)
    v["binding"] = binding
    p(f"{tag}: KL [{v.L3_KL.min():.4f}, {v.L3_KL.max():.4f}]  nESS [{v.L3_nESS.min():.4f}, {v.L3_nESS.max():.4f}]  "
      f"R3 [{v.R_3.min():.4f}, {v.R_3.max():.4f}]  binding {int(v.binding.sum())}/{len(v)}")
    for y in (2018, 2021, 2024):
        g = v[v.year == y]
        if len(g):
            p(f"   {y}: n={len(g)} KL<= {g.L3_KL.max():.4f} nESS>= {g.L3_nESS.min():.4f} disp [{g.disp.min():.3f},{g.disp.max():.3f}] R3 [{g.R_3.min():.4f},{g.R_3.max():.4f}]")
    allf = c[c.L3_feasible == 1]
    p(f"   all L3-run cells feasible {int((c.L3_feasible==1).sum())}/{int(c.L3_feasible.notna().sum())}")

for tag, base in (("NEW", NEW),):
    p(f"\n{tag} L3 diagnostics table:")
    p((base / "PAPER_xstate_v3/figures/tables/tab_layer3_diagnostics.tex").read_text().split("\\midrule", 1)[1].split("\\bottomrule")[0])

for tag, base in (("NEW", NEW),):
    s = base / "PAPER_xstate_v3/sensitivity"
    p(f"\n===== {tag} kappa2 curve / B0 ladder =====")
    p(pd.read_csv(s / "sensitivity_kappa2_curve.csv").to_string(index=False))
    p(pd.read_csv(s / "B0_ladder_postproj.csv").to_string(index=False))
    fl = s / "sensitivity_flips.csv"
    if fl.exists():
        f = pd.read_csv(fl); p("sensitivity_flips rows", len(f)); p(f.head(10).to_string(index=False))
    g = pd.read_csv(s / "sensitivity_grid.csv"); p("sensitivity_grid shape", g.shape); p(g.head(3).to_string(index=False))

for tag, base in (("NEW", NEW),):
    j = json.loads((base / "PAPER_xstate_v3/figures/tables/transfer_stats.json").read_text())
    p(f"\n{tag} transfer:", json.dumps({k: j[k] for k in ("n_valid_cells", "marginal")}, indent=None)[:900])
    p("   joint_65plus:", {k: round(v, 4) if isinstance(v, float) else v for k, v in j.get("joint_65plus", {}).items()})

# CA 2018 single-share example: highest income class of the selected prior
p("\n===== single-share example (Household_Income_Class==High) =====")
for tag, base in (("NEW", NEW),):
    b = pd.read_csv(base / "PAPER_xstate_v3/exp_ii_scenario_budget_post.csv")
    for r in b[b.theta_dagger.notna()].itertuples():
        t = pd.read_csv(base / f"layer3_paperW_{r.theta_dagger}_{r.state}_{r.year}/layer3_target_summary.csv").set_index("target")
        h = t.loc["Household_Income_Class==High"]
        w = t.loc["Commute_Mode==Worked_at_home"] if "Commute_Mode==Worked_at_home" in t.index else None
        p(f"{tag} {r.state} {r.year} {r.theta_dagger}: High proj {h.projected_mean:.4f} prior {h.prior_mean:.4f} realized {h.target_m:.4f} sigma {h.sigma:.4f} "
          f"z {(h.target_m-h.projected_mean)/h.sigma:.2f}" + (f" | WFH proj {w.projected_mean:.4f} realized {w.target_m:.4f}" if w is not None else ""))
