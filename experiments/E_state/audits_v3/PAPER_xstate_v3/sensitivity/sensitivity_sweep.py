"""
Sensitivity sweep (v3 / v3 protocol, 2026-09-02) — alpha x kappa_1_support x kappa_2 grid.

Port of experiments/E_state/audits/PAPER_xstate/sensitivity/sensitivity_sweep.py to the
audits_v3 outputs (9 uniform model tags x 3 states x 3 horizons = 81 cells, K=50).
Re-evaluates the layered-test verdicts under each parameter combo; audit outputs are
reused, only the test cutoffs change.  Grid (manuscript sec 4 / 6.4 / App. D.5):
    alpha in {0.01, 0.05, 0.10} x kappa_1_supp in {0.20,...,0.40} x kappa_2 in {2, 5, 10, 20}
Fixed: kappa_1_marg=0.05, kappa_1_dep=0.10, kappa_1_inv=0.005, tau_a=1.0, B0=0.30.

Gate formulas mirror decision_certificate_xstate_v3.py (paper-faithful): L1 uses the
DKW slack as an UPPER confidence bound (kappa - slack), L2 uses the Student-t quantile
times the empirical SE of the per-realization chi_H^(r), and the budget quantity is
R_2^MA = 0.5 * ||sigma||_{2,w} * sqrt(2 chibar_H) with the |C_a|-weighted sigma norm.
(The May-2026 script pre-dated the July bug fixes and used kappa+slack, a normal-z
analytic SE and an unweighted sigma norm; those are NOT reproduced here so that the
baseline grid point reproduces the v3 certificate exactly.)

Outputs (OUTDIR = audits_v3/PAPER_xstate_v3/sensitivity/):
  sensitivity_grid.csv          one row per (alpha, kappa_1_support, kappa_2)
  sensitivity_flips.csv         VALID flips vs the baseline (0.05, 0.30, 5.0)
  sensitivity_cases.csv         one row per (state, model, year) with raw stats
  sensitivity_kappa2_curve.csv  valid count vs kappa_2 at baseline alpha/kappa_1 (figF panel b)
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[5])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(_PA.R(""))
BASE = ROOT / "experiments/E_state/audits_v3"
OUTDIR = BASE / "PAPER_xstate_v3" / "sensitivity"   # canonical output (read by figures + summary)
CERT_VALID = BASE / "PAPER_xstate_v3" / "xstate_valid_only.csv"

MODELS = ["semapop_vae", "ttvae", "bn", "wgan_zcr", "ctgan", "flowsyn",
          "borysov", "tabddpm", "tvae"]
STATE_MODELS = {st: list(MODELS) for st in ("CA", "OH", "TX")}
STATE_N_AGENTS = {"CA": 330_109, "OH": 64_112, "TX": 212_756}
YEARS = ["2018", "2021", "2024"]

# Fixed (non-swept) tolerances
KAPPA_1_MARG = 0.05
KAPPA_1_DEP = 0.10
INVALID_MASS_TOL = 0.005
TAU_DEFAULT = 1.0
B0 = 0.30

BASELINE = (0.05, 0.30, 5.0)
GRID_ALPHA = [0.01, 0.05, 0.10]
GRID_K1S = [0.20, 0.25, 0.30, 0.35, 0.40]
GRID_K2 = [2.0, 5.0, 10.0, 20.0]
K2_CURVE = [2.0, 3.5, 4.0, 5.0, 6.0, 7.0, 10.0, 20.0]   # superset of make_figures_v2 figF(b) grid


def dkw_critical(alpha, n, n_categories=2):
    factor = max(n_categories - 1, 1)
    return float(np.sqrt(np.log(2.0 * factor / alpha) / (2.0 * n)))


def l1_path(state, model):
    return BASE / f"layer1_paper_{model}_{state}" / "layer1_ensemble_summary.json"


def l2_dir(state, model, year):
    return BASE / f"layer2_{model}_{state}_{year}"


def l3_dir(state, model, year):
    for d in [BASE / f"layer3_paperW_{model}_{state}_{year}",
              BASE / f"layer3_paper_{model}_{state}_{year}"]:
        if (d / "layer3_summary.json").exists():
            return d
    return None


def load_all_cases():
    rows = []
    for state, models in STATE_MODELS.items():
        for model in models:
            for year in YEARS:
                row = {"state": state, "model": model, "year": year}
                p1 = l1_path(state, model)
                if p1.exists():
                    j = json.loads(p1.read_text())
                    row["L1_marg"] = float(j["marginal_total_discrepancy_mean"])
                    row["L1_dep"] = float(j["dependency_total_discrepancy_mean"])
                    row["L1_supp"] = float(j["ensemble_support_coverage"])
                    row["L1_invalid_mass"] = float(j.get("invalid_mass_mean", 0.0))
                    row["L1_K"] = int(j["n_realizations"])
                d2 = l2_dir(state, model, year)
                p2 = d2 / "layer2_summary.json"
                if p2.exists():
                    j = json.loads(p2.read_text())
                    row["L2_chi_H"] = float(j["prior_mean_scenario_loss"])
                    row["L2_K"] = int(j["n_realizations"])
                    row["L2_J"] = int(j["n_targets"])
                    pw = d2 / "layer2_per_realization_weights.csv"
                    if pw.exists():
                        losses = pd.read_csv(pw)["scenario_loss"].to_numpy(float)
                        if len(losses) > 1:
                            row["L2_K"] = int(len(losses))
                            row["L2_emp_se"] = float(np.std(losses, ddof=1) / np.sqrt(len(losses)))
                sp = d2 / "layer2_historical_sigma_summary.csv"
                if sp.exists():
                    sdf = pd.read_csv(sp)
                    sig2 = sdf["sigma"].to_numpy(float) ** 2
                    cnt = sdf.groupby("attr")["sigma"].transform("count").to_numpy(float)  # |C_a| per row
                    row["sigma_l2_norm"] = float(np.sqrt(np.sum(sig2)))
                    row["sigma_l2_norm_w"] = float(np.sqrt(np.sum(cnt * sig2)))
                d3 = l3_dir(state, model, year)
                if d3:
                    j = json.loads((d3 / "layer3_summary.json").read_text())
                    row["L3_feasibility_rate"] = float(j["feasibility_rate"])
                    row["L3_KL"] = j.get("mean_distortion")
                    pr = pd.read_csv(d3 / "layer3_per_realization_diagnostics.csv")
                    apr = pr.iloc[0].get("attribute_projected_losses_json", "")
                    try:
                        attr_proj = json.loads(apr) if isinstance(apr, str) and apr not in ("", "nan") else {}
                    except (TypeError, json.JSONDecodeError):
                        attr_proj = {}
                    row["L3_max_chi_a"] = max(attr_proj.values()) if attr_proj else float("inf")
                    row["L3_n_attrs"] = len(attr_proj) if attr_proj else 7
                rows.append(row)
    return pd.DataFrame(rows)


def evaluate(cases, alpha, kappa_1_supp, kappa_2, tau=TAU_DEFAULT):
    """Apply test cutoffs with overridden (alpha, kappa_1_supp, kappa_2). Return df with VALID + R_2^MA."""
    out = cases.copy()

    def l1_t(r):
        if pd.isna(r.get("L1_K")):
            return False, "no_l1"
        if r["L1_invalid_mass"] > INVALID_MASS_TOL:
            return False, "invalid_mass"
        alpha_each = alpha / 3
        K = r["L1_K"]
        N = STATE_N_AGENTS[r["state"]]
        slack_marg = dkw_critical(alpha_each, n=K * N, n_categories=10)
        slack_dep = dkw_critical(alpha_each, n=K * N, n_categories=10)
        slack_supp = dkw_critical(alpha_each, n=N, n_categories=1)
        failed = []
        if r["L1_marg"] > KAPPA_1_MARG - slack_marg: failed.append("marg")
        if r["L1_dep"] > KAPPA_1_DEP - slack_dep: failed.append("dep")
        if (1.0 - r["L1_supp"]) > kappa_1_supp - slack_supp: failed.append("supp")
        return (len(failed) == 0), (",".join(failed) if failed else "")

    def l2_t(r):
        if pd.isna(r.get("L2_chi_H")):
            return False
        if pd.notna(r.get("L2_emp_se")):
            K = int(r["L2_K"])
            eta2 = float(student_t.ppf(1.0 - alpha, df=K - 1)) * float(r["L2_emp_se"])
        else:
            eta2 = 0.0
        cutoff = kappa_2 + eta2
        return r["L2_chi_H"] <= cutoff

    def l3_t(r):
        if pd.isna(r.get("L3_feasibility_rate")):
            return False
        feasible = r["L3_feasibility_rate"] >= 1.0 - alpha
        chi_ok = r.get("L3_max_chi_a", float("inf")) <= tau + 1e-3
        return feasible and chi_ok

    l1_results = out.apply(l1_t, axis=1, result_type="expand")
    out["L1_accept"] = l1_results[0]
    out["L1_failed"] = l1_results[1]
    out["L2_accept"] = out.apply(l2_t, axis=1)
    out["L3_accept"] = out.apply(l3_t, axis=1)
    out["VALID"] = out["L1_accept"] & out["L2_accept"] & out["L3_accept"]
    out["R_2_MA"] = out.apply(
        lambda r: 0.5 * r["sigma_l2_norm_w"] * np.sqrt(2.0 * r["L2_chi_H"])
        if pd.notna(r.get("sigma_l2_norm_w")) and pd.notna(r.get("L2_chi_H")) else None,
        axis=1,
    )
    return out


def _summarize(v):
    """Budget verdict on the VALID subset v (R_2^MA vs B0)."""
    if len(v) and pd.notna(v["R_2_MA"].max()):
        mx = float(v["R_2_MA"].max())
        verdict = mx <= B0
        arg = v.loc[v["R_2_MA"].idxmax()]
        arg_str = f"{arg['state']}×{arg['model']}×{arg['year']}"
    else:
        mx = float("nan"); verdict = None; arg_str = ""
    return mx, verdict, arg_str


def sweep():
    cases = load_all_cases()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cases.to_csv(OUTDIR / "sensitivity_cases.csv", index=False)
    print(f"Loaded {len(cases)} cases (state × model × year combos)")
    print(f"  CA: {(cases['state']=='CA').sum()}, OH: {(cases['state']=='OH').sum()}, TX: {(cases['state']=='TX').sum()}")

    grid_rows = []
    flip_rows = []
    baseline_ev = evaluate(cases, *BASELINE)
    baseline_valid_set = set(
        (r["state"], r["model"], r["year"])
        for _, r in baseline_ev[baseline_ev["VALID"]].iterrows()
    )
    # consistency check against the v3 certificate
    if CERT_VALID.exists():
        cv = pd.read_csv(CERT_VALID)
        cert_set = set((r["state"], r["model"], str(r["year"])) for _, r in cv.iterrows())
        if cert_set == baseline_valid_set:
            print(f"Baseline VALID set matches xstate_valid_only.csv ({len(cert_set)} cells)")
        else:
            print(f"WARNING: baseline VALID set differs from certificate: "
                  f"sweep-only={sorted(baseline_valid_set - cert_set)} cert-only={sorted(cert_set - baseline_valid_set)}")

    for alpha in GRID_ALPHA:
        for k1s in GRID_K1S:
            for k2 in GRID_K2:
                ev = evaluate(cases, alpha, k1s, k2)
                v = ev[ev["VALID"]]
                mx, verdict, arg_str = _summarize(v)
                grid_rows.append({
                    "alpha": alpha, "k1_supp": k1s, "k2": k2,
                    "n_valid_total": int(len(v)),
                    "n_valid_CA": int((v["state"] == "CA").sum()),
                    "n_valid_OH": int((v["state"] == "OH").sum()),
                    "n_valid_TX": int((v["state"] == "TX").sum()),
                    "n_valid_2018": int((v["year"] == "2018").sum()),
                    "n_valid_2021": int((v["year"] == "2021").sum()),
                    "n_valid_2024": int((v["year"] == "2024").sum()),
                    "max_R2MA": round(mx, 4) if not np.isnan(mx) else None,
                    "max_R2MA_arg": arg_str,
                    "verdict_ADOPT": verdict,
                })
                # Identify VALID flips vs baseline
                cur_valid_set = set(
                    (r["state"], r["model"], r["year"])
                    for _, r in v.iterrows()
                )
                added = cur_valid_set - baseline_valid_set
                removed = baseline_valid_set - cur_valid_set
                for s, m, y in added:
                    flip_rows.append({"alpha": alpha, "k1_supp": k1s, "k2": k2,
                                       "direction": "+VALID", "state": s, "model": m, "year": y})
                for s, m, y in removed:
                    flip_rows.append({"alpha": alpha, "k1_supp": k1s, "k2": k2,
                                       "direction": "-VALID", "state": s, "model": m, "year": y})

    gdf = pd.DataFrame(grid_rows)
    fdf = pd.DataFrame(flip_rows, columns=["alpha", "k1_supp", "k2", "direction", "state", "model", "year"])
    gdf.to_csv(OUTDIR / "sensitivity_grid.csv", index=False)
    fdf.to_csv(OUTDIR / "sensitivity_flips.csv", index=False)

    # kappa_2 curve at baseline alpha / kappa_1 (figF panel b)
    curve_rows = []
    for k2 in K2_CURVE:
        ev = evaluate(cases, BASELINE[0], BASELINE[1], k2)
        v = ev[ev["VALID"]]
        mx, verdict, arg_str = _summarize(v)
        curve_rows.append({
            "k2": k2, "n_valid_total": int(len(v)),
            "n_valid_CA": int((v["state"] == "CA").sum()),
            "n_valid_OH": int((v["state"] == "OH").sum()),
            "n_valid_TX": int((v["state"] == "TX").sum()),
            "n_valid_2018": int((v["year"] == "2018").sum()),
            "n_valid_2021": int((v["year"] == "2021").sum()),
            "n_valid_2024": int((v["year"] == "2024").sum()),
            "max_R2MA": round(mx, 4) if not np.isnan(mx) else None,
            "max_R2MA_arg": arg_str,
            "verdict_ADOPT": verdict,
        })
    cdf = pd.DataFrame(curve_rows)
    cdf.to_csv(OUTDIR / "sensitivity_kappa2_curve.csv", index=False)
    return cases, gdf, fdf, cdf, baseline_valid_set


if __name__ == "__main__":
    cases, gdf, fdf, cdf, baseline_set = sweep()
    print(f"\nGrid sweep: {len(gdf)} param combos × {len(cases)} cases each")
    print(f"Baseline (α=0.05, κ1_supp=0.30, κ2=5.0): {len(baseline_set)} valid cases")
    print(f"\n=== Verdict counts (R_2^MA vs B0={B0}) ===")
    print(f"  ADOPT: {(gdf['verdict_ADOPT']==True).sum()}/{len(gdf)} combos")
    print(f"  DEFER: {(gdf['verdict_ADOPT']==False).sum()}/{len(gdf)} combos")
    print(f"\n=== Max R_2^MA range across sweep ===")
    base_row = gdf[(gdf['alpha']==0.05)&(gdf['k1_supp']==0.30)&(gdf['k2']==5.0)].iloc[0]
    print(f"  min: {gdf['max_R2MA'].min():.4f}  max: {gdf['max_R2MA'].max():.4f}  baseline: {base_row['max_R2MA']:.4f} at {base_row['max_R2MA_arg']}")
    print(f"  n_valid range: {gdf['n_valid_total'].min()}..{gdf['n_valid_total'].max()}")
    print(f"\n=== Invariance checks ===")
    for col in ["alpha", "k1_supp", "k2"]:
        piv = gdf.groupby(col)["n_valid_total"].agg(["min", "max"])
        print(f"  n_valid by {col}:\n{piv.to_string()}")
    print(f"\n=== kappa_2 curve (alpha=0.05, kappa_1_supp=0.30) ===")
    print(cdf.to_string(index=False))
    print(f"\n=== Flips summary ===")
    print(f"  Total flip events (combo × case): {len(fdf)}")
    if len(fdf):
        print(fdf.groupby(["model", "state", "year", "direction"]).size().sort_values(ascending=False).head(20))
    # side checks on the non-swept tolerances (tau, B0) at baseline gates
    ev = evaluate(cases, *BASELINE)
    l12 = ev[ev["L1_accept"] & ev["L2_accept"]]
    print(f"\n=== tau side-check (cells passing L1&L2 at baseline: {len(l12)}) ===")
    print(f"  max_a chi_a^proj over these cells: min={l12['L3_max_chi_a'].min():.4f} max={l12['L3_max_chi_a'].max():.4f}")
    for tau in [0.5, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0]:
        evt = evaluate(cases, *BASELINE, tau=tau)
        print(f"  tau={tau:<5}: n_valid={int(evt['VALID'].sum())}")
    v = ev[ev["VALID"]]
    print(f"\n=== B0 side-check (baseline valid set, n={len(v)}) ===")
    print(f"  R_2^MA over valid: min={v['R_2_MA'].min():.4f} max={v['R_2_MA'].max():.4f}; ADOPT iff B0 >= {v['R_2_MA'].max():.4f}")
    for b0 in [0.30, 0.50, 0.75, 1.0]:
        print(f"  B0={b0}: {int((v['R_2_MA'] <= b0).sum())}/{len(v)} valid cells under budget; uniform verdict={'ADOPT' if v['R_2_MA'].max() <= b0 else 'DEFER'}")
    print(f"\nSaved -> {OUTDIR}/ (sensitivity_grid.csv, sensitivity_flips.csv, sensitivity_cases.csv, sensitivity_kappa2_curve.csv)")
