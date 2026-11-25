"""
Baseline reproduction: Beckman 1996 (SRMSE) + Bigi 2024 (multi-metric).
[v3 port, v3 protocol 2026-09-02: reads audits_v3 / PAPER_xstate_v3, 9 uniform model
 tags, 3 horizons -> 81 (state x model x year) cases; algorithms/thresholds unchanged.]

For each of the 81 (state × model × year) audit cases, compute:
  - SRMSE_marg                                  (Beckman 1996 marginal goodness-of-fit)
  - Bigi 4-metric vector = (SRMSE_marg, marg_TVD, dep_TVD, 1−support_coverage)

Decision rules applied:
  - Beckman: ACCEPT iff SRMSE_marg ≤ 0.30 (literature-standard threshold).
  - Bigi-strict: ACCEPT iff (SRMSE_marg ≤ 0.20) AND (marg_TVD ≤ 0.05)
                              AND (dep_TVD ≤ 0.10) AND (1−support_coverage ≤ 0.30).
  - Bigi-Pareto: per (state, year) cell, list the non-dominated subset on
                 (SRMSE_marg, marg_TVD, dep_TVD, 1−support_coverage).

Output: comparison table vs our framework's VALID flag.
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[4])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(_PA.R(""))
BASE = Path(_PA.R("/main_experiments/audits_v3"))
PAPER = BASE / "PAPER_xstate_v3"

MODELS = ["semapop_vae", "ttvae", "tabddpm", "bn", "wgan_zcr",
          "ctgan", "flowsyn", "borysov", "tvae"]           # strong -> weak, uniform in every state
STATE_MODELS = {
    "CA": list(MODELS),
    "OH": list(MODELS),
    "TX": list(MODELS),
}
YEARS = ["2018", "2021", "2024"]

# Standard literature thresholds for the two baselines
BECKMAN_SRMSE_THRESHOLD = 0.30   # common synthesis-paper "acceptance" cutoff
BIGI_SRMSE_MARG_THRESHOLD = 0.20
BIGI_MARG_TVD_THRESHOLD = 0.05
BIGI_DEP_TVD_THRESHOLD = 0.10
BIGI_SUPPORT_GAP_THRESHOLD = 0.30


def l1_path(state, model):
    return BASE / f"layer1_paper_{model}_{state}" / "layer1_ensemble_summary.json"


def l2_target_path(state, model, year):
    return BASE / f"layer2_{model}_{state}_{year}" / "layer2_target_summary.csv"


def compute_srmse_marg(target_summary_path):
    """
    Beckman 1996 SRMSE on the marginal target set.

    SRMSE = sqrt( (1/J) * sum_j (m_j - t_j)^2 ) / mean_j(t_j)

    where m_j = generated population marginal share, t_j = target marginal share,
    J = total number of marginal cells across all target attributes.
    """
    df = pd.read_csv(target_summary_path)
    m = df["prior_mean"].to_numpy(dtype=float)
    t = df["target_m"].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean((m - t) ** 2)))
    tbar = float(np.mean(t))
    if tbar <= 0:
        return float("nan")
    return rmse / tbar


def load_framework_verdict(case_csv):
    """Load existing framework VALID flag from xstate output."""
    df = pd.read_csv(case_csv)
    return df


def main():
    # Load existing framework verdicts
    fw = pd.read_csv(PAPER / "xstate_certificate_all.csv")
    fw["year"] = fw["year"].astype(int)

    rows = []
    for state, models in STATE_MODELS.items():
        for model in models:
            # Load L1 once per (state, model)
            p1 = l1_path(state, model)
            if not p1.exists():
                continue
            j1 = json.loads(p1.read_text())
            marg_TVD = float(j1["marginal_total_discrepancy_mean"])
            dep_TVD = float(j1["dependency_total_discrepancy_mean"])
            supp_cov = float(j1["ensemble_support_coverage"])
            supp_gap = 1.0 - supp_cov

            for year in YEARS:
                p2 = l2_target_path(state, model, year)
                if not p2.exists():
                    srmse = float("nan")
                else:
                    srmse = compute_srmse_marg(p2)

                # Framework verdict
                row_fw = fw[(fw["state"] == state) &
                            (fw["model"] == model) &
                            (fw["year"] == int(year))]
                fw_valid = bool(row_fw["VALID"].iloc[0]) if len(row_fw) else None
                fw_l1 = bool(row_fw["L1_accept"].iloc[0]) if len(row_fw) else None
                fw_R2MA = float(row_fw["R_2_MA"].iloc[0]) if len(row_fw) and pd.notna(row_fw["R_2_MA"].iloc[0]) else None

                # Beckman decision
                beckman_accept = (not np.isnan(srmse)) and (srmse <= BECKMAN_SRMSE_THRESHOLD)

                # Bigi-strict decision
                bigi_strict = (
                    (not np.isnan(srmse)) and (srmse <= BIGI_SRMSE_MARG_THRESHOLD) and
                    (marg_TVD <= BIGI_MARG_TVD_THRESHOLD) and
                    (dep_TVD <= BIGI_DEP_TVD_THRESHOLD) and
                    (supp_gap <= BIGI_SUPPORT_GAP_THRESHOLD)
                )

                rows.append({
                    "state": state, "model": model, "year": int(year),
                    "srmse_marg": round(srmse, 4) if not np.isnan(srmse) else None,
                    "marg_TVD": round(marg_TVD, 4),
                    "dep_TVD": round(dep_TVD, 4),
                    "supp_gap": round(supp_gap, 4),
                    "framework_VALID": fw_valid,
                    "framework_L1_accept": fw_l1,      # structural-only scorecard (tab:baseline-counts)
                    "framework_R2MA": round(fw_R2MA, 4) if fw_R2MA is not None else None,
                    "Beckman_ACCEPT": beckman_accept,
                    "Bigi_strict_ACCEPT": bigi_strict,
                })

    df = pd.DataFrame(rows)

    # Compute Bigi-Pareto per (state, year) cell on the 4-metric vector
    # (lower-is-better on all 4)
    df["Bigi_Pareto"] = False
    for (s, y), g in df.groupby(["state", "year"]):
        # Drop cases with NaN srmse from the comparison
        g_clean = g.dropna(subset=["srmse_marg"]).copy()
        if g_clean.empty:
            continue
        V = g_clean[["srmse_marg", "marg_TVD", "dep_TVD", "supp_gap"]].to_numpy()
        n = len(V)
        flags = np.ones(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if all(V[j] <= V[i] + 1e-12) and any(V[j] < V[i] - 1e-12):
                    flags[i] = False
                    break
        for idx, flag in zip(g_clean.index, flags):
            df.loc[idx, "Bigi_Pareto"] = bool(flag)

    print("=" * 110)
    print("HEAD-TO-HEAD: Framework vs Beckman 1996 vs Bigi-strict vs Bigi-Pareto")
    print("=" * 110)
    print(df.to_string(index=False))

    print()
    print("=" * 110)
    print("AGGREGATE: Acceptance counts (per method)")
    print("=" * 110)
    print(f"  Total cases evaluated:         {len(df)}")
    print(f"  Framework VALID:                {int(df['framework_VALID'].sum())}/{len(df)}")
    print(f"  Structural-only (L1 pass):      {int(df['framework_L1_accept'].sum())}/{len(df)}")
    print(f"  Beckman ACCEPT  (SRMSE ≤ 0.30): {int(df['Beckman_ACCEPT'].sum())}/{len(df)}")
    print(f"  Bigi-strict ACCEPT (all 4 thr): {int(df['Bigi_strict_ACCEPT'].sum())}/{len(df)}")
    print(f"  Bigi-Pareto (non-dominated):    {int(df['Bigi_Pareto'].sum())}/{len(df)}")

    print()
    print("=" * 110)
    print("DISAGREEMENT MATRIX (Framework vs each baseline)")
    print("=" * 110)
    for method_col in ["Beckman_ACCEPT", "Bigi_strict_ACCEPT", "Bigi_Pareto"]:
        agree_both_valid = ((df["framework_VALID"] == True) & (df[method_col] == True)).sum()
        agree_both_reject = ((df["framework_VALID"] == False) & (df[method_col] == False)).sum()
        fw_only_valid = ((df["framework_VALID"] == True) & (df[method_col] == False)).sum()
        baseline_only_valid = ((df["framework_VALID"] == False) & (df[method_col] == True)).sum()
        print(f"\n  vs {method_col}:")
        print(f"    Both VALID:                       {agree_both_valid}")
        print(f"    Both REJECT:                      {agree_both_reject}")
        print(f"    Framework VALID, baseline REJECT: {fw_only_valid}")
        print(f"    Baseline VALID, framework REJECT: {baseline_only_valid}")
        agreement = (agree_both_valid + agree_both_reject) / len(df) * 100
        print(f"    Agreement rate:                   {agreement:.1f}%")

    print()
    print("=" * 110)
    print("CASES WHERE FRAMEWORK and BECKMAN DIFFER")
    print("=" * 110)
    diff = df[df["framework_VALID"] != df["Beckman_ACCEPT"]].copy()
    if not diff.empty:
        print(diff[["state","model","year","srmse_marg","marg_TVD","dep_TVD","supp_gap",
                    "framework_VALID","Beckman_ACCEPT"]].to_string(index=False))
    else:
        print("  (none)")

    print()
    print("=" * 110)
    print("CASES WHERE FRAMEWORK and BIGI-strict DIFFER")
    print("=" * 110)
    diff = df[df["framework_VALID"] != df["Bigi_strict_ACCEPT"]].copy()
    if not diff.empty:
        print(diff[["state","model","year","srmse_marg","marg_TVD","dep_TVD","supp_gap",
                    "framework_VALID","Bigi_strict_ACCEPT"]].to_string(index=False))
    else:
        print("  (none)")

    # Save to PAPER_xstate_v3
    paper_out = PAPER / "baselines"
    paper_out.mkdir(parents=True, exist_ok=True)
    df.to_csv(paper_out / "baseline_comparison.csv", index=False)
    print(f"\nSaved → {paper_out}/baseline_comparison.csv")


if __name__ == "__main__":
    main()
