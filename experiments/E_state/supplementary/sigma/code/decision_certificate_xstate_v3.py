"""
v3 cross-state decision certificate (v3, 2026-09-02).

Same gates/formulas as decision_certificate_xstate_paper.py (paper-faithful L1
Age_Group joint-TV, L2 t-quantile empirical-SE gate, |C_a|-weighted sigma norm,
decision = R_2^MA vs B0 only).  Only the audit source and the model set change:
  - reads experiments/E_state/audits_v3/  (hpo_v3 seed-42 K=50 ensembles, 13-attr build)
  - 9 primary models (uniform tags, same list in every state); --extras switches to
    the diagnostic rows (as-shipped configs, TabDDPM ml_eff, SemaPop-VAE+ M-U arm,
    TTVAE zero-context arm).
Outputs -> experiments/E_state/audits_v3/PAPER_xstate_v3[_extras]/
"""

from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[5])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import json
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(_PA.R(""))
BASE = ROOT / "experiments/E_state/audits_v3"
import sys as _sys
EXTRAS = "--extras" in _sys.argv
OUT_DIR = BASE / ("PAPER_xstate_v3_extras" if EXTRAS else "PAPER_xstate_v3")

# === Gate constants (paper sec 5.3) ===
ALPHA = (0.05, 0.05, 0.05)
KAPPA_1_MARG, KAPPA_1_DEP, KAPPA_1_SUPPORT, INVALID_MASS_TOL = 0.05, 0.10, 0.30, 0.005
KAPPA_2 = 5.0
TAU_DEFAULT = 1.0
B0 = 0.30

PRIMARY_MODELS = ["semapop_vae", "ttvae", "bn", "wgan_zcr", "ctgan", "flowsyn",
                  "borysov", "tabddpm", "tvae"]
EXTRA_MODELS = ["ctgan_defaults", "tvae_defaults", "borysov_defaults", "tabddpm_mleff",
                "semapop_vae_mu", "ttvae_prior"]
MODELS = EXTRA_MODELS if EXTRAS else PRIMARY_MODELS
STATE_MODELS = {st: list(MODELS) for st in ("CA", "TX", "OH")}
STATE_N_AGENTS = {"CA": 330_109, "OH": 64_112, "TX": 212_756}
YEARS = ["2018", "2021", "2024"]


def dkw_critical(alpha: float, n: int, n_categories: int = 2) -> float:
    factor = max(n_categories - 1, 1)
    return float(np.sqrt(np.log(2.0 * factor / alpha) / (2.0 * n)))


def _l2_dir(state, model, year):
    return BASE / f"layer2_{model}_{state}_{year}"


def load_layer1(state, model):
    # paper-faithful L1 output
    p = BASE / f"layer1_paper_{model}_{state}" / "layer1_ensemble_summary.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    return {
        "marg_TVD":     float(j["marginal_total_discrepancy_mean"]),
        "dep_TVD":      float(j["dependency_total_discrepancy_mean"]),
        "support_cov":  float(j["ensemble_support_coverage"]),
        "invalid_mass": float(j.get("invalid_mass_mean", 0.0)),
        "n_realizations": int(j["n_realizations"]),
    }


def load_layer2(state, model, year):
    d = _l2_dir(state, model, year)
    p = d / "layer2_summary.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    out = {
        "chi_H":     float(j["prior_mean_scenario_loss"]),
        "n_realizations": int(j["n_realizations"]),
        "n_targets": int(j["n_targets"]),
        "per_real_loss": None,
    }
    pr = d / "layer2_per_realization_weights.csv"
    if pr.exists():
        try:
            out["per_real_loss"] = pd.read_csv(pr)["scenario_loss"].to_numpy(float)
        except Exception:
            pass
    return out


def load_layer3(state, model, year):
    for prefix in ("layer3_paperW_", "layer3_paper_"):
        d = BASE / f"{prefix}{model}_{state}_{year}"
        if (d / "layer3_summary.json").exists():
            j = json.loads((d / "layer3_summary.json").read_text())
            pr = pd.read_csv(d / "layer3_per_realization_diagnostics.csv")
            attr_raw = pr.iloc[0].get("attribute_projected_losses_json", None)
            attr_proj = {}
            if isinstance(attr_raw, str) and attr_raw not in ("", "nan"):
                try:
                    attr_proj = {k: float(v) for k, v in json.loads(attr_raw).items()}
                except (TypeError, json.JSONDecodeError):
                    pass
            KL = j.get("mean_distortion")
            ness = j.get("mean_agent_nESS", j.get("mean_nESS"))
            return {
                "feasibility_rate": float(j["feasibility_rate"]),
                "KL": float(KL) if KL is not None else None,
                "nESS": float(ness) if ness is not None else None,
                "attr_proj": attr_proj,
            }
    return None


def compute_R2_MA(state, model, year, l2):
    """R_2^MA = (1/2) * ||sigma||_{2,w} * sqrt(2 chibar_H), ||.||_{2,w}=sqrt(sum_a sum_c |C_a| sigma^2)."""
    if l2 is None or l2.get("chi_H") is None:
        return None, None
    sigma_csv = _l2_dir(state, model, year) / "layer2_historical_sigma_summary.csv"
    if not sigma_csv.exists():
        return None, None
    try:
        sdf = pd.read_csv(sigma_csv)
        cnt = sdf.groupby("attr")["sigma"].transform("count").to_numpy(float)   # |C_a| per row
        sig2 = sdf["sigma"].to_numpy(float) ** 2
        norm_w = float(np.sqrt(np.sum(cnt * sig2)))
        r2ma = 0.5 * norm_w * float(np.sqrt(2.0 * float(l2["chi_H"])))
        return r2ma, norm_w
    except Exception:
        return None, None


# ---------- gates ----------
def layer1_test(l1, alpha, n_agents):
    if l1["invalid_mass"] > INVALID_MASS_TOL:
        return {"accept": False, "failed": ["invalid_mass"]}
    a = alpha / 3
    K = l1["n_realizations"]
    slack_marg = dkw_critical(a, n=K * n_agents, n_categories=10)
    slack_dep = dkw_critical(a, n=K * n_agents, n_categories=10)
    slack_supp = dkw_critical(a, n=n_agents, n_categories=1)
    tests = [
        ("marg_TVD",      l1["marg_TVD"],          KAPPA_1_MARG - slack_marg),
        ("dep_TVD",       l1["dep_TVD"],           KAPPA_1_DEP - slack_dep),
        ("1-support_cov", 1.0 - l1["support_cov"], KAPPA_1_SUPPORT - slack_supp),
    ]
    failed = [n for n, v, c in tests if v > c]
    return {"accept": len(failed) == 0, "failed": failed}


def layer2_test(l2, alpha):
    chibar = l2["chi_H"]
    losses = l2.get("per_real_loss")
    if losses is not None and len(losses) > 1:
        K = len(losses)
        emp_se = float(np.std(losses, ddof=1) / np.sqrt(K))
        eta2 = float(student_t.ppf(1 - alpha, df=K - 1)) * emp_se
    else:
        eta2 = 0.0
    cutoff = KAPPA_2 + eta2
    return {"accept": bool(chibar <= cutoff), "cutoff": cutoff, "eta2": eta2}


def layer3_test(l3, alpha, tau=TAU_DEFAULT):
    if not l3 or not l3["attr_proj"]:
        return {"accept": False, "failed_attrs": ["infeasible"]}
    cutoff = tau + 1e-3
    failed = [a for a, c in l3["attr_proj"].items() if c > cutoff]
    return {"accept": bool(l3["feasibility_rate"] >= 1 - alpha and not failed),
            "failed_attrs": failed}


def compute_risk_vector(l1, l2, l3):
    T1 = (l1["marg_TVD"] + l1["dep_TVD"] + (1.0 - l1["support_cov"])) if l1 else None  # paper T_1
    R2 = float(np.sqrt(2.0 * l2["chi_H"])) if l2 and l2["chi_H"] is not None else None
    R3 = float(np.sqrt(l3["KL"] / 2.0)) if l3 and l3.get("KL") is not None else None
    return T1, R2, R3


def main():
    rows = []
    for state, models in STATE_MODELS.items():
        n_agents = STATE_N_AGENTS[state]
        for model in models:
            l1 = load_layer1(state, model)
            if l1 is None:
                rows.append({"state": state, "model": model, "year": None,
                             "L1_loaded": False, "VALID": None})
                continue
            t1 = layer1_test(l1, ALPHA[0], n_agents)
            for year in YEARS:
                l2 = load_layer2(state, model, year)
                l3 = load_layer3(state, model, year)
                t2 = layer2_test(l2, ALPHA[1]) if l2 else None
                t3 = layer3_test(l3, ALPHA[2]) if l3 else None
                T1, R2, R3 = compute_risk_vector(l1, l2, l3)
                r2ma, norm_w = compute_R2_MA(state, model, year, l2)
                # first-failed gate
                if not t1["accept"]:
                    first_fail = "L1:" + ",".join(t1["failed"])
                elif t2 is None or not t2["accept"]:
                    first_fail = "L2"
                elif t3 is None or not t3["accept"]:
                    first_fail = "L3"
                else:
                    first_fail = ""
                valid = (t1["accept"] and t2 and t2["accept"] and t3 and t3["accept"])
                rows.append({
                    "state": state, "model": model, "year": year, "L1_loaded": True,
                    "L1_accept": t1["accept"], "L1_failed": ",".join(t1["failed"]),
                    "L1_marg": round(l1["marg_TVD"], 6), "L1_dep": round(l1["dep_TVD"], 6),
                    "L1_support": round(l1["support_cov"], 6), "L1_invalid": round(l1["invalid_mass"], 6),
                    "L2_accept": (t2["accept"] if t2 else None),
                    "L2_chi_H": (round(l2["chi_H"], 5) if l2 else None),
                    "L2_cutoff": (round(t2["cutoff"], 5) if t2 else None),
                    "L3_accept": (t3["accept"] if t3 else None),
                    "L3_KL": (round(l3["KL"], 6) if l3 and l3.get("KL") is not None else None),
                    "L3_nESS": (round(l3["nESS"], 4) if l3 and l3.get("nESS") is not None else None),
                    "L3_feasible": (round(l3["feasibility_rate"], 4) if l3 else None),
                    "first_fail": first_fail,
                    "VALID": bool(valid),
                    "T1": (round(T1, 5) if T1 is not None else None),
                    "R_2": (round(R2, 5) if R2 is not None else None),
                    "R_3": (round(R3, 5) if R3 is not None else None),
                    "R_2_MA": (round(r2ma, 5) if r2ma is not None else None),
                    "sigma_norm_w": (round(norm_w, 5) if norm_w is not None else None),
                })

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "xstate_certificate_all.csv", index=False)
    loaded = df[df["L1_loaded"] & df["year"].notna()].copy()
    loaded["VALID"] = loaded["VALID"].astype(bool)   # object dtype when some L1 rows are missing
    valid = loaded[loaded["VALID"]].copy()
    valid.to_csv(OUT_DIR / "xstate_valid_only.csv", index=False)

    # ---- headline summary ----
    print("=" * 90)
    print(f"O1 v3 CERTIFICATE ({'EXTRAS' if EXTRAS else 'PRIMARY'}): {len(MODELS)} models x 3 states x 3 horizons; cases loaded={len(loaded)}")
    print("=" * 90)
    n_cases = len(loaded)
    n_valid = int(loaded["VALID"].sum())
    print(f"Total cases: {n_cases} | VALID: {n_valid}")
    print("\nBy horizon (valid / cases):")
    for y in YEARS:
        sub = loaded[loaded["year"] == y]
        print(f"  {y}: {int(sub['VALID'].sum())} / {len(sub)}")
    print("\nBy model (valid / 9):")
    for m in MODELS:
        sub = loaded[loaded["model"] == m]
        print(f"  {m:18s}: {int(sub['VALID'].sum())} / {len(sub)}")
    print("\nFirst-failed-gate counts:")
    ff = loaded[~loaded["VALID"]]["first_fail"].apply(lambda s: s.split(":")[0])
    print(ff.value_counts().to_string())
    # layerwise: first fail L1 vs L2 vs L3
    n_l1 = int((ff == "L1").sum()); n_l2 = int((ff == "L2").sum()); n_l3 = int((ff == "L3").sum())
    print(f"  => L1 reject={n_l1}, L2 reject={n_l2}, L3 reject={n_l3}, valid={n_valid}")
    print("\nModel x first-failed gate (counts over the 9 state-horizon cases):")
    ff_full = loaded["first_fail"].apply(lambda s: s.split(":")[0] if s else "VALID")
    ct = pd.crosstab(loaded["model"], ff_full).reindex(MODELS).fillna(0).astype(int)
    print(ct.to_string())
    ct.to_csv(OUT_DIR / "xstate_model_by_gate.csv")
    print("\nL1 detail per (model,state):  marg / dep / support / invalid  [gates: marg<=0.05-eta, dep<=0.10-eta, 1-supp<=0.30-eta, inv<=0.005]")
    l1d = loaded.drop_duplicates(["state","model"])[["model","state","L1_marg","L1_dep","L1_support","L1_invalid","L1_failed"]]
    print(l1d.sort_values(["model","state"]).to_string(index=False))

    if len(valid):
        r2ma = valid["R_2_MA"].dropna()
        mx = float(r2ma.max()); arg = valid.loc[valid["R_2_MA"].idxmax()]
        verdict = "ADOPT" if mx <= B0 else "DEFER"
        print(f"\nDeployment decision (R_2^MA vs B0={B0}):")
        print(f"  max R_2^MA over valid = {mx:.4f}  at {arg['state']} x {arg['model']} x {arg['year']}")
        print(f"  flat-budget verdict: {verdict}")
        print(f"  R_2^MA range over valid = [{r2ma.min():.4f}, {mx:.4f}]")
        # per-model robustness
        rob = valid.groupby("model").agg(n_valid=("VALID", "size"),
                                         max_R2MA=("R_2_MA", "max")).sort_values("n_valid", ascending=False)
        rob.to_csv(OUT_DIR / "xstate_per_model_robustness.csv")
    print(f"\nSaved -> {OUT_DIR}/  (xstate_certificate_all.csv, xstate_valid_only.csv)")


if __name__ == "__main__":
    main()
