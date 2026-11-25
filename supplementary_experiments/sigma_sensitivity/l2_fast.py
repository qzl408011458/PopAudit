"""Fast, exact Layer-2 recomputation under a sigma variant.

The per-realization target shares s_j^(r) do not depend on sigma, and every term of
Eq. sigma-scale (|mean Delta|, sd Delta, sampling SE, H) is stored per target in the
paper's layer2_historical_sigma_summary.csv.  This applies the same arithmetic as
layer2_historical_sigma.apply_historical_dynamics_sigmas and
compute_realization_attribute_losses (attribute_normalized loss) and writes a Layer-2
directory in the same format, so sigma_sens.do_l3 (r* resolution, cells7, Layer 3)
runs unchanged.  Verified against full runs of the copied Layer-2 script
(verify_fast_vs_script.py).  Reads audits_v3/ (read-only).

usage: l2_fast.py --variant floor0025 mu2 ...
"""
import sys, json, math, argparse
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sigma_sens import VARIANTS, SRC, cases, vdir


def variant_sigma(sig, p):
    if p.get("y0"):
        md, sd = [], []
        for r in sig.itertuples():
            yrs = [int(x) for x in r.historical_years.split("|")]
            vals = np.array([float(x) for x in r.historical_values.split("|")])
            vals = vals[[y >= p["y0"] for y in yrs]]
            d = np.diff(vals); md.append(float(np.mean(d))); sd.append(float(np.std(d, ddof=1)))
        sig = sig.copy(); sig["historical_mean_delta"] = md; sig["historical_sd_delta"] = sd
    H = sig["historical_projection_horizon"].to_numpy(float) if p["H"] is None else np.full(len(sig), float(p["H"]))
    dyn = (p["mu"] * H * np.abs(sig["historical_mean_delta"].to_numpy(float))
           + p["lam"] * np.sqrt(H) * sig["historical_sd_delta"].to_numpy(float)
           + p["beta"] * sig["historical_sampling_se"].to_numpy(float))
    return np.maximum(p["floor"], dyn), H


def one(v, m, st, y):
    p = VARIANTS[v]
    src = Path(SRC) / f"layer2_{m}_{st}_{y}"
    sig = pd.read_csv(src / "layer2_historical_sigma_summary.csv")
    w = pd.read_csv(src / "layer2_per_realization_weights.csv")
    s_new, H = variant_sigma(sig, p)
    sig = sig.copy(); sig["sigma"] = s_new; sig["historical_projection_horizon"] = H
    sig["historical_sigma_floor"] = p["floor"]; sig["historical_drift_scale"] = p["mu"]
    sig["historical_vol_scale"] = p["lam"]; sig["historical_sampling_scale"] = p["beta"]
    loss = np.zeros(len(w))
    attr_cols = {}
    for a, g in sig.groupby("attr", sort=False):
        z2 = np.zeros(len(w))
        for r in g.itertuples():
            z2 += ((w[r.target].to_numpy(float) - float(r.target_m)) / float(r.sigma)) ** 2
        attr_cols[a] = 0.5 * z2 / len(g)
        loss += attr_cols[a]
    keep = ["realization_id", "n_agents"] + list(sig.target)
    out_w = w[keep].copy(); out_w["scenario_loss"] = loss
    for a, vals in attr_cols.items():
        out_w[f"attribute_loss__{a}"] = vals
    d = vdir(v) / f"layer2_{m}_{st}_{y}"; d.mkdir(parents=True, exist_ok=True)
    out_w.to_csv(d / "layer2_per_realization_weights.csv", index=False)
    sig.to_csv(d / "layer2_historical_sigma_summary.csv", index=False)
    pm = np.array([w[t].mean() for t in sig.target])
    ts = pd.DataFrame({"target": sig.target, "attr": sig.attr, "target_m": sig.target_m, "sigma": s_new,
                       "prior_mean": pm, "abs_err_prior": np.abs(pm - sig.target_m.to_numpy(float))})
    ts.to_csv(d / "layer2_target_summary.csv", index=False)
    json.dump({"n_realizations": int(len(w)), "n_targets": int(len(sig)),
               "prior_mean_scenario_loss": float(loss.mean()), "variant": v, "params": p,
               "source": "l2_fast"}, open(d / "layer2_summary.json", "w"), indent=1)
    return float(loss.mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", nargs="+", required=True); a = ap.parse_args()
    for v in a.variant:
        for c in cases():
            one(v, *c)
        print(v, "done")
