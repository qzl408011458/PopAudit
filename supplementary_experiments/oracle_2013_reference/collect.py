"""Collect the oracle-reference table and the production context (selected priors, D-dagger)."""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[2])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, _PA.R("/main_experiments/audits_v3/PAPER_xstate_v3"))
sys.path.insert(0, _PA.R("/main_experiments"))
from dA_union import dA_union_l3, dA_union_case
import decision_certificate_xstate_v3 as C
OUT = Path(__file__).parent
AUD = Path(_PA.R("/main_experiments/audits_v3"))
STATES, YEARS = ["CA", "OH", "TX"], ["2018", "2021", "2024"]
ATTRS = ["Age_Group", "Household_Size", "Number_of_children", "Number_of_cars_of_household",
         "Employment_status", "Household_Income_Class", "Commute_Mode"]
cert = pd.read_csv(AUD / "PAPER_xstate_v3/xstate_certificate_all.csv")
cert["year"] = cert["year"].astype(str)

rows = []
for var in ["full", "test"]:
    for st in STATES:
        for y in YEARS:
            l2d = OUT / f"runs/layer2_{var}_{st}_{y}"; l3d = OUT / f"runs/layer3_{var}_{st}_{y}"
            j2 = json.load(open(l2d / "layer2_summary.json"))
            pr = pd.read_csv(l2d / "layer2_per_realization_weights.csv").iloc[0]
            j3 = json.load(open(l3d / "layer3_summary.json"))
            d3 = pd.read_csv(l3d / "layer3_per_realization_diagnostics.csv").iloc[0]
            ap = json.loads(d3["attribute_projected_losses_json"])
            dd = dA_union_l3(l3d, st, y)
            l3_ok = (j3["feasibility_rate"] >= 0.95) and max(ap.values()) <= 1.0 + 1e-3
            r = dict(variant=var, state=st, year=y, n=int(pr["n_agents"]),
                     chi_H=j2["prior_mean_scenario_loss"], chi_le_5=j2["prior_mean_scenario_loss"] <= 5.0,
                     **{f"chi_{a}": float(pr[f"attribute_loss__{a}"]) for a in ATTRS},
                     L3_feasible=j3["feasibility_rate"], L3_max_attr_proj=max(ap.values()), L3_gate=l3_ok,
                     KL=j3["mean_distortion"], nESS=j3["mean_agent_nESS"],
                     D=dd["d_A"], D_ref=dd["d_A_ref"], mu_out=dd["mu_out"],
                     R2_MA=None)
            # R2^MA as the certificate computes it (|C_a|-weighted sigma norm, chi of the single realization)
            sdf = pd.read_csv(l2d / "layer2_historical_sigma_summary.csv")
            cnt = sdf.groupby("attr")["sigma"].transform("count").to_numpy(float)
            r["R2_MA"] = 0.5 * float(np.sqrt(np.sum(cnt * sdf["sigma"].to_numpy(float) ** 2))) * np.sqrt(2 * r["chi_H"])
            rows.append(r)
orc = pd.DataFrame(rows)
orc.to_csv(OUT / "a9_oracle_all.csv", index=False)

# production context per scenario
ctx = []
for st in STATES:
    for y in YEARS:
        c = cert[(cert.state == st) & (cert.year == y)].copy()
        c["D"] = [dA_union_case(m, st, y)["d_A"] for m in c.model]
        val = c[c.VALID == True]
        l1p = c[c.L1_accept == True]
        dag = val.loc[val.D.idxmin()] if len(val) else None
        ctx.append(dict(state=st, year=y, cutoff_range=f"{c.L2_cutoff.min():.3f}-{c.L2_cutoff.max():.3f}",
                        n_valid=len(val), theta_dagger=(dag.model if dag is not None else ""),
                        D_dagger=(dag.D if dag is not None else np.nan),
                        chi_dagger=(dag.L2_chi_H if dag is not None else np.nan),
                        D_valid_range=(f"{val.D.min():.4f}-{val.D.max():.4f}" if len(val) else ""),
                        chi_min_L1pass=l1p.L2_chi_H.min(), model_chi_min_L1pass=l1p.loc[l1p.L2_chi_H.idxmin(), "model"],
                        D_min_L1pass=l1p.D.min(), model_D_min_L1pass=l1p.loc[l1p.D.idxmin(), "model"],
                        chi_min_all=c.L2_chi_H.min(), D_min_all=c.D.min()))
ctx = pd.DataFrame(ctx); ctx.to_csv(OUT / "a9_production_context.csv", index=False)

m = orc[orc.variant == "full"].merge(orc[orc.variant == "test"][["state", "year", "chi_H", "D", "KL", "nESS"]],
                                     on=["state", "year"], suffixes=("", "_test")).merge(ctx, on=["state", "year"])
m["D_minus_Ddagger"] = m["D"] - m["D_dagger"]
cols = ["state", "year", "chi_H", "chi_H_test", "chi_Commute_Mode", "chi_Number_of_cars_of_household",
        "chi_Household_Income_Class", "chi_le_5", "L3_feasible", "L3_max_attr_proj", "L3_gate", "KL", "KL_test",
        "nESS", "nESS_test", "D", "D_test", "mu_out", "R2_MA", "n_valid", "theta_dagger", "chi_dagger", "D_dagger",
        "D_minus_Ddagger", "chi_min_L1pass", "model_chi_min_L1pass", "D_min_L1pass", "model_D_min_L1pass"]
m[cols].to_csv(OUT / "a9_oracle_table.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(m[cols].round(4).to_string(index=False))
print()
print(orc[["variant", "state", "year", "n"] + [f"chi_{a}" for a in ATTRS]].round(3).to_string(index=False))
