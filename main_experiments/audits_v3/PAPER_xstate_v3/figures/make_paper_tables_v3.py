"""
v3 port (v3 protocol, 2026-09-02) of
  main_experiments/audits/PAPER_xstate/figures/make_paper_tables.py
to the audits_v3 / PAPER_xstate_v3 results (9 models x 3 states x 3 horizons = 81 cells).

Paper-ready deliverables:

  (T1) Layer-wise outcome table  — proves the three-layer audit separates
       failure modes (each cell attributed to the FIRST gate it fails).
  (T2) Attribute-level chi_H decomposition table + figure — shows the
       Layer-2 scenario loss is Commute_Mode-dominated.
  (T3) Layer-3 projection diagnostics table — feasibility, median R_3,
       normalized ESS (weight concentration) for the surviving valid models.
  (T4) NEW: horizon-validity table (tab:horizon-validity) from the certificate.

Outputs:
  figures/tables/tab_layerwise_outcomes.tex
  figures/tables/tab_chiH_decomposition.tex
  figures/tables/tab_layer3_diagnostics.tex
  figures/tables/tab_horizon_validity.tex
  figures/{png,pdf}/fig12_chiH_decomposition.{png,pdf}

Edits vs the v2 script (algorithms / formulas byte-identical):
  * paths -> audits_v3 / PAPER_xstate_v3 ; figures split into png/ pdf/ tables/
  * 9 uniform model tags + display names; l2dir uniform (no layer2_pwgtp_ split)
  * cell counts 72 -> 81 (and drift-cell count computed, not hard-coded)
  * LaTeX skeleton (caption / label / column headers / row labels) aligned to the
    manuscript's CURRENT tab:layerwise_outcomes, tab:chiH_decomposition,
    tab:layer3_diagnostics and tab:horizon-validity so the files are drop-in.
  * Layer-3 table 'survivors' block = models with >=1 valid cell (same rule as v2).

Run with the project env, from the repo root:
  python main_experiments/audits_v3/PAPER_xstate_v3/figures/make_paper_tables_v3.py
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[4])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(_PA.R(""))
AUD  = Path(_PA.R("/main_experiments/audits_v3"))
DATA = AUD / "PAPER_xstate_v3"
FIG  = DATA / "figures"
TAB  = FIG / "tables"
PNG  = FIG / "png"
PDF  = FIG / "pdf"
for _d in (TAB, PNG, PDF):
    _d.mkdir(parents=True, exist_ok=True)

# ───────────── style (match make_figures.py) ─────────────
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "grid.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42,
})

STATES = ["CA", "OH", "TX"]
YEARS  = [2018, 2021, 2024]
# Paper model order (strong -> weak): survivors first, then static-fidelity failers.
MODEL_ORDER = ["semapop_vae", "ttvae", "tabddpm", "bn", "wgan_zcr",
               "ctgan", "flowsyn", "borysov", "tvae"]
MODEL_DISPLAY = {
    "semapop_vae": "SemaPop-VAE+", "ttvae": "TTVAE", "tabddpm": "TabDDPM",
    "bn": "BN", "wgan_zcr": "WGAN-GP-ZCR", "ctgan": "CTGAN",
    "flowsyn": "FlowSyn", "borysov": "Borysov-VAE", "tvae": "TVAE",
}
# attribute labels as used in the manuscript's tab:chiH_decomposition
ATTR_DISPLAY = {
    "Commute_Mode": "Commute mode", "Household_Income_Class": "Household income",
    "Number_of_cars_of_household": "Household cars", "Age_Group": "Age",
    "Number_of_children": "Children", "Household_Size": "Household size",
    "Employment_status": "Employment",
}
# Layer-1 failure-mode vocabulary of the manuscript (marginal / dependency discrepancy)
FAIL_DISPLAY = {
    "invalid_mass": "invalid mass", "marg_TVD": "marginal discrepancy",
    "dep_TVD": "dependency discrepancy", "1-support_cov": "support coverage",
}
N_MODELS = len(MODEL_ORDER)
N_CELLS  = N_MODELS * len(STATES) * len(YEARS)      # 81

def dn(m):  return MODEL_DISPLAY.get(m, m)
def da(a):  return ATTR_DISPLAY.get(a, a.replace("_", " "))

def l2dir(s, m, y):
    return AUD / f"layer2_{m}_{s}_{y}"

def l3dir(s, m, y):
    for p in ("paperW", "paper"):
        d = AUD / f"layer3_{p}_{m}_{s}_{y}"
        if (d / "layer3_summary.json").exists():
            return d
    return None

cert = pd.read_csv(DATA / "xstate_certificate_all.csv")
cert = cert[cert["L1_loaded"] & cert["year"].notna()].copy()
cert["year"] = cert["year"].astype(int)
for _c in ("L1_accept", "L2_accept", "L3_accept", "VALID"):
    cert[_c] = cert[_c].astype(bool)
assert len(cert) == N_CELLS, f"expected {N_CELLS} cells, got {len(cert)}"


# ═══════════════════════════════════════════════════════════════════════
# (T1) Layer-wise outcome table — sequential-gate failure-mode separation
# ═══════════════════════════════════════════════════════════════════════
def table1_layerwise():
    rows = []
    for m in MODEL_ORDER:
        sub = cert[cert["model"] == m]
        n = len(sub)
        l1_rej = int((~sub["L1_accept"]).sum())
        l2_rej = int((sub["L1_accept"] & ~sub["L2_accept"]).sum())
        l3_rej = int((sub["L1_accept"] & sub["L2_accept"] & ~sub["L3_accept"]).sum())
        valid  = int(sub["VALID"].sum())
        # dominant L1 failure mode (among L1-rejected cells)
        fm = sub.loc[~sub["L1_accept"], "L1_failed"].dropna()
        if len(fm):
            toks = pd.Series([t for s in fm for t in str(s).split(",")])
            top = toks.value_counts().idxmax()
            l1mode = FAIL_DISPLAY.get(top, top)
        else:
            l1mode = "--"
        # first gate that rejects this model (characteristic failure layer)
        if valid == n:                       gate = "none (valid)"
        elif l1_rej >= max(l2_rej, l3_rej) and l1_rej > 0: gate = "L1"
        elif l2_rej > 0:                     gate = "L2"
        else:                                gate = "L3"
        rows.append(dict(model=m, n=n, l1_rej=l1_rej, l1mode=l1mode,
                         l2_rej=l2_rej, l3_rej=l3_rej, valid=valid, gate=gate))
    R = pd.DataFrame(rows)
    tot = dict(model="\\textbf{Total}", n=R.n.sum(), l1_rej=R.l1_rej.sum(),
               l1mode="--", l2_rej=R.l2_rej.sum(), l3_rej=R.l3_rej.sum(),
               valid=R.valid.sum(), gate="--")

    L = []
    L.append(r"\begin{table}[h] %htbp")
    L.append(r"\centering")
    L.append(rf"\caption{{Layer-wise audit outcomes across all \({N_MODELS}\times3\times3={N_CELLS}\) "
             r"model--state--horizon cells. Each cell is assigned to the first gate it fails.}")
    L.append(r"\label{tab:layerwise_outcomes}")
    L.append(r"\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.15}")
    L.append(r"\begin{tabular}{lcccccc}")
    L.append(r"\toprule")
    L.append(r"Model & Cells & \multicolumn{2}{c}{Layer 1} & Layer 2 & Layer 3 & Valid \\")
    L.append(r"\cmidrule(lr){3-4}")
    L.append(r" & ($n$) & rej. & failure mode & rej.\ & rej.\ & ($n$) \\")
    L.append(r"\midrule")
    for _, r in R.iterrows():
        L.append(f"{dn(r['model'])} & {r['n']} & {r['l1_rej']} & {r['l1mode']} "
                 f"& {r['l2_rej']} & {r['l3_rej']} & {r['valid']} \\\\")
    L.append(r"\midrule")
    L.append(f"{tot['model']} & {tot['n']} & {tot['l1_rej']} & {tot['l1mode']} "
             f"& {tot['l2_rej']} & {tot['l3_rej']} & \\textbf{{{tot['valid']}}} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    (TAB / "tab_layerwise_outcomes.tex").write_text("\n".join(L) + "\n")
    print("  wrote tables/tab_layerwise_outcomes.tex")
    print(R.to_string(index=False))
    print(f"  TOTAL: n={tot['n']} L1={tot['l1_rej']} L2={tot['l2_rej']} L3={tot['l3_rej']} valid={tot['valid']}")
    return R


# ═══════════════════════════════════════════════════════════════════════
# chi_H decomposition core — per (state,model,year) attribute contributions
#   chi_H = (1/K) sum_r sum_a chi_{H,a}^{(r)}   (Eq. chiH-ensemble; attribute_normalized)
#   Attribute contribution = per-realization attribute loss averaged over the K realizations
#   (2026-11-24: was 0.5*mean z_err_prior^2, i.e. chi at the ensemble-mean marginals).
# ═══════════════════════════════════════════════════════════════════════
def compute_chiH_decomp():
    rows = []
    for s in STATES:
        for m in MODEL_ORDER:
            for y in YEARS:
                pr = pd.read_csv(l2dir(s, m, y) / "layer2_per_realization_weights.csv")
                cols = [k for k in pr.columns if k.startswith("attribute_loss__")]
                c = pd.Series({k[len("attribute_loss__"):]: float(pr[k].mean()) for k in cols})
                d = {"state": s, "model": m, "year": y, "chi_H": float(c.sum())}
                for a, v in c.items():
                    d[a] = float(v)
                rows.append(d)
    df = pd.DataFrame(rows).fillna(0.0)
    df = df.merge(cert[["state", "model", "year", "L1_accept", "L2_accept", "VALID", "L2_chi_H"]],
                  on=["state", "model", "year"])
    # consistency check vs the certificate's chi_H (informational only)
    dev = float((df["chi_H"] - df["L2_chi_H"]).abs().max())
    print(f"  decomposition chi_H vs certificate L2_chi_H: max |diff| = {dev:.2e} over {len(df)} cells")
    return df


# ═══════════════════════════════════════════════════════════════════════
# (T2) Attribute-level chi_H decomposition table
# ═══════════════════════════════════════════════════════════════════════
def table2_decomp(df):
    attrs = [a for a in ATTR_DISPLAY if a in df.columns]
    share = df[attrs].div(df["chi_H"], axis=0)
    drift = df[df["L1_accept"] & ~df["L2_accept"]]            # pure scenario drift
    share_d = drift[attrs].div(drift["chi_H"], axis=0)

    order = (share.mean().sort_values(ascending=False)).index.tolist()
    L = []
    L.append(r"\begin{table}[h] %htbp")
    L.append(r"\centering")
    L.append(r"\caption{Attribute-level decomposition of the Layer-2 scenario loss "
             r"\(\overline{\chi}_H\). Shares are averaged over all cells and over "
             r"scenario-drift cells that pass Layer~1 but fail Layer~2.}")
    L.append(r"\label{tab:chiH_decomposition}")
    L.append(r"\small")
    L.append(r"\begin{tabular}{lcc}")
    L.append(r"\toprule")
    L.append(r"Attribute & \multicolumn{2}{c}{Share of \(\overline{\chi}_H\) (\%)} \\")
    L.append(r"\cmidrule(lr){2-3}")
    L.append(r" & All cells & Scenario-drift cells \\")
    L.append(r"\midrule")
    for a in order:
        ma = share[a].mean() * 100
        md = share_d[a].mean() * 100
        L.append(f"{da(a)} & {ma:.1f} & {md:.1f} \\\\")
    L.append(r"\midrule")
    L.append(f"Mean \\(\\overline{{\\chi}}_H\\) & {df['chi_H'].mean():.2f} & {drift['chi_H'].mean():.2f} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    (TAB / "tab_chiH_decomposition.tex").write_text("\n".join(L) + "\n")
    print("  wrote tables/tab_chiH_decomposition.tex")
    summ = pd.DataFrame({"all_%": share.mean() * 100,
                         "drift_%": share_d.mean() * 100}).loc[order].round(1)
    print(summ.to_string())
    print(f"  n_all={len(df)}  n_drift={len(drift)}  mean chi_H all={df['chi_H'].mean():.2f} drift={drift['chi_H'].mean():.2f}")
    return order


# ═══════════════════════════════════════════════════════════════════════
# (F12) chi_H decomposition figure — grouped bar of attribute shares
# ═══════════════════════════════════════════════════════════════════════
def figure_decomp(df, order):
    attrs = order
    share = df[attrs].div(df["chi_H"], axis=0).mean() * 100
    drift = df[df["L1_accept"] & ~df["L2_accept"]]
    share_d = drift[attrs].div(drift["chi_H"], axis=0).mean() * 100

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    x = np.arange(len(attrs)); w = 0.38
    bars1 = ax.bar(x - w/2, share.values,   w, label=f"All cells ($n{{=}}{len(df)}$)",
                   color="#9ecae1", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + w/2, share_d.values, w, label=f"Scenario-drift cells ($n{{=}}{len(drift)}$)",
                   color="#fc9272", edgecolor="black", linewidth=0.5)
    # highlight commute mode
    for bars, vals in ((bars1, share.values), (bars2, share_d.values)):
        bars[0].set_edgecolor("#67000d"); bars[0].set_linewidth(1.6)
    ax.set_xticks(x)
    ax.set_xticklabels([da(a) for a in attrs], rotation=30, ha="right")
    ax.set_ylabel(r"Share of $\overline{\chi}_H$ (%)")
    ax.set_title(r"Attribute decomposition of the Layer-2 scenario loss $\overline{\chi}_H$",
                 fontsize=9)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, max(share_d.max(), share.max()) * 1.18)
    for b in bars2:
        ax.annotate(f"{b.get_height():.0f}", (b.get_x()+b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(PNG / "fig12_chiH_decomposition.png", dpi=300, bbox_inches="tight")
    fig.savefig(PDF / "fig12_chiH_decomposition.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote png/fig12_chiH_decomposition.png, pdf/fig12_chiH_decomposition.pdf")


# ═══════════════════════════════════════════════════════════════════════
# (T3) Layer-3 projection diagnostics table
# ═══════════════════════════════════════════════════════════════════════
def table3_layer3():
    rows = []
    for s in STATES:
        for m in MODEL_ORDER:
            for y in YEARS:
                d = l3dir(s, m, y)
                if d is None:
                    continue
                j = json.loads((d / "layer3_summary.json").read_text())
                feas = float(j["feasibility_rate"])
                is_feas = feas >= 0.999
                dist = j.get("median_distortion", j.get("mean_distortion"))
                def g(k):  # coerce missing / None -> nan
                    v = j.get(k)
                    return float(v) if v is not None else np.nan
                rows.append(dict(
                    state=s, model=m, year=y, feas=feas, is_feas=is_feas,
                    # R_3 and weight-concentration diagnostics only defined where feasible
                    R3=(float(np.sqrt(dist / 2.0)) if (is_feas and dist is not None) else np.nan),
                    nESS=(g("mean_agent_nESS") if is_feas else np.nan),
                    topk=(g("mean_topk_mass")  if is_feas else np.nan),
                ))
    d3 = pd.DataFrame(rows)
    d3 = d3.merge(cert[["state", "model", "year", "VALID"]],
                  on=["state", "model", "year"])

    # Aggregate per model: feasibility over all 9 cells; R3/nESS/topk over feasible cells.
    agg = (d3.groupby("model")
             .agg(n=("feas", "size"),
                  n_feas=("is_feas", "sum"),
                  feas=("feas", "mean"),
                  R3_med=("R3", "median"),
                  nESS=("nESS", "mean"),
                  topk=("topk", "mean"))
             .reindex(MODEL_ORDER))

    L = []
    L.append(r"\begin{table}[htbp]")
    L.append(r"\centering")
    L.append(r"\caption{Layer-3 projection diagnostics by model. Feasibility, median "
             r"\(R_3=\sqrt{\mathrm{KL}/2}\), and agent-level nESS are computed over "
             r"feasible projected cells.}")
    L.append(r"\label{tab:layer3_diagnostics}")
    L.append(r"\small")
    L.append(r"\begin{tabular}{lccc}")
    L.append(r"\toprule")
    L.append(r"Model & Feasible & Median \(R_3\) & Agent nESS \\")
    L.append(r"\midrule")
    def fmt(x, p):  return "--" if pd.isna(x) else f"{x:.{p}f}"
    # Layer-1/2 survivors = models with >= 1 layer-wise valid cell (v3 certificate)
    survivors = {"semapop_vae", "ttvae", "tabddpm", "bn", "wgan_zcr"}
    prev_surv = True
    for m in MODEL_ORDER:
        if prev_surv and m not in survivors:
            L.append(r"\midrule")
            prev_surv = False
        r = agg.loc[m]
        L.append(f"\\texttt{{{dn(m)}}} & {int(r['n_feas'])}/{int(r['n'])} "
                 f"& {fmt(r['R3_med'],3)} & {fmt(r['nESS'],3)} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    (TAB / "tab_layer3_diagnostics.tex").write_text("\n".join(L) + "\n")
    print("  wrote tables/tab_layer3_diagnostics.tex")
    print(agg.round(3).to_string())
    return agg


# ═══════════════════════════════════════════════════════════════════════
# (T4) NEW — horizon-validity table (manuscript tab:horizon-validity)
# ═══════════════════════════════════════════════════════════════════════
BASE_YEAR = 2013
HORIZON_INTERP = {
    2018: "pre-COVID reference horizon",
    2021: "peak-COVID disruption",
    2024: "post-emergency partial recovery",
}
def table4_horizon():
    rows = []
    for y in YEARS:
        sub = cert[cert["year"] == y]
        rows.append(dict(year=y, H=y - BASE_YEAR, valid=int(sub["VALID"].sum()), n=len(sub)))
    R = pd.DataFrame(rows)
    L = []
    L.append(r"\begin{table}[htbp]")
    L.append(r"\centering")
    L.append(r"\caption{Layer-wise validity by deployment horizon. A cell is valid only if it")
    L.append(r"passes Layers~1--3.}")
    L.append(r"\label{tab:horizon-validity}")
    L.append(r"\begin{tabular}{lccc}")
    L.append(r"\toprule")
    L.append(r"Deployment year & Horizon & Valid cells & Interpretation \\")
    L.append(r"\midrule")
    for _, r in R.iterrows():
        hs = f"\\(H={r['H']}\\)"
        pad = " " if r["H"] < 10 else ""
        L.append(f"{r['year']} & {hs}{pad} & {r['valid']}/{r['n']} & {HORIZON_INTERP[int(r['year'])]} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    (TAB / "tab_horizon_validity.tex").write_text("\n".join(L) + "\n")
    print("  wrote tables/tab_horizon_validity.tex")
    print(R.to_string(index=False))
    return R


if __name__ == "__main__":
    print("== (T1) layer-wise outcomes ==")
    table1_layerwise()
    print("\n== chi_H decomposition ==")
    decomp = compute_chiH_decomp()
    print("\n== (T2) decomposition table ==")
    order = table2_decomp(decomp)
    print("\n== (F12) decomposition figure ==")
    figure_decomp(decomp, order)
    print("\n== (T3) layer-3 diagnostics ==")
    table3_layer3()
    print("\n== (T4) horizon validity ==")
    table4_horizon()
    print("\nDone.")
