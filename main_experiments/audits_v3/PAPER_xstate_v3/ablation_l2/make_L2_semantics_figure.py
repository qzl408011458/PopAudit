"""[v3 port 2026-09-02: PAPER_xstate_v3; GOOD tier = ttvae/semapop_vae (old ttvae/tvae_llm+); docstring counts are from the v2 run]
Figure: Layer-2 measures scenario ABNORMALITY-vs-history, an axis orthogonal to
elapsed-drift / hold-out infidelity. Three panels:
 (a) chi_H vs horizon per prior -> NON-monotone: spikes at the 2021 COVID shock,
     recovers by 2024, model-independently (v3: 16/16 priors peak at 2021 and recover by 2024).
 (b) hold-out full-joint SRMSE_J vs horizon per prior -> ~monotone elapsed drift
     (worst at 2024, the horizon furthest from the 2013 base year; v3: 13/16 peak at 2024).
 (c) chi_H vs SRMSE_J scatter -> the two axes are only weakly associated (v3 within-prior mean r=+0.35; 2021 overlaps 2018/2024 on SRMSE_J),
     which is why hold-out fidelity cannot validate the Layer-2 compatibility screen.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
c = pd.read_csv(V2 / "xstate_certificate_all.csv"); c = c[c.L1_accept == True].copy()
c["year"] = c.year.astype(int); c["prior"] = c.state + "/" + c.model
e = pd.read_csv(HERE / "ensemble_results.csv"); e["prior"] = e.state + "/" + e.model
m = c.merge(e[["state", "model", "year", "SRMSE_J"]], on=["state", "model", "year"])
# 2026-11-23 revision: the strong/weak base-year grouping (GOOD set) is dropped; one colour per prior
# (6 Layer-1-passing priors, 3 state lines each). Panel titles are neutral descriptions.
PRIOR_ORDER = ["semapop_vae", "ttvae", "tabddpm", "bn", "wgan_zcr", "flowsyn"]
DISP = {"semapop_vae": "SemaPop-VAE+", "ttvae": "TTVAE", "tabddpm": "TabDDPM", "bn": "BN",
        "wgan_zcr": "WGAN-GP-ZCR", "flowsyn": "FlowSyn"}
PCOL = dict(zip(PRIOR_ORDER, ["#1f77b4", "#ff7f0e", "#8c564b", "#e377c2", "#7f7f7f", "#17becf"]))
YEARS = [2018, 2021, 2024]
KAPPA2 = 5.0

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4)); fig.subplots_adjust(wspace=0.30, bottom=0.22)

# ---- (a) chi_H vs horizon ----
for p, g in m.groupby("prior"):
    g = g.set_index("year").reindex(YEARS)
    col = PCOL[g.model.dropna().iloc[0]]
    ax[0].plot(range(3), g.L2_chi_H.values, "-o", color=col, alpha=0.8, ms=4, lw=1.2)
_gate = ax[0].axhline(KAPPA2, ls="--", color="k", lw=1, label=r"gate $\kappa_2=5$")   # in figure legend (in-axes text overlapped lines)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(YEARS)
ax[0].set_ylabel(r"Layer-2 mismatch  $\bar\chi_H$")
ax[0].set_title("(a) Layer-2 mismatch by horizon", fontsize=10)
_h = [ax[0].plot([], [], "-o", color=PCOL[k], ms=4, lw=1.2, label=DISP[k])[0] for k in PRIOR_ORDER]
fig.legend(handles=_h + [_gate], loc="lower center", bbox_to_anchor=(0.38, 0.02), ncol=7, fontsize=8,
           frameon=False, title="prior (one line per state)", title_fontsize=8)

# ---- (b) SRMSE_J vs horizon ----
for p, g in m.groupby("prior"):
    g = g.set_index("year").reindex(YEARS)
    col = PCOL[g.model.dropna().iloc[0]]
    ax[1].plot(range(3), g.SRMSE_J.values, "-o", color=col, alpha=0.8, ms=4, lw=1.2)
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(YEARS)
ax[1].set_ylabel(r"full-joint SRMSE  $\mathrm{SRMSE}_J$")
ax[1].set_title("(b) Realized-reference joint SRMSE by horizon", fontsize=10)

# ---- (c) scatter chi_H vs SRMSE_J ----
cmap = {2018: "#2ca02c", 2021: "#d62728", 2024: "#9467bd"}
for yr in YEARS:
    s = m[m.year == yr]
    ax[2].scatter(s.L2_chi_H, s.SRMSE_J, c=cmap[yr], s=42, alpha=0.85, edgecolor="w", lw=0.5, label=str(yr))
wr = m.groupby("prior").apply(lambda g: g.L2_chi_H.corr(g.SRMSE_J), include_groups=False).mean()
ax[2].set_xlabel(r"Layer-2 mismatch  $\bar\chi_H$")
ax[2].set_ylabel(r"$\mathrm{SRMSE}_J$")
# v3: title is data-driven (v2 hard-coded 'nearly uncorrelated' at r=-0.08; v3 gives r=+0.35)
ax[2].set_title("(c) Mismatch against joint SRMSE", fontsize=10)
ax[2].legend(fontsize=8, title="horizon")

for a in ax:
    for sp in ["top", "right"]:
        a.spines[sp].set_visible(False)

for d in [V2 / "figures/png", V2 / "figures/pdf"]:
    d.mkdir(parents=True, exist_ok=True)
fig.savefig(V2 / "figures/png/fig_L2_abnormality_vs_drift.png", dpi=150, bbox_inches="tight")
fig.savefig(V2 / "figures/pdf/fig_L2_abnormality_vs_drift.pdf", bbox_inches="tight")
# v3: no copy to repo root (old manuscript figure must not be overwritten)
print("wrote fig_L2_abnormality_vs_drift (png+pdf)")
print(f"within-prior mean corr(chi_H, SRMSE_J) = {wr:+.3f}")
