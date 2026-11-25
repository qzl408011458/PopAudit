"""LaTeX table of D(tau), nESS(tau), displacement for SemaPop-VAE+ (the prior selected at tau_a=1) in the three
2024 scenarios with admitted priors, plus the scenario-minimum D, and a D(tau) figure with every admitted prior.
g1fix re-run (2026-11-24): California 2024 is the only deferred-after-admission scenario; Ohio and Texas 2024 deploy.
D = d_A on the category union (dA_union.py), from tau_ladder_all.csv / tau_scenario_selected.csv (collect.py)."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
H = Path(__file__).resolve().parent; d = pd.read_csv(H/"tau_ladder_all.csv")
cr = pd.read_csv(H/"tau_crossing_B0.csv").set_index("cell"); sc = pd.read_csv(H/"tau_scenario_selected.csv")
mc = pd.read_csv(H/"tau_scenario_crossing.csv").set_index("state")
T = [1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001]
L = [r"\begin{table}[htbp]", r"\centering", r"\small",
r"\caption{Post-projection residual \(D\), nESS, and displacement coefficient \(\sqrt{1/\mathrm{nESS}-1}\) of \texttt{SemaPop-VAE+}, the prior selected at \(\tau_a=1\), in the three 2024 scenarios with admitted priors as the tolerance \(\tau_a=\tau\) is tightened uniformly for all \(a\in\Aset\). California~2024 is deferred after admission; Ohio and Texas~2024 are deployed. \(\min_\theta D\) is the smallest residual among the four admitted California priors. The row \(\tau=1\) reproduces Table~\ref*{tab:selected-prior-budget} of the main text. All projections are feasible. The last row gives the largest tolerance \(\tau^{\ast}\) at which \(D\le B_0=0.30\), interpolated linearly between ladder points.}",
r"\label{tab:tau-ladder}", r"\begin{tabular}{lccccccccc}", r"\toprule",
r" & \multicolumn{4}{c}{California 2024} & \multicolumn{2}{c}{Ohio 2024} & \multicolumn{2}{c}{Texas 2024} \\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
r"\(\tau\) & \(D\) & nESS & \(\sqrt{1/\mathrm{nESS}-1}\) & \(\min_\theta D\) & \(D\) & nESS & \(D\) & nESS \\", r"\midrule"]
def g(st, t): return d[(d.cell==f"semapop_vae_{st}_2024")&(np.isclose(d.tau, t))].iloc[0]
def row(t):
    c, o, x = g("CA", t), g("OH", t), g("TX", t); mn = sc[(sc.state=="CA")&(np.isclose(sc.tau, t))].D_sel.iloc[0]
    return f"{t:g} & {c.D:.3f} & {c.nESS:.3f} & {c.disp:.3f} & {mn:.3f} & {o.D:.3f} & {o.nESS:.3f} & {x.D:.3f} & {x.nESS:.3f} \\\\"
for t in T: L.append(row(t))
c = cr.loc["semapop_vae_CA_2024"]
L += [r"\midrule",
      f"\\(\\tau^{{\\ast}}\\) & \\multicolumn{{4}}{{c}}{{\\({c.tau_cross_interp:.2f}\\), nESS \\({c.nESS_cross_interp:.3f}\\), \\({c.disp_cross_interp:.2f}\\)}} & \\multicolumn{{2}}{{c}}{{\\(D\\le B_0\\) for all \\(\\tau\\)}} & \\multicolumn{{2}}{{c}}{{\\(D\\le B_0\\) for all \\(\\tau\\)}} \\\\",
      r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(H/"tab_tau_ladder.tex").write_text("\n".join(L)+"\n"); print("\n".join(L))
# figure: D(tau) of every admitted prior, one panel per 2024 scenario
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
sty = {"semapop_vae":("SemaPop-VAE+","-","o"), "ttvae":("TTVAE","--","s"), "bn":("BN",":","^"), "tabddpm":("TabDDPM","-.","d")}
name = {"CA":"California 2024 (deferred)", "OH":"Ohio 2024 (deployed)", "TX":"Texas 2024 (deployed)"}
for k, st in enumerate(["CA","OH","TX"]):
    for m,(lab,ls,mk) in sty.items():
        gg = d[(d.state==st)&(d.model==m)&(d.tau>=0.01)].sort_values("tau")
        if len(gg): ax[k].plot(gg.tau, gg.D, ls=ls, marker=mk, ms=4, label=lab)
    ax[k].axhline(0.30, color="k", lw=1, ls="--"); ax[k].axvline(1.0, color="#999", lw=0.8)
    if st == "CA": ax[k].axvline(c.tau_cross_interp, color="#c44", lw=0.8, ls=":")
    ax[k].set_xscale("log"); ax[k].set_xlabel(r"tolerance $\tau$ (log scale)"); ax[k].set_title(name[st]); ax[k].invert_xaxis()
    ax[k].legend(fontsize=8)
ax[0].set_ylabel(r"post-projection residual $D$")
for a in ax:
    for sp in ("top","right"): a.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(H/"figS_tau_ladder_D.pdf"); fig.savefig(H/"figS_tau_ladder_D.png", dpi=150)
