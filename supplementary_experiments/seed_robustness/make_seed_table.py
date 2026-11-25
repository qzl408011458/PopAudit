"""LaTeX table for Supplementary S3.8 (tab:seed-robustness) from results/*.csv (2026-11-24 g1fix rerun).
Format follows the current _supplementary.tex table (Panels A near-threshold, B valid counts,
C theta_dagger / D_dagger / decision).  Writes results/seed_robustness_table.tex."""
import os
import numpy as np, pandas as pd
from pathlib import Path
H = Path(os.environ.get("SEED_OUT", Path(__file__).resolve().parent / "results"))
d = pd.read_csv(H / "seed_certificate_cases.csv")
sc = pd.read_csv(H / "seed_scenario_decisions.csv")
v81 = pd.read_csv(H / "seed_valid81.csv").set_index("seed")
NAME = {"semapop_vae": "SemaPop-VAE+", "ttvae": "TTVAE", "bn": "BN", "tabddpm": "TabDDPM", "wgan_zcr": "WGAN-GP-ZCR"}
SEEDS = (42, 142, 242)
TOL_INV = 0.005
# Panel A rows: (model, state, year or None for the invalid-mass row)
ROWS = [("tabddpm", "OH", None), ("tabddpm", "OH", 2018), ("tabddpm", "OH", 2024), ("tabddpm", "TX", 2024),
        ("wgan_zcr", "OH", 2018), ("wgan_zcr", "CA", 2018), ("semapop_vae", "CA", 2024),
        ("ttvae", "CA", 2024), ("tabddpm", "CA", 2024)]


def cell(m, st, y, s):
    r = d[(d.model == m) & (d.state == st) & (d.seed == s) & (d.year == (y or 2018))]
    if not len(r):
        return "--"
    r = r.iloc[0]
    if y is None:
        v, bad = r.L1_invalid, r.L1_invalid > TOL_INV
        txt = f"{v:.4f}"
    else:
        v, bad = r.chi_H, r.chi_H > r.L2_cutoff
        txt = f"{v:.2f}"
    return f"\\(\\mathbf{{{txt}}}\\)" if bad else f"\\({txt}\\)"


L = []
L.append(r"\begin{tabular}{lllll}")
L.append(r"\toprule")
L.append(r" & Statistic or decision & Seed 42 & Seed 142 & Seed 242 \\")
L.append(r"\midrule")
L.append(r"\multicolumn{5}{l}{\emph{A. Cases closest to a threshold}}\\")
for m, st, y in ROWS:
    lab = f"\\texttt{{{NAME[m]}}}, {st}, " + ("all horizons" if y is None else str(y))
    stat = r"invalid mass (tolerance \(0.005\))" if y is None else r"\(\overline{\chi}_H\)"
    L.append(f"{lab} & {stat} & " + " & ".join(cell(m, st, y, s) for s in SEEDS) + r" \\")
L.append(r"\midrule")
L.append(r"\multicolumn{5}{l}{\emph{B. Valid cases}}\\")
n = int(v81.loc[142, "n_regenerated"])
reg = d[d.seed.isin(SEEDS)].pivot_table(index=["state", "model", "year"], columns="seed", values="VALID", aggfunc="first").dropna()
L.append(f"\\multicolumn{{2}}{{l}}{{{len(reg)} regenerated cases}} & " + " & ".join(str(int(reg[s].astype(bool).sum())) for s in SEEDS) + r" \\")
L.append(r"\multicolumn{2}{l}{all 81 cases} & " + " & ".join(str(int(v81.loc[s, "VALID81"])) for s in SEEDS) + r" \\")
L.append(r"\multicolumn{2}{l}{\quad 2018 / 2021 / 2024} & " + " & ".join(
    f"{int(v81.loc[s, 'VALID_2018'])}/{int(v81.loc[s, 'VALID_2021'])}/{int(v81.loc[s, 'VALID_2024'])}" for s in SEEDS) + r" \\")
L.append(r"\midrule")
L.append(r"\multicolumn{5}{l}{\emph{C. Selected prior \(\theta^\dagger(S_H)\), residual \(D^\dagger(S_H)\), and flat decision at \(B_0=0.30\)}}\\")
for st, y in [("CA", 2018), ("OH", 2018), ("TX", 2018), ("CA", 2024), ("OH", 2024), ("TX", 2024)]:
    cells, decs = [], []
    for s in SEEDS:
        r = sc[(sc.seed == s) & (sc.state == st) & (sc.year == y)].iloc[0]
        if r.n_adm == 0:
            cells.append("none"); decs.append("Defer"); continue
        c = f"\\texttt{{{NAME[r.theta_dagger]}}} {r.D_dagger:.3f}"
        if r.decision == "Defer":
            c += f" (\\(\\gamma_{{\\min}}={r.gamma_min:.3f}\\))"
        cells.append(c); decs.append(r.decision)
    dec = decs[0] if len(set(decs)) == 1 else "/".join(decs)
    L.append(f"{st} {y} & {dec} & " + " & ".join(cells) + r" \\")
L.append(r"\bottomrule")
L.append(r"\end{tabular}")
out = "\n".join(L) + "\n"
(H / "seed_robustness_table.tex").write_text(out)
print(out)
