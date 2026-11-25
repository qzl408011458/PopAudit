"""LaTeX table for the supplementary sigma-sensitivity subsection -> results/tab_sigma_sensitivity.tex
g1fix (2026-11-24): California 2024 is admitted (deferred) at the baseline after the fix, so it gets a
column (CA~24); California 2021 stays in the caption (admitted only at floor 0.01).  D is on the
category union (analyze.py).  The boundary probes are written to results/tab_sigma_probes.tex."""
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
T = pd.read_csv(HERE / "results/sigma_sens_summary.csv").set_index("variant")
S = pd.read_csv(HERE / "results/sigma_sens_scenarios.csv")
ORDER = [("base", r"Baseline"),
         ("floor0025", r"\(\sigma_{\mathrm{floor},a}=0.0025\)"),
         ("floor01", r"\(\sigma_{\mathrm{floor},a}=0.01\)"),
         ("mu05", r"\(\mu_a=0.5\)"), ("mu2", r"\(\mu_a=2\)"),
         ("lam05", r"\(\lambda_a=0.5\)"), ("lam2", r"\(\lambda_a=2\)"),
         ("beta0", r"\(\beta_a=0\)"),
         ("hfree", r"No horizon scaling (\(H=5\) for all)"),
         ("win1013", r"Window 2010--2013")]


def cell(v, st, y):
    r = S[(S.variant == v) & (S.state == st) & (S.year == y)].iloc[0]
    if r.n_adm == 0:
        return "--"
    s = f"{r.D_dagger:.3f}"
    return s if r.decision == "Deploy" else s + "\\(^{\\ast}\\)"


SC = (("CA", 2018), ("OH", 2018), ("TX", 2018), ("OH", 2021), ("TX", 2021), ("CA", 2024), ("OH", 2024), ("TX", 2024))
PROBES = [("floor00525", r"\(\sigma_{\mathrm{floor},a}=0.00525\)"), ("floor0055", r"\(\sigma_{\mathrm{floor},a}=0.0055\)"),
          ("mu13", r"\(\mu_a=1.3\)"), ("mu145", r"\(\mu_a=1.45\)"),
          ("lam145", r"\(\lambda_a=1.45\)"), ("lam16", r"\(\lambda_a=1.6\)")]


def table(rows, fn):
    L = [r"\begin{tabular}{@{}lccccccccccc@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{Valid cases} & \multicolumn{8}{c}{\(D^\dagger(S_H)\) by scenario} \\",
         r"\cmidrule(lr){2-4}\cmidrule(l){5-12}",
         r"Specification & 2018 & 2021 & 2024 & CA~18 & OH~18 & TX~18 & OH~21 & TX~21 & CA~24 & OH~24 & TX~24 \\", r"\midrule"]
    for v, lab in rows:
        if v not in T.index:
            continue
        t = T.loc[v]
        cs = [cell(v, st, y) for st, y in SC]
        L.append(f"{lab} & {t.V2018} & {t.V2021} & {t.V2024} & " + " & ".join(cs) + r" \\")
        if v == "base":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "results" / fn).write_text("\n".join(L) + "\n")
    print("\n".join(L))


table(ORDER, "tab_sigma_sensitivity.tex")
table([ORDER[0]] + PROBES, "tab_sigma_probes.tex")
