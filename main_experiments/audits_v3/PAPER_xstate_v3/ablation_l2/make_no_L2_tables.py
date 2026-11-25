"""[v3 2026-09-02] Regenerate the two markdown tables that in v2 were written ad hoc
(no_L2_performance_table.md from ablation_results.csv; no_L2_ensemble_table.md from
ensemble_results.csv).  Format verified against the v2 files: rows sorted by year then
chi_H; Prec/Rec/F1 are already in percent in the CSVs; the L2 column is check/cross in
the performance table and PASS/drop in the ensemble table.  Display names = 9 v3 tags."""
import os
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
DISP = {"semapop_vae": "SemaPop-VAE+", "ttvae": "TTVAE", "tabddpm": "TabDDPM", "bn": "BN",
        "wgan_zcr": "WGAN-GP-ZCR", "ctgan": "CTGAN", "flowsyn": "FlowSyn",
        "borysov": "Borysov-VAE", "tvae": "TVAE"}


def table(df, cols, l2map, sfx=""):
    df = df.sort_values(["year", "L2_chi_H"])
    head = ["Yr", "Model", "St", "L2", "χ_H", "L1marg", "nESS"] + [c for c, _ in cols]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in df.iterrows():
        row = [str(int(r.year)), DISP.get(r.model, r.model), r.state, l2map[bool(r.L2pass)],
               f"{r.L2_chi_H:.1f}", f"{r.L1_marg:.3f}", f"{r.L3_nESS:.2f}"]
        row += [f"{r[c + sfx]:.{p}f}" for c, p in cols]
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


a = pd.read_csv(f"{HERE}/ablation_results.csv")
perf = table(a, [("SRMSE_U", 3), ("SRMSE_M", 3), ("SRMSE_J", 3), ("Precision", 1), ("Recall", 1), ("F1", 1)],
             {True: "✓", False: "✗"}, sfx="_mean").replace("| Precision |", "| Prec |").replace("| Recall |", "| Rec |")
open(f"{HERE}/no_L2_performance_table.md", "w").write(perf)
e = pd.read_csv(f"{HERE}/ensemble_results.csv")
ens = table(e, [("SRMSE_U", 3), ("SRMSE_J", 3), ("Precision", 1), ("Recall", 1), ("F1", 1)],
            {True: "PASS", False: "drop"}).replace("| Precision |", "| Prec |").replace("| Recall |", "| Rec |")
open(f"{HERE}/no_L2_ensemble_table.md", "w").write(ens)
print(f"wrote no_L2_performance_table.md ({len(a)} rows) and no_L2_ensemble_table.md ({len(e)} rows)")
print(perf)
