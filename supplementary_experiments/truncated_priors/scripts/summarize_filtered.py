"""Supplementary diagnostic (filter): certificate rows for the truncated priors, control check, post-projection
residual D, and the Table-4 scenario decisions when the truncated priors join the candidate set.

Gate logic = shadow copy of decision_certificate_xstate_v3.py (unchanged formulas), only BASE/OUT_DIR
redirected. D = 0.5 * sum_targets projected_abs_err_mean (make_figE_postproj.py).  Outputs in ../certificate/.
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[3])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import sys
from pathlib import Path
import numpy as np, pandas as pd
S = Path(_PA.R("/supplementary_experiments/truncated_priors"))
V3 = Path(_PA.R("/main_experiments/audits_v3"))
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(S / "scripts"))
import decision_certificate_xstate_v3 as DC
assert DC.BASE == S / "audit"
B0 = 0.30
TAGS = ["ctgan_filt", "tvae_filt", "borysov_filt"]
OUT = S / "certificate"; OUT.mkdir(exist_ok=True)


# FIX 3 (2026-11-24): D = d_A(Q*,S_H) on the union of Q* and reference categories (shared helper
# PAPER_xstate_v3/dA_union.py, as make_figE_postproj.py); the pre-projection value stays on the targets.
sys.path.insert(0, str(V3 / "PAPER_xstate_v3")); from dA_union import D_union


def D_post(base, model, st, y):
    t3 = base / f"layer3_paperW_{model}_{st}_{y}" / "layer3_target_summary.csv"
    if not t3.exists():
        raise FileNotFoundError(t3)
    t2 = pd.read_csv(base / f"layer2_{model}_{st}_{y}" / "layer2_target_summary.csv")
    return float(D_union(model, st, y, base)), 0.5 * t2["abs_err_prior"].sum()


# ---- certificate rows (filtered + control) ----
DC.STATE_MODELS = {"CA": TAGS, "TX": TAGS, "OH": TAGS + ["ctgan_ctrl"]}
DC.MODELS = TAGS + ["ctgan_ctrl"]
DC.OUT_DIR = OUT
DC.main()
cert = pd.read_csv(OUT / "xstate_certificate_all.csv"); cert = cert[cert.year.notna()].copy(); cert["year"] = cert.year.astype(int)
Ds = []
for r in cert.itertuples():
    try:
        d, dpre = D_post(DC.BASE, r.model, r.state, r.year)
    except FileNotFoundError:
        d, dpre = np.nan, np.nan
    Ds.append((d, dpre))
cert["D_post"] = [x[0] for x in Ds]; cert["dA_pre_ensmean"] = [x[1] for x in Ds]

paper = pd.read_csv(V3 / "PAPER_xstate_v3" / "xstate_certificate_all.csv"); paper = paper[paper.year.notna()].copy(); paper["year"] = paper.year.astype(int)
pD = []
for r in paper.itertuples():
    try:
        pD.append(D_post(V3, r.model, r.state, r.year)[0])
    except FileNotFoundError:
        pD.append(np.nan)
paper["D_post"] = pD

# ---- control: unfiltered CTGAN-Ohio through the shadow pipeline must reproduce audits_v3 ----
cols = ["L1_marg", "L1_dep", "L1_support", "L1_invalid", "L2_chi_H", "L3_KL", "L3_nESS", "D_post", "first_fail"]
a = cert[cert.model == "ctgan_ctrl"].set_index("year")[cols]
b = paper[(paper.model == "ctgan") & (paper.state == "OH")].set_index("year")[cols]
ctrl = pd.concat({"shadow": a, "audits_v3": b}, axis=1); ctrl.to_csv(OUT / "control_ctgan_OH.csv")
num = [c for c in cols if c != "first_fail"]
maxdiff = float((a[num].astype(float) - b[num].astype(float)).abs().max().max())
print("\n[control] unfiltered CTGAN-Ohio, shadow vs audits_v3:\n", ctrl.to_string(), f"\nmax |diff| = {maxdiff:.2e}")

# ---- filtered vs unfiltered side by side ----
f = cert[cert.model.isin(TAGS)].copy(); f["base_model"] = f.model.str.replace("_filt", "", regex=False)
u = paper[paper.model.isin(["ctgan", "tvae", "borysov"])].rename(columns={"model": "base_model"})
side = f.merge(u, on=["state", "base_model", "year"], suffixes=("_filt", "_unfilt"))
keep = ["base_model", "state", "year",
        "L1_marg_unfilt", "L1_marg_filt", "L1_dep_unfilt", "L1_dep_filt", "L1_support_unfilt", "L1_support_filt",
        "L1_invalid_unfilt", "L1_invalid_filt", "L2_chi_H_unfilt", "L2_chi_H_filt", "L2_cutoff_filt",
        "L3_nESS_unfilt", "L3_nESS_filt", "D_post_unfilt", "D_post_filt", "first_fail_unfilt", "first_fail_filt", "VALID_filt"]
side = side[keep].sort_values(["base_model", "state", "year"]); side.to_csv(OUT / "filtered_vs_unfiltered.csv", index=False)
pd.set_option("display.width", 300)
print("\n", side.round(4).to_string(index=False))

# ---- scenario decisions: paper valid set  vs  paper valid set + admitted truncated priors ----
def decide(valid):
    rows = []
    for y in (2018, 2021, 2024):
        for st in ("CA", "OH", "TX"):
            g = valid[(valid.state == st) & (valid.year == y)].sort_values("D_post")
            if len(g) == 0:
                rows.append(dict(state=st, year=y, n_adm=0, theta_dagger=None, D_dagger=None, flat="Defer")); continue
            rows.append(dict(state=st, year=y, n_adm=len(g), theta_dagger=g.model.iloc[0], D_dagger=round(g.D_post.iloc[0], 4),
                             flat="Deploy" if g.D_post.iloc[0] <= B0 else "Defer",
                             admitted=";".join(f"{m}:{d:.3f}" for m, d in zip(g.model, g.D_post))))
    return pd.DataFrame(rows)

pv = paper[paper.VALID == True]
fv = f[f.VALID == True]
base_dec = decide(pv); new_dec = decide(pd.concat([pv, fv]))
dec = base_dec.merge(new_dec, on=["state", "year"], suffixes=("_paper", "_with_filtered"))
dec.to_csv(OUT / "scenario_decisions_swap.csv", index=False)
print("\n", dec.to_string(index=False))
print(f"\nVALID: paper {int(paper.VALID.sum())}/81 ; truncated priors admitted {int(f.VALID.sum())}/27 "
      f"; L1 pass {int(f.drop_duplicates(['model','state']).L1_accept.sum())}/9")
print("first-fail of truncated priors:\n", f.first_fail.replace("", "VALID").value_counts().to_string())
# g1fix 2026-11-24: None != None is True in pandas, which listed the 2021 rows as "changed"; compare with fillna
changed = dec[(dec.theta_dagger_paper.fillna("") != dec.theta_dagger_with_filtered.fillna(""))
              | (dec.flat_paper.fillna("") != dec.flat_with_filtered.fillna(""))]
print("\nscenarios whose theta_dagger or flat decision changes:\n", changed.to_string(index=False) if len(changed) else "  none")

# ---- g1fix 2026-11-24: each admitted truncated prior vs the selected prior of its scenario (rank, gap) ----
allv = pd.concat([pv, fv])
rk = []
for r in fv.itertuples():
    g = allv[(allv.state == r.state) & (allv.year == r.year)].sort_values("D_post").reset_index(drop=True)
    sel = g.iloc[0] if g.model.iloc[0] != r.model else g.iloc[1]
    pos = int(g.index[g.model == r.model][0]) + 1
    nxt = g.iloc[pos] if pos < len(g) else None
    rk.append(dict(state=r.state, year=r.year, model=r.model, chi_H=r.L2_chi_H, cutoff=r.L2_cutoff, D_filt=r.D_post,
                   theta_dagger=sel.model, D_dagger=sel.D_post, gap=r.D_post - sel.D_post, rank=pos, n_admitted=len(g),
                   next_model=None if nxt is None else nxt.model, D_next=np.nan if nxt is None else nxt.D_post))
rk = pd.DataFrame(rk).sort_values(["year", "state"]); rk.to_csv(OUT / "truncated_vs_selected.csv", index=False)
print("\nadmitted truncated priors vs selected prior:\n", rk.round(4).to_string(index=False))
