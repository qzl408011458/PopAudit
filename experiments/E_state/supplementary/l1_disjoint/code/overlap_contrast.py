"""Overlap effect at matched composition: (a) hhdisjoint_trainval minus mean(placebo_s1, placebo_s2), per pair and
statistic, against the placebo-to-placebo spread |s1 - s2| (reference-sampling noise). Also the excess of each prior over
the real-data yardstick (test split scored against the same reference)."""
import os, pandas as pd, numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(f"{HERE}/results/l1_disjoint_compare.csv"); y = pd.read_csv(f"{HERE}/results/l1_real_yardsticks.csv")
y = y[y.real_set == "test_split"].assign(one_minus_cov=lambda t: 1 - t.support_cov)
out = []
for (tag, st), g in d.groupby(["tag", "state"]):
    g = g.set_index("variant"); r = dict(tag=tag, state=st, prior=g.prior.iloc[0], accept_prod=bool(g.accept_prod.iloc[0]))
    for s, col in [("marg", "marg_shadow"), ("dep", "dep_shadow"), ("one_minus_cov", "one_minus_cov_shadow")]:
        a, p1, p2 = g.loc["hhdisjoint_trainval", col], g.loc["placebo_s1", col], g.loc["placebo_s2", col]
        r[f"{s}_a"] = a; r[f"{s}_placebo_mean"] = (p1 + p2) / 2; r[f"{s}_a_minus_placebo"] = a - (p1 + p2) / 2; r[f"{s}_placebo_spread"] = abs(p1 - p2)
        for v in ["orig", "hhdisjoint_trainval", "hhdisjoint_trainval_ipw", "placebo_s1"]:
            yy = y[(y.state == st) & (y.variant == v)][s if s != "one_minus_cov" else "one_minus_cov"].iloc[0]
            r[f"{s}_excess_over_real_{v}"] = g.loc[v, col] - yy
    out.append(r)
o = pd.DataFrame(out); o.to_csv(f"{HERE}/results/overlap_contrast.csv", index=False)
pd.set_option("display.width", 250)
print(o[["tag", "state", "marg_a", "marg_placebo_mean", "marg_a_minus_placebo", "marg_placebo_spread", "dep_a_minus_placebo", "dep_placebo_spread",
         "one_minus_cov_a_minus_placebo", "one_minus_cov_placebo_spread", "marg_excess_over_real_orig", "marg_excess_over_real_hhdisjoint_trainval",
         "marg_excess_over_real_hhdisjoint_trainval_ipw"]].round(4).to_string(index=False))
for s in ["marg", "dep", "one_minus_cov"]:
    print(s, "mean(a-placebo)=%.4f  mean|a-placebo|=%.4f  mean placebo spread=%.4f  share a<placebo=%.2f" % (
        o[f"{s}_a_minus_placebo"].mean(), o[f"{s}_a_minus_placebo"].abs().mean(), o[f"{s}_placebo_spread"].mean(), (o[f"{s}_a_minus_placebo"] < 0).mean()))
