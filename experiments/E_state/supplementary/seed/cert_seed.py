"""Per-seed certificate with the exact gate code of decision_certificate_xstate_v3.py
(imported; only BASE is redirected) + post-projection D (make_figE_postproj definition)
+ selected prior theta_dagger and flat decision per scenario.
Seed 42 rows for priors not re-audited here are read from the ORIGINAL audits_v3."""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[4])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import sys, os, json
from pathlib import Path
import pandas as pd, numpy as np
E = _PA.R("/experiments/E_state"); sys.path.insert(0, E)
HERE = Path(os.path.dirname(os.path.abspath(__file__)))
import decision_certificate_xstate_v3 as C
# FIX 3 (2026-11-24): D = d_A(Q*,S_H) on the union of Q* and reference categories (shared helper)
sys.path.insert(0, f"{E}/audits_v3/PAPER_xstate_v3"); from dA_union import D_union
ORIG = Path(f"{E}/audits_v3")
MODELS = ["semapop_vae", "ttvae", "bn", "wgan_zcr", "tabddpm"]
B0 = 0.30
# v(H) of the production budget (exp_ii_vH.csv, unweighted ||sigma(H)||_2); gamma_min as in
# build_budget_table_v3.py: (D_dagger/B0 - 1)/v(H) if D_dagger > B0 else 0  (2026-11-24 g1fix rerun)
VH = pd.read_csv(f"{E}/audits_v3/PAPER_xstate_v3/exp_ii_vH.csv").set_index("year")["v_H"].to_dict()
OUT = Path(os.environ.get("SEED_OUT", HERE / "results"))

def rows_for(base, seed):
    C.BASE = Path(base); out = []
    for st in ("CA", "TX", "OH"):
        for m in MODELS:
            l1 = C.load_layer1(st, m)
            if l1 is None: continue
            t1 = C.layer1_test(l1, C.ALPHA[0], C.STATE_N_AGENTS[st])
            for y in C.YEARS:
                l2 = C.load_layer2(st, m, y); l3 = C.load_layer3(st, m, y)
                if l2 is None: continue
                t2 = C.layer2_test(l2, C.ALPHA[1]); t3 = C.layer3_test(l3, C.ALPHA[2]) if l3 else None
                valid = bool(t1["accept"] and t2["accept"] and t3 and t3["accept"])
                ff = ("L1:" + ",".join(t1["failed"])) if not t1["accept"] else ("L2" if not t2["accept"] else ("L3" if not (t3 and t3["accept"]) else ""))
                D = np.nan
                p3 = Path(base) / f"layer3_paperW_{m}_{st}_{y}" / "layer3_target_summary.csv"
                if p3.exists():
                    D = D_union(m, st, y, base)
                out.append(dict(seed=seed, state=st, model=m, year=int(y), L1_marg=l1["marg_TVD"], L1_dep=l1["dep_TVD"],
                                L1_support=l1["support_cov"], L1_invalid=l1["invalid_mass"], L1_accept=t1["accept"],
                                chi_H=l2["chi_H"], L2_cutoff=t2["cutoff"], L2_accept=t2["accept"],
                                L3_accept=(t3["accept"] if t3 else None), L3_nESS=(l3 or {}).get("nESS"),
                                first_fail=ff, VALID=valid, D=D))
    return pd.DataFrame(out)

def scenario(df):
    r = []
    for (seed, st, y), g in df.groupby(["seed", "state", "year"]):
        v = g[g.VALID].sort_values("D")
        r.append(dict(seed=seed, state=st, year=y, n_adm=len(v), admitted=",".join(v.model),
                      theta_dagger=(v.model.iloc[0] if len(v) else ""), D_dagger=(v.D.iloc[0] if len(v) else np.nan),
                      runner_up=(v.model.iloc[1] if len(v) > 1 else ""), D_runner=(v.D.iloc[1] if len(v) > 1 else np.nan),
                      decision=("Deploy" if len(v) and v.D.iloc[0] <= B0 else "Defer"),
                      v_H=VH[int(y)],
                      gamma_min=((v.D.iloc[0] / B0 - 1) / VH[int(y)] if len(v) and v.D.iloc[0] > B0 else (0.0 if len(v) else np.nan))))
    return pd.DataFrame(r)

if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [42, 142, 242]
    frames = []
    for s in seeds:
        base = ORIG if s == 42 else HERE / f"audits_s{s}"
        frames.append(rows_for(base, s))
    df = pd.concat(frames, ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(OUT / "seed_certificate_cases.csv", index=False)
    sc = scenario(df); sc.to_csv(OUT / "seed_scenario_decisions.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(df.round(5).to_string(index=False)); print(); print(sc.round(4).to_string(index=False))
