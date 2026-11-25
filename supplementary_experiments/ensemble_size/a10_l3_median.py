"""Layer 3 + D at the median-chi_H realization instead of r* = argmin chi_H^(r).
Mirrors rerun_audit_new.do_l3 (same cells7 build, same Layer-3 command line, same sigma_H json from the
production Layer-2 output) but writes only under this directory:
   cells7_<tag>/<case>/..._cells7.csv,  l3_<tag>_<model>_<ST>_<Y>/
tag = 'med' (lower median, 25th of 50) or 'rstar' (pipeline check: must reproduce production D).
D = dA_union_l3 (category union), exactly as make_figE_postproj.py.
"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[2])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import sys, os, json, glob, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
ROOT = _PA.R("")
PY = _PA.PYTHON
OUT = Path(__file__).resolve().parent
BASE = Path(_PA.R("/main_experiments/audits_v3"))
sys.path.insert(0, str(BASE / "PAPER_xstate_v3"))
from dA_union import dA_union_l3
FINAL = _PA.S("/hpo_v3/final")
PRIMARY = {"semapop_vae": "semapop_vae_{st}_train", "ttvae": "ttvae_{st}", "bn": "bn_pgmpy_{st}",
           "wgan_zcr": "wgan_zcr_{st}", "tabddpm": "tabddpm_{st}_valsrmse"}
AGE_BINS = [0, 18, 35, 50, 65, 200]; AGE_LABELS = ["0-17", "18-34", "35-49", "50-64", "65+"]
CELL_KEYS = ["Age", "Age_Group", "Household_Size", "Number_of_children", "Number_of_cars_of_household",
             "Employment_status", "Household_Income_Class", "Commute_Mode"]
ATTRS = ["Age", "Household_Size", "Number_of_children", "Number_of_cars_of_household",
         "Employment_status", "Household_Income_Class", "Commute_Mode"]
CHECK = ("Household_Size==", "Commute_Mode==", "Employment_status==", "Number_of_cars_of_household==",
         "Household_Income_Class==")


def gen_dir(model, st):
    s = PRIMARY[model].format(st=st)
    return Path(f"{FINAL}/{s}/{s}_final_s42/generated_ensembles")


def pick(model, st, y, tag):
    w = pd.read_csv(BASE / f"layer2_{model}_{st}_{y}/layer2_per_realization_weights.csv")
    order = np.argsort(w.scenario_loss.to_numpy(), kind="stable")
    rid = int(w.realization_id.iloc[order[0] if tag == "rstar" else order[24]])
    files = sorted(gen_dir(model, st).glob("*.csv"))
    assert len(files) == len(w) == 50
    f = files[rid]; row = w.loc[w.realization_id == rid].iloc[0]
    cols = [c for c in w.columns if c.startswith(CHECK)]
    d = pd.read_csv(f, usecols=sorted({c.split("==")[0] for c in cols}))
    err = 0.0
    for c in cols:
        a, v = c.split("==")
        try: msk = pd.to_numeric(d[a], errors="coerce") == float(v)
        except ValueError: msk = d[a].astype(str) == v
        err = max(err, abs(float(msk.mean()) - float(row[c])))
    assert err < 1e-9, (model, st, y, tag, err)
    return rid, f, float(row.scenario_loss)


def run_case(model, st, y, tag):
    case = f"{model}_{st}_{y}"
    rid, f, chi = pick(model, st, y, tag)
    cdir = OUT / f"cells7_{tag}" / case; cdir.mkdir(parents=True, exist_ok=True)
    for s in glob.glob(f"{cdir}/*_cells7.csv"): os.remove(s)
    df = pd.read_csv(f)
    df["Age_Group"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    g = df.groupby(CELL_KEYS, observed=True).size().reset_index(name="base_count")
    g.insert(0, "realization_id", rid)
    cells7 = cdir / f"{f.stem}_cells7.csv"; g.to_csv(cells7, index=False)
    l3 = OUT / f"l3_{tag}_{case}"
    import shutil; shutil.rmtree(l3, ignore_errors=True); l3.mkdir(parents=True)
    sig = pd.read_csv(BASE / f"layer2_{case}/layer2_historical_sigma_summary.csv")
    sj = l3 / "sigma_H_targets.json"
    json.dump({r.target: float(r.sigma) for r in sig.itertuples()}, open(sj, "w"))
    ref = f"{ROOT}/data/pums_states/{st}/reference_{y}_{st}.csv"
    cmd = [PY, "-u", _PA.R("/audit/layer3_projection_uncertainty_branches.py"), "--realizations_csv", str(cells7),
           "--scenario_reference_csv", ref, "--target_attrs", *ATTRS, "--base_weight_col", "base_count",
           "--reference_weight_col", "PWGTP", "--output_dir", str(l3), "--projection_mode", "uncertainty_set",
           "--uncertainty_aggregation", "attribute_constraints", "--uncertainty_tau", "1.0", "--solver", "SCALABLE",
           "--save_projected_weights", "--sigma_mode", "scenario_uncertainty", "--scenario_target_sigma_json", str(sj),
           "--sigma_floor", "0.005"]
    with open(f"{l3}.relog", "w") as lg:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=lg, stderr=subprocess.STDOUT).returncode
    assert rc == 0, (case, tag, rc)
    lv = pd.read_csv(l3 / "layer3_scenario_sigma_summary.csv").sigma_level
    assert (lv == "target").all()
    j = json.loads((l3 / "layer3_summary.json").read_text())
    pr = pd.read_csv(l3 / "layer3_per_realization_diagnostics.csv")
    ap = json.loads(pr.iloc[0]["attribute_projected_losses_json"])
    l3ok = j["feasibility_rate"] >= 0.95 and all(float(v) <= 1.0 + 1e-3 for v in ap.values())
    u = dA_union_l3(l3, st, int(y))
    return {"state": st, "model": model, "year": int(y), "tag": tag, "realization": rid, "chi_H_r": chi,
            "L3_accept": bool(l3ok), "KL": j.get("mean_distortion"), "D": u["d_A"], "D_ref": u["d_A_ref"]}


if __name__ == "__main__":
    tags = sys.argv[1].split(",") if len(sys.argv) > 1 else ["med"]
    v = pd.read_csv(BASE / "PAPER_xstate_v3/xstate_valid_only.csv")
    jobs = [(r.model, r.state, str(int(r.year)), t) for t in tags for r in v.itertuples()]
    with ThreadPoolExecutor(12) as ex:
        res = list(ex.map(lambda a: run_case(*a), jobs))
    out = pd.DataFrame(res)
    fn = OUT / f"a10_l3_{'_'.join(tags)}.csv"; out.to_csv(fn, index=False)
    print(out.round(5).to_string(index=False))
