#!/usr/bin/env python
"""Freeze the pop_audit v3 evaluation protocol (13-attr build, 12 modelled columns).

Port of the SemaPop repository's scripts/freeze_eval_config.py (not shipped here)
to the pop_audit `pums_states` layout and `cols_config_v3.yaml` (Region_Category excluded).

One frozen EvalConfig is the single matching space for the HPO objective and
every model-fidelity table: stored category lists make N_j byte-identical
across models, seeds and states.  Categories are fitted on the UNION of the
three FULL-STATE 2013 person-level files, so one file serves CA/TX/OH and no
split can surface an unseen category.

The `_weighted` variant is NOT produced: testdata_pers.csv carries no PWGTP
(the primary protocol is unweighted; the audit layers weight scenario targets with
PWGTP through their own machinery).

  python utils/freeze_eval_config_v3.py                # writes configs/eval_v3_pums12*.yaml
  python utils/freeze_eval_config_v3.py --sanity CA    # + real CA sanity run
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import pandas as pd
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from utils import eval_v2 as ev  # noqa: E402

DATA = os.path.join(REPO, "data", "pums_states")
STATES = ("CA", "TX", "OH")
OUT_DIR = os.path.join(REPO, "configs")

# Age 5-year bins primary; 1/10-year as the resolution-sensitivity axis.
# Tail merges follow the same policy as the v2 freeze (HH size >=10, children >=8);
# Number_of_cars_of_household spans only 0..6 and is kept unmerged.
AGE_VARIANTS = {"": 5, "_age1": 1, "_age10": 10}
TAIL_MERGE = {"Household_Size": 10, "Number_of_children": 8}


def load_cols(state: str) -> dict:
    with open(f"{DATA}/{state}/cols_config_v3.yaml") as fh:
        return yaml.safe_load(fh)


def build(age_width: int, frames: Dict[str, pd.DataFrame], cfg0: dict) -> ev.EvalConfig:
    num_cols: List[str] = list(cfg0["num_cols"])
    cat_cols: List[str] = list(cfg0["cat_cols"])
    specs: Dict[str, ev.BinningSpec] = {}
    for c in num_cols:
        if c == "Age":
            specs[c] = ev.BinningSpec(name=c, kind="binned", bin_width=float(age_width))
        else:
            specs[c] = ev.BinningSpec(name=c, kind="integer", tail_merge_at=TAIL_MERGE.get(c))
    for c in cat_cols:
        specs[c] = ev.BinningSpec(name=c, kind="categorical")
    order = num_cols + cat_cols
    cfg = ev.EvalConfig(specs=specs, order=order,
                        blocks={k: list(v) for k, v in cfg0["blocks"].items()},
                        weight_col=None,
                        name=f"eval_v3_pums12{'' if age_width == 5 else f'_age{age_width}'}")
    union = pd.concat([frames[s][order] for s in STATES], ignore_index=True)
    cfg.fit(union)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--sanity", default="", help="state code: run a real evaluation with the frozen config")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg0 = load_cols(STATES[0])
    order = list(cfg0["num_cols"]) + list(cfg0["cat_cols"])
    frames = {s: pd.read_csv(f"{args.data}/{s}/person_level_2013_{s}.csv", usecols=order)
              for s in STATES}
    for s in STATES:
        print(f"[load] {s}: {len(frames[s]):,} persons (full state 2013)")

    written = []
    for suffix, width in AGE_VARIANTS.items():
        cfg = build(width, frames, cfg0)
        path = os.path.join(args.out_dir, f"eval_v3_pums12{suffix}.yaml")
        cfg.to_yaml(path)
        written.append((path, cfg))
        n_cells = 1
        for c in cfg.order:
            n_cells *= len(cfg.specs[c].categories)
        print(f"[write] {path}  (Age bin {width}y, full-tuple cells {n_cells:.3e})")

    cfg = written[0][1]
    print("\n%-30s %-12s %6s" % ("column", "kind", "n_cat"))
    for c in cfg.order:
        print("%-30s %-12s %6d" % (c, cfg.specs[c].kind, len(cfg.specs[c].categories)))
    print("\n[check] per-state categories covered by the frozen union:")
    bad = 0
    for s in STATES:
        for c in cfg.order:
            if cfg.specs[c].kind != "categorical":
                continue
            missing = set(frames[s][c].astype(str).unique()) - set(cfg.specs[c].categories)
            if missing:
                print(f"  !! {s}.{c}: {sorted(missing)}")
                bad += 1
    print(f"  missing categories: {bad}")

    if args.sanity:
        s = args.sanity
        print(f"\n[sanity] {s}: gen = traindata_pers, reference = testdata_pers, frozen config")
        cfg = ev.EvalConfig.from_yaml(written[0][0])
        need = list(cfg.order)
        gen = pd.read_csv(f"{args.data}/{s}/traindata_pers.csv", usecols=need)
        ref = pd.read_csv(f"{args.data}/{s}/testdata_pers.csv", usecols=need)
        for k in (1, 2):
            r = ev.srmse_kway(ref, gen, cfg, k)
            print(f"  SRMSE-{k}: mean {r['mean']:.6f}  max {r['max']:.6f}  over {r['n_subsets']} subsets")
        pr = ev.block_precision_recall(ref, gen, cfg)
        for b, v in pr.items():
            print(f"  {b:<14} P {v['precision']:.4f} R {v['recall']:.4f} F1 {v['f1']:.4f} "
                  f"uniq_ref {v['n_unique_ref']:,} uniq_gen {v['n_unique_gen']:,} "
                  f"singleton_ref {v.get('singleton_share_ref', float('nan')):.4f}")


if __name__ == "__main__":
    main()
