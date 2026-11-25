"""Retrain the HPO winner at 3 seeds, aggregate, and emit the budget table.

    python -m hpo.finalize --model semapop_gan --state CA --gpu 0
    python -m hpo.finalize --budget-table-only          # rebuild the table from the studies

What this does, in order:

1. Read the study's best trial (best_params) from the SQLite storage written by
   ``hpo.run``.  Refuse if the study has no COMPLETE trial.
2. Retrain that configuration at the pre-registered seeds (``common.seeds_for``:
   3 per state under the v3 protocol; ``--n-seeds`` may shorten but never below 3).
   The sampling seed moves with the training seed by default (see
   ``retrain_seed``), so a trainer whose fit is deterministic does not report a
   zero-width CI.
3. Score each retrain's samples against the reference population with the same
   frozen EvalConfig used for the val objective (``reference_metrics_for``).
   This is the ONLY place in the pipeline that opens the reference split, it
   happens after selection is finished, and it is computed here rather than
   trusted from each trainer so the definition is identical across models.
4. Aggregate the per-seed metrics with ``utils.eval_v2.aggregate_seeds``
   (mean / sd / n / 95% CI) into ``$SSD/hpo/final/<study>/aggregate.json``.
5. Emit the budget table row: model x state x n_trials x wall-clock x the
   val-vs-reference objective gap -- the number that shows tuning on val did
   not quietly optimise the reference.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[1])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import optuna

from . import common as C
from . import objective as OBJ
from . import run as R
from . import spaces as S

sys.path.insert(0, C.REPO)


def _ev():
    from utils import eval_v2 as ev
    return ev


# --------------------------------------------------------------------------- #
def load_study(name: str) -> optuna.Study:
    return optuna.load_study(study_name=name, storage=R.storage_url(name, dry_run=False))


def best_params(study: optuna.Study, reselect: Optional[str] = None,
                f1_weight: float = 0.0) -> Dict[str, Any]:
    comp = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not comp:
        raise SystemExit(f"study {study.study_name!r} has no COMPLETE trial")
    if not reselect:
        return dict(study.best_trial.params)
    # v3 protocol: re-rank the SAME trial pool under a different criterion (e.g. the
    # uniform val_srmse), so a study searched under an author's own criterion can
    # also be finalized under the audit's fidelity criterion without a new search.
    # Caveat recorded by the caller: pruning happened under the original criterion.
    import utils.criteria as _CR
    scored = []
    for t in comp:
        try:
            scored.append((_CR.score(reselect, t.user_attrs, f1_weight), t))
        except ValueError:
            continue
    if not scored:
        raise SystemExit(f"study {study.study_name!r}: no trial scoreable under {reselect!r}")
    best_s, best_t = max(scored, key=lambda x: x[0])
    print(f"[final] RESELECTED under {reselect!r}: trial #{best_t.number} score={best_s:+.4f} "
          f"(study's own best was #{study.best_trial.number}; "
          f"srmse_1 {best_t.user_attrs.get('srmse_1')} vs {study.best_trial.user_attrs.get('srmse_1')})")
    return dict(best_t.params)


# --------------------------------------------------------------------------- #
def retrain_seed(spec: S.ModelSpec, ctx: C.TrialContext, params: Dict[str, Any],
                 seed: int, gpu: Optional[int], study: str,
                 with_reference: bool, timeout_s: int, ensemble: bool = False,
                 vary_sample_seed: bool = True) -> Dict[str, Any]:
    """One final retrain, at one seed.

    ``vary_sample_seed`` moves the SAMPLING seed with the training seed.  During
    HPO the sampling seed is pinned (1234) so trials differ only in their
    hyper-parameters; for the final sweep it must move, otherwise a trainer whose
    fit is deterministic given the data -- BN is exactly that -- produces five
    identical runs and a 95% CI of width zero, which would be a fabricated error
    bar.  With it on, the reported spread covers the whole fit+generate pipeline.
    """
    run_name = f"{study}_final_s{seed}"
    ctx.seed = seed
    if vary_sample_seed:
        ctx.sample_seed = 1234 + seed
    ctx.out_root = os.path.join(C.FINAL_DIR, study)
    ctx.extra["run_name"] = run_name
    argv = [C.PYEXE, spec.script_path()] + spec.argv(spec.full_params(params), ctx)
    if ensemble:
        # v3 protocol: the audit certificate is computed on this seed's K=50 ensemble
        # (N = |reference| per realization, seeds from audit_config.ENSEMBLE_SEEDS).
        # N is passed EXPLICITLY (not 0) so no trainer has to open the reference
        # file to resolve it -- ctgan's argv carries --no-ref-eval and its
        # fail-fast rejects --ensemble-n 0 in that combination.
        n_ref = C.split_rows(ctx.state, "reference", ctx.condition)
        argv += ["--ensemble", "--ensemble-n", str(n_ref)]
    env = C.trial_env(gpu=gpu, forbid_reference=False)
    log = os.path.join(C.HPO_ROOT, "logs", study, f"final_s{seed}.log")
    print(f"[final] seed={seed} -> {os.path.join(ctx.out_root, run_name)}")
    res = R._stream_trial(argv, env, log, timeout_s, trial=None, prunable=False)
    out: Dict[str, Any] = {"seed": seed, "run_name": run_name,
                           "run_dir": os.path.join(ctx.out_root, run_name),
                           "seconds": res["seconds"], "returncode": res["returncode"],
                           "timed_out": res["timed_out"], "log": log}
    if res["returncode"] != 0 or res["timed_out"]:
        out["error"] = res["tail"][-1200:]
        return out
    m = OBJ.read_metrics_json(out["run_dir"])
    out["val_metrics"] = m.get("final_val_metrics") or {}
    out["val_objective"] = OBJ.score_from_metrics(out["val_metrics"], ctx.f1_weight)
    if with_reference:
        # THE ONE PLACE the reference population is read in this pipeline, and it
        # happens after selection is finished.  Computed here (rather than trusting
        # each trainer's own reference block) so the number is identical in
        # definition for every model.
        ref = reference_metrics_for(out["run_dir"], ctx.state, ctx.eval_config,
                                    ctx.condition)
        if ref:
            out["reference_metrics"] = ref
            try:
                out["reference_objective"] = OBJ.score_from_metrics(ref, ctx.f1_weight)
                out["val_minus_reference"] = (out["val_objective"]
                                              - out["reference_objective"])
            except ValueError:
                pass
    out["trainer_wall_clock_s"] = (m.get("run") or {}).get("wall_clock_s")
    return out


def reference_metrics_for(run_dir: str, state: str, eval_config: str,
                          condition: Optional[str] = None) -> Dict[str, Any]:
    """SRMSE-1/2 + mean block F1 of a finished run's samples vs the reference.

    Unweighted, frozen EvalConfig -- the same definition as the val objective, so
    ``val_objective - reference_objective`` is a like-for-like gap.
    """
    import pandas as pd

    ev = _ev()
    samples = os.path.join(run_dir, "generated_samples.csv")
    if not os.path.exists(samples):
        print(f"[warn] no generated_samples.csv in {run_dir}; skipping reference gap")
        return {}
    cfg = ev.EvalConfig.from_yaml(eval_config)
    cols = list(cfg.order)
    # The condition's OWN reference table.  Without it this reads the 20-attribute file,
    # which does not have the extended columns -- a loud failure here, but the same
    # omission in a metric that tolerated missing columns would be a silent one.
    ref_path = os.path.join(C.data_dir(state, condition), "referencedata_pers.csv")
    df_ref = pd.read_csv(ref_path, usecols=cols)
    df_gen = pd.read_csv(samples, usecols=cols)
    m: Dict[str, Any] = {
        "n_reference": int(len(df_ref)), "n_generated": int(len(df_gen)),
        "srmse_1": float(ev.srmse_kway(df_ref, df_gen, cfg, k=1)["mean"]),
        "srmse_2": float(ev.srmse_kway(df_ref, df_gen, cfg, k=2)["mean"]),
    }
    pr = ev.block_precision_recall(df_ref, df_gen, cfg)
    f1s = []
    for b, v in pr.items():
        if b == "all" or not isinstance(v, dict) or "f1" not in v:
            continue
        m[f"block_f1_{b}"] = float(v["f1"])
        f1s.append(float(v["f1"]))
    if f1s:
        m["block_f1"] = float(sum(f1s) / len(f1s))
    return m


# --------------------------------------------------------------------------- #
def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ev = _ev()
    ok = [r for r in records if "val_metrics" in r]
    if not ok:
        return {}
    agg: Dict[str, Any] = {
        "n_seeds": len(ok),
        "seeds": [r["seed"] for r in ok],
        "val": ev.aggregate_seeds([r["val_metrics"] for r in ok]),
        "val_objective": ev.aggregate_seeds([{"objective": r["val_objective"]} for r in ok]),
    }
    refs = [r["reference_metrics"] for r in ok if r.get("reference_metrics")]
    if refs:
        agg["reference"] = ev.aggregate_seeds(refs)
    gaps = [{"gap": r["val_minus_reference"]} for r in ok if "val_minus_reference" in r]
    if gaps:
        agg["val_minus_reference_objective"] = ev.aggregate_seeds(gaps)
    return agg


# --------------------------------------------------------------------------- #
def budget_row(study: optuna.Study, agg: Dict[str, Any],
               finals: List[Dict[str, Any]]) -> Dict[str, Any]:
    ua = dict(study.user_attrs)
    secs = [t.user_attrs.get("seconds", 0.0) for t in study.trials
            if t.user_attrs.get("seconds")]
    comp = [t for t in study.trials if t.state.name == "COMPLETE"]
    pruned = [t for t in study.trials if t.state.name == "PRUNED"]
    fin_s = [f["seconds"] for f in finals if f.get("seconds")]
    gap = (agg.get("val_minus_reference_objective", {}) or {}).get("gap", {})
    return {
        "model": ua.get("model"), "state": ua.get("state"), "study": study.study_name,
        "search_space_dims": ua.get("search_space_dims"),
        "budget_class": ua.get("budget_class"),
        "n_trials_target": ua.get("n_trials_target"),
        "n_trials_recorded": len(study.trials),
        "n_complete": len(comp), "n_pruned": len(pruned),
        "tuning_wall_clock_h": round(sum(secs) / 3600, 3) if secs else None,
        "median_trial_min": round(sorted(secs)[len(secs) // 2] / 60, 2) if secs else None,
        "max_trial_min": round(max(secs) / 60, 2) if secs else None,
        "n_train_rows": ua.get("n_train_rows"), "n_val_rows": ua.get("n_val_rows"),
        "tuned_on_full_train_split": ua.get("tuned_on_full_train_split"),
        "warm_start": ua.get("warm_start"),
        "best_val_objective": study.best_value if comp else None,
        "n_final_seeds": agg.get("n_seeds"),
        "final_wall_clock_h": round(sum(fin_s) / 3600, 3) if fin_s else None,
        "val_objective_mean": (agg.get("val_objective", {}).get("objective", {}) or {}).get("mean"),
        "val_objective_sd": (agg.get("val_objective", {}).get("objective", {}) or {}).get("sd"),
        "val_minus_reference_objective_mean": gap.get("mean"),
        "val_minus_reference_objective_ci95_lo": gap.get("ci95_lo"),
        "val_minus_reference_objective_ci95_hi": gap.get("ci95_hi"),
        "objective": ua.get("objective"),
    }


def write_budget_table(rows: List[Dict[str, Any]], path: str) -> None:
    import csv
    if not rows:
        print("[budget] no rows")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[budget] wrote {path} ({len(rows)} rows)")


def collect_all_studies() -> List[optuna.Study]:
    out = []
    if not os.path.isdir(C.STUDY_DIR):
        return out
    for fn in sorted(os.listdir(C.STUDY_DIR)):
        if not fn.endswith(".db"):
            continue
        name = fn[:-3]
        try:
            out.append(load_study(name))
        except Exception as exc:                            # noqa: BLE001
            print(f"[warn] cannot load study {name}: {exc}")
    return out


# --------------------------------------------------------------------------- #
def _criterion_of(study, spec) -> str:
    """The criterion a study was actually SEARCHED with, never today's registry value.

    Studies from before 2026-08-25 have no objective_criterion attr.  Defaulting those to
    spec.criterion would retrain a val_srmse study under if1 or ml_eff and pick checkpoints
    the search never ranked -- the reported winner would not be the winner.  An absent attr
    means the study predates per-model criteria, so it was searched under the old shared
    validation-SRMSE objective, and that is what it must be finalised under.
    """
    rec = study.user_attrs.get("objective_criterion")
    if rec is None:
        print(f"[finalize] study {study.study_name!r} records no objective_criterion "
              f"(pre-2026-08-25); finalising under 'val_srmse', not {spec.criterion!r}")
        return "val_srmse"
    if rec != spec.criterion:
        print(f"[finalize] study was searched under {rec!r}; the registry now says "
              f"{spec.criterion!r}. Using the study's, so the retrains match the search.")
    return rec


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="finalize a SemaPop v2 HPO study")
    p.add_argument("--model", choices=sorted(S.REGISTRY))
    p.add_argument("--state", choices=list(C.STATES))
    p.add_argument("--condition", default="",
                   help="extended-set condition, e.g. L_w100; must match the study's")
    # 'mu', matching hpo.run's default.  These two build the SAME study name from this
    # flag, so a disagreement means finalize looks for a study that was never created:
    # every semapop finalize died with KeyError 'Record does not exist.' looking for
    # semapop_gan_CA_mixed_MR while the search had written ..._MU.  run.py's default is the
    # deliverable path (prior sampling); 'mr' remains available for the oracle diagnostic.
    p.add_argument("--cond-mode", default="mu", choices=["mr", "mu", "train"])
    p.add_argument("--tag", default="")
    p.add_argument("--n-seeds", type=int, default=0, help="0 = the pre-registered set")
    p.add_argument("--n-samples", type=int, default=0,
                   help="rows each retrain emits; 0 = |reference| on the mixed tier, "
                        "|val| elsewhere (see the note in main)")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--timeout", type=int, default=0)
    p.add_argument("--fixed-sample-seed", action="store_true",
                   help="keep sample_seed=1234 for every final seed (only for a "
                        "training-seed-only variance decomposition; the default "
                        "moves it with the training seed)")
    p.add_argument("--reselect-criterion", default="",
                   help="re-rank the study's COMPLETE trials under this utils.criteria "
                        "name (e.g. val_srmse) instead of the study's own objective")
    p.add_argument("--out-tag", default="",
                   help="suffix for the OUTPUT dir (final/<study>_<out-tag>) while still "
                        "loading the study by its original name; required with "
                        "--reselect-criterion so the original finalize is not overwritten")
    p.add_argument("--sampling-arm", default="",
                   help="condition-source DUAL ARM (O1 plan, diagnostic only): retrain the "
                        "study's winner but sample from the prior side -- semapop_vae: "
                        "'mu' (GMM persona prior + subspace); ttvae: 'prior' (N(0,I) "
                        "latent instead of triangular). The study is still loaded by its "
                        "ORIGINAL name/cond-mode; requires --out-tag")
    p.add_argument("--seeds", default="",
                   help="explicit comma list of training seeds, bypassing the pre-registered "
                        "floor of 3 (diagnostic arms only, e.g. --seeds 42)")
    p.add_argument("--certificate-seed", type=int, default=42,
                   help="the retrain at this training seed also writes the K=50 audit "
                        "ensemble (O1 rework; -1 disables)")
    p.add_argument("--no-reference", action="store_true",
                   help="skip the post-selection reference evaluation")
    p.add_argument("--budget-table-only", action="store_true",
                   help="rebuild the budget table from every study on disk")
    p.add_argument("--budget-csv", default=os.path.join(C.HPO_ROOT, "budget_table.csv"))
    p.add_argument("--print-commands", action="store_true",
                   help="print the one-command-per-(model,state) launch list and exit")
    a = p.parse_args(argv)

    if a.print_commands:
        print(launch_commands(a.condition or C.MIXED_CONDITION))
        return 0

    if a.budget_table_only:
        rows = []
        for st in collect_all_studies():
            agg_p = os.path.join(C.FINAL_DIR, st.study_name, "aggregate.json")
            agg = json.load(open(agg_p)) if os.path.exists(agg_p) else {}
            fin_p = os.path.join(C.FINAL_DIR, st.study_name, "seed_runs.json")
            fin = json.load(open(fin_p)) if os.path.exists(fin_p) else []
            rows.append(budget_row(st, agg, fin))
        write_budget_table(rows, a.budget_csv)
        return 0

    if not (a.model and a.state):
        p.error("--model and --state are required unless --budget-table-only/--print-commands")

    if a.reselect_criterion and not a.out_tag:
        p.error("--reselect-criterion requires --out-tag (would overwrite the original finalize)")
    if a.sampling_arm and not a.out_tag:
        p.error("--sampling-arm requires --out-tag (would overwrite the main-arm finalize)")
    if a.sampling_arm and a.model not in ("semapop_vae", "ttvae"):
        p.error("--sampling-arm is defined only for semapop_vae (mu) and ttvae (prior)")
    spec = S.get(a.model)
    if not spec.available:
        raise SystemExit(f"{a.model} is a stub: {spec.notes}")
    name = R.study_name(a.model, a.state, a.cond_mode, a.tag, a.condition)
    if spec.no_hpo:
        # ctgan and bn are reproduced at their reference implementation's defaults, so
        # there is no study and no winner to read -- best_params would be asking which of
        # zero trials was best.  They still need the rest of this file: the same seeds, the
        # same reference-sized sample, the same aggregation.  Without this branch the
        # ctgan row simply could not be produced, and its finalize died on a study Optuna
        # had never been asked to create.
        study = None
        params = dict(spec.defaults)
        print(f"[final] {a.model} is no_hpo: author defaults, no study read")
        print(f"[final] params={json.dumps(params, default=str)}")
    else:
        study = load_study(name)
        params = best_params(study, reselect=a.reselect_criterion or None,
                             f1_weight=float(study.user_attrs.get("objective_w_f1", OBJ.W_F1)))
        print(f"[final] study={name} best trial #{study.best_trial.number} "
              f"score={study.best_value:+.4f}")
        print(f"[final] params={json.dumps(params, default=str)}")
    ua = dict(study.user_attrs) if study is not None else {}

    # The condition is read back from the study rather than re-specified: finalize builds
    # its own context, and the device bug of 2026-08-21 came from exactly this duplication.
    cond = ua.get("condition") or (a.condition or None)
    if cond == "pums20":
        cond = None

    # FIVE SEEDS ON THE MIXED TIER, FOR BOTH STATES, WITHOUT NEEDING THE FLAG.
    #
    # (Must come AFTER cond is resolved: this line short-circuits on --n-seeds, so the
    # 2026-08-28 launches, which all passed the flag, never evaluated the right-hand
    # operand -- and the UnboundLocalError it hid killed all four flag-less VAE finalizes
    # on 2026-08-29.  A default that is only exercised when the flag is absent has to be
    # tested without the flag.)
    #
    # The pre-registered sets are asymmetric -- CA 10, check states 5 -- and that asymmetry
    # was right when CA was the only table.  The mixed tier's design is uniform five-seed
    # rows; seeds_for(state, 5) returns the SAME five seeds for both states by the nesting
    # invariant in common.py, so a CA/FL difference cannot be a seed difference.
    # v3 protocol: 3 pre-registered seeds per state; floor lowered 5 -> 3.
    n_seeds = a.n_seeds
    seeds = C.seeds_for(a.state) if not n_seeds else C.seeds_for(a.state, max(3, n_seeds))
    if a.seeds:
        seeds = [int(x) for x in a.seeds.split(",")]
        print(f"[final] --seeds override (diagnostic arm): {seeds}")
    print(f"[final] {len(seeds)} seeds: {seeds}", flush=True)
    prior = C.prior_assets(a.state, ua.get("prior_family", "gmm"), cond) \
        if spec.needs_prior else {"prior": None, "subspace": None}
    # HOW MANY ROWS A FINAL RETRAIN EMITS.  A trial emits |val|, because that is what it
    # is scored against and nothing larger would be read.  A FINAL sample is a synthetic
    # POPULATION, and on the mixed tier it has to be the size of the population it stands
    # for.  Mutual information estimated from a finite sample is biased upwards and the
    # bias grows as the sample shrinks, so scoring a 20,002-row synthetic population
    # against a 311,401-row reference charges the generator for a bias it did not cause --
    # and hands a model that UNDERSTATES dependence that bias back as a bonus.  Measured on
    # the BN row: NMI-RMSE 0.0479 at 20k against an oracle real draw's 0.0620, i.e. better
    # than real data; at matched size it is 0.0376 against 0.0084, which is the true
    # ordering.  scripts/oracle_floor_mixed.py and scripts/run_bn_mixed.sh already generate
    # |reference|, so anything that does not is simply not on the same table.
    #
    # Only the mixed tier: the 50-attribute studies were finalised at |val| and changing
    # what their numbers mean after the fact would be worse than leaving them alone.
    n_samples = a.n_samples
    if not n_samples:
        n_samples = (C.split_rows(a.state, "reference", cond)
                     if cond == C.MIXED_CONDITION else 0)
    if n_samples:
        print(f"[final] each retrain emits {n_samples:,} rows "
              f"(|reference| for {a.state}/{cond})", flush=True)

    ctx = C.TrialContext(
        condition=cond,
        n_samples=n_samples,
        state=a.state, device=C.normalise_device(a.device, a.gpu),
        eval_config=ua.get("eval_config_path") or C.eval_config_for(cond),
        out_root=os.path.join(C.FINAL_DIR, name + (f"_{a.out_tag}" if a.out_tag else "")),
        f1_weight=float(ua.get("objective_w_f1", OBJ.W_F1)),
        # read back from the STUDY, not from the registry: a study run before a criterion
        # change must be finalised under the criterion it was actually searched with, or
        # the retrains pick checkpoints the search never ranked
        criterion=_criterion_of(study, spec) if study is not None else spec.criterion,
        cond_mode=ua.get("cond_mode", a.cond_mode),
        prior_path=ua.get("prior_path") or prior["prior"],
        subspace_path=prior["subspace"],
    )
    if a.sampling_arm:
        # Same winner, other condition source.  semapop_vae's argv gates the prior +
        # subspace on ctx.cond_mode == "mu"; ttvae's argv reads extra["sample_method"].
        if a.model == "semapop_vae":
            ctx.cond_mode = a.sampling_arm
            if a.sampling_arm == "mu" and not ctx.prior_path:
                raise SystemExit(f"[final] mu arm needs a fitted prior for {a.state}; "
                                 f"C.prior_assets returned none")
        else:
            ctx.extra["sample_method"] = a.sampling_arm
        print(f"[final] SAMPLING-ARM override: {a.model} -> {a.sampling_arm} "
              f"(prior={ctx.prior_path}, subspace={ctx.subspace_path})")
    t0 = time.time()
    out_name = name + (f"_{a.out_tag}" if a.out_tag else "")
    finals = [retrain_seed(spec, ctx, params, s, a.gpu, out_name,
                           ensemble=(s == a.certificate_seed),
                           with_reference=not a.no_reference,
                           timeout_s=a.timeout or spec.timeout_s,
                           vary_sample_seed=not a.fixed_sample_seed) for s in seeds]
    agg = aggregate(finals)
    # A finalize that produced NOTHING must not exit 0.  All ten CTGAN retrains died of
    # "invalid device ordinal" and this still printed an empty aggregate, wrote a budget
    # row and returned success -- the same silent-failure shape as trainer crashes being
    # recorded as PRUNED.  A budget row for zero usable retrains is worse than no row.
    ok = [r for r in finals if "val_metrics" in r]
    if not ok:
        tails = [str(r.get("tail", ""))[-400:] for r in finals[:1]]
        raise SystemExit(
            f"[final] FAILED: 0 of {len(finals)} retrains produced metrics for {name}. "
            f"First failure tail:\n{tails[0] if tails else '(no tail captured)'}")
    if len(ok) < len(finals):
        print(f"[final] WARNING: only {len(ok)} of {len(finals)} retrains produced "
              f"metrics; the aggregate below is over those {len(ok)}.")
    outdir = os.path.join(C.FINAL_DIR, name + (f"_{a.out_tag}" if a.out_tag else ""))
    C.jdump(finals, os.path.join(outdir, "seed_runs.json"))
    C.jdump({"study": name, "best_params": params,
             "objective": OBJ.criterion_doc(spec.criterion),
             "objective_criterion": spec.criterion,
             "reselected_under": a.reselect_criterion or None,
             "sampling_arm": a.sampling_arm or None,
             "reselect_caveat": ("trial pool was searched and PRUNED under the study's own "
                                 "criterion; reselection picks the best COMPLETE trial of that "
                                 "pool under the new criterion, not a fresh search")
                                if a.reselect_criterion else None,
             "seeds": seeds,
             "sample_seed_varies_with_training_seed": not a.fixed_sample_seed,
             "finished": datetime.now().isoformat(timespec="seconds"),
             "total_wall_clock_h": round((time.time() - t0) / 3600, 3),
             "aggregate": agg}, os.path.join(outdir, "aggregate.json"))
    print(json.dumps({k: v for k, v in agg.items() if k != "val"}, indent=2, default=str)[:2000])
    if study is not None:
        write_budget_table([budget_row(study, agg, finals)],
                           os.path.join(outdir, "budget_row.csv"))
    else:
        # No search, so no search budget to report.  budget_row reads study.user_attrs and
        # would have raised on None -- AFTER the five retrains, which is the most expensive
        # possible place to discover it.
        print("[final] no_hpo model: no budget row (there was no search to cost)")
    return 0


def launch_commands(condition: str = C.MIXED_CONDITION) -> str:
    """The deliverable: one command per (model, state).

    The condition is threaded through every line rather than left to the caller: a study
    resumed without it reopens the 20-attribute study of the same name (see run.study_name),
    and the omission is invisible until the results are already in.
    """
    cond = f" --condition {condition}" if condition else ""
    lines = [f"# SemaPop v2 fair HPO -- one command per (model, state), "
             f"condition={condition or 'pums20'}.",
             "# GPU assignment is a suggestion; --device cpu for bn.",
             "cd " + _PA.R("")]
    gpu = 0
    for m, spec in S.REGISTRY.items():
        if not spec.available:
            lines.append(f"# {m}: STUB -- {spec.notes.splitlines()[0]}")
            continue
        if spec.no_hpo:
            lines.append(f"# {m}: NO HPO -- author defaults, no search to run")
            continue
        for st in C.STATES:
            lines.append(f"{C.PYEXE} -m hpo.run --model {m} --state {st}{cond} "
                         f"--gpu {gpu} --device cuda:0")
        gpu = (gpu + 1) % 4
    lines.append("")
    lines.append("# then, per (model, state), the >=5-seed retrain + budget row:")
    for m, spec in S.REGISTRY.items():
        if spec.available:
            for st in C.STATES:
                lines.append(f"{C.PYEXE} -m hpo.finalize --model {m} --state {st}{cond}"
                             + (" --n-seeds 5" if condition == C.MIXED_CONDITION else "")
                             + (" --device cpu" if m == "bn" else " --gpu 0"))
    lines.append(f"{C.PYEXE} -m hpo.finalize --budget-table-only")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
