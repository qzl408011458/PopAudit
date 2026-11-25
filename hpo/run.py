"""Subprocess-per-trial Optuna driver for the SemaPop v2 fair HPO.

One command per (model, state):

    python -m hpo.run --model semapop_gan --state CA --gpu 0
    python -m hpo.run --model ctgan       --state CA --gpu 1
    python -m hpo.run --model bn          --state CA --device cpu

Properties (fairness of the comparison):

* **val-only objective** -- see ``hpo/objective.py``; the reference split is made
  unopenable inside the trial by ``hpo/guard/sitecustomize.py``.
* **full-data tuning** -- the whole 40k train split; ``--subsample`` exists only
  under ``--dry-run``.  Recorded in the study's ``user_attrs``.
* **no warm start** -- nothing is enqueued, ever.
* **resumable** -- SQLite study under ``$SSD/hpo/studies``; re-running the same
  command tops the study up to ``--trials`` and never over-shoots.
* **per-trial timeout** -- the subprocess is killed and the trial recorded as
  PRUNED with ``prune_reason=timeout``.
* **pruning** -- the trainers already print one val line per eval epoch; the
  driver parses it, calls ``trial.report`` and honours ``should_prune()`` by
  terminating the subprocess.  No trainer was modified to make this work.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[1])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import optuna

from . import common as C
from . import objective as OBJ
from . import spaces as S


# --------------------------------------------------------------------------- #
def study_name(model: str, state: str, cond_mode: str, tag: str = "",
               condition: str = "") -> str:
    """A study name must name everything that changes what is being tuned.

    The condition belongs here because Optuna resumes by name (load_if_exists=True).
    Without it, `--condition L_w100` reopened the 20-attribute study `semapop_gan_CA_mu`,
    restamped its provenance with the extended-set data directory, eval config and prior,
    and reported that study's best trial as if it were an extended-set result.  It ran no
    trials, so nothing was mixed -- but finalize.py reads the condition back OUT of the
    study to decide which dataset to retrain on, so the next finalize would have retrained
    the 20-attribute winner against the 50-attribute data and filed it as E1.
    """
    n = f"{model}_{state}"
    if condition:
        n += f"_{condition}"
    if model.startswith("semapop"):
        n += f"_{cond_mode}"      # always explicit: the mode changes what is being tuned
    if tag:
        n += f"_{tag}"
    return n


def storage_url(name: str, dry_run: bool) -> str:
    d = os.path.join(C.STUDY_DIR, "dryrun") if dry_run else C.STUDY_DIR
    os.makedirs(d, exist_ok=True)
    return f"sqlite:///{os.path.join(d, name + '.db')}"


# --------------------------------------------------------------------------- #
def _stream_trial(cmd: List[str], env: Dict[str, str], log_path: str,
                  timeout_s: int, trial: Optional[optuna.Trial],
                  prunable: bool) -> Dict[str, Any]:
    """Run one trainer, tee its stdout, report intermediates, honour pruning."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    t0 = time.time()
    out: Dict[str, Any] = {"returncode": None, "timed_out": False,
                           "pruned_early": False, "n_reports": 0, "tail": ""}
    tail: List[str] = []
    step_counter = 0
    proc = subprocess.Popen(cmd, cwd=C.REPO, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            start_new_session=True)
    # Hard watchdog: the read loop below can only notice a timeout when a new
    # line arrives, so a silently hung trainer would block forever without this.
    watchdog = threading.Timer(timeout_s, _kill, args=(proc,))
    watchdog.daemon = True
    watchdog.start()
    try:
        with open(log_path, "w", encoding="utf-8", buffering=1) as log:
            log.write("# " + " ".join(cmd) + "\n")
            for line in proc.stdout:                       # type: ignore[union-attr]
                log.write(line)
                tail.append(line)
                if len(tail) > 60:
                    tail.pop(0)
                if time.time() - t0 > timeout_s:
                    out["timed_out"] = True
                    break
                if trial is None or not prunable:
                    continue
                got = OBJ.parse_intermediate(line)
                if got is None:
                    continue
                step, value = got
                step_counter += 1
                step = step if step >= 0 else step_counter
                trial.report(value, step)
                out["n_reports"] += 1
                if trial.should_prune():
                    out["pruned_early"] = True
                    out["prune_step"] = step
                    out["prune_value"] = value
                    break
        if out["timed_out"] or out["pruned_early"]:
            _kill(proc)
        else:
            proc.wait(timeout=120)
    finally:
        watchdog.cancel()
        _kill(proc)
        out["returncode"] = proc.returncode
        out["seconds"] = round(time.time() - t0, 1)
        if out["seconds"] >= timeout_s:
            out["timed_out"] = True
        out["tail"] = "".join(tail)[-2500:]
    return out


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:                                       # noqa: BLE001
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:                                   # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
def make_objective(spec: S.ModelSpec, ctx: C.TrialContext, args) -> Any:
    def _objective(trial: optuna.Trial) -> float:
        params = spec.suggest(trial)
        # include the study name (model+state+cond_mode+tag) and the dry-run marker, so
        # two studies of the same model can never write into the same trial directory.
        run_name = f"{ctx.study}_t{trial.number:04d}"
        ctx.extra["run_name"] = run_name
        run_dir = os.path.join(ctx.out_root, run_name)
        argv = [C.PYEXE, spec.script_path()] + spec.argv(params, ctx)
        env = C.trial_env(gpu=args.gpu, forbid_reference=True)
        log_path = os.path.join(C.HPO_ROOT, "logs", f"{trial.study.study_name}",
                                f"trial_{trial.number:04d}.log")
        trial.set_user_attr("run_dir", run_dir)
        trial.set_user_attr("log", log_path)
        trial.set_user_attr("cmd", " ".join(argv))

        print(f"\n[trial {trial.number}] {spec.name}/{ctx.state} -> {run_dir}")
        print("  " + " ".join(argv))
        res = _stream_trial(argv, env, log_path, args.timeout or spec.timeout_s,
                            trial, spec.prunable)
        trial.set_user_attr("seconds", res["seconds"])
        trial.set_user_attr("returncode", res["returncode"])
        trial.set_user_attr("n_intermediate_reports", res["n_reports"])

        if res["timed_out"]:
            trial.set_user_attr("prune_reason", "timeout")
            raise optuna.TrialPruned(
                f"timeout > {args.timeout or spec.timeout_s}s ({res['seconds']}s elapsed)")
        if res["pruned_early"]:
            trial.set_user_attr("prune_reason", "median_pruner")
            raise optuna.TrialPruned(
                f"pruned at step {res.get('prune_step')} value={res.get('prune_value'):.4f}")
        if res["returncode"] != 0:
            # A CRASH IS NOT A PRUNE.  Raising TrialPruned here made a study whose
            # every trial died of `CUDA error: invalid device ordinal` report as
            # "20 pruned" -- indistinguishable from an aggressive pruner, and the
            # whole budget was spent in 60 seconds without a single usable trial.
            # Optuna records a raised exception as FAIL, which is what this is.
            trial.set_user_attr("fail_reason", f"rc={res['returncode']}")
            trial.set_user_attr("tail", res["tail"])
            raise RuntimeError(f"trainer rc={res['returncode']}\n{res['tail']}")

        try:
            score, info = OBJ.objective_from_run(run_dir, w_f1=ctx.f1_weight,
                                                 criterion=spec.criterion)
        except Exception as exc:                            # noqa: BLE001
            trial.set_user_attr("prune_reason", f"{type(exc).__name__}: {exc}")
            trial.set_user_attr("tail", res["tail"])
            raise optuna.TrialPruned(f"unscoreable trial: {exc}")
        for k, v in info.items():
            trial.set_user_attr(k, v)
        print(f"[trial {trial.number}] score={score:+.4f} "
              f"(SRMSE-1 {info['srmse_1']}, SRMSE-2 {info['srmse_2']}, "
              f"blockF1 {info['block_f1']}) in {res['seconds']}s")
        return score

    return _objective


# --------------------------------------------------------------------------- #
def build_context(args, spec=None) -> C.TrialContext:
    # spec carries this model's own selection criterion; without it the trainer would
    # pick its checkpoint on a different quantity than Optuna ranks trials by
    spec = spec if spec is not None else S.get(args.model)
    cond = args.condition or None
    # Only for the models that read it -- see ModelSpec.needs_prior.  --no-prior is an
    # ablation switch and also suppresses the lookup, so an ablation does not require the
    # artefact it is ablating.
    prior = ({"prior": None, "subspace": None, "source": "not_used_by_this_model"}
             if (not spec.needs_prior or args.no_prior)
             else C.prior_assets(args.state, args.prior_family, cond))
    # An explicit --eval-config still wins, but the default must follow the condition:
    # tuning an extended-set model against the 20-attribute config would run, score, and
    # be meaningless.
    eval_config = (args.eval_config if args.eval_config != C.EVAL_CONFIG
                   else C.eval_config_for(cond))
    ctx = C.TrialContext(
        state=args.state,
        condition=cond,
        device=args.device,
        seed=args.seed,
        sample_seed=args.sample_seed,
        eval_config=eval_config,
        out_root=args.out_root,
        subsample=args.subsample,
        subsample_val=args.subsample_val,
        max_epochs=args.max_epochs,
        eval_every=args.eval_every,
        n_samples=args.n_samples,
        f1_weight=args.f1_weight,
        criterion=spec.criterion,
        cond_mode=args.cond_mode,
        prior_family=args.prior_family,
        prior_path=None if args.no_prior else prior["prior"],
        subspace_path=None if args.no_prior else prior["subspace"],
    )
    ctx.extra["prior_source"] = "none" if args.no_prior else prior["source"]
    # CUDA_VISIBLE_DEVICES renumbers devices inside the subprocess: with --gpu 1 the
    # card is cuda:0 there, not cuda:1.  Passing the physical index through --device
    # is what made every GPU-1 trial die of "invalid device ordinal" -- 20 tabddpm
    # trials burned in 60 seconds.  --gpu 0 worked only by coincidence.  Normalise
    # HERE rather than at trial time so --print-cmd shows what will actually run.
    ctx.device = C.normalise_device(ctx.device, getattr(args, "gpu", None))
    ctx.study = study_name(args.model, args.state, args.cond_mode, args.tag,
                           args.condition)
    if getattr(args, "dry_run", False):
        ctx.study += "_dryrun"
    return ctx


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SemaPop v2 fair-HPO driver (F8 / R3-5)")
    p.add_argument("--model", required=True, choices=sorted(S.REGISTRY))
    p.add_argument("--state", required=True, choices=list(C.STATES))
    p.add_argument("--trials", type=int, default=0,
                   help="0 = the pre-registered budget for this model/state")
    p.add_argument("--gpu", type=int, default=None,
                   help="CUDA_VISIBLE_DEVICES for the trial subprocess")
    p.add_argument("--device", default="cuda:0",
                   help="value passed to the trainer's --device (use cpu for bn)")
    p.add_argument("--timeout", type=int, default=0, help="0 = the model's default")
    p.add_argument("--tag", default="")
    p.add_argument("--seed", type=int, default=42, help="the ONE tuning seed")
    p.add_argument("--sample-seed", type=int, default=1234)
    p.add_argument("--sampler-seed", type=int, default=42, help="TPE seed")
    p.add_argument("--f1-weight", type=float, default=OBJ.W_F1,
                   help="weight on mean block F1; 0 since 2026-08-25 (see objective.py)")
    p.add_argument("--eval-config", default=C.EVAL_CONFIG,
                   help="explicit override; by default it follows --condition")
    p.add_argument("--condition", default="",
                   help="extended-set condition, e.g. L_w100; empty = the 20-attribute build")
    p.add_argument("--out-root", default=C.TRIAL_DIR)
    # DEFAULT IS 'mu', NOT 'mr'.  draw_cond('mr') takes conditioning vectors from the
    # VALIDATION embeddings, and the HPO objective is itself computed on validation --
    # tuning in that mode selects hyper-parameters for an oracle path that cannot be
    # delivered (no real personas exist for a population you have not observed).  The
    # measured cost of that path: switching a tuned-on-M-R model to prior sampling
    # degrades a conditional-independence decoder 4.16x on SRMSE-1.  'mr' remains
    # available for the separate diagnostic/oracle study.
    p.add_argument("--cond-mode", default="mu", choices=["mr", "mu", "train"],
                   help="semapop only: sampling-time conditioning source")
    p.add_argument("--prior-family", default="gmm", choices=["gmm", "flow"])
    p.add_argument("--no-prior", action="store_true",
                   help="ignore the fitted persona prior (ablation studies)")
    p.add_argument("--no-prune", action="store_true")
    p.add_argument("--prune-startup", type=int, default=5,
                   help="MedianPruner n_startup_trials")
    p.add_argument("--prune-min-trials", type=int, default=5,
                   help="MedianPruner n_min_trials: how many trials must have reported at "
                        "a step before the median at that step is allowed to prune. Optuna "
                        "defaults to 1, and because `epochs` is itself searched over "
                        "100-800 only ~20%% of trials ever report past step 700 -- so a "
                        "long trial could be killed after ~90%% of its cost against a "
                        "single sample.")
    p.add_argument("--prune-warmup", type=int, default=50,
                   help="MedianPruner n_warmup_steps, in the trainer's own step "
                        "units (epoch number for the GAN/DDPM trainers): no trial "
                        "is pruned before this many epochs. Raised 20 -> 50 on "
                        "2026-08-25: the 20 was justified on the val_srmse criterion at a "
                        "350-400 epoch cap (Spearman of step-20 value against the final "
                        "objective: +0.50 wgan_zcr, +0.78 tabddpm, both on surviving "
                        "trials only). With the cap at 800 and two new criteria that have "
                        "almost no dynamic range early -- if1 reads 0.010 at epoch 1 "
                        "against 0.986 for real-vs-real -- step 20 is a coin flip.")
    # dry run
    p.add_argument("--dry-run", action="store_true",
                   help="tiny CPU-safe smoke test into studies/dryrun/")
    p.add_argument("--subsample", type=int, default=0)
    p.add_argument("--subsample-val", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=0,
                   help="0 = |val| (the frozen setting); >0 only for dry runs")
    p.add_argument("--print-cmd", action="store_true",
                   help="print one example trial command line and exit")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    spec = S.get(args.model)
    if not spec.available:
        print(f"[abort] {args.model} is a STUB: {spec.script} does not exist yet.\n"
              f"        {spec.notes}", file=sys.stderr)
        return 2
    if getattr(spec, "no_hpo", False):
        # Not a failure and not a stub: this model is reproduced at its reference
        # implementation's defaults, so there is nothing to search.  Refusing here rather
        # than quietly running a search keeps the decision auditable.
        print(f"[abort] {args.model} is deliberately NOT hyperparameter-searched.\n"
              f"        {spec.notes}\n"
              f"        Produce its row with a single run:\n"
              f"          python " + os.path.relpath(os.path.join(_PA.R("/training"), spec.script), C.REPO)
              + " --data-dir <dir> --eval-config <yaml>",
              file=sys.stderr)
        return 2
    if not os.path.exists(spec.script_path()):
        print(f"[abort] trainer not found: {spec.script_path()}", file=sys.stderr)
        return 2

    if args.dry_run:
        # 8192, not 2000.  tabddpm searches batch_size over [1024, 2048, 4096] and its
        # loader uses drop_last=True, so a 2000-row dry run drew zero batches for two of
        # the three choices: the loss accumulators stayed empty, np.mean([]) printed nan,
        # and an untrained network was written out as "best".  The floor has to clear
        # twice the largest batch any space can suggest, or the preflight passes on a run
        # that never trained.
        args.subsample = args.subsample or 8192
        args.subsample_val = args.subsample_val or 2000
        args.max_epochs = args.max_epochs or 2
        args.eval_every = args.eval_every or 1
        args.n_samples = args.n_samples or 1000
        # CPU unless a card was named.  A 1000-step reverse diffusion over 1,000 rows is
        # minutes on CPU, so the tabddpm and semapop_ddpm dry runs never finished a single
        # trial there -- a preflight nobody can afford to wait for is a preflight nobody
        # runs.  `--dry-run --gpu 0` keeps the same tiny-and-separate study on a card and
        # takes about a minute.
        if args.gpu is None:
            args.device = "cpu"
        else:
            args.device = "cuda:0"
        args.trials = args.trials or 2
        args.timeout = args.timeout or 1800

    ctx = build_context(args, spec)
    n_trials = args.trials or spec.n_trials(args.state)
    name = study_name(args.model, args.state, args.cond_mode, args.tag,
                      args.condition)
    if args.dry_run:
        name += "_dryrun"

    if args.print_cmd:
        probe_params = spec.suggest(_RandomProbe(args.sampler_seed))
        ctx.extra["run_name"] = f"{getattr(ctx, 'study', spec.name + '_' + args.state)}_t0000"
        print(" ".join([C.PYEXE, spec.script_path()] + spec.argv(probe_params, ctx)))
        return 0

    storage_uri = storage_url(name, args.dry_run)
    # An RDBStorage with a SQLite busy timeout, not the bare URI.  One study can be worked
    # by MORE THAN ONE PROCESS -- that is how a single sequential study stops being the
    # makespan's floor -- and two writers on a bare sqlite:/// URI raise "database is
    # locked" the first time two trials report at the same moment.  60 s of waiting is
    # free; a study that dies 30 hours in is not.  Single-worker runs are unaffected.
    storage = optuna.storages.RDBStorage(
        url=storage_uri, engine_kwargs={"connect_args": {"timeout": 60}})
    pruner = (optuna.pruners.NopPruner() if (args.no_prune or not spec.prunable)
              else optuna.pruners.MedianPruner(n_startup_trials=args.prune_startup,
                                               n_warmup_steps=args.prune_warmup,
                                               n_min_trials=args.prune_min_trials))
    study = optuna.create_study(
        direction="maximize", study_name=name, storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed, multivariate=True),
        pruner=pruner,
    )
    # Defence in depth behind the study name.  Resuming a study whose recorded condition,
    # eval config or data directory differs from this run's means the two sets of trials
    # are not comparable, and the attrs below are about to overwrite the older run's
    # provenance -- which is how the 20-attribute semapop study lost its own on
    # 2026-08-23.  Refuse instead of restamping.
    if study.trials:
        want = {"condition": ctx.condition or "pums20",
                "eval_config_path": ctx.eval_config,
                "data_dir": ctx.data_dir,
                # A study searched under val_srmse and resumed under if1 would hold two
                # incomparable objective scales in one TPE surrogate, and nothing in the
                # results would show it.  Studies from before 2026-08-25 have no such attr
                # and read None, which this comparison treats as "no clash" -- correct for
                # a genuinely absent field, and the reason finalize.py must not silently
                # substitute today's criterion for a study that never recorded one.
                "objective_criterion": spec.criterion}
        # A MISSING attr normally means "this study predates the field", which is not a
        # clash -- except for objective_criterion, where absence is itself information:
        # every study without one was searched under the old shared validation-SRMSE
        # objective. Reading it as None-means-anything is what would let an archived
        # composite-F1 study be resumed and extended under if1 or ml_eff, mixing two
        # incomparable objective scales inside one TPE surrogate with nothing in the
        # results to show it.
        on_record = dict(study.user_attrs)
        on_record.setdefault("objective_criterion", None)
        if on_record["objective_criterion"] is None:
            on_record["objective_criterion"] = "val_srmse"
        clash = {k: (on_record.get(k), v) for k, v in want.items()
                 if on_record.get(k) not in (None, v)}
        if clash:
            lines = "\n".join(f"    {k}: on record {old!r}, this run {new!r}"
                               for k, (old, new) in clash.items())
            raise SystemExit(
                f"[abort] study {name!r} already holds {len(study.trials)} trials from a "
                f"different setup:\n{lines}\n"
                f"  Use a distinct --tag, or delete the study, but do not mix them.")

    attrs = {
        "condition": ctx.condition or "pums20",
        "objective": OBJ.criterion_doc(spec.criterion),
        "objective_criterion": spec.criterion,
        "objective_w_f1": args.f1_weight,
        "model": args.model, "state": args.state,
        "trainer_script": spec.script,
        "search_space_dims": spec.n_dims,
        "search_space_notes": spec.notes,
        "budget_class": spec.weight,
        "n_trials_target": n_trials,
        "per_trial_timeout_s": args.timeout or spec.timeout_s,
        "tuning_seed": args.seed, "sampling_seed": args.sample_seed,
        "sampler": f"TPESampler(seed={args.sampler_seed}, multivariate=True)",
        "pruner": type(pruner).__name__,
        "subsample": ctx.subsample, "subsample_val": ctx.subsample_val,
        "dry_run": bool(args.dry_run),
        "cond_mode": ctx.cond_mode,
        "prior_source": ctx.extra["prior_source"],
        "prior_path": ctx.prior_path,
        "started": datetime.now().isoformat(timespec="seconds"),
        **C.provenance(args.state, ctx.eval_config, ctx.condition),
    }
    if not args.dry_run and ctx.subsample:
        raise SystemExit("refusing to run a real study on a subsample (plan F8)")
    for k, v in attrs.items():
        study.set_user_attr(k, v)

    done = sum(1 for t in study.trials
               if t.state.name in ("COMPLETE", "PRUNED", "FAIL"))
    todo = max(0, n_trials - done)
    print(f"[{name}] storage={storage_uri}")
    print(f"[{name}] {done} trials on record, target {n_trials} -> running {todo}")
    print(f"[{name}] criterion={spec.criterion}  {OBJ.criterion_doc(spec.criterion)}")
    print(f"[{name}] tuning on the FULL train split: "
          f"n_train={attrs['n_train_rows']} n_val={attrs['n_val_rows']}")
    if todo:
        study.optimize(make_objective(spec, ctx, args), n_trials=todo,
                       gc_after_trial=True, catch=())

    _report(study, name, spec, args)
    return 0


class _RandomProbe:
    """Minimal trial-like object for --print-cmd (no study, no storage)."""

    def __init__(self, seed: int):
        import random
        self.r = random.Random(seed)

    def suggest_categorical(self, name, choices):
        return self.r.choice(list(choices))

    def suggest_int(self, name, lo, hi, step=1, log=False):
        return lo + step * self.r.randrange(0, (hi - lo) // step + 1)

    def suggest_float(self, name, lo, hi, step=None, log=False):
        return self.r.uniform(lo, hi)


def _report(study: optuna.Study, name: str, spec: S.ModelSpec, args) -> None:
    comp = [t for t in study.trials if t.state.name == "COMPLETE"]
    pruned = [t for t in study.trials if t.state.name == "PRUNED"]
    secs = [t.user_attrs.get("seconds", 0.0) for t in study.trials
            if t.user_attrs.get("seconds")]
    print(f"\n=== {name}: {len(comp)} complete, {len(pruned)} pruned ===")
    if secs:
        secs_sorted = sorted(secs)
        print(f"trial wall-clock s: min={min(secs):.0f} "
              f"median={secs_sorted[len(secs_sorted)//2]:.0f} max={max(secs):.0f} "
              f"total={sum(secs)/3600:.2f} h")
    if not comp:
        print("no COMPLETE trial; nothing to export")
        return
    best = study.best_trial
    print(f"best trial #{best.number} score={best.value:+.4f}")
    print(f"  SRMSE-1={best.user_attrs.get('srmse_1')} "
          f"SRMSE-2={best.user_attrs.get('srmse_2')} "
          f"blockF1={best.user_attrs.get('block_f1')}")
    print(f"  params={json.dumps(best.params, default=str)}")
    out = {
        "study": name, "model": spec.name, "state": args.state,
        "objective": OBJ.criterion_doc(spec.criterion),
        "objective_criterion": spec.criterion,
        "n_trials_recorded": len(study.trials),
        "n_complete": len(comp), "n_pruned": len(pruned),
        "wall_clock_h": round(sum(secs) / 3600, 3) if secs else None,
        "best_trial": best.number, "best_score": best.value,
        "best_params": best.params,
        "best_user_attrs": dict(best.user_attrs),
        "study_user_attrs": dict(study.user_attrs),
    }
    p = os.path.join(C.HPO_ROOT, "best", f"{name}.json")
    C.jdump(out, p)
    print(f"wrote {p}")


if __name__ == "__main__":
    raise SystemExit(main())
