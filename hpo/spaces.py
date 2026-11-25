"""Optuna search spaces + subprocess argv builders, one entry per model.

DESIGN RULES (every deviation must be written down)
===================================================
1. **No warm start.**  No ``enqueue_trial``, no seeded/"known-good" trial, no
   narrowed range centred on a published config.  The pop_audit harness gave
   its own model a 13-dim space centred on the May checkpoint (``MAY_WGANLLM``
   enqueued as trial 0) while the baselines got wide uninformed spaces; that is
   the single biggest fairness defect being repaired here.
2. **Every range brackets the trainer default.**  A default that fell outside
   its range would be an unfair *dis*advantage; every entry below is checked by
   ``python -m hpo.spaces --check-defaults`` (asserts the trainer's argparse
   default is reachable inside the declared range).  Reachable != enqueued.
3. **Every model's space is EXHAUSTIVE of its trainer's real knobs.**  The rule
   is not "same number of dimensions" -- it is "nothing a model could be tuned
   on was withheld".  Actual counts (``python -m hpo.spaces --table``):
   semapop_gan 19, tabddpm 17, wgan_zcr 16, ctgan 12, bn 8.  CTGAN and BN are
   smaller because their trainers expose 12 and 8 knobs in total; both spaces
   cover every one of them.  Note the direction of any residual asymmetry: at a
   FIXED 20-trial budget a larger space is a *handicap*, not a favour, so
   SemaPop's 19 dims cannot be an unfair advantage.  Equalisation is enforced
   where it can be: identical trial budgets, identical objective, identical
   data, identical selection rule, identical timeout.
4. **Same axes for everyone.**  Each deep model searches capacity (2 axes),
   architecture (1-2), optimiser lr (1-2), regularisation/decay (2-3), batch
   size (1) and training length (1); on top of that each gets its own
   method-specific knobs -- SemaPop 4 (lambda_m, lambda_m_prior, lambda_cal,
   cond_jitter), TabDDPM 8 (parametrisation/schedule/loss/time sampler and the
   two loss weights), WGAN-ZCR 4 (bd/ad weights, zcr scale, zcr period),
   CTGAN 2 (pac, log_frequency -- all it has).
5. **Fixed, NOT searched, for every model**: the data split (full 40k train),
   the eval config, ``--f1-weight 1.0``, the tuning seed (42), the sampling
   seed (1234), and the number of generated rows (= |val|).  Anything that
   would change the objective's definition is frozen.
6. **The epoch cap is 800, not 400.**  Under a 400 cap the convergence probe of
   2026-08-25 found that EVERY probed model peaked at or near the cap and improved
   when allowed to run to 800: WGAN-GP-ZCR +0.0118 (best epoch 705), TabDDPM +0.0357
   (600), and SemaPop-GAN's best trial peaked at exactly its own 150-epoch draw.  The
   improvements are one to two times the spread between models, so a 400-cap table
   would have compared training budgets rather than methods -- and would have reported
   TabDDPM as weak while our own data showed it under-trained, i.e. insufficient
   tuning.  Widening costs roughly 2x per trial and is the
   reason the 400-cap studies are archived rather than deleted: they are the evidence
   for that finding.
7. **The prior fit is not free.**  SemaPop's M-U runs consume a pre-fitted
   persona prior; its fitting cost is charged to SemaPop in the budget table
   (``hpo/budget_probe.py``).
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import common as C

# --------------------------------------------------------------------------- #
# shared architecture menus -- identical option sets wherever a model has an
# MLP trunk, so "capacity" means the same thing across models.
# --------------------------------------------------------------------------- #
MLP_TRUNKS = ["256 256 256", "256 512 256", "512 512 512", "512 1024 512"]
DDPM_TRUNKS = ["512 512 512", "512 1024 512", "512 1024 1024 512", "1024 1024 1024 1024"]
CTGAN_TRUNKS = ["256 256", "512 512", "256 512 256", "512 512 512"]


def _ints(s: str) -> List[str]:
    return s.split()


@dataclass
class ModelSpec:
    """One tunable model: its script, its space, and how a trial is launched."""
    name: str
    script: str
    weight: str                                   # "heavy" | "light" -> trial budget
    timeout_s: int                                # per-trial wall-clock cap
    suggest: Callable[[Any], Dict[str, Any]]
    argv: Callable[[Dict[str, Any], C.TrialContext], List[str]]
    n_dims: int
    prunable: bool = True
    notes: str = ""
    available: bool = True                        # False => stub, run.py refuses
    #: True => this model is deliberately NOT searched, and run.py refuses with the reason.
    #: Distinct from ``available``: the trainer exists and works, there is simply nothing
    #: to search, because the reference implementation exposes no tuned hyperparameters.
    no_hpo: bool = False
    #: key into utils.criteria.CRITERIA -- the quantity THIS model's own authors select on.
    #: Ranking every model by one metric of ours would favour our own model.
    criterion: str = "val_srmse"
    #: True only for the models that actually consume the persona prior.  build_context
    #: used to resolve it for every model, so `hpo.run --model wgan_zcr --dry-run` died
    #: with "no prior for CA/mixed" -- a baseline that never opens a persona could not be
    #: smoke-tested until the whole persona pipeline had finished.  That is backwards: the
    #: preflight exists to run BEFORE the expensive stages.
    needs_prior: bool = False
    defaults: Dict[str, Any] = field(default_factory=dict)

    def script_path(self) -> str:
        return os.path.join(C.REPO, self.script)

    def full_params(self, p: Dict[str, Any]) -> Dict[str, Any]:
        """Searched params on top of this model's defaults.

        Optuna records only the values a trial actually SUGGESTED, so any knob pinned
        to a constant inside ``suggest`` -- lambda_cal for SemaPop, zcr_cat_scale and
        zcr_every for WGAN-GP-ZCR -- is absent from ``study.best_params``.  The argv
        builders index those keys directly, so reading a study back for the final
        retrain died with KeyError: 'lambda_cal'.  Merge here so a pinned constant and
        a searched value reach the trainer by the same path.
        """
        return {**self.defaults, **p}

    def n_trials(self, state: str) -> int:
        """Dimension-scaled, NOT the legacy flat per-weight-class table.

        C.n_trials_for is the flat {heavy: CA 20, TX 15, OH 15} table that produced the
        confound in plan 14.5: BN searched 8 dimensions with 30 trials (3.75/dim) while
        SemaPop searched 19 with 21 (1.11/dim), so a 12.9% deficit could not be split
        between method and budget.  Any caller that omits --trials -- scripts/queue_hpo.sh
        did -- silently reinstated it: 20/15/15 instead of 60/32/32.  Routing through
        n_trials_for_dims here means there is one budget, not two.
        """
        return C.n_trials_for_dims(self.n_dims, state)


# --------------------------------------------------------------------------- #
# argv fragments shared by every trainer (the frozen part of a trial)
# --------------------------------------------------------------------------- #
def _base_argv(ctx: C.TrialContext, tag: str, run_name: str) -> List[str]:
    a = [
        "--data-dir", ctx.data_dir,
        "--eval-config", ctx.eval_config,
        "--f1-weight", str(ctx.f1_weight),
        "--seed", str(ctx.seed),
        "--sample-seed", str(ctx.sample_seed),
        "--device", ctx.device,
        "--out-root", ctx.out_root,
        "--tag", tag,
        "--run-name", run_name,
        "--n-samples", str(ctx.n_samples or C.split_rows(ctx.state, "val", ctx.condition)),
    ]
    if ctx.subsample:
        a += ["--subsample", str(ctx.subsample)]
    if ctx.subsample_val:
        a += ["--subsample-val", str(ctx.subsample_val)]
    # the criterion the TRAINER selects its checkpoint by; must match what Optuna
    # ranks trials by, or per-epoch and between-trial selection disagree
    a += ["--select-on", ctx.criterion]
    return a


# =========================================================================== #
# 1. semapop_gan  (heavy, 17 searched dims)
# =========================================================================== #
def _semapop_suggest(t) -> Dict[str, Any]:
    return dict(
        # --- capacity -------------------------------------------------------
        noise_dim=t.suggest_categorical("noise_dim", [64, 128, 256]),
        cond_dim=t.suggest_categorical("cond_dim", [64, 128, 256]),
        adapter_hidden=t.suggest_categorical("adapter_hidden", [512, 1024, 2048]),
        g_hidden=t.suggest_categorical("g_hidden", MLP_TRUNKS),
        d_hidden=t.suggest_categorical("d_hidden", MLP_TRUNKS),
        # --- regularisation -------------------------------------------------
        adapter_dropout=t.suggest_float("adapter_dropout", 0.0, 0.3),
        dropout_g=t.suggest_float("dropout_g", 0.0, 0.3),
        dropout_d=t.suggest_float("dropout_d", 0.0, 0.3),
        # FIXED at 10 (Gulrajani et al.) since 2026-08-28, no longer searched.  The mixed
        # round searched [1, 20] and measured the response FLAT: Spearman of the objective
        # against lambda_gp was |rho| <= 0.14 (p >= 0.63) on three of the four studies, the
        # winners scattered to 2.72/14.07/3.98/1.61 with no cross-state consistency, and
        # retraining the CA winner with 10 in place of its searched 2.72 moved every
        # reported metric by <= 0.5 sigma of the seed spread (results/gp_probe).  Searching
        # a flat dimension dilutes the per-dim budget and hands the paper a "2.72" nobody
        # can defend.
        lambda_gp=10.0,
        # --- optimisation ---------------------------------------------------
        batch_size=t.suggest_categorical("batch_size", [256, 512, 1024]),
        epochs=t.suggest_int("epochs", 100, 800, step=50),
        n_critic=t.suggest_int("n_critic", 1, 5),
        lr_g=t.suggest_float("lr_g", 1e-5, 5e-4, log=True),
        lr_d=t.suggest_float("lr_d", 1e-5, 5e-4, log=True),
        # --- method-specific (the knobs this paper introduces) --------------
        gumbel_tau=t.suggest_float("gumbel_tau", 0.1, 1.0),
        lambda_m=t.suggest_float("lambda_m", 0.0, 1.0),
        # D24 adopted: 0.4 is the measured optimum (t=-8.78 vs 0).  Searched in a
        # narrow window around it rather than from 0, since 0 is known to be worse
        # than no conditioning at all.
        lambda_m_prior=t.suggest_float("lambda_m_prior", 0.2, 0.8),
        # lambda_cal is FIXED at 0: a full 2x3 design under EMA found no effect in
        # either mode (|t| <= 1.35, n=3) with lambda_prior as a positive control in
        # the same setup (t=-8.78).  Searching it would spend budget on a knob
        # measured to do nothing.
        lambda_cal=0.0,
        cond_jitter=t.suggest_float("cond_jitter", 0.0, 0.3),
    )


_SEMAPOP_DEFAULTS = dict(noise_dim=128, cond_dim=128, adapter_hidden=1024,
                         g_hidden="256 512 256", d_hidden="256 512 256",
                         adapter_dropout=0.1, dropout_g=0.0, dropout_d=0.0,
                         lambda_gp=10.0, batch_size=512, epochs=300, n_critic=5,
                         lr_g=2e-4, lr_d=2e-4, gumbel_tau=0.667, lambda_m=0.4,
                         lambda_m_prior=0.4, lambda_cal=0.0, cond_jitter=0.0)


def _semapop_argv(p: Dict[str, Any], ctx: C.TrialContext) -> List[str]:
    emb = C.emb_files(ctx.state)
    a = _base_argv(ctx, "semapop_gan_v2", ctx.extra["run_name"]) + [
        "--emb-file", emb["train"], "--emb-val-file", emb["val"],
        "--cond-source", "persona", "--cond-mode", ctx.cond_mode,
        "--noise-dim", str(p["noise_dim"]),
        "--cond-dim", str(p["cond_dim"]),
        "--adapter-hidden", str(p["adapter_hidden"]),
        "--adapter-dropout", f"{p['adapter_dropout']:.6g}",
        "--dropout-g", f"{p['dropout_g']:.6g}",
        "--dropout-d", f"{p['dropout_d']:.6g}",
        "--lambda-gp", f"{p['lambda_gp']:.6g}",
        "--batch-size", str(p["batch_size"]),
        "--epochs", str(ctx.cap_epochs(p["epochs"])),
        "--n-critic", str(p["n_critic"]),
        "--lr-g", f"{p['lr_g']:.6g}", "--lr-d", f"{p['lr_d']:.6g}",
        "--gumbel-tau", f"{p['gumbel_tau']:.6g}",
        "--lambda-m", f"{p['lambda_m']:.6g}",
        "--lambda-cal", f"{p['lambda_cal']:.6g}",
        "--cond-jitter", f"{p['cond_jitter']:.6g}",
        "--g-hidden", *_ints(p["g_hidden"]),
        "--d-hidden", *_ints(p["d_hidden"]),
    ]
    # The prior-conditioned marginal term and the M-U sampler both need a prior.
    if ctx.prior_path:
        a += ["--prior-path", ctx.prior_path, "--prior-family", ctx.prior_family,
              "--lambda-m-prior", f"{p['lambda_m_prior']:.6g}"]
        if ctx.subspace_path:
            a += ["--cond-subspace", ctx.subspace_path]
    if ctx.eval_every:
        a += ["--eval-every", str(ctx.eval_every)]
    return a


# =========================================================================== #
# 2. ctgan  (heavy, 16 searched dims)
# =========================================================================== #
def _ctgan_suggest(t) -> Dict[str, Any]:
    return dict(
        embedding_dim=t.suggest_categorical("embedding_dim", [64, 128, 256]),
        generator_dim=t.suggest_categorical("generator_dim", CTGAN_TRUNKS),
        discriminator_dim=t.suggest_categorical("discriminator_dim", CTGAN_TRUNKS),
        generator_decay=t.suggest_float("generator_decay", 1e-8, 1e-4, log=True),
        discriminator_decay=t.suggest_float("discriminator_decay", 1e-8, 1e-4, log=True),
        # batch_size stays a multiple of every pac option -> no invalid pairs
        batch_size=t.suggest_categorical("batch_size", [500, 1000, 2000]),
        epochs=t.suggest_int("epochs", 100, 800, step=50),
        discriminator_steps=t.suggest_int("discriminator_steps", 1, 5),
        generator_lr=t.suggest_float("generator_lr", 1e-5, 5e-4, log=True),
        discriminator_lr=t.suggest_float("discriminator_lr", 1e-5, 5e-4, log=True),
        pac=t.suggest_categorical("pac", [1, 2, 5, 10]),
        log_frequency=t.suggest_categorical("log_frequency", [True, False]),
    )


_CTGAN_DEFAULTS = dict(embedding_dim=128, generator_dim="256 256",
                       discriminator_dim="256 256", generator_decay=1e-6,
                       discriminator_decay=1e-6, batch_size=500, epochs=300,
                       discriminator_steps=1, generator_lr=2e-4,
                       discriminator_lr=2e-4, pac=10, log_frequency=True)


def _ctgan_argv(p: Dict[str, Any], ctx: C.TrialContext) -> List[str]:
    a = _base_argv(ctx, "ctgan_v2", ctx.extra["run_name"]) + [
        "--cols-config", "cols_config_v3.yaml",
        "--no-ref-eval",                      # objective is val-only
        "--embedding-dim", str(p["embedding_dim"]),
        "--generator-dim", *_ints(p["generator_dim"]),
        "--discriminator-dim", *_ints(p["discriminator_dim"]),
        "--generator-decay", f"{p['generator_decay']:.6g}",
        "--discriminator-decay", f"{p['discriminator_decay']:.6g}",
        "--batch-size", str(p["batch_size"]),
        "--epochs", str(ctx.cap_epochs(p["epochs"])),
        "--discriminator-steps", str(p["discriminator_steps"]),
        "--generator-lr", f"{p['generator_lr']:.6g}",
        "--discriminator-lr", f"{p['discriminator_lr']:.6g}",
        "--pac", str(p["pac"]),
    ]
    if not p["log_frequency"]:
        a += ["--no-log-frequency"]
    if ctx.eval_every:
        a += ["--eval-every", str(ctx.eval_every)]
    return a


# =========================================================================== #
# 3. tabddpm  (heavy, 16 searched dims)
# =========================================================================== #
def _tabddpm_suggest(t) -> Dict[str, Any]:
    return dict(
        hidden=t.suggest_categorical("hidden", DDPM_TRUNKS),
        dim_t=t.suggest_categorical("dim_t", [64, 128, 256]),
        dropout=t.suggest_float("dropout", 0.0, 0.3),
        weight_decay=t.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
        batch_size=t.suggest_categorical("batch_size", [1024, 2048, 4096]),
        epochs=t.suggest_int("epochs", 100, 800, step=50),
        lr=t.suggest_float("lr", 1e-4, 5e-3, log=True),
        grad_clip=t.suggest_float("grad_clip", 0.0, 5.0),
        # method-specific: the TabDDPM parametrisation choices
        timesteps=t.suggest_categorical("timesteps", [500, 1000, 2000]),
        schedule=t.suggest_categorical("schedule", ["cosine", "linear"]),
        parametrization=t.suggest_categorical("parametrization", ["x0", "direct"]),
        gaussian_parametrization=t.suggest_categorical("gaussian_parametrization",
                                                       ["eps", "x0"]),
        gaussian_loss=t.suggest_categorical("gaussian_loss", ["mse", "kl"]),
        time_sampler=t.suggest_categorical("time_sampler", ["uniform", "importance"]),
        lambda_multi=t.suggest_float("lambda_multi", 0.25, 4.0, log=True),
        lambda_gauss=t.suggest_float("lambda_gauss", 0.25, 4.0, log=True),
        num_transform=t.suggest_categorical("num_transform", ["quantile", "standard"]),
    )


_TABDDPM_DEFAULTS = dict(hidden="512 1024 1024 512", dim_t=128, dropout=0.0,
                         weight_decay=1e-8, batch_size=4096, epochs=200, lr=1e-3,
                         grad_clip=0.0, timesteps=1000, schedule="cosine",
                         parametrization="x0", gaussian_parametrization="eps",
                         gaussian_loss="mse", time_sampler="uniform",
                         lambda_multi=1.0, lambda_gauss=1.0, num_transform="quantile")


def _tabddpm_argv(p: Dict[str, Any], ctx: C.TrialContext) -> List[str]:
    a = _base_argv(ctx, "tabddpm_v2", ctx.extra["run_name"]) + [
        "--cols-config", "cols_config_v3.yaml",
        "--hidden", *_ints(p["hidden"]),
        "--dim-t", str(p["dim_t"]),
        "--dropout", f"{p['dropout']:.6g}",
        "--weight-decay", f"{p['weight_decay']:.6g}",
        "--batch-size", str(p["batch_size"]),
        "--epochs", str(ctx.cap_epochs(p["epochs"])),
        "--lr", f"{p['lr']:.6g}",
        "--grad-clip", f"{p['grad_clip']:.6g}",
        "--timesteps", str(p["timesteps"]),
        "--schedule", p["schedule"],
        "--parametrization", p["parametrization"],
        "--gaussian-parametrization", p["gaussian_parametrization"],
        "--gaussian-loss", p["gaussian_loss"],
        "--time-sampler", p["time_sampler"],
        "--lambda-multi", f"{p['lambda_multi']:.6g}",
        "--lambda-gauss", f"{p['lambda_gauss']:.6g}",
        "--num-transform", p["num_transform"],
    ]
    if ctx.eval_every:
        a += ["--eval-every", str(ctx.eval_every)]
    # Kotelnikov et al. do NO in-training model selection: Trainer.run_loop runs a fixed
    # number of steps and saves the final (and EMA) weights, with `steps` itself a
    # searched hyperparameter.  Taking a val-selected best checkpoint would add a
    # selection rule they never used.  Per-epoch ml_eff is still computed, because
    # pruning has to see the trajectory of the criterion being optimised.
    a += ["--load-final"]
    return a


# =========================================================================== #
# borysov  (Borysov, Rich & Pereira 2019 population-synthesis VAE -- our transcription)
# =========================================================================== #
BORYSOV_HIDDEN = ["25", "50", "100", "50 25", "100 50", "100 50 25"]   # paper Sec. 5.1 grid
_BORYSOV_DEFAULTS = dict(hidden_layers_shape="100 50 25", latent_dim=10, beta=0.5,
                         lr=1e-3, batch_size=64, epochs=100, n_bins=25)  # selected in 5/6 of their cases; 25 bins = as-shipped grid


def _borysov_suggest(t) -> Dict[str, Any]:
    # The paper's OWN grid for architecture / latent / beta (Sec. 5.1), plus a modest
    # window on the optimiser knobs the notebook fixed (lr 1e-3, batch 64, 100 epochs).
    return dict(
        hidden_layers_shape=t.suggest_categorical("hidden_layers_shape", BORYSOV_HIDDEN),
        latent_dim=t.suggest_categorical("latent_dim", [5, 10, 25]),
        beta=t.suggest_categorical("beta", [0.01, 0.05, 0.1, 0.5, 1.0, 10.0, 100.0]),
        lr=t.suggest_float("lr", 3e-4, 3e-3, log=True),
        batch_size=t.suggest_categorical("batch_size", [32, 64, 128]),
        epochs=t.suggest_int("epochs", 50, 300, step=25),
        # Sec. 5.2: "a sufficiently fine grid" for numerics -- the grid IS a design
        # choice of the encoding, so it is searched: 25 quantile bins (the SemaPop repository's
        # continuous-income setting), 50, or 95 (>= Age's 92 distinct values, i.e. every
        # numeric becomes an exact-value one-hot -- the finest grid).  Small-integer
        # counts are exact-value at every setting (see the trainer's Encoding).
        n_bins=t.suggest_categorical("n_bins", [25, 50, 95]),
    )


def _borysov_argv(p: Dict[str, Any], ctx: C.TrialContext, load_final: bool = False) -> List[str]:
    a = _base_argv(ctx, "borysov_vae_v2", ctx.extra["run_name"]) + [
        "--cols-config", "cols_config_v3.yaml",
        "--encoding", "discretised", "--n-bins", str(p.get("n_bins", 25)),   # Sec. 5.2 encoding; grid searched
        "--hidden-layers-shape", *_ints(p["hidden_layers_shape"]),
        "--latent-dim", str(p["latent_dim"]),
        "--beta", f"{p['beta']:.6g}",
        "--lr", f"{p['lr']:.6g}",
        "--batch-size", str(p["batch_size"]),
        "--epochs", str(ctx.cap_epochs(p["epochs"])),
    ]
    if load_final:
        a += ["--load-final"]          # faithful: no model selection in the released notebook
    if ctx.eval_every:
        a += ["--eval-every", str(ctx.eval_every)]
    return a


def _borysov_defaults_argv(p: Dict[str, Any], ctx: C.TrialContext) -> List[str]:
    return _borysov_argv(p, ctx, load_final=True)


def _borysov_defaults_suggest(t) -> Dict[str, Any]:
    return dict(_BORYSOV_DEFAULTS)


# =========================================================================== #
# flowsyn  (official FlowSyn protocol, NOT searched -- v3 protocol)
# =========================================================================== #
_FLOWSYN_DEFAULTS: Dict[str, Any] = {}   # the trainer's argparse defaults ARE the official protocol


def _flowsyn_suggest(t) -> Dict[str, Any]:
    # no_hpo: never called by run.py; present so ModelSpec is well-formed.
    return dict(_FLOWSYN_DEFAULTS)


def _flowsyn_argv(p: Dict[str, Any], ctx: C.TrialContext) -> List[str]:
    # train_flowsyn_v3.py's defaults already encode the official protocol
    # (VAE lr 1e-2 / flow 5e-4, caps 900/5000, val-latent early stopping,
    # masked-denoising VAE, QuantileTransformer, gMLP velocity + Beta source);
    # only the frozen context is passed.
    a = [
        "--data-dir", ctx.data_dir,
        "--cols-config", "cols_config_v3.yaml",
        "--eval-config", ctx.eval_config,
        "--f1-weight", str(ctx.f1_weight),
        "--seed", str(ctx.seed),
        "--sample-seed", str(ctx.sample_seed),
        "--device", ctx.device,
        "--out-root", ctx.out_root,
        "--tag", "flowsyn_v3",
        "--run-name", ctx.extra["run_name"],
        "--n-samples", str(ctx.n_samples or C.split_rows(ctx.state, "val", ctx.condition)),
    ]
    if ctx.subsample:
        a += ["--subsample", str(ctx.subsample)]
    if ctx.subsample_val:
        a += ["--subsample-val", str(ctx.subsample_val)]
    if ctx.max_epochs:
        a += ["--vae-epochs", str(ctx.max_epochs), "--flow-epochs", str(ctx.max_epochs)]
    return a


# =========================================================================== #
# tvae_ref  (reference-implementation TVAE, NOT searched -- v3 protocol)
# =========================================================================== #
_TVAE_REF_DEFAULTS = dict(embedding_dim=128, compress_dims="128 128",
                          decompress_dims="128 128", l2scale=1e-5,
                          batch_size=500, epochs=300, loss_factor=2.0)


def _tvae_ref_suggest(t) -> Dict[str, Any]:
    # no_hpo: never called by run.py; present so ModelSpec is well-formed.
    return dict(_TVAE_REF_DEFAULTS)


def _tvae_ref_argv(p: Dict[str, Any], ctx: C.TrialContext, load_final: bool = False) -> List[str]:
    a = _base_argv(ctx, "tvae_ref_v2", ctx.extra["run_name"]) + [
        "--cols-config", "cols_config_v3.yaml",
        "--no-ref-eval",                      # objective is val-only
        "--embedding-dim", str(p["embedding_dim"]),
        "--compress-dims", *_ints(p["compress_dims"]),
        "--decompress-dims", *_ints(p["decompress_dims"]),
        "--l2scale", f"{p['l2scale']:.6g}",
        "--batch-size", str(p["batch_size"]),
        "--epochs", str(ctx.cap_epochs(p["epochs"])),
        "--loss-factor", f"{p['loss_factor']:.6g}",
    ]
    if load_final:
        # Xu et al. (2019) do no model selection for TVAE; the as-shipped row takes
        # the last epoch.  The SEARCHED row uses the per-epoch val_srmse checkpoint.
        a += ["--load-final"]
    if ctx.eval_every:
        a += ["--eval-every", str(ctx.eval_every)]
    return a


def _tvae_ref_defaults_argv(p: Dict[str, Any], ctx: C.TrialContext) -> List[str]:
    return _tvae_ref_argv(p, ctx, load_final=True)


TVAE_TRUNKS = ["128 128", "256 256", "128 256 128", "256 256 256"]


def _tvae_ref_search_suggest(t) -> Dict[str, Any]:
    # 7 dims over the package's own knobs, centred on the published values
    # (embedding 128, 128-128 compress/decompress, l2scale 1e-5, batch 500,
    # epochs 300, loss_factor 2).
    return dict(
        embedding_dim=t.suggest_categorical("embedding_dim", [64, 128, 256]),
        compress_dims=t.suggest_categorical("compress_dims", TVAE_TRUNKS),
        decompress_dims=t.suggest_categorical("decompress_dims", TVAE_TRUNKS),
        l2scale=t.suggest_float("l2scale", 1e-7, 1e-3, log=True),
        batch_size=t.suggest_categorical("batch_size", [250, 500, 1000]),
        epochs=t.suggest_int("epochs", 100, 800, step=50),
        loss_factor=t.suggest_float("loss_factor", 0.5, 8.0, log=True),
    )


# =========================================================================== #
# 4. wgan_zcr  (heavy, 16 searched dims)
# =========================================================================== #
def _wganzcr_suggest(t) -> Dict[str, Any]:
    return dict(
        noise_dim=t.suggest_categorical("noise_dim", [64, 128, 256]),
        g_hidden=t.suggest_categorical("g_hidden", MLP_TRUNKS),
        d_hidden=t.suggest_categorical("d_hidden", MLP_TRUNKS),
        dropout_g=t.suggest_float("dropout_g", 0.0, 0.3),
        dropout_d=t.suggest_float("dropout_d", 0.0, 0.3),
        # FIXED at 10, same evidence and date as the semapop_gan arm: rho=-0.06 (CA) and
        # +0.03 (FL) against the objective, winners at 3.98 and 1.61.  See above.
        lambda_gp=10.0,
        batch_size=t.suggest_categorical("batch_size", [256, 512, 1024]),
        epochs=t.suggest_int("epochs", 100, 800, step=50),
        n_critic=t.suggest_int("n_critic", 1, 5),
        # ceiling raised to 1e-2 so the paper's Adam lr of 0.005 (Kim & Bansal
        # Fig. 9) is reachable; the previous 5e-4 excluded it by 10x.
        lr_g=t.suggest_float("lr_g", 1e-5, 1e-2, log=True),
        lr_d=t.suggest_float("lr_d", 1e-5, 1e-2, log=True),
        gumbel_tau=t.suggest_float("gumbel_tau", 0.1, 1.0),
        # method-specific: the ZCR (zero-centred regularisation) knobs
        # Search ONLY what Kim & Bansal expose as hyperparameters.  Eq.(11)'s
        # gamma_BD and gamma_AD are calibrated in the paper (its Fig. 9), so tuning
        # them is a fair search.  The lower bound is lifted
        # off 0 because bd_weight=ad_weight=0 IS plain WGAN-GP, and a "WGAN-GP-ZCR"
        # row that silently tuned its way back to vanilla would be mislabelled --
        # that ablation is already reported separately through --no-zcr.
        # The range must CONTAIN the published calibration (discrete, both terms
        # active: gamma_BD 0.5, gamma_AD 0.005) or the "tuned baseline" is tuned
        # inside a box that excludes the paper's own setting.  The previous
        # ad_weight floor of 0.05 missed 0.005 by 10x.  Log scale, because gamma_AD
        # spans three orders across the paper's configurations (0.005 discrete,
        # 0.05 embedded, 0.9 for the VAE).
        bd_weight=t.suggest_float("bd_weight", 0.05, 5.0, log=True),
        ad_weight=t.suggest_float("ad_weight", 0.001, 2.0, log=True),
        # zcr_cat_scale and zcr_every are OUR additions, not the paper's; pin them at
        # the values that reproduce it (1.0 = no re-weighting; every generator step,
        # as Eq.(11) implies).  Searching them would strengthen the baseline at the
        # cost of it no longer being the published method.
        zcr_cat_scale=1.0,
        zcr_every=1,
    )


_WGANZCR_DEFAULTS = dict(noise_dim=128, g_hidden="256 512 256", d_hidden="256 512 256",
                         dropout_g=0.0, dropout_d=0.0, lambda_gp=10.0, batch_size=512,
                         epochs=300, n_critic=5, lr_g=2e-4, lr_d=2e-4,
                         gumbel_tau=0.667, bd_weight=0.5, ad_weight=0.005,
                         zcr_cat_scale=1.0, zcr_every=1)


def _wganzcr_argv(p: Dict[str, Any], ctx: C.TrialContext) -> List[str]:
    a = _base_argv(ctx, "wgan_zcr_v2", ctx.extra["run_name"]) + [
        "--cols-config", "cols_config_v3.yaml",
        "--noise-dim", str(p["noise_dim"]),
        "--g-hidden", *_ints(p["g_hidden"]),
        "--d-hidden", *_ints(p["d_hidden"]),
        "--dropout-g", f"{p['dropout_g']:.6g}",
        "--dropout-d", f"{p['dropout_d']:.6g}",
        "--lambda-gp", f"{p['lambda_gp']:.6g}",
        "--batch-size", str(p["batch_size"]),
        "--epochs", str(ctx.cap_epochs(p["epochs"])),
        "--n-critic", str(p["n_critic"]),
        "--lr-g", f"{p['lr_g']:.6g}", "--lr-d", f"{p['lr_d']:.6g}",
        "--gumbel-tau", f"{p['gumbel_tau']:.6g}",
        "--bd-weight", f"{p['bd_weight']:.6g}",
        "--ad-weight", f"{p['ad_weight']:.6g}",
        "--zcr-cat-scale", f"{p['zcr_cat_scale']:.6g}",
        "--zcr-every", str(p["zcr_every"]),
    ]
    if ctx.eval_every:
        a += ["--eval-every", str(ctx.eval_every)]
    return a


# =========================================================================== #
# 5. bn  (light, 8 searched dims -- the model's full set of real knobs)
# =========================================================================== #
def _bn_suggest(t) -> Dict[str, Any]:
    return dict(
        max_parents=t.suggest_int("max_parents", 1, 5),
        score=t.suggest_categorical("score", ["bic", "bdeu"]),
        ess=t.suggest_float("ess", 1.0, 100.0, log=True),
        dirichlet_alpha=t.suggest_float("dirichlet_alpha", 0.01, 10.0, log=True),
        alpha_mode=t.suggest_categorical("alpha_mode", ["fixed", "bdeu"]),
        numeric_decode=t.suggest_categorical("numeric_decode", ["empirical", "midpoint"]),
        max_iter=t.suggest_categorical("max_iter", [200, 400, 800]),
        tol=t.suggest_float("tol", 1e-8, 1e-4, log=True),
    )


_BN_DEFAULTS = dict(max_parents=3, score="bic", ess=10.0, dirichlet_alpha=1.0,
                    alpha_mode="fixed", numeric_decode="empirical", max_iter=400,
                    tol=1e-6)


def _bn_argv(p: Dict[str, Any], ctx: C.TrialContext) -> List[str]:
    # NOTE: the trainer accepts nargs="+" for max-parents / score / dirichlet-alpha
    # and would run its OWN internal grid.  We pass exactly one value each so the
    # trial budget is what Optuna spends, not what the trainer spends internally.
    a = [
        "--data-dir", ctx.data_dir,
        "--eval-config", ctx.eval_config,
        "--f1-weight", str(ctx.f1_weight),
        "--seed", str(ctx.seed),
        "--sample-seed", str(ctx.sample_seed),
        "--out-root", ctx.out_root,
        "--tag", "bn_v2",
        "--run-name", ctx.extra["run_name"],
        "--n-samples", str(ctx.n_samples or C.split_rows(ctx.state, "val", ctx.condition)),
        "--ref-file", "",                     # objective is val-only
        "--no-coverage",
        "--cols-config", "cols_config_v3.yaml",
        # 13-attr build: referencedata has no PWGTP; harmless in trials
        # (--ref-file ''), required when finalize opens the reference
        "--ref-weight-col", "",
        "--max-parents", str(p["max_parents"]),
        "--score", p["score"],
        "--ess", f"{p['ess']:.6g}",
        "--dirichlet-alpha", f"{p['dirichlet_alpha']:.6g}",
        "--alpha-mode", p["alpha_mode"],
        "--numeric-decode", p["numeric_decode"],
        "--max-iter", str(p["max_iter"]),
        "--tol", f"{p['tol']:.6g}",
    ]
    if ctx.subsample:
        a += ["--subsample", str(ctx.subsample)]
    if ctx.subsample_val:
        a += ["--subsample-val", str(ctx.subsample_val)]
    return a


# --------------------------------------------------------------------------- #
# 5b. bn_pgmpy = the AUDITED BN since 2026-11-23; bn_pgmpy_bic = diagnostic only
# the SemaPop repository's latest BN (train_bn_pgmpy.py @ e2b27f1),
# ported with the K=50 ensemble writer.  bn_pgmpy = the SemaPop repository's reported configuration
# (hpo/spaces_paper.BN_REPORTED: BDeu, max in-degree 4, ess 66.6383, MLE CPTs, empirical
# decoding); bn_pgmpy_bic = the same trainer at its literature defaults (BIC, max in-degree
# 3, MLE) with empirical decoding, to separate the backend effect from the config effect.
# --------------------------------------------------------------------------- #
_BN_PGMPY_LLMPOP = dict(score="bdeu", max_indegree=4, ess=66.6383, cpt="mle",
                        decode="empirical")
_BN_PGMPY_BIC = dict(score="bic", max_indegree=3, ess=10.0, cpt="mle", decode="empirical")


def _bn_pgmpy_suggest(t) -> Dict[str, Any]:
    return {}                                   # no_hpo: nothing is searched


def _bn_pgmpy_argv(p: Dict[str, Any], ctx: C.TrialContext) -> List[str]:
    a = [
        "--data-dir", ctx.data_dir,
        "--eval-config", ctx.eval_config,
        "--seed", str(ctx.seed),
        "--sample-seed", str(ctx.sample_seed),
        "--out-root", ctx.out_root,
        "--tag", "bn_pgmpy",
        "--run-name", ctx.extra["run_name"],
        "--n-samples", str(ctx.n_samples or C.split_rows(ctx.state, "val", ctx.condition)),
        "--cols-config", "cols_config_v3.yaml",
        "--score", p["score"],
        "--max-indegree", str(p["max_indegree"]),
        "--ess", f"{p['ess']:.6g}",
        "--cpt", p["cpt"],
        "--decode", p["decode"],
    ]
    return a


# =========================================================================== #
# 6. semapop_ddpm  (STUB -- trainer not written yet, plan W3-W4)
# =========================================================================== #
def _semapop_ddpm_suggest(t) -> Dict[str, Any]:
    p = _tabddpm_suggest(t)
    p.update(
        cond_dim=t.suggest_categorical("cond_dim", [64, 128, 256]),
        adapter_hidden=t.suggest_categorical("adapter_hidden", [512, 1024, 2048]),
        adapter_dropout=t.suggest_float("adapter_dropout", 0.0, 0.3),
        cond_jitter=t.suggest_float("cond_jitter", 0.0, 0.3),
        lambda_m=t.suggest_float("lambda_m", 0.0, 1.0),
        # D24 adopted: 0.4 is the measured optimum (t=-8.78 vs 0).  Searched in a
        # narrow window around it rather than from 0, since 0 is known to be worse
        # than no conditioning at all.
        lambda_m_prior=t.suggest_float("lambda_m_prior", 0.2, 0.8),
    )
    return p


def _semapop_ddpm_argv(p: Dict[str, Any], ctx: C.TrialContext) -> List[str]:
    emb = C.emb_files(ctx.state)
    a = _tabddpm_argv(p, ctx)
    a[a.index("--tag") + 1] = "semapop_ddpm_v2"
    a += ["--emb-file", emb["train"], "--emb-val-file", emb["val"],
          "--cond-source", "persona", "--cond-mode", ctx.cond_mode,
          "--cond-dim", str(p["cond_dim"]),
          "--adapter-hidden", str(p["adapter_hidden"]),
          "--adapter-dropout", f"{p['adapter_dropout']:.6g}",
          "--cond-jitter", f"{p['cond_jitter']:.6g}",
          "--lambda-m", f"{p['lambda_m']:.6g}"]
    if ctx.prior_path:
        a += ["--prior-path", ctx.prior_path, "--prior-family", ctx.prior_family,
              "--lambda-m-prior", f"{p['lambda_m_prior']:.6g}"]
        # Same reason as the GAN arm: the prior samples in a PCA subspace, so the model
        # has to be TRAINED on conditions in that subspace or it is sampled off a
        # manifold it never saw.  Omitting this is a silent M-U penalty, not an error.
        if ctx.subspace_path:
            a += ["--cond-subspace", ctx.subspace_path]
    return a


# =========================================================================== #
# registry
# =========================================================================== #

# =========================================================================== #
# 7. semapop_vae  (the ITSC 2026 SemaPop-VAE+, model copied verbatim; searched
#    around its PUBLISHED configuration, which is also the registry default)
# =========================================================================== #
def _semapop_vae_suggest(t) -> Dict[str, Any]:
    return dict(
        # --- capacity: published latent 128 / cond 128 / enc=dec 512x3 -------
        latent_dim=t.suggest_categorical("latent_dim", [64, 128, 256]),
        # 32 included below the published 128: on this tier both GAN-arm states
        # pushed cond_dim to their bottom choice, so the range must extend past it
        cond_dim=t.suggest_categorical("cond_dim", [32, 64, 128, 256]),
        hidden=t.suggest_categorical("hidden", [256, 512, 1024]),
        depth=t.suggest_int("depth", 2, 4),
        adapter_hidden=t.suggest_categorical("adapter_hidden", [512, 1024, 2048]),
        # --- regularisation ---------------------------------------------------
        adapter_dropout=t.suggest_float("adapter_dropout", 0.0, 0.3),
        dropout=t.suggest_float("dropout", 0.0, 0.3),
        # --- optimisation -----------------------------------------------------
        lr=t.suggest_float("lr", 1e-5, 1e-3, log=True),
        batch_size=t.suggest_categorical("batch_size", [256, 512, 1024]),
        epochs=t.suggest_int("epochs", 100, 800, step=50),
        # --- method-specific --------------------------------------------------
        # beta is THE VAE knob (posterior-collapse vs reconstruction); log around
        # the published 1.0
        beta=t.suggest_float("beta", 0.1, 4.0, log=True),
        # published lambda_m=2; searched log around it, 0 excluded (the ablation
        # ladder owns the off switch, not the search)
        lambda_m=t.suggest_float("lambda_m", 0.25, 4.0, log=True),
        # pinned at the published values, not searched:
        weight_decay=0.0, beta_warmup_epochs=0, grad_clip=1.0,
    )


_SEMAPOP_VAE_DEFAULTS = dict(latent_dim=128, cond_dim=128, hidden=512, depth=3,
                             adapter_hidden=1024, adapter_dropout=0.1, dropout=0.1,
                             lr=2e-4, batch_size=512, epochs=300, beta=1.0,
                             lambda_m=2.0, weight_decay=0.0, beta_warmup_epochs=0,
                             grad_clip=1.0)


def _semapop_vae_argv(p: Dict[str, Any], ctx: C.TrialContext) -> List[str]:
    emb = C.emb_files(ctx.state)
    a = _base_argv(ctx, "semapop_vae_v2", ctx.extra["run_name"]) + [
        "--cols-config", "cols_config_v3.yaml",
        "--emb-file", emb["train"], "--emb-val-file", emb["val"],
        "--cond-source", "persona", "--cond-mode", ctx.cond_mode,
        "--latent-dim", str(p["latent_dim"]),
        "--cond-dim", str(p["cond_dim"]),
        "--enc-hidden", str(p["hidden"]), "--dec-hidden", str(p["hidden"]),
        "--enc-depth", str(p["depth"]), "--dec-depth", str(p["depth"]),
        "--adapter-hidden", str(p["adapter_hidden"]),
        "--adapter-dropout", f"{p['adapter_dropout']:.6g}",
        "--dropout", f"{p['dropout']:.6g}",
        "--lr", f"{p['lr']:.6g}",
        "--weight-decay", f"{p['weight_decay']:.6g}",
        "--batch-size", str(p["batch_size"]),
        "--epochs", str(ctx.cap_epochs(p["epochs"])),
        "--beta", f"{p['beta']:.6g}",
        "--beta-warmup-epochs", str(p["beta_warmup_epochs"]),
        "--lambda-m", f"{p['lambda_m']:.6g}",
        "--grad-clip", f"{p['grad_clip']:.6g}",
    ]
    # v3 protocol: the prior + subspace projection belong ONLY to the mu (M-U) arm.
    # The audited MAIN arm is cond-mode train (the published ITSC procedure:
    # conditions resampled from RAW training embeddings); passing --cond-subspace
    # there would train on projected embeddings and silently change the arm.
    if ctx.cond_mode == "mu" and ctx.prior_path:
        a += ["--prior-path", ctx.prior_path, "--prior-family", ctx.prior_family]
        if ctx.subspace_path:
            a += ["--cond-subspace", ctx.subspace_path]
    if ctx.eval_every:
        a += ["--eval-every", str(ctx.eval_every)]
    return a



# =========================================================================== #
# 8. tvae_plain -- the with/without-framework CONTROL for the VAE backbone.
#    Same trainer and architecture as semapop_vae; conditioning is a constant
#    zero vector (cond-source none), so FiLM and the PriorNet receive no
#    information and the marginal regulariser is off.  What remains is the bare
#    TVAE backbone under an honest search of its own -- the third with/without
#    pair (tabddpm/semapop_ddpm and wgan_plain/semapop_gan are the other two).
#    Zero-information conditioning rather than surgically removing the modules
#    keeps parameter count and code path identical, so the pair isolates the
#    INFORMATION, not architecture capacity.
# =========================================================================== #
def _tvae_plain_suggest(t) -> Dict[str, Any]:
    return dict(
        latent_dim=t.suggest_categorical("latent_dim", [64, 128, 256]),
        hidden=t.suggest_categorical("hidden", [256, 512, 1024]),
        depth=t.suggest_int("depth", 2, 4),
        dropout=t.suggest_float("dropout", 0.0, 0.3),
        lr=t.suggest_float("lr", 1e-5, 1e-3, log=True),
        batch_size=t.suggest_categorical("batch_size", [256, 512, 1024]),
        epochs=t.suggest_int("epochs", 100, 800, step=50),
        beta=t.suggest_float("beta", 0.1, 4.0, log=True),
        # dead-weight conditioning modules pinned minimal; framework knobs OFF
        cond_dim=32, adapter_hidden=512, adapter_dropout=0.0,
        lambda_m=0.0, weight_decay=0.0, beta_warmup_epochs=0, grad_clip=1.0,
    )


_TVAE_PLAIN_DEFAULTS = dict(latent_dim=128, hidden=512, depth=3, dropout=0.1,
                            lr=2e-4, batch_size=512, epochs=300, beta=1.0,
                            cond_dim=32, adapter_hidden=512, adapter_dropout=0.0,
                            lambda_m=0.0, weight_decay=0.0, beta_warmup_epochs=0,
                            grad_clip=1.0)


def _tvae_plain_argv(p: Dict[str, Any], ctx: C.TrialContext) -> List[str]:
    a = _base_argv(ctx, "tvae_plain_v2", ctx.extra["run_name"]) + [
        # cond-source none feeds a constant zero vector; cond-mode mr then draws
        # those zeros back at sampling time, so NO PRIOR is involved anywhere.
        "--cond-source", "none", "--cond-mode", "mr",
        "--latent-dim", str(p["latent_dim"]),
        "--cond-dim", str(p["cond_dim"]),
        "--enc-hidden", str(p["hidden"]), "--dec-hidden", str(p["hidden"]),
        "--enc-depth", str(p["depth"]), "--dec-depth", str(p["depth"]),
        "--adapter-hidden", str(p["adapter_hidden"]),
        "--adapter-dropout", f"{p['adapter_dropout']:.6g}",
        "--dropout", f"{p['dropout']:.6g}",
        "--lr", f"{p['lr']:.6g}",
        "--weight-decay", f"{p['weight_decay']:.6g}",
        "--batch-size", str(p["batch_size"]),
        "--epochs", str(ctx.cap_epochs(p["epochs"])),
        "--beta", f"{p['beta']:.6g}",
        "--beta-warmup-epochs", str(p["beta_warmup_epochs"]),
        "--lambda-m", "0",
        "--grad-clip", f"{p['grad_clip']:.6g}",
    ]
    if ctx.eval_every:
        a += ["--eval-every", str(ctx.eval_every)]
    return a


# =========================================================================== #
# 9. ttvae  (third-party transformer VAE + MMD, heavy, 12 searched dims)
# =========================================================================== #
def _ttvae_suggest(t) -> Dict[str, Any]:
    # Ported VERBATIM (same ranges) from the old harness,
    # experiments/hpo/search_spaces.py::_ttvae_suggest.
    return dict(
        d_model=t.suggest_int("d_model", 64, 256, step=32),
        nhead=t.suggest_categorical("nhead", [4, 8, 16]),
        num_layers=t.suggest_int("num_layers", 1, 4),
        dim_ff=t.suggest_int("dim_ff", 256, 1536, step=256),
        dropout=t.suggest_float("dropout", 0.0, 0.3, step=0.05),
        z_dim=t.suggest_int("z_dim", 32, 192, step=32),
        lr=t.suggest_float("lr", 1e-5, 1e-3, log=True),
        weight_decay=t.suggest_float("weight_decay", 1e-7, 1e-4, log=True),
        batch_size=t.suggest_categorical("batch_size", [256, 512, 1024]),
        epochs=t.suggest_int("epochs", 60, 300, step=20),
        beta_mmd=t.suggest_float("beta_mmd", 0.1, 10.0, log=True),
        patience=t.suggest_int("patience", 8, 20),
    )


# The trainer's built-in defaults (train_ttvae_CA.py; flag > TTVAE_* env > these).
# --check-defaults NOTE: `epochs` deliberately fails rule 2 -- the built-in 1000
# is an early-stop CAP (patience 10 ends training long before it), not a tuned
# value, and the searched 60-300 range is the old harness's, ported verbatim as
# required.  Every other default is reachable inside its range.
_TTVAE_DEFAULTS = dict(d_model=128, nhead=8, num_layers=3, dim_ff=512,
                       dropout=0.1, z_dim=128, lr=2e-5, weight_decay=1e-6,
                       batch_size=512, epochs=1000, beta_mmd=1.0, patience=10)


def _ttvae_argv(p: Dict[str, Any], ctx: C.TrialContext) -> List[str]:
    # NOT _base_argv: this trainer takes the STATE as a positional first
    # argument and has no --data-dir/--tag -- it resolves its data
    # repo-relative (data/pums_states/<ST>), the same location ctx.data_dir
    # points at through the repo's data/ directory (a symlink is fine).
    a = [
        ctx.state,
        "--cols-config", "cols_config_v3.yaml",
        "--eval-config", ctx.eval_config,
        "--f1-weight", str(ctx.f1_weight),
        "--select-on", ctx.criterion,          # accepted + recorded by the trainer
        "--seed", str(ctx.seed),
        "--sample-seed", str(ctx.sample_seed),
        "--device", ctx.device,
        "--out-root", ctx.out_root,
        # the trainer writes its run dir EXACTLY at <out_root>/<run_name> and
        # metrics.json inside -- what run.py's objective_from_run reads; without
        # --ensemble/--hpo-out the trainer keeps the dir and skips the ensemble
        "--run-name", ctx.extra["run_name"],
        "--n-samples", str(ctx.n_samples or C.split_rows(ctx.state, "val", ctx.condition)),
        # the published arm by default; finalize --sampling-arm prior overrides
        "--sample-method", ctx.extra.get("sample_method", "triangular"),
        "--d-model", str(p["d_model"]),
        "--nhead", str(p["nhead"]),
        "--num-layers", str(p["num_layers"]),
        "--dim-feedforward", str(p["dim_ff"]),
        "--dropout", f"{p['dropout']:.6g}",
        "--z-dim", str(p["z_dim"]),
        "--lr", f"{p['lr']:.6g}",
        "--weight-decay", f"{p['weight_decay']:.6g}",
        "--batch-size", str(p["batch_size"]),
        "--epochs", str(ctx.cap_epochs(p["epochs"])),
        "--beta-mmd", f"{p['beta_mmd']:.6g}",
        "--patience", str(p["patience"]),
    ]
    if ctx.subsample:
        a += ["--subsample", str(ctx.subsample)]
    if ctx.subsample_val:
        a += ["--subsample-val", str(ctx.subsample_val)]
    return a


REGISTRY: Dict[str, ModelSpec] = {
    "tvae_plain": ModelSpec(
        available=False,  # EXCLUDED: replaced by the reference-implementation TVAE (train_tvae_ref_v2.py)
        name="tvae_plain", script="train_semapop_vae_v2.py", weight="heavy",
        timeout_s=4 * 3600, suggest=_tvae_plain_suggest, argv=_tvae_plain_argv,
        n_dims=8, defaults=_TVAE_PLAIN_DEFAULTS,
        needs_prior=False,
        criterion="val_srmse",
        notes="Bare-TVAE control for the VAE with/without-framework pair: zero-information "
              "conditioning, lambda_m off, same architecture and search protocol as "
              "semapop_vae, selected on the same criterion.",
    ),
    "semapop_vae": ModelSpec(
        name="semapop_vae", script="train_semapop_vae_v2.py", weight="heavy",
        timeout_s=4 * 3600, suggest=_semapop_vae_suggest, argv=_semapop_vae_argv,
        n_dims=12, defaults=_SEMAPOP_VAE_DEFAULTS,
        needs_prior=True,
        criterion="val_srmse",
        notes="SemaPop-VAE+ (ITSC 2026): TVAE + FiLM + PriorNet, model file copied "
              "verbatim from pop_audit; searched around the published configuration. "
              "Selection follows the family's val_srmse lineage.",
    ),
    "semapop_gan": ModelSpec(
        available=False,  # EXCLUDED from the v3 audit: method outside this paper
        name="semapop_gan", script="train_semapop_gan_v2.py", weight="heavy",
        timeout_s=4 * 3600, suggest=_semapop_suggest, argv=_semapop_argv,
        # 17, corrected from 19 on 2026-08-28: lambda_cal was never searched (fixed 0,
        # measured no-effect) yet was counted, and lambda_gp is now fixed at 10.  n_dims
        # feeds n_trials_for_dims, so an overcount is a hidden budget subsidy -- at 19
        # dims this model drew 48 CA trials for 17 searched dimensions (2.8/dim) while
        # wgan_zcr got 2.5/dim, exactly the confound the per-dim rule exists to prevent.
        n_dims=17, defaults=_SEMAPOP_DEFAULTS,
        needs_prior=True,
        notes="persona-conditioned WGAN-GP + FiLM adapter. lambda_m_prior/"
              "cond_subspace only take effect when a fitted prior is present; "
              "the prior-fit cost is charged to this model in the budget table."),
    "ctgan": ModelSpec(
        name="ctgan", script="train_ctgan_v2.py", weight="heavy",
        timeout_s=4 * 3600, suggest=_ctgan_suggest, argv=_ctgan_argv,
        n_dims=12, defaults=_CTGAN_DEFAULTS, criterion="val_srmse",
        notes="SEARCHED (O1 rework, user decision 2026-09-01). The author-default row "
              "is kept as the separate 'ctgan_defaults' entry (the as-shipped "
              "disclosure); this row answers R2-1.2 ('were the baselines optimally "
              "tuned?') head-on. llm_pop's 12-dim space, centred on the published "
              "configuration; llm_pop's own searched CTGAN reached reference SRMSE-1 "
              "0.053 on 20 attributes against 0.20 for the defaults here, so the "
              "gap is configuration, not implementation."),
    "ctgan_defaults": ModelSpec(
        name="ctgan_defaults", script="train_ctgan_v2.py", weight="heavy",
        timeout_s=4 * 3600, suggest=_ctgan_suggest, argv=_ctgan_argv,
        n_dims=0, prunable=False, no_hpo=True, defaults=_CTGAN_DEFAULTS,
        criterion="val_srmse",
        notes="As-shipped row: one run at train_ctgan_v2.py's argparse defaults (= the "
              "published Xu et al. 2019 configuration, reused unchanged across their 15 "
              "datasets; ctgan 0.11.1's fit() has no validation, no early stopping). "
              "NOTE: the row already produced used the trainer's per-epoch val_srmse "
              "checkpoint (argv carries no --load-final), i.e. it is the defaults plus "
              "best-epoch selection -- disclosed as such."),
    "borysov": ModelSpec(
        name="borysov", script="train_borysov_vae_v2.py", weight="light",
        timeout_s=2 * 3600, suggest=_borysov_suggest, argv=_borysov_argv,
        n_dims=7, defaults=_BORYSOV_DEFAULTS, criterion="val_srmse",
        notes="Borysov-style VAE (OUR PyTorch transcription of stasmix/popsynth cells "
              "36-38; their notebook cannot execute). SEARCHED over the paper's own "
              "Sec. 5.1 grid (hidden shape x latent x beta) plus a modest optimiser "
              "window; criterion val_srmse matches the paper's SRMSE-based selection. "
              "Encoding = 'discretised' (25 quantile bins per numeric, one-hot all), "
              "the variant Sec. 5.2 recommends and Tables 2-3 prefer. Per-epoch "
              "val_srmse checkpoint. Label in tables: 'Borysov-style VAE (our impl.)'."),
    "borysov_defaults": ModelSpec(
        name="borysov_defaults", script="train_borysov_vae_v2.py", weight="light",
        timeout_s=2 * 3600, suggest=_borysov_defaults_suggest, argv=_borysov_defaults_argv,
        n_dims=0, prunable=False, no_hpo=True, defaults=_BORYSOV_DEFAULTS,
        criterion="val_srmse",
        notes="As-shipped row: the configuration the paper selected in 5 of 6 cases "
              "(100-50-25, latent 10, beta 0.5, RMSprop 1e-3, batch 64, 100 epochs), "
              "--load-final (no selection in the released notebook)."),
    "flowsyn": ModelSpec(
        name="flowsyn", script="train_flowsyn_v3.py", weight="heavy",
        timeout_s=6 * 3600, suggest=_flowsyn_suggest, argv=_flowsyn_argv,
        n_dims=0, prunable=False, no_hpo=True, defaults=_FLOWSYN_DEFAULTS,
        notes="NOT SEARCHED -- the official FlowSyn protocol IS one fixed "
              "configuration (official LRs 1e-2/5e-4, epoch caps 900/5000 with "
              "val-side early stopping inside the trainer). One run at the "
              "trainer's defaults per state/seed. train_flowsyn_v3.py also fixes "
              "two leaks the old pop_audit trainer had (encoders fitted on "
              "train+val+TEST; HPO quick mode scored against the reference); "
              "generation is ode_sample from Beta-source noise -- no record reuse. "
              "No tail_val_metrics by design (selection inside the trainer is on "
              "reconstruction/CFM loss per the authors); the objective reads "
              "final_val_metrics."),
    "tvae_ref": ModelSpec(
        name="tvae_ref", script="train_tvae_ref_v2.py", weight="heavy",
        timeout_s=4 * 3600, suggest=_tvae_ref_search_suggest, argv=_tvae_ref_argv,
        n_dims=7, defaults=_TVAE_REF_DEFAULTS, criterion="val_srmse",
        notes="SEARCHED (O1 rework, user decision 2026-09-01) over the reference "
              "package's own 7 knobs, published values inside every range; per-epoch "
              "val_srmse checkpoint (the tqdm hook makes it available). The as-shipped "
              "row is the separate 'tvae_ref_defaults' entry."),
    "tvae_ref_defaults": ModelSpec(
        name="tvae_ref_defaults", script="train_tvae_ref_v2.py", weight="heavy",
        timeout_s=4 * 3600, suggest=_tvae_ref_suggest, argv=_tvae_ref_defaults_argv,
        n_dims=0, prunable=False, no_hpo=True, defaults=_TVAE_REF_DEFAULTS,
        criterion="val_srmse",
        notes="As-shipped row: ctgan 0.11.1's TVAE defaults (= published Xu et al. 2019 "
              "values) with --load-final (no per-epoch selection, as the authors)."),
    "ttvae": ModelSpec(
        name="ttvae", script="train_ttvae_CA.py", weight="heavy",
        timeout_s=4 * 3600, suggest=_ttvae_suggest, argv=_ttvae_argv,
        n_dims=12, defaults=_TTVAE_DEFAULTS,
        # no per-epoch eval_v2 "[ep N] ... score=" line to parse -- the model's
        # own selection is its internal val-loss early stopping (patience), so
        # there is nothing for the median pruner to read
        prunable=False,
        criterion="val_srmse",
        notes="Third-party TTVAE (transformer VAE + MMD), state as positional "
              "argv[0], data resolved repo-relative (data/pums_states/<ST> == "
              "ctx.data_dir via the data/ -> SSD symlink; no --data-dir flag). "
              "--sample-method triangular is the paper's PUBLISHED sampling "
              "(encode real rows, Dirichlet convex combos of posterior latents, "
              "decode with contextual h) and is the main audited arm -- HPO "
              "searches under it; the prior arm (z~N(0,I), h_ctx=0) is produced "
              "at finalize time from the same checkpoint with --sample-method "
              "prior, never searched separately. Space ported verbatim from the "
              "old harness (experiments/hpo/search_spaces.py). Trials never "
              "pass --ensemble, so the trainer skips the K=50 loop, keeps "
              "<out_root>/<run_name>/metrics.json and never opens the "
              "reference split."),
    "tabddpm": ModelSpec(
        name="tabddpm", script="train_tabddpm_v2.py", weight="heavy",
        timeout_s=4 * 3600, suggest=_tabddpm_suggest, argv=_tabddpm_argv,
        n_dims=17, defaults=_TABDDPM_DEFAULTS,
        criterion="ml_eff"),
    "wgan_zcr": ModelSpec(
        name="wgan_zcr", script="train_wgan_zcr_v2.py", weight="heavy",
        timeout_s=4 * 3600, suggest=_wganzcr_suggest, argv=_wganzcr_argv,
        # 13: lambda_gp fixed (see the space), and the 16 also overcounted zcr_cat_scale
        # and zcr_every, which are pinned constants in the space (searched nowhere).
        n_dims=13, defaults=_WGANZCR_DEFAULTS,
        criterion="if1"),
    "bn": ModelSpec(
        name="bn", script="train_bn_v2.py", weight="light",
        timeout_s=2 * 3600, suggest=_bn_suggest, argv=_bn_argv,
        n_dims=0, prunable=False, no_hpo=True, defaults=_BN_DEFAULTS,
        notes="DEPRECATED (2026-11-23), kept only to document the pre-swap runs in "
              "hpo_v3/final/bn_{CA,TX,OH}. NOT the audited BN any more: the paper's BN row is "
              "bn_pgmpy (run_audit_v3.PRIMARY['bn'] -> bn_pgmpy_{st}). What this entry actually "
              "ran: train_bn_v2.py with its own numpy hill-climbing backend (structure_backend="
              "numpy_hc, NOT pgmpy), BIC, at most 3 parents, Laplace-smoothed CPTs (Dirichlet "
              "alpha=1, NOT MLE), empirical decoding, fixed at _BN_DEFAULTS. An earlier version "
              "of this note wrongly said 'pgmpy hill-climb + BIC + MLE CPTs'. Old audit outputs: "
              "experiments/E_state/audits_v3/_backup_bn_v2_2026-11-23/."),
    "bn_pgmpy": ModelSpec(
        name="bn_pgmpy", script="train_bn_pgmpy.py", weight="light",
        timeout_s=4 * 3600, suggest=_bn_pgmpy_suggest, argv=_bn_pgmpy_argv,
        n_dims=0, prunable=False, no_hpo=True, defaults=_BN_PGMPY_LLMPOP,
        notes="AUDITED BN ROW (since 2026-11-23; the paper's 'BN', audit tag bn, "
              "run_audit_v3.PRIMARY['bn'] -> bn_pgmpy_{st}). The Bayesian-network approach of "
              "Sun and Erath (2015), run with llm_pop's train_bn_pgmpy.py @ e2b27f1 unchanged "
              "(pgmpy 1.1.2 HillClimbSearch, BDeu ess 66.6383, max in-degree 4, MLE CPTs, "
              "forward sampling, empirical decoding) + K=50 ensemble writer. Fixed "
              "configuration, not searched here."),
    "bn_pgmpy_bic": ModelSpec(
        name="bn_pgmpy_bic", script="train_bn_pgmpy.py", weight="light",
        timeout_s=4 * 3600, suggest=_bn_pgmpy_suggest, argv=_bn_pgmpy_argv,
        n_dims=0, prunable=False, no_hpo=True, defaults=_BN_PGMPY_BIC,
        notes="DIAGNOSTIC (2026-11-23): the same pgmpy trainer at its literature "
              "defaults (BIC, max in-degree 3, MLE, empirical decoding), separating the "
              "backend change from the configuration change."),
    "semapop_ddpm": ModelSpec(
        available=False,  # EXCLUDED from the v3 audit: method outside this paper
        name="semapop_ddpm", script="train_semapop_ddpm_v2.py", weight="heavy",
        timeout_s=4 * 3600, suggest=_semapop_ddpm_suggest, argv=_semapop_ddpm_argv,
        n_dims=23,
        needs_prior=True,
        notes="SemaPop with a diffusion backbone (R2-4): the same persona encoder, "
              "PersonaAdapter and fitted prior as semapop_gan, with the WGAN-GP "
              "generator replaced by the Gaussian-multinomial diffusion of tabddpm. "
              "Shares tabddpm's training loop so the two arms differ in the backbone "
              "and nothing else. Space = the TabDDPM space plus the SemaPop "
              "Space = the TabDDPM space plus the SemaPop conditioning knobs, so "
              "the two are directly comparable once the trainer lands."),
}


def get(model: str) -> ModelSpec:
    if model not in REGISTRY:
        raise KeyError(f"unknown model {model!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[model]


# --------------------------------------------------------------------------- #
# self-checks (rule 2 and rule 3 above)
# --------------------------------------------------------------------------- #
class _DefaultProbe:
    """A fake Optuna trial that records every declared range and returns the
    trainer default, so we can assert the default is inside its own space."""

    def __init__(self, defaults: Dict[str, Any]):
        self.defaults = defaults
        self.seen: Dict[str, Any] = {}
        self.errors: List[str] = []

    def _take(self, name, ok: bool, rng: str):
        d = self.defaults.get(name, "<no default recorded>")
        self.seen[name] = rng
        if not ok:
            self.errors.append(f"{name}: default {d!r} outside {rng}")
        return d if name in self.defaults else None

    def suggest_categorical(self, name, choices):
        return self._take(name, self.defaults.get(name) in choices, f"choices={choices}")

    def suggest_int(self, name, lo, hi, step=1, log=False):
        d = self.defaults.get(name)
        ok = d is not None and lo <= d <= hi
        return self._take(name, ok, f"int[{lo},{hi}]step{step}")

    def suggest_float(self, name, lo, hi, step=None, log=False):
        d = self.defaults.get(name)
        ok = d is not None and lo <= d <= hi
        return self._take(name, ok, f"float[{lo},{hi}]{'log' if log else ''}")


def check_defaults(verbose: bool = True) -> int:
    import utils.criteria as _CR
    bad = 0
    for name, spec in REGISTRY.items():
        if spec.criterion not in _CR.CRITERIA:
            print(f"{name:14s} BAD criterion {spec.criterion!r}; "
                  f"available: {sorted(_CR.CRITERIA)}")
            bad += 1
        if not spec.defaults:
            if verbose:
                print(f"{name:14s} SKIP (stub, no defaults recorded)")
            continue
        probe = _DefaultProbe(spec.defaults)
        spec.suggest(probe)
        n = len(probe.seen)
        if n != spec.n_dims and verbose:
            print(f"{name:14s} WARN declared n_dims={spec.n_dims} but space has {n}")
        if probe.errors:
            bad += len(probe.errors)
            for e in probe.errors:
                print(f"{name:14s} FAIL {e}")
        elif verbose:
            print(f"{name:14s} OK   {n} dims, every trainer default reachable")
    return bad


def table() -> str:
    hdr = "/".join(C.STATES)
    rows = [f"model          dims  weight  trials {hdr:<9s} timeout  available",
            "-" * 74]
    for name, s in REGISTRY.items():
        t = "/".join(str(s.n_trials(st)) for st in C.STATES)
        rows.append(f"{name:14s} {s.n_dims:4d}  {s.weight:6s}  {t:>15s}  "
                    f"{s.timeout_s // 3600:>5d}h  {str(s.available):>9s}")
    return "\n".join(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="inspect the HPO search spaces")
    ap.add_argument("--check-defaults", action="store_true")
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    if a.table or not a.check_defaults:
        print(table())
    if a.check_defaults:
        raise SystemExit(1 if check_defaults() else 0)
