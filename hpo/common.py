"""Shared constants, paths and small helpers for the SemaPop v2 HPO harness.

Nothing in this module reads the reference population.  The reference split
(``referencedata_pers.csv`` and its ``testdata_pers.csv`` symlink) is touched
exactly once in the whole pipeline: by :mod:`hpo.finalize`, AFTER the winning
configuration has been chosen, to report the val-vs-reference objective gap.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[1])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# roots
# --------------------------------------------------------------------------- #
REPO = _PA.R("")
PYEXE = _PA.PYTHON
SSD = _PA.S("")

DATA_ROOT = _PA.S("/data/pums_states")
SML_ROOT = _PA.S("/data/pums_sml_ds")      # one directory per S/M/L condition
#: The mixed tier is one flat directory, not one per condition: it has a single attribute
#: set and a single word budget, so there is nothing for a condition suffix to select.
MIXED_ROOT = _PA.S("/data/pums_mixed_ds")
MIXED_CONDITION = "mixed"
HPO_ROOT = _PA.S("/hpo_v3")
STUDY_DIR = f"{HPO_ROOT}/studies"
TRIAL_DIR = f"{HPO_ROOT}/trials"          # per-trial run dirs (models + metrics.json)
FINAL_DIR = f"{HPO_ROOT}/final"           # finalize.py seed retrains
RESULT_DIR = f"{REPO}/results/hpo"        # small json/csv summaries that go in the paper

EVAL_CONFIG = f"{REPO}/configs/eval_v3_pums12.yaml"   # FROZEN, unweighted primary (13-attr build)


def eval_config_for(condition: Optional[str]) -> str:
    """The frozen config matching a condition, or the 20-attribute one when unset.

    S, M and L have separate configs whose bin edges and category lists are byte-identical
    where they overlap, so a difference between tiers cannot come from the metric.
    """
    if not condition:
        return EVAL_CONFIG
    if condition == MIXED_CONDITION:
        # The SELECTION config (20 columns, continuous quantile-binned), not the reporting
        # one.  Optuna maximises a single scalar and that scalar must see the continuous
        # half; the reported continuous number is the bin-free Wasserstein distance from
        # scripts/report_metrics_mixed.py.  See eval_config_report_for.
        return f"{REPO}/configs/eval_mixed_X_hpo.yaml"
    return f"{REPO}/configs/eval_sml_{condition.split('_')[0]}.yaml"

def eval_config_report_for(condition: Optional[str]) -> str:
    """The config the PAPER's tables are computed under.

    Identical to eval_config_for everywhere except the mixed tier, where selection runs on
    a binned surrogate of the continuous columns and reporting does not.
    """
    if condition == MIXED_CONDITION:
        return f"{REPO}/configs/eval_mixed_X.yaml"
    return eval_config_for(condition)


#: CA is the main table; FL is the generalisation check.  TX and OH belong to the
#: 50-attribute round and are kept so its studies still resolve.
STATES = ("CA", "TX", "OH")
STATES_SML = ("CA", "TX", "OH")

# The two file names that must never be opened during a trial.
REFERENCE_BASENAMES = ("referencedata_pers.csv", "testdata_pers.csv")

# Training seeds for the final seed sweep.
#
# The short set is a PREFIX of the long one, deliberately: a five-seed run on the primary
# state then uses the same five seeds as a five-seed run on a check state, so a difference
# between the two rows cannot be a difference in seeds.  _check_seed_nesting below refuses
# to let that invariant be edited away silently.
# v3 protocol: 3 pre-registered training seeds per state; the audit
# certificate is computed on seed 42's K=50 ensemble, the fidelity table on all 3.
SEEDS_CA = [42, 142, 242]
SEEDS_OTHER = [42, 142, 242]

if list(SEEDS_OTHER) != list(SEEDS_CA[:len(SEEDS_OTHER)]):
    raise RuntimeError(
        "SEEDS_OTHER must be a prefix of SEEDS_CA: seeds_for(state, n) returns the first n "
        "of the state's own set, and the two states are only comparable at a shortened "
        f"budget while the sets are nested.  Got {SEEDS_OTHER} vs {SEEDS_CA[:len(SEEDS_OTHER)]}.")


def seeds_for(state: str, n: Optional[int] = None) -> List[int]:
    """The pre-registered training seeds for a state, or the first ``n`` of them.

    ``n`` shortens THIS STATE's set.  The previous version returned ``SEEDS_CA[:n]``
    whatever the state, so ``seeds_for("FL", 10)`` handed back ten seeds for a state whose
    pre-registration has five -- silently, and outside the registration.  It never bit,
    because SEEDS_OTHER happens to equal SEEDS_CA[:5] and nothing asked for more than five
    on a check state; the moment either of those stopped being true it would have been a
    seed set nobody chose.  Asking for more than a state has is now an error.
    """
    base = SEEDS_CA if state == "CA" else SEEDS_OTHER
    if n is None:
        return list(base)
    if n > len(base):
        raise ValueError(
            f"{state} has {len(base)} pre-registered seeds; asked for {n}.  Extending a "
            f"check state's seed set is a change to the pre-registration, not a flag.")
    return list(base[:n])


# --------------------------------------------------------------------------- #
# trial budgets: CA is the primary state, TX/OH generalisation
# --------------------------------------------------------------------------- #
#   heavy = one GPU trainer trial costs > ~30 min at full 40k rows
#   light = everything else
TRIALS = {
    "heavy": {"CA": 20, "TX": 15, "OH": 15},
    "light": {"CA": 30, "TX": 15, "OH": 15},
}

#: Trials per searched dimension, for the extended set.
#:
#: The flat per-class table above is what produced the confound recorded in plan 14.5: on
#: the 20-attribute run BN got 30 trials over 8 dimensions (3.75 per dimension, close to
#: exhaustive) while SemaPop got 21 over 19 (1.11).  All three confounds identified there
#: pointed the same way, and SemaPop's 12.9% deficit against ZCR could not be split
#: between method and budget.  Allocating by dimension removes that argument: every model
#: is searched at the same density, and the density is the one the best-served baseline
#: had.  It costs more, which is the point -- the cheaper setting was cheap because it
#: under-searched the models with the most knobs.
#:
#: 3.75 -> 2.5 on 2026-08-26, for the mixed tier.  3.75 was set to match the density the
#: best-served baseline had on the 20-attribute round, and it still does what it was for:
#: every model is searched at the SAME density, so no gap between two models can be
#: attributed to one of them having had more trials.  That property is a ratio, and it
#: survives the cut intact -- what changes is only how much search everybody gets.
#:
#: The cut is a wall-clock decision, priced before it was taken.  At 3.75 the eight studies
#: come to 168 GPU-hours; on the three cards left after one was handed back that is 59
#: hours, and the schedule cannot beat the longest single study (semapop_ddpm CA, 86 trials
#: run sequentially in one process) however the rest is packed.  2.5 brings it to 112
#: GPU-hours and 39 hours of wall clock.
TRIALS_PER_DIM = 2.5
#: FL is NOT scaled with it.  1.5/dim already leaves the check state 24-34 trials, and the
#: two constants answer different questions: CA's density is what has to be equal ACROSS
#: MODELS, FL's is how much a generalisation check is worth spending on at all.  Cutting
#: both would have taken semapop_ddpm FL to 23 trials over 23 dimensions, which is one
#: sample per dimension and not a search.  The CA:FL ratio narrows from 2.5x to 1.7x as a
#: consequence, and that is the intended direction: the gap was never a design target.
#: TX/OH are generalisation checks, not the main table, so they are searched at a lower
#: density than CA.  Cut 2.0 -> 1.5 on 2026-08-26: the per-trial costs measured off
#: the ep800 archive turned out to be much higher than the budget was planned against
#: (wgan_zcr 45.4 min over COMPLETED trials against the 27.6 all-trial mean, tabddpm
#: 40.8 against 15.6), before adding the val top-up to 20,002 rows and the new
#: per-model criteria. Spending the difference on CA, which is the table the paper
#: reports, is the right trade; the two check states still get 24-34 trials each.
TRIALS_PER_DIM_OTHER = 1.5


def n_trials_for_dims(dims: int, state: str = "CA") -> int:
    per = TRIALS_PER_DIM if state == "CA" else TRIALS_PER_DIM_OTHER
    return int(round(dims * per))


def normalise_device(device: str, gpu) -> str:
    """Device index as seen INSIDE a subprocess that has CUDA_VISIBLE_DEVICES set.

    Masking renumbers the cards: with ``--gpu 1`` the visible card is ``cuda:0`` in the
    child, not ``cuda:1``.  Passing the physical index through ``--device`` is what made
    every GPU-1 trial die of "CUDA error: invalid device ordinal"; ``--gpu 0`` worked
    only because the two indices coincide there.  run.py and finalize.py both build
    their own context, so this lives here rather than in either of them -- the same bug
    was fixed in run.py first and then reappeared in finalize.py untouched.
    """
    if gpu not in (None, "") and str(device).startswith("cuda"):
        return "cuda:0"
    return device


def n_trials_for(weight: str, state: str) -> int:
    return TRIALS[weight][state]


# --------------------------------------------------------------------------- #
# per-state assets
# --------------------------------------------------------------------------- #
def data_dir(state: str, condition: Optional[str] = None) -> str:
    """The 20-attribute directory, or one extended-set condition's directory.

    A condition directory (built by scripts/data_build/make_dataset_sml.py) holds the same
    file names the trainers already expect, so nothing below this line changes.
    """
    if condition == MIXED_CONDITION:
        return f"{MIXED_ROOT}/{state}"
    return f"{SML_ROOT}/{state}/{condition}" if condition else f"{DATA_ROOT}/{state}"


def emb_files(state: str) -> Dict[str, str]:
    return {"train": "embPers.pkl", "val": "embPers_val.pkl"}


def prior_assets(state: str, family: str = "gmm",
                 condition: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Locate the pre-fitted, val-selected persona prior + PCA subspace.

    v3 protocol: priors are fitted on pop_audit's own 13-attr-build embeddings
    (data/pums_states/<ST>/embPers.pkl) into results/prior_v3/<ST>_gmm/.
    Used only by the SemaPop-VAE+ PRIOR arm (--cond-mode mu); the main audited
    arm (--cond-mode train) needs no prior.
    """
    fam_file = {"gmm": "prior_gmm.npz", "flow": "prior_flow.pt"}[family]
    d = f"{REPO}/results/prior_v3/{state}_{family}"
    p, sub = os.path.join(d, fam_file), os.path.join(d, "subspace.npz")
    if os.path.exists(p):
        return {"prior": p, "subspace": sub if os.path.exists(sub) else None,
                "source": f"prior_v3_{state}_{family}"}
    return {"prior": None, "subspace": None, "source": "missing"}


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def count_rows(path: str) -> int:
    """Row count of a csv excluding the header.

    NOT a raw line count: the 13-attr build's *_pers.csv files carry free-text
    ``prompt``/``persona`` columns with embedded newlines, so ``wc -l`` style
    counting inflates val from 10,000 to ~250,000 (measured).  Parse the csv
    properly, reading no field content beyond the first column.
    """
    import csv as _csv
    n = 0
    with open(path, newline="", encoding="utf-8") as f:
        r = _csv.reader(f)
        next(r, None)                       # header
        for _ in r:
            n += 1
    return n


# --------------------------------------------------------------------------- #
# trial context
# --------------------------------------------------------------------------- #
@dataclass
class TrialContext:
    """Everything a trial needs that is NOT a searched hyperparameter.

    ``subsample`` is 0 for every real study -- tuning happens on the FULL 40k
    train split (fixing an earlier defect where heavy models were tuned on a
    15k subsample and produced degenerate winners).  It is non-zero
    only in ``--dry-run`` mode.
    """
    state: str
    device: str = "cuda:0"
    seed: int = 42                      # ONE fixed tuning seed; the seed sweep is finalize.py
    sample_seed: int = 1234
    eval_config: str = EVAL_CONFIG
    out_root: str = TRIAL_DIR
    subsample: int = 0                  # 0 = full 40k train split
    subsample_val: int = 0              # 0 = full 10k val split
    max_epochs: int = 0                 # >0 caps epochs (dry runs only)
    n_samples: int = 0                  # 0 = |val| (frozen); >0 only for dry runs/probes
    eval_every: int = 0                 # >0 overrides the trainer default (dry runs only)
    # Per-epoch checkpoint selection inside a trial must optimise the SAME quantity as
    # between-trial selection, or the two disagree about what "best" means; the default
    # therefore follows objective.W_F1 rather than being set independently.
    f1_weight: float = 0.0              # objective weight on mean block F1 (see objective.py)
    #: which criterion this model is selected by -- key into utils.criteria.CRITERIA.  Set
    #: from the ModelSpec in BOTH run.build_context and finalize's context: if only the
    #: first is set, the final retrains pick their checkpoint on a different quantity than
    #: the search ranked configurations by, and the reported winner is not the winner.
    criterion: str = "val_srmse"
    cond_mode: str = "mu"               # semapop only: sampling-time conditioning source.
                                        # 'mu' (prior sampling) is the DELIVERABLE path and the
                                        # default; 'mr' draws from the val embeddings, i.e. the
                                        # same split the objective scores -- oracle tuning.
    study: str = ""                     # study name; also prefixes trial dirs so two studies
                                        # of one model cannot overwrite each other
    prior_family: str = "gmm"
    prior_path: Optional[str] = None
    subspace_path: Optional[str] = None
    condition: Optional[str] = None      # e.g. "L_w100"; None = the 20-attribute build
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def data_dir(self) -> str:
        return data_dir(self.state, self.condition)

    def cap_epochs(self, n: int) -> int:
        return min(n, self.max_epochs) if self.max_epochs > 0 else n

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["data_dir"] = self.data_dir
        return d


def jdump(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


# --------------------------------------------------------------------------- #
# split sizes (used to pin --n-samples so every model's final val sample has the
# same size, and to record the tuning-data size in the study user_attrs)
# --------------------------------------------------------------------------- #
_ROWS_CACHE: Dict[str, int] = {}


def split_rows(state: str, split: str = "train", condition: Optional[str] = None) -> int:
    fn = {"train": "traindata_pers.csv", "val": "valdata_pers.csv",
          "reference": "referencedata_pers.csv"}[split]
    key = f"{state}/{condition or '-'}/{fn}"
    if key not in _ROWS_CACHE:
        _ROWS_CACHE[key] = count_rows(os.path.join(data_dir(state, condition), fn))
    return _ROWS_CACHE[key]


# --------------------------------------------------------------------------- #
# the fairness guard
# --------------------------------------------------------------------------- #
GUARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guard")


def trial_env(gpu: Optional[int] = None, forbid_reference: bool = True,
              base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Environment for one trial subprocess.

    ``hpo/guard/sitecustomize.py`` is put first on ``PYTHONPATH`` so CPython
    imports it at start-up; with ``SEMAPOP_HPO_FORBID_REF=1`` it patches
    ``open`` / ``io.open`` / ``os.open`` to raise on the reference basenames.
    That is how "the objective never sees the reference" is *asserted* rather
    than promised.  ``finalize.py`` calls this with ``forbid_reference=False``.
    """
    env = dict(os.environ if base is None else base)
    env["PYTHONUNBUFFERED"] = "1"
    pp = [GUARD_DIR, REPO] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    env["PYTHONPATH"] = os.pathsep.join(pp)
    if forbid_reference:
        env["SEMAPOP_HPO_FORBID_REF"] = "1"
        env["SEMAPOP_HPO_FORBID_BASENAMES"] = ",".join(REFERENCE_BASENAMES)
    else:
        env.pop("SEMAPOP_HPO_FORBID_REF", None)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Cap BLAS/OpenMP fan-out.  The box has 56 cores and queue_hpo.sh packs three
    # trials at once, so an uncapped isotree (if1) or HistGradientBoosting (ml_eff)
    # asks for 56 threads three ways and thrashes: measured 7.4 s per ml_efficiency
    # call at 16 threads against 100-293 s uncapped.  Set only when the caller has
    # not already chosen a value.
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS"):
        env.setdefault(_v, os.environ.get(_v, "12"))
    return env


def provenance(state: str, eval_config: str = EVAL_CONFIG,
               condition: Optional[str] = None) -> Dict[str, Any]:
    """Everything that must be reproducible about a study, for user_attrs."""
    return {
        "git_commit": git_commit(),
        "condition": condition or "pums20",
        "eval_config_path": eval_config,
        "eval_config_sha1": file_sha1(eval_config) if os.path.exists(eval_config) else None,
        "data_dir": data_dir(state, condition),
        "n_train_rows": split_rows(state, "train", condition),
        "n_val_rows": split_rows(state, "val", condition),
        "tuning_split": "train (fit) + val (objective); reference NEVER opened",
        "tuned_on_full_train_split": True,
        "warm_start": False,
        "enqueued_best": False,
    }
