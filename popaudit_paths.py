"""Central path configuration for the PopAudit release.

Every script that originally hard-coded an absolute location now resolves it through this
module.  Defaults are relative to the repository root (the folder that holds this file);
each one can be overridden with an environment variable:

  POPAUDIT_ROOT       repository root                         default: folder of this file
                      (must be this code checkout: the trainers and hpo/ take code and configs/ from it;
                      relocate data or outputs with POPAUDIT_DATA / MODELS / HPO instead)
  POPAUDIT_DATA       processed PUMS / persona data            default: $POPAUDIT_ROOT/data
  POPAUDIT_MODELS     trainer output root (checkpoints)        default: $POPAUDIT_ROOT/modelSave
  POPAUDIT_HPO        HPO records (studies/, best/, final/)    default: $POPAUDIT_ROOT/hpo/records
  POPAUDIT_ENSEMBLES  finalize tree with the K=50 ensembles     default: $POPAUDIT_HPO/final
                      (final/<study>/<study>_final_s42/generated_ensembles/*.csv; available on request)
  POPAUDIT_SSD        fallback root for paths of the authors' external disk that are not data,
                      models or HPO records (see S())          default: $POPAUDIT_ROOT
  POPAUDIT_PYTHON     interpreter used for subprocess calls    default: the running interpreter
  POPAUDIT_LAYOUT     'release' (default) or 'production'; 'production' maps the release folder
                      names back to the authors' original tree (see _PRODUCTION_LAYOUT).

Scripts that build paths relative to their own location, or relative to the repository root
(``data/...``), are unchanged; keep the data under ``$POPAUDIT_ROOT/data`` (a symlink is fine).

Code folders.  The audit layers live in ``audit/`` (AUDIT) and the trainers in ``training/`` (TRAINING);
in the authors' tree both sat at the repository root.  Importing this module appends both folders to
``sys.path`` (appended, so they never shadow anything), which is how scripts that ``import layer1`` or
``import train_tabddpm_v2`` find them.  Every script in ``training/`` starts with a one-line bootstrap that
puts the repository root first on ``sys.path`` and imports this module, so ``model.*``, ``utils.*``,
``hpo.*`` and ``audit_config`` resolve as they did at the root.  Scripts that call a layer or a trainer as a
subprocess take its path from ``R("/audit/...")`` / ``R("/training")``.
"""
import os
import sys
from pathlib import Path

ROOT = os.environ.get("POPAUDIT_ROOT") or str(Path(__file__).resolve().parent)
DATA = os.environ.get("POPAUDIT_DATA") or ROOT + "/data"
MODELS = os.environ.get("POPAUDIT_MODELS") or ROOT + "/modelSave"
HPO = os.environ.get("POPAUDIT_HPO") or ROOT + "/hpo/records"
ENSEMBLES = os.environ.get("POPAUDIT_ENSEMBLES") or HPO + "/final"
SSD = os.environ.get("POPAUDIT_SSD") or ROOT
PYTHON = os.environ.get("POPAUDIT_PYTHON") or sys.executable
LAYOUT = os.environ.get("POPAUDIT_LAYOUT", "release")

# release-relative folder -> folder in the authors' original tree (used only when POPAUDIT_LAYOUT=production;
# most specific first).  main_experiments/ is their experiments/E_state/, supplementary_experiments/ their
# experiments/E_state/audits_v3_referee/ (with the subfolder names on the right); audit/ and training/ were
# their repository root.
_PRODUCTION_LAYOUT = [
    ("/supplementary_experiments/household_disjoint/code/compare_l1_disjoint_g1fix.py", "/experiments/E_state/audits_v3_referee/misc_g1fix/code/compare_l1_disjoint_g1fix.py"),
    ("/supplementary_experiments/iid_reference/code/check_scenario_refs_g1fix.py", "/experiments/E_state/audits_v3_referee/misc_g1fix/code/check_scenario_refs_g1fix.py"),
    ("/supplementary_experiments/iid_reference/code/compare_iid_g1fix.py", "/experiments/E_state/audits_v3_referee/misc_g1fix/code/compare_iid_g1fix.py"),
    ("/supplementary_experiments/household_disjoint/corrected", "/experiments/E_state/audits_v3_referee/misc_g1fix/l1_disjoint"),
    ("/supplementary_experiments/iid_reference/corrected", "/experiments/E_state/audits_v3_referee/misc_g1fix/iid_test"),
    ("/supplementary_experiments/seed_robustness/verify", "/experiments/E_state/audits_v3_referee/group1/referee_verify/seed"),
    ("/supplementary_experiments/scenario_only_ablation", "/experiments/E_state/audits_v3_referee/review28/a27_ablation"),
    ("/supplementary_experiments/oracle_2013_reference", "/experiments/E_state/audits_v3_referee/review28/a9_oracle"),
    ("/supplementary_experiments/household_disjoint", "/experiments/E_state/audits_v3_referee/l1_disjoint"),
    ("/supplementary_experiments/sigma_calibration", "/experiments/E_state/audits_v3_referee/group1/release/check/indep"),
    ("/supplementary_experiments/sigma_sensitivity", "/experiments/E_state/audits_v3_referee/sigma"),
    ("/supplementary_experiments/income_deflation", "/experiments/E_state/audits_v3_referee/group1/adjinc"),
    ("/supplementary_experiments/truncated_priors", "/experiments/E_state/audits_v3_referee/filter"),
    ("/supplementary_experiments/if1_metric_check", "/experiments/E_state/audits_v3_referee/if1_check"),
    ("/supplementary_experiments/srmse_threshold", "/experiments/E_state/audits_v3_referee/srmse"),
    ("/supplementary_experiments/seed_robustness", "/experiments/E_state/audits_v3_referee/seed"),
    ("/supplementary_experiments/jwtrns_coding", "/experiments/E_state/audits_v3_referee/group1/jwtrns_g1fix"),
    ("/supplementary_experiments/iid_reference", "/experiments/E_state/audits_v3_referee/iid_test"),
    ("/supplementary_experiments/ensemble_size", "/experiments/E_state/audits_v3_referee/review28/a10_K"),
    ("/supplementary_experiments/tau_ladder", "/experiments/E_state/audits_v3_referee/tau"),
    ("/supplementary_experiments", "/experiments/E_state/audits_v3_referee"),
    ("/main_experiments/tables", "/docs/manuscript/revision/tables_v3"),
    ("/main_experiments", "/experiments/E_state"),
    ("/training", ""),
    ("/audit", ""),
]


def _under(rel, seg):
    return rel == seg or rel.startswith(seg + "/")


def R(rel=""):
    """Path below the repository root; ``rel`` starts with '/' (or is empty)."""
    if _under(rel, "/data"):
        return DATA + rel[len("/data"):]
    if _under(rel, "/modelSave"):
        return MODELS + rel[len("/modelSave"):]
    if LAYOUT == "production":
        for a, b in _PRODUCTION_LAYOUT:
            if _under(rel, a):
                rel = b + rel[len(a):]
                break
    return ROOT + rel


AUDIT = R("/audit")          # layer1.py, layer2_historical_sigma.py, layer3_projection_uncertainty_branches.py
TRAINING = R("/training")    # train_*.py
for _d in (AUDIT, TRAINING):
    if _d not in sys.path:
        sys.path.append(_d)


def S(rel=""):
    """Path below the authors' external data disk, mapped onto the configured locations."""
    for seg, base in (("/hpo_v3/final", ENSEMBLES), ("/hpo_v3", HPO), ("/data", DATA), ("/modelSave", MODELS)):
        if _under(rel, seg):
            return base + rel[len(seg):]
    return SSD + rel


if __name__ == "__main__":
    for k in ("ROOT", "DATA", "MODELS", "HPO", "ENSEMBLES", "SSD", "PYTHON", "LAYOUT", "AUDIT", "TRAINING"):
        print(f"{k:10s} {globals()[k]}")
