"""Central path configuration for the PopAudit release.

Every script that originally hard-coded an absolute location now resolves it through this
module.  Defaults are relative to the repository root (the folder that holds this file);
each one can be overridden with an environment variable:

  POPAUDIT_ROOT       repository root                         default: folder of this file
  POPAUDIT_DATA       processed PUMS / persona data            default: $POPAUDIT_ROOT/data
  POPAUDIT_MODELS     trainer output root (checkpoints)        default: $POPAUDIT_ROOT/modelSave
  POPAUDIT_HPO        HPO records (studies/, best/, final/)    default: $POPAUDIT_ROOT/results/hpo_v3
  POPAUDIT_ENSEMBLES  finalize tree with the K=50 ensembles     default: $POPAUDIT_HPO/final
                      (final/<study>/<study>_final_s42/generated_ensembles/*.csv; available on request)
  POPAUDIT_SSD        fallback root for paths of the authors' external disk that are not data,
                      models or HPO records (see S())          default: $POPAUDIT_ROOT
  POPAUDIT_PYTHON     interpreter used for subprocess calls    default: the running interpreter
  POPAUDIT_LAYOUT     'release' (default) or 'production'; 'production' maps the release folder
                      names back to the authors' original tree (see _PRODUCTION_LAYOUT).

Scripts that build paths relative to their own location, or relative to the repository root
(``data/...``), are unchanged; keep the data under ``$POPAUDIT_ROOT/data`` (a symlink is fine).
"""
import os
import sys
from pathlib import Path

ROOT = os.environ.get("POPAUDIT_ROOT") or str(Path(__file__).resolve().parent)
DATA = os.environ.get("POPAUDIT_DATA") or ROOT + "/data"
MODELS = os.environ.get("POPAUDIT_MODELS") or ROOT + "/modelSave"
HPO = os.environ.get("POPAUDIT_HPO") or ROOT + "/results/hpo_v3"
ENSEMBLES = os.environ.get("POPAUDIT_ENSEMBLES") or HPO + "/final"
SSD = os.environ.get("POPAUDIT_SSD") or ROOT
PYTHON = os.environ.get("POPAUDIT_PYTHON") or sys.executable
LAYOUT = os.environ.get("POPAUDIT_LAYOUT", "release")

# release-relative folder -> folder in the authors' original tree (used only when POPAUDIT_LAYOUT=production)
_PRODUCTION_LAYOUT = [
    ("/experiments/E_state/supplementary/jwtrns", "/experiments/E_state/audits_v3_referee/group1/jwtrns_g1fix"),
    ("/experiments/E_state/supplementary/adjinc", "/experiments/E_state/audits_v3_referee/group1/adjinc"),
    ("/experiments/E_state/supplementary/sigma_calibration", "/experiments/E_state/audits_v3_referee/group1/release/check/indep"),
    ("/experiments/E_state/supplementary/seed/verify", "/experiments/E_state/audits_v3_referee/group1/referee_verify/seed"),
    ("/experiments/E_state/supplementary", "/experiments/E_state/audits_v3_referee"),
    ("/results/tables", "/docs/manuscript/revision/tables_v3"),
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


def S(rel=""):
    """Path below the authors' external data disk, mapped onto the configured locations."""
    for seg, base in (("/hpo_v3/final", ENSEMBLES), ("/hpo_v3", HPO), ("/data", DATA), ("/modelSave", MODELS)):
        if _under(rel, seg):
            return base + rel[len(seg):]
    return SSD + rel


if __name__ == "__main__":
    for k in ("ROOT", "DATA", "MODELS", "HPO", "ENSEMBLES", "SSD", "PYTHON", "LAYOUT"):
        print(f"{k:10s} {globals()[k]}")
