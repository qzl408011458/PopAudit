# PopAudit

Code and audit outputs for the paper

> **Auditing Generative Population Priors for Scenario-Constrained Transportation Simulation under Distributional Shift**

The paper asks whether a generative population prior trained on base-year data can serve as the base
population of a static scenario projection. It proposes a three-layer scenario-readiness audit: Layer 1
tests base-year validity (marginals, dependencies, hard rules), Layer 2 tests whether the scenario stays
within historically plausible change for its horizon, and Layer 3 tests whether KL reweighting can meet
the scenario and how strongly the prior must be reweighted. Priors that pass are compared with an error
budget to deploy or defer each scenario, and an audit certificate records the evidence. The study audits
nine priors trained on 2013 ACS PUMS data for California, Ohio and Texas against 2018, 2021 and 2024
scenarios (81 prior-state-horizon cases). This repository holds the code for every step, the per-case
audit outputs and the hyper-parameter search records.

## Repository map

| folder or file | contents |
|---|---|
| `audit/` | the three audit layers: `layer1.py`, `layer2_historical_sigma.py`, `layer3_projection_uncertainty_branches.py` |
| `training/` | trainers of the candidate priors (`train_*.py`) |
| `model/`, `utils/` | model code and shared utilities (evaluation, K = 50 ensemble writer) |
| `audit_config.py`, `popaudit_paths.py` | ensemble settings (K = 50, seeds 42 + 100 i), read by the trainers and the ensemble writer `utils/ensemble_v2.py`; central path configuration |
| `_pums_proc.py`, `_gen_pub_personas_traintest.py` | PUMS processing and the Qwen3-8B persona generator (imported by `data_prep/`) |
| `configs/` | frozen evaluation protocol |
| [`data_prep/`](data_prep/README.md) | download, processing, state slices, splits, personas and embeddings |
| [`hpo/`](hpo/README.md) | hyper-parameter search, final training, fidelity tables; `hpo/records/` holds the search records |
| [`main_experiments/`](main_experiments/README.md) | audit drivers, certificate, per-case outputs, and the scripts and outputs of every main-text table and figure |
| [`supplementary_experiments/`](supplementary_experiments/README.md) | one folder per Supplementary analysis, scripts next to their results |
| `RELEASE_MANIFEST.tsv` | every file with its md5, role and the manuscript items it supports |

## Quick start

```
conda env create -f environment.yml && conda activate popaudit
```

Python 3.12 on Linux; the full pinned list is in `environment.yml`. The audit, certificate and table
scripts run on CPU. Training needs one GPU per run, and persona generation needs a GPU with at least 24 GB.

Paths default to folders inside the repository and can be changed with environment variables
(`POPAUDIT_DATA` for the processed data, default `data/`; `POPAUDIT_HPO` for the HPO records, default
`hpo/records/`; `POPAUDIT_ENSEMBLES` for the generated ensembles, default `hpo/records/final/`; see
`popaudit_paths.py` for the full list). Keep the processed data under `data/` (a symlink is fine), because
some scripts read it relative to the repository root. `POPAUDIT_ROOT` must be this checkout. Run commands
from the repository root, e.g. `python training/train_ctgan_v2.py --help` or `python audit/layer1.py --help`;
the import mechanism is described in `popaudit_paths.py`. Other scripts below the root that import `model.*`,
`utils.*` or `hpo.*` add the root to `sys.path` themselves or need `PYTHONPATH=$POPAUDIT_ROOT`.

**From the shipped outputs** (no large inputs needed), regenerate the certificate, the Sec. 6.1 counts, Figures 1
and 4, the sensitivity analyses, the headline numbers and the Supplementary tables that work from
released CSVs:

```
bash main_experiments/audits_v3/regen_from_release.sh   # all of the above; logs in main_experiments/audits_v3/regen_logs_release/
(cd main_experiments && python decision_certificate_xstate_v3.py)        # certificate only
python main_experiments/audits_v3/PAPER_xstate_v3/headline_numbers.py    # headline numbers only (step 24 above)
```

**Full pipeline**: data chain (`bash data_prep/run_data_chain.sh`), search and training
(`bash hpo/run_all.sh search`, then `finalize`), the 81-case audit (`python main_experiments/run_audit_v3.py`),
then all main-text artifacts (`bash main_experiments/audits_v3/regen_after_g1fix.sh`). The folder READMEs
give the steps, their order and the inputs each one needs.

## Large inputs available on request

Trained models, generated ensembles, persona texts and embeddings, and Layer-3 projected weights are too
large for this repository. They are available from the authors on request.

| item | size (approx.) | needed for (label in the folder READMEs) |
|---|---|---|
| raw ACS PUMS (public, `data_prep/download_pums.sh`) | 19 GB | [RAW] |
| processed state datasets and references | 1.7 GB | [RAW] |
| persona texts / embeddings / GMM priors | 0.2 / 2.3 / 0.5 GB | [RAW] (SemaPop-VAE+ training) |
| checkpoints (seed 42 / all seeds) | 1 / 56 GB | [CKPT] |
| generated ensembles (K = 50 per prior and state) | 29 GB | [ENS] |
| Layer-3 r* cells7 inputs | 0.4 GB | [ENS] |
| Layer-3 projected weights (81 primary cases) | 0.4 GB | [PW] |

[REPO] marks steps that run from this repository alone. Logs and some per-case JSON files were written in
the authors' working tree; absolute paths in them were replaced by `<POPAUDIT_ROOT>` / `<POPAUDIT_HPO>`, and
their folder names map to this layout as listed in `popaudit_paths.py` (`_PRODUCTION_LAYOUT`). In that tree
the audit layers and the trainers sat at the repository root, so records such as
`hpo/records/finalize_commands.tsv` name them without `audit/` or `training/`.

## License and citation

MIT License, see `LICENSE`. Code adapted from TabDDPM, FlowSyn and popsynth (all MIT) keeps the original
notices, listed in `THIRD_PARTY_NOTICES.md`. `CITATION.cff` lists the authors (GitHub shows it under "Cite
this repository"); the journal reference will be added once the paper is published.
