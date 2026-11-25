# PopAudit

Code and audit outputs for the paper

> **Auditing Generative Population Priors for Scenario-Constrained Transportation Simulation under Distributional Shift**

The repository contains everything needed to trace each number of the paper and its Supplementary Material
back to code:

- the processing of the ACS PUMS files into the California, Texas and Ohio datasets;
- the generation of the persona texts and embeddings used by SemaPop-VAE+ (Qwen3-8B);
- the hyper-parameter search and training of the nine candidate priors;
- the three audit layers and the decision certificate;
- the scripts behind every table and figure of the main text and the Supplementary Material, including
  Supplementary S2, S3.6-S3.9, the S4.1 harmonization checks and the HPO tables;
- the Layer-1 outputs of all 27 prior-state pairs and the Layer-2 and Layer-3 outputs of all
  81 prior-state-horizon cases;
- the hyper-parameter search records.

Trained models, generated ensembles, persona texts and embeddings, and Layer-3 projected weights are too
large for this repository. They are **available from the authors on request** (see
[Large inputs available on request](#large-inputs-available-on-request)).

> **License.** MIT License, see `LICENSE`. Code adapted from TabDDPM, FlowSyn and popsynth (all MIT)
> keeps the original notices, listed in `THIRD_PARTY_NOTICES.md`.
>
> **Citation.** `CITATION.cff` lists the authors (GitHub shows it under "Cite this repository"). The
> journal reference and a BibTeX entry will be added here once the paper is published.

---

## 1. Repository layout

```
popaudit_paths.py            central path configuration (environment variables, see Section 3)
layer1.py                    Layer 1: base-year validity (marginals, dependencies, hard rules)
layer2_historical_sigma.py   Layer 2: horizon-scaled historical uncertainty sigma(H), chi_H test
layer3_projection_uncertainty_branches.py   Layer 3: KL projection onto the scenario constraint set
audit_config.py              K = 50 realizations, ensemble seeds 42 + 100 i
_pums_proc.py                raw ACS PUMS -> national compact person tables (also imported by other scripts)
_gen_pub_personas_traintest.py   Qwen3-8B persona generator (imported by data_prep/pums_states/*/run_persona_*.py)
train_*.py, model/, utils/   trainers of the candidate priors and shared utilities
configs/                     frozen evaluation protocol (eval_v3_pums12.yaml + age1/age10 variants)
hpo/                         Optuna search, finalize (3-seed retrain + K = 50 ensemble), fidelity tables
data_prep/                   download, processing, state slicing, splits, personas, embeddings, column configs
experiments/E_state/
  run_audit_v3.py            runs Layers 1-3 for the 81 cases (+ diagnostic extras)
  rerun_l1_paper.py, rerun_audit_new.py, l1_invalid_fix_shadow.py   audit drivers
  decision_certificate_xstate_v3.py   gates, first failed gate, R_2^MA, certificate tables
  audits_v3/
    regen_after_g1fix.sh     the authors' regeneration driver for every main-text artifact (steps 01-22)
    regen_from_release.sh    re-runs every step that works from this repository alone
    regen_logs_g1fix/        logs of the authors' last full run (numbers printed by each step)
    layer1_paper_<prior>_<ST>/, layer2_<prior>_<ST>_<Y>/, layer3_paperW_<prior>_<ST>_<Y>/   per-case outputs
    PAPER_xstate_v3/         certificate, budget/decision tables, figures, sensitivity, ablations (scripts + outputs)
    PAPER_xstate_v3_extras/  certificate of the diagnostic rows (App. B, Supplementary S4.1, Table S12)
    bn_pgmpy_compare/        BN backend comparison (diagnostic, not reported in the paper)
  supplementary/             supplementary analyses, scripts next to their result files:
    srmse/ tau/ sigma/ seed/ filter/ iid_test/ if1_check/ misc_g1fix/ l1_disjoint/ jwtrns/ adjinc/
    sigma_calibration/ household/
results/
  hpo_v3/                    Optuna study databases, best trials, finalize summaries, budget table, finalize argv
  tables/                    two LaTeX tables that the generators write outside the audit tree (Table 4, Table S11)
  figures/                   TikZ source of Figure A1 (fig_framework_flow.tex)
RELEASE_MANIFEST.tsv         every file: md5, role, production source, manuscript items
environment.yml, CITATION.cff
```

Folder names follow the authors' working tree, because many scripts locate their inputs relative to
their own location. In the authors' tree, `experiments/E_state/supplementary` is called `audits_v3_referee`,
`supplementary/jwtrns` is `group1/jwtrns_g1fix`, `results/tables` is `docs/manuscript/revision/tables_v3`
and `data_prep/pums_states` is the script part of `data/pums_states`.

Labels used below: **[REPO]** re-runs from this repository alone; **[PW]** needs the Layer-3 projected
weights and the processed scenario references; **[ENS]** needs the generated ensembles or the r* cells7
inputs of Layer 3; **[CKPT]** needs trained checkpoints; **[RAW]** needs raw or processed PUMS files.

## 2. Environment

```
conda env create -f environment.yml && conda activate popaudit
```

The results were produced with Python 3.12.12 on Linux, with PyTorch 2.3.0 (CUDA 12.1), pandas 2.3.3,
numpy 2.3.5, cvxpy 1.9.0, Optuna 4.5.0, pgmpy 1.1.2, ctgan 0.11.1 / SDV 1.32.0, transformers 4.51.1 and
matplotlib 3.10.7 (the full pinned list is in `environment.yml`). The audit, certificate and table
scripts run on CPU. Training uses one GPU per run. Persona generation and embedding (Qwen/Qwen3-8B,
bf16) need a GPU with at least 24 GB of memory.

## 3. Paths

Scripts that originally hard-coded absolute locations resolve them through `popaudit_paths.py` (Python) or
an equivalent header (shell). The defaults are relative to the repository root:

| variable | meaning | default |
|---|---|---|
| `POPAUDIT_ROOT` | repository root | folder of `popaudit_paths.py` |
| `POPAUDIT_DATA` | processed PUMS, splits, persona files | `$POPAUDIT_ROOT/data` |
| `POPAUDIT_MODELS` | trainer output root | `$POPAUDIT_ROOT/modelSave` |
| `POPAUDIT_HPO` | HPO records | `$POPAUDIT_ROOT/results/hpo_v3` |
| `POPAUDIT_ENSEMBLES` | finalize tree with the K = 50 ensembles (`<study>/<study>_final_s42/generated_ensembles/`) | `$POPAUDIT_HPO/final` |
| `POPAUDIT_SSD` | fallback root for the few remaining paths of the authors' external disk that are not data, models or HPO records | `$POPAUDIT_ROOT` |
| `POPAUDIT_SUPP` | supplementary folder (shell scripts only) | `$POPAUDIT_ROOT/experiments/E_state/supplementary` |
| `POPAUDIT_PYTHON` | interpreter for subprocess calls | running interpreter (`python` in shell) |

Many scripts also read `data/...` relative to the repository root, so keep the data under
`$POPAUDIT_ROOT/data` (a symlink is fine). If you obtain the ensembles on request, either unpack them into
`results/hpo_v3/final/` or point `POPAUDIT_ENSEMBLES` at their folder. The HPO harness (`hpo/common.py`)
reads the state datasets from `$POPAUDIT_DATA/pums_states` and keeps its studies, trials and finalize runs
under `$POPAUDIT_HPO`; a **new** search or finalize run therefore writes into `results/hpo_v3/` unless
`POPAUDIT_HPO` points elsewhere (do that to keep the shipped records intact).

Scripts below the repository root that import repository modules (`model.*`, `utils.*`, `hpo.*`) either
add the root to `sys.path` themselves or must be run from the root with `PYTHONPATH=$POPAUDIT_ROOT`, for
example `supplementary/seed/code/train_ttvae_CA_bigbatch.py`.

## 4. Reproduction pipeline

| step | command | needs |
|---|---|---|
| 1. Download ACS 1-year PUMS 2009-2013, 2018, 2021, 2024 | `bash data_prep/download_pums.sh` | internet, about 19 GB |
| 2. National compact tables, 2013 PWGTP resample, state slices, splits | `bash data_prep/run_data_chain.sh` (steps 1-4) | [RAW] |
| 3. Persona texts and embeddings (Qwen3-8B) | `run_data_chain.sh` steps 5-6 | GPU |
| 4. Frozen evaluation protocol | `python utils/freeze_eval_config_v3.py` | 2013 slices |
| 5. Hyper-parameter search | `bash hpo/run_all.sh search` | GPU, data |
| 6. 3-seed training and K = 50 generation | `bash hpo/run_all.sh finalize` (+ `diagnostics`) | GPU, data |
| 7. Audit Layers 1-3 (81 cases) | `python experiments/E_state/run_audit_v3.py [--extras]` | [ENS] [RAW] |
| 8. Certificate, tables, figures | `bash experiments/E_state/audits_v3/regen_after_g1fix.sh` | [PW] [ENS] for some steps |
| 8'. Everything that works from this repository | `bash experiments/E_state/audits_v3/regen_from_release.sh` | [REPO] |
| 9. Supplementary analyses | scripts in `experiments/E_state/supplementary/*/` (Section 6) | mostly [ENS] [PW] |

Notes on the pipeline:

- **Data chain (steps 1-4).** `_pums_proc.py` writes income classes in constant 2013 dollars (ADJINC x
  CPI-U; `--nominal-income` restores nominal dollars) and maps the ACS 2019+ JWTRNS codes onto the JWTR
  categories (chosen from the person-file header). The 2013 training base is a PWGTP-proportional resample
  with replacement (`--seed 42 --drop-wgtp`). Before release, the authors re-ran these steps read-only:
  the 2013 resample, all 27 state slices (`data_prep/build_state_slices.py --check`), the splits
  (`dataset_ids.pkl`) and the three `configs/eval_v3_pums12*.yaml` files all reproduce byte for byte.
- **Splits.** The splits are 40,000 training and 10,000 validation records with seed 42; the rest is the
  test split (330,109 CA, 212,756 TX and 64,112 OH records). `referencedata_pers.csv` is a symlink to
  `testdata_pers.csv`. The OH/TX split scripts write `testdata_pers.csv` themselves; CA writes it in
  `run_persona_CA.py`. The OH/TX split scripts also copy `data/pums/cols_config.yaml`, a 9-column legacy
  config shipped as `data_prep/pums/cols_config.yaml`.
- **Persona texts are not bit-reproducible.** Qwen3-8B sampling (temperature 0.9, top-p 0.9, at most
  512 new tokens) is not seeded, and the number of shards used for CA and TX was not recorded. The exact
  texts and embeddings used in the paper are available on request. The GMM persona prior
  (`model/persona_prior.py`, `run_data_chain.sh` step 8) is used only by the diagnostic M-U row. Its
  command line was reconstructed from the stored prior metadata (K = 64, full covariance).
- **SemaPop-VAE+ arm.** The audited SemaPop-VAE+ is the M-R arm: `--cond-mode train` resamples
  training-record embeddings. The argparse default of `train_semapop_vae_v2.py` is `--cond-mode mu` (the
  M-U arm, Supplementary Table S12 only). `hpo/run_all.sh` and `results/hpo_v3/finalize_commands.tsv`
  pass the audited setting explicitly.
- **Finalize settings.** `results/hpo_v3/finalize_commands.tsv` gives the exact trainer argv of every
  finalize run: all 81 primary prior-state-seed runs and the diagnostic rows. `$POPAUDIT_OUT/hpo_v3` there
  stands for the HPO output root. The as-shipped CTGAN and TVAE runs are missing from that table because
  their logs were overwritten; their configuration is in
  `results/hpo_v3/final/{ctgan,tvae_ref}_defaults_<ST>/*_final_s42/metrics.json` (`config`). TabDDPM uses
  the trainer default `--sample-batch 4096`, which the finalize argv does not pass. For the reselected
  TabDDPM (`tabddpm_<ST>_valsrmse`), `aggregate.json` holds the configuration that was actually retrained;
  it differs from `best/tabddpm_<ST>.json`, which is the ML-efficacy optimum.
- **Audit (step 7).** `run_audit_v3.py --l23-only` re-runs Layer 2, the r* cells7 inputs and Layer 3 and
  reuses Layer 1. `layer1.py` itself implements the seven hard rules of Supplementary S4.2, so a fresh
  Layer-1 run reproduces every value of the shipped `layer1_ensemble_summary.json` (the fresh file also
  has `rule_child_not_employed_std` and lacks the provenance key `invalid_rules_note`, and its keys are
  ordered differently); `l1_invalid_fix_shadow.py` is the
  independent re-implementation of the same rules that the authors used to check it (it writes
  `audits_v3_l1fix/`, identical for all 51 pairs). In every `layer1_paper_*` folder the summary and
  per-realization CSVs were regenerated with the released `layer1.py` before release (the authors' earlier
  copies of these two CSVs predated the seven-rule set); the other Layer-1 files were reproduced byte for
  byte by that run. The certificate reads only the JSON. The primary BN row is audited under the
  tag `bn` (study `bn_pgmpy_<ST>`). The diagnostic `--extras` run audits the same study again under the tag
  `bn_pgmpy`, so `layer*_bn_*` and `layer*_bn_pgmpy_*` hold identical outputs. Both are shipped because
  the certificate reads the first and `kappa_null_sim.py` the second.
- **Ordering.** In `regen_after_g1fix.sh`, step 11 runs before step 08 (figure 4(b) reads the sweep), and
  step 09 must run after step 08: `make_figures_v3.py` writes the pre-projection
  `tableE_decision_dA.csv` and `figE_decision.*`, and step 09 replaces them with the post-projection
  versions used in the paper. Without projected weights, use `regen_from_release.sh`, which protects these
  files. `bn_pgmpy_compare/compare_bn.py` (log 23) and `PAPER_xstate_v3/headline_numbers.py` (log 24) run
  outside the driver.
- **Release check.** Before release, the authors ran `regen_from_release.sh` on a clean copy of this
  repository. Every regenerated table, CSV, PNG and LaTeX file was byte-identical to the shipped one; the
  eight regenerated PDFs differed only in their embedded creation date, and `sensitivity_flips.csv`
  differed only in row order (its row order is not deterministic). From the ensembles and projected
  weights they also re-ran Layer 1 for all 51 prior-state pairs, Layers 2-3 for 18 cases,
  `regen_after_g1fix.sh` steps 09 and 12, the S4.1 fresh-reference and household-disjoint collectors
  and the Table S10 budget step; all outputs matched the shipped ones (byte for byte, except the Layer-1
  JSON, which matched value by value as described above).
- **Headline log.** Only the `NEW` sections of `regen_logs_g1fix/24_headline_dump.log` correspond to the
  paper. The `OLD` sections are the authors' comparison with the outputs before the correction of the
  reference data (JWTRNS crosswalk, constant-dollar income) and of D (category union); they are not
  reproducible from this repository. `PAPER_xstate_v3/headline_numbers.py` prints the `NEW` part only.

## 5. Main text: item -> script -> output

Paths are relative to `experiments/E_state/audits_v3/` unless they start with `experiments/`, `hpo/`,
`data_prep/` or `results/`. "Step NN" refers to `regen_after_g1fix.sh`.

| manuscript item | script | output | needs |
|---|---|---|---|
| Sec. 5.2 PUMS processing, constant-2013-dollar income, JWTRNS crosswalk | `_pums_proc.py`, `data_prep/run_data_chain.sh` | `data/pums*/person_level_merged_compact_*.csv` | [RAW] |
| Sec. 5.2 Ohio 2024 income example (42.4 / 28.2 / 22.4 %) | `experiments/E_state/supplementary/adjinc/check/raw_shares.py`, `adjinc/code/relabel.py` | `adjinc/check/raw_shares.{csv,log}`, `adjinc/results/income_shares_by_definition.csv` | [RAW] |
| Sec. 5.2-5.3, App. B datasets and splits | `data_prep/build_state_slices.py`, `data_prep/pums_states/<ST>/sample_and_split.py` | `data/pums_states/<ST>/...` | [RAW] |
| Sec. 5.3, App. B priors, HPO, training, personas | `hpo/`, `train_*.py`, `data_prep/pums_states/<ST>/run_persona_*.py`, `run_emb_*.py` | `results/hpo_v3/` | GPU |
| Sec. 5.3 common settings (alpha, kappa1, kappa2, tau, B0, sigma floor) | `experiments/E_state/decision_certificate_xstate_v3.py`, `experiments/E_state/rerun_audit_new.py` | constants | - |
| Sec. 6, all Layer 1-3 numbers | `experiments/E_state/run_audit_v3.py` | `layer1_paper_*`, `layer2_*`, `layer3_paperW_*` | [ENS] [RAW] |
| Decision certificate | `experiments/E_state/decision_certificate_xstate_v3.py` (steps 01/02) | `PAPER_xstate_v3/xstate_{certificate_all,valid_only,model_by_gate,per_model_robustness}.csv` | [REPO] |
| Table 3 (tab:horizon-validity), Sec. 6.1 counts | `PAPER_xstate_v3/figures/make_paper_tables_v3.py` (step 07) | `PAPER_xstate_v3/figures/tables/tab_horizon_validity.tex` (the manuscript says "cases" where the script says "cells") | [REPO] |
| Figure 1 (fig:admission) | `PAPER_xstate_v3/figures/make_figures_v3.py` (step 08) | `PAPER_xstate_v3/figures/pdf/figA_admission.pdf` | [REPO] |
| Sec. 6.1 CTGAN base-year SRMSE | `hpo/fidelity_table_tex.py` | `results/tables/tab_fidelity_v3.tex` | [REPO] |
| Sec. 6.2 Layer-1 rejections | certificate (`first_fail`, `L1_*`), `layer1_ensemble_summary.json` (`rule_*_mean`) | certificate | [REPO] |
| Sec. 6.2 attribute shares of chi_H | `make_paper_tables_v3.py` | `PAPER_xstate_v3/figures/tables/tab_chiH_decomposition.tex` | [REPO] |
| Figure 2 (fig:l2-commute) and WFH shares | `PAPER_xstate_v3/ablation_l2/make_L2_commute_figure.py` (step 20) | `PAPER_xstate_v3/figures/pdf/fig_L2_wfh_unprojectable.pdf`; numbers in `regen_logs_g1fix/20_commute_figure.log` | [PW] |
| Sec. 6.2 chi_H vs SRMSE_J | `ablation_l2/run_ensemble_table.py` (16), `make_L2_semantics_figure.py` (17) | `ablation_l2/ensemble_results.csv`, `figures/pdf/fig_L2_abnormality_vs_drift.pdf` | [PW] [ENS]; figure [REPO] |
| Sec. 6.2 Layer 3 of the valid cases, displacement, horizon contrast | `PAPER_xstate_v3/headline_numbers.py`, `sensitivity/kappa_recheck.py` (step 10) | `regen_logs_g1fix/24_headline_dump.log`, `sensitivity/kappa_recheck.md` | [REPO] |
| Sec. 6.2 stress projections of rejected priors | `make_paper_tables_v3.py` | `figures/tables/tab_layer3_diagnostics.tex` | [REPO] |
| Sec. 6.2 counterfactual stress test | `PAPER_xstate_v3/l3_counterfactual_telework.py` (05/06) | `l3_counterfactual/*/l3_counterfactual_curve.csv` | [ENS] [RAW] |
| Decision-relevant residual D on the category union (D = d_A,ref + mu_out/2, where mu_out is the projected mass on categories absent from the reference), R_2^MA > D | `PAPER_xstate_v3/figures/make_figE_postproj.py` (09) + `PAPER_xstate_v3/dA_union.py` | `figures/tables/tableE_decision_dA.csv` | [PW] |
| Figure 3 (fig:certificate) | `make_figures_v3.py` (08), then `make_figE_postproj.py` (09) | `figures/pdf/figE_decision.pdf` | [PW] |
| Table 4 (tab:selected-prior-budget), Sec. 6.3 | `make_figE_postproj.py` (09) | `results/tables/tab_selected_prior_budget_v3.tex`, `PAPER_xstate_v3/exp_ii_scenario_budget_post.csv` (the manuscript caption says "post-admission budget rule") | [PW] |
| Sec. 6.3 v(11) | `PAPER_xstate_v3/rebuild_lever_data.py` (01b) | `PAPER_xstate_v3/exp_ii_vH.csv` | [REPO] |
| Sec. 6.3 marginal transfer | `figures/make_transfer_figures_v3.py` (12) | `figures/tables/{tableG,tableH}_*.csv`, `transfer_stats.json` | [PW] |
| Sec. 6.3 single shares | `headline_numbers.py` | `24_headline_dump.log` | [REPO] |
| Sec. 6.3 tau paragraph | `experiments/E_state/supplementary/tau/` | see Table S6 | [ENS] [PW] |
| Sec. 6.4 kappa2 ladder, Figure 4(b), B0 ladder | `sensitivity/sensitivity_sweep.py` (11), `kappa_recheck.py` (10), `kappa_null_sim.py` (10b) | `sensitivity/*.csv`, `kappa_recheck.md` (Chinese headings; English copy `kappa_recheck_EN.md`) | [REPO] |
| Sec. 6.5, Table 5 (tab:baseline-counts) | `baselines/baseline_reproduction.py` (01d), `make_figures_v3.py` figF | `baselines/baseline_comparison.csv`, `figures/tables/tableF_comparison.csv` (the table was typed from these) | [REPO] |
| Sec. 6.5 SRMSE threshold facts | `experiments/E_state/supplementary/srmse/threshold_sweep.py` | `supplementary/srmse/threshold_sweep.{csv,log}` | [REPO] |
| Figure 4 (fig:baseline-comparison) | `make_figures_v3.py` (08, after 11) | `figures/pdf/figF_comparison_sensitivity.pdf` | [REPO] |
| Sec. 7 seed and sigma sensitivity | Supplementary S3.7, S3.8 | see Section 6 below | - |
| App. B, Table B1 taxi footnote | `experiments/E_state/supplementary/jwtrns/verify_crosswalk.py` | `jwtrns/results/footprint_taxi.csv` | [RAW] |
| App. B BN configuration | `hpo/spaces.py` (BDeu, equivalent sample size 66.6383, maximum in-degree 4), `train_bn_pgmpy.py` | `results/hpo_v3/final/bn_pgmpy_<ST>/` | - |
| (diagnostic, not reported in the paper) BN backend comparison | certificate `--extras`, `bn_pgmpy_compare/compare_bn.py` | `PAPER_xstate_v3_extras/`, `bn_pgmpy_compare/` (since the primary `bn` tag audits the study `bn_pgmpy_<ST>`, its `bn` and `bn_pgmpy` audit rows are identical) | [REPO]; compare_bn D part [PW] |
| Figure A1 (fig:framework-flow) | `results/figures/fig_framework_flow.tex` (TikZ, standalone; `pdflatex` gives the manuscript's `fig_framework_flow.pdf`) | - | - |
| Tables 1, 2, C1 | static | - | - |

## 6. Supplementary Material: item -> script -> output

Supplementary scripts sit in `experiments/E_state/supplementary/<name>/` next to their results.

| item | script | output | needs |
|---|---|---|---|
| S1 Figure (fig:joint-transfer) | `PAPER_xstate_v3/figures/make_transfer_figures_v3.py` | `fig_appB_joint_transfer.pdf`, `tableH_joint_transfer.csv`, `transfer_stats.json` | [PW] |
| S2 Table S1 (tab:srmse-sweep) | `srmse/threshold_sweep.py` | `srmse/threshold_sweep.csv`, `per_case_srmse.csv`, `threshold_sweep.log` | [REPO] |
| S3.1 Table S2 (tab:layerwise_outcomes) | `make_paper_tables_v3.py` | `figures/tables/tab_layerwise_outcomes.tex` | [REPO] |
| S3.2 Table S3, Figure (fig:l2-abnormality) | `make_paper_tables_v3.py`; `ablation_l2/run_ensemble_table.py`, `make_L2_semantics_figure.py` | `tab_chiH_decomposition.tex`; `ensemble_results.csv`, `fig_L2_abnormality_vs_drift.pdf` | [REPO]; SRMSE_J [ENS] [PW] |
| S3.3 Table S4, Figure (fig:l3-stress) | `make_paper_tables_v3.py`; `PAPER_xstate_v3/l3_stress_case.py`, `l3_counterfactual_telework.py`, `make_figures_v3.py` | `tab_layer3_diagnostics.tex`; `l3_stress/`, `l3_counterfactual/`, `figD_L3_stress.pdf` | [REPO]; curves [ENS] |
| S3.4 Table S5 (tab:wfh-age-tx) | `ablation_l2/make_wfh_age_tables.py` (21) | `ablation_l2/wfh_age_{TX,OH,CA}_2024.{tex,csv}` | [PW] [RAW] |
| S3.5 Figure (fig:lipschitz-transfer) | `make_transfer_figures_v3.py` | `fig_cor1_marginal_transfer.pdf`, `tableG_marginal_transfer.csv` | [PW] |
| S3.6 gate sweep, kappa ladders, B0 ladder, null reference | `sensitivity/sensitivity_sweep.py`, `kappa_recheck.py`, `kappa_null_sim.py` | `sensitivity/*.csv`, `kappa_recheck.md` | [REPO] |
| S3.6 Table S6 (tab:tau-ladder) | `tau/tau_ladder.py`, `launch.sh`, `collect.py`, `tau_loose.py`, `launch_loose.sh`, `collect_loose.py`, `make_table.py` | `tau/tab_tau_ladder.tex`, `tau_*.csv`, `figS_tau_ladder_D.pdf` | runs [ENS] [PW]; table [REPO] |
| S3.7 Table S7 (tab:sigma-sensitivity), exact thresholds | `sigma/sigma_sens.py`, `l2_fast.py`, `l3_all_g1fix.py`, `ladder.py`, `analyze.py`, `make_table.py`, `verify_fast_vs_script.py`, `boundary_search.py`, `l2check.py`, `code/*` | `sigma/results/*` | Layer 2 [REPO]; Layer 3 / D [ENS] [PW] |
| S3.8 Table S8 (tab:seed-robustness) | `seed/` (generation `cmds/*.sh`, `launch_*.sh`, `ttvae_gen_only.py`, `tabddpm_gen_*.py`, `code/train_ttvae_CA_bigbatch.py` (run with `PYTHONPATH=$POPAUDIT_ROOT`); audit `audit_seed.py`, `g1fix_audit_*.sh`; `cert_seed.py`, `analyze_seeds.py`, `make_seed_table.py`) | `seed/results/*` | [CKPT] [ENS] [PW]; table [REPO] |
| S3.8 TTVAE record reuse across seeds (share of generated records equal to a training record; share identical across seeds) | `seed/verify/ttvae_records.py` (CA, OH, TX), `seed/verify/ttvae_ca_records.py` (CA only, the original check) | `seed/verify/ttvae_records.log`, `ttvae_ca_records.log` | [ENS] [RAW] |
| S3.9 Table S9 (tab:truncated-priors) | `filter/scripts/*` | `filter/certificate/*.csv`, `filter/diag/*.csv` | [ENS] [PW] |
| S4.1 household / group quarters | `household/household_gq_share.py` | `household/household_gq_share.log` | [RAW] |
| S4.1 income in constant dollars | `_pums_proc.py`; `adjinc/code/relabel.py`, `adjinc/code/extract_hh.py` | `adjinc/results/income_shares_by_definition.csv` | [RAW] |
| S4.1 commute-mode coding (JWTRNS) | `jwtrns/verify_crosswalk.py`, `build_refs.py`, `run_jw.py`, `analyze_jw.py`, `taxi_contrib.py` | `jwtrns/results/*.csv` | [RAW] [ENS] [PW] |
| S4.1 splits and overlap rates (observed and expected under independent draws) | `iid_test/code/reproduce_resample.py`, `verify_design.py`, `overlap_tv.py`; `iid_test/independence/code/independence_test.py` | `iid_test/out/*`, `iid_test/results/overlap.csv`, `marginal_tv_fresh_vs_prod.csv`, `iid_test/independence/out/*`, `iid_test/logs/*` | [RAW] |
| S4.1 fresh test reference: Layer-1 statistics and base-year fidelity | `iid_test/code/make_fresh_test.py`, `run_l1_iid.py` (uses `l1_disjoint/code/rerun_l1_paper_disjoint.py`), `control_check_iid.py`, `fidelity_iid.py` | `iid_test/results/{l1_compare,l1_marginal_attr_compare,control_reproduction,fidelity_*}.csv` | [RAW] [ENS] [CKPT] |
| S4.1 fresh test reference: valid count, selected priors, deployment decisions | `misc_g1fix/code/compare_iid_g1fix.py` (replaces `iid_test/code/compare_iid.py`, which computed D before the category-union correction), `misc_g1fix/code/check_scenario_refs_g1fix.py` | `misc_g1fix/iid_test/results/{certificate_L1_fresh,l1_headline,selected_priors}.csv`, `misc_g1fix/iid_test/*.log`, `scenario_refs_check.json` | [RAW] [ENS] [PW] |
| S4.1 "removing the overlapping records would not give a valid reference" (household-disjoint, IPW and placebo references) | `l1_disjoint/code/*` (see `l1_disjoint/README.md`), `misc_g1fix/code/compare_l1_disjoint_g1fix.py` | `l1_disjoint/results/*`, `l1_disjoint/refs/*.csv`, `misc_g1fix/l1_disjoint/` | [RAW] [ENS] [CKPT] |
| S4.1 hyper-parameter search (IF1, rank agreement) | `hpo/`, `if1_check/if1_check.py`, `if1_leakfree.py`, `check/verify_if1.py` | `if1_check/*.csv` | [RAW] [ENS] |
| S4.1 diagnostic rows | `hpo/fidelity_table.py`, certificate `--extras` | `PAPER_xstate_v3_extras/xstate_certificate_all.csv` | [REPO] |
| Table S10 (tab:hpo-budget-v3) | `python -m hpo.finalize --budget-table-only` (or `bash hpo/run_all.sh budget`) | `results/hpo_v3/budget_table.csv` (the LaTeX table was typed from it) | [REPO] |
| Table S11 (tab:fidelity-v3) | `hpo/fidelity_table_tex.py` | `results/tables/tab_fidelity_v3.tex` | [REPO] |
| Table S11 caption, oracle floor (about 0.022, 0.015 to 0.030 across draws) | `iid_test/code/fidelity_iid.py`, `l1_disjoint/code/fidelity_disjoint.py` | `iid_test/results/fidelity_yardsticks.csv` (rows `oracle_prodtest`, column `srmse_1_orig`), `l1_disjoint/results/fidelity_oracle_floor.csv` (column `srmse_1_orig`); the caption text itself is a fixed string in `fidelity_table_tex.py` | [RAW] |
| Table S12 (tab:condition-source-v3) | inputs: `results/hpo_v3/final/{semapop_vae_<ST>_train_mu,ttvae_<ST>_prior}/aggregate.json`, `PAPER_xstate_v3_extras/xstate_certificate_all.csv` (no generator) | - | [REPO] |
| S4.2 hard validity rules | `layer1.py`, `experiments/E_state/l1_invalid_fix_shadow.py`, `filter/scripts/verify_ens_g1fix.py` | `filter/diag/reference_rule_violations.csv` | [RAW] for the reference check |
| Table S13 (tab:sigma-calibration) | upper block: defaults of `layer2_historical_sigma.py` / `rerun_audit_new.py`; lower block: `sigma_calibration/sigma_lower_block.py` | `sigma_calibration/sigma_lower_block.csv` | [REPO] |

## 7. Large inputs available on request

| item | size (approx.) | used by |
|---|---|---|
| raw ACS PUMS (public, `data_prep/download_pums.sh`) | 19 GB | data chain |
| processed state datasets and references | 1.7 GB | Layers 2-3, data-dependent checks |
| persona texts / embeddings / GMM priors | 0.2 / 2.3 / 0.5 GB | SemaPop-VAE+ training |
| checkpoints (seed 42 / all seeds) | 1 / 56 GB | [CKPT] |
| generated ensembles (K = 50 per prior and state) | 29 GB | [ENS], Layer 1-3 re-runs |
| Layer-3 r* cells7 inputs | 0.4 GB | [ENS] (stress, counterfactual, tau, sigma, JWTRNS) |
| Layer-3 projected weights (81 primary cases) | 0.4 GB | [PW] (D on the category union, Figures 2-3, Table 4, transfer) |

The per-case folders contain all other Layer-3 outputs. `layer3_projected_weight_files.json` lists the
weight files that are not shipped.

## 8. Known limitations of this release

- A few numbers are not printed by any shipped script; they were read from the shipped outputs:
  - the Appendix A allowance values (computed inline by `dkw_critical` in the certificate);
  - "SRMSE_J monotone in all 18 pairs" and its 0.97-3.35 range (from `ensemble_results.csv`);
  - the 5/18/8 crossing counts of S3.2;
  - the per-rule breakdown of the Layer-1 rejections and the invalid-mass ratios (from the certificate
    and `layer1_ensemble_summary.json`);
  - the commute constraint binding in the 2024 scenarios (from `attribute_projected_losses_json`);
  - the work-from-home share at tau* (from the tau runs);
  - the observed category counts of Table B1;
  - the commute term 4.43 of S3.7;
  - the HPO facts of S4.1 (25 % / 9 %, trials #17/#16, extra trials, CTGAN pac/log-frequency; from the
    Optuna databases in `results/hpo_v3/studies/`).
- Tables 5, S10 and S12 were typed by hand from the shipped CSVs. The captions of Tables 3 and 4 were
  edited by hand after generation.
- `supplementary/sigma/code/make_figE_postproj.py` writes its LaTeX table to the same file as step 09
  (`results/tables/tab_selected_prior_budget_v3.tex`). The authors ran step 09 after it; do the same.
- Some code strings still contain the authors' internal labels, for example print strings in Chinese in
  `kappa_recheck.py`, an assert message in `model/wgan_llm.py`, argparse descriptions that refer to plan
  items of an earlier project, and literal metrics keys. They are code and were left unchanged so that
  behaviour and outputs stay identical. The same holds for the `notes` of the search-space definitions in
  `hpo/spaces.py`, the provenance strings in `utils/criteria.py`, `train_bn_v2.py` and `hpo/run.py`, and
  the study attributes stored in the Optuna databases of `results/hpo_v3/studies/` (`search_space_notes`,
  `objective`): they keep the authors' internal labels, including decision dates and reviewer-item codes
  (such as "R2-1.2" or "R3-5") of the review of an earlier paper on SemaPop, which these baselines were
  first rebuilt for. In comments and docstrings, such references were removed or reworded; the
  enumerators F2-F8 that remain in some trainer docstrings are the authors' internal item numbers.
- The Table S10 and S12 inputs and the HPO facts of S4.1 are in the shipped records; the BN
  configuration of App. B (BDeu, equivalent sample size 66.6383, maximum in-degree 4) was taken, as
  Supplementary S4.1 states, from an earlier validation-split search on a different data build, whose
  records are not part of this release.
- Manuscript numbers that rely on the Layer-3 projected weights (marked [PW]) can be checked against the
  shipped tables and logs but not regenerated without the weights (Section 7).
- `supplementary/seed/cmds/*.sh` pass `--sample-batch 32768` for TabDDPM. The audited CA/TX seed runs
  regenerated the ensembles at 4096 with `tabddpm_gen_shard.py` (see `g1fix_audit_*.sh`, `launch_*_gen.sh`).
- The logs in `regen_logs_g1fix/` and in `supplementary/*/` are the authors' runs. Absolute paths in them
  were replaced by `<POPAUDIT_ROOT>` / `<POPAUDIT_HPO>` placeholders.
