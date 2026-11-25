# Supplementary experiments

One folder per analysis of the Supplementary Material, with the scripts next to their results. Items of
the Supplementary Material that come from the main pipeline (S1, S3.1-S3.5, the S3.6 gate, kappa and B0
sweeps, Tables S10-S12 and the upper block of Table S13) are produced in `main_experiments/`, `hpo/` and
`audit/`, `training/` and the shared modules at the root; the table at the end lists them too.

Labels: [REPO] runs from this repository alone; [PW] needs the Layer-3 projected weights and the processed
scenario references; [ENS] needs the generated ensembles or the r* cells7 inputs of Layer 3; [CKPT] needs
checkpoints; [RAW] needs raw or processed PUMS files. The large inputs are available on request (see the top-level README). Most
analyses are shadow runs: they call the production audit code with only the output root (and the one
setting under study) changed, and never write into `main_experiments/`. Run the scripts from their own
folder unless a usage line says otherwise.

| folder | Supplementary item |
|---|---|
| `srmse_threshold/` | S2, Table S1 |
| `tau_ladder/` | S3.6, Table S6; Sec. 6.3 tau paragraph |
| `sigma_sensitivity/` | S3.7, Table S7; Sec. 7 |
| `seed_robustness/` | S3.8, Table S8; Sec. 7 |
| `truncated_priors/` | S3.9, Table S9; S4.2 hard-rule check |
| `household_gq/`, `income_deflation/`, `jwtrns_coding/` | S4.1 harmonization checks |
| `iid_reference/`, `household_disjoint/` | S4.1 splits, fresh test reference, household-disjoint references |
| `if1_metric_check/` | S4.1 HPO criterion (IF1) |
| `sigma_calibration/` | S4.2, Table S13 lower block |

## Analyses

**srmse_threshold** (S2, Table S1; Sec. 6.5). Sweeps the threshold of the two reduced protocols
(marginal-only SRMSE) over the 81 cases. `python threshold_sweep.py` [REPO] writes `threshold_sweep.csv`,
`per_case_srmse.csv` and `threshold_sweep.log`.

**tau_ladder** (S3.6, Table S6). Re-projects the 10 valid 2024 cases over a grid of Layer-3 tolerances tau
and records D, nESS, R_3 and feasibility. `bash launch.sh` runs `tau_ladder.py` for every case
(`launch_loose.sh` / `tau_loose.py`: tau > 1); `python collect.py`, `python collect_loose.py`, then
`python make_table.py` write `tau_ladder_all.csv`, `tau_*.csv`, `tab_tau_ladder.tex` and
`figS_tau_ladder_D.pdf`. Runs [ENS] [PW]; table [REPO].

**sigma_sensitivity** (S3.7, Table S7, exact admission thresholds). Varies the historical-uncertainty
scale sigma(H) one setting at a time and re-runs Layers 2-3 for the 54 Layer-1-passing cases with copies
of the audit code in `code/`. `sigma_sens.py --variant <v> --stage all` (or the fast path `l2_fast.py`
plus `l3_all_g1fix.py`), `ladder.py`, `boundary_search.py` and `l2check.py` (thresholds), then
`analyze.py` and `make_table.py`; `verify_fast_vs_script.py` checks the fast path. Outputs in `results/`,
logs in `logs_g1fix/`. Layer 2 [REPO]; Layer 3 and D [ENS] [PW].

**seed_robustness** (S3.8, Table S8). Regenerates the K = 50 ensembles of five priors with training seeds
142 and 242 and audits them. Generation: `cmds/*.sh` (built by `build_cmds.py`), `launch_*.sh`,
`ttvae_gen_only.py`, `tabddpm_gen_*.py`, `code/train_ttvae_CA_bigbatch.py` (run with
`PYTHONPATH=$POPAUDIT_ROOT`). Audit: `audit_seed.py --seed 142` (and `g1fix_audit_*.sh`), then
`cert_seed.py`, `analyze_seeds.py`, `make_seed_table.py`. Outputs in `results/`. Runs [CKPT] [ENS] [PW];
table [REPO]. `verify/ttvae_records.py` (all states) and `verify/ttvae_ca_records.py` (California) measure
TTVAE record reuse across seeds [ENS] [RAW].

**truncated_priors** (S3.9, Table S9). Drops the records that break a hard rule from the three
Layer-1-rejected priors and audits the truncated priors. `scripts/filter_ensembles.py`,
`run_filtered_audit.py` (shadow copies of the drivers in `scripts/`), `summarize_filtered.py`;
`verify_ens_g1fix.py` also scores the splits and references against the seven rules (S4.2). Outputs in
`certificate/` and `diag/` [ENS] [PW].

**household_gq** (S4.1). Group-quarters share of the scenario references:
`python supplementary_experiments/household_gq/household_gq_share.py` [RAW]. This is a release helper
derived from the authors' verification script, so it has no entry in `_PRODUCTION_LAYOUT`.

**income_deflation** (S4.1; Sec. 5.2 Ohio 2024 example). Household income classes under nominal,
survey-year and constant-2013 dollars. `code/extract_hh.py`, `code/relabel.py`; independent check
`check/raw_shares.py`. Outputs `results/income_shares_by_definition.csv`, `check/raw_shares.{csv,log}`
[RAW].

**jwtrns_coding** (S4.1; Table B1 footnote). Checks the Census JWTRNS crosswalk in the references and
audits two light-rail alternatives. `verify_crosswalk.py`, `build_refs.py`, `run_jw.py --variant ...`,
`analyze_jw.py`, `taxi_contrib.py`. Outputs in `results/` [RAW] [ENS] [PW].

**iid_reference** (S4.1 splits and fresh test reference). Reproduces the 2013 resample and the split
design (`code/reproduce_resample.py`, `verify_design.py`, `overlap_tv.py`,
`independence/code/independence_test.py`), draws a fresh test reference (`make_fresh_test.py`,
`make_fresh_ref.py`) and re-runs Layer 1 and base-year fidelity against it (`run_l1_iid.py`,
`control_check_iid.py`, `fidelity_iid.py`). `code/compare_iid_g1fix.py` and
`check_scenario_refs_g1fix.py` give the valid count, selected priors and decisions, with outputs in
`corrected/`; they replace `compare_iid.py`, which computed D before the category-union correction.
[RAW] [ENS] [CKPT] [PW].

**household_disjoint** (S4.1, "removing the overlapping records would not give a valid reference").
Layer 1 against household-disjoint, IPW and placebo references. See `household_disjoint/README.md`;
current comparison `code/compare_l1_disjoint_g1fix.py`, outputs in `corrected/`. [RAW] [ENS] [CKPT].

**if1_metric_check** (S4.1 HPO criterion). Exact-tuple IF1 and its rank agreement with the HPO criterion.
`if1_check.py`, `if1_leakfree.py`; independent re-computation `check/verify_if1.py` [RAW] [ENS].

**sigma_calibration** (S4.2, Table S13). `python sigma_lower_block.py` recomputes the lower block from
the shipped Layer-2 outputs [REPO]; the upper block is the defaults of `audit/layer2_historical_sigma.py` and
`main_experiments/rerun_audit_new.py`.

## Item -> script -> output

Paths are relative to `supplementary_experiments/` unless they start with another top-level folder.
`V3` stands for `main_experiments/audits_v3/PAPER_xstate_v3`.

| item | script | output | needs |
|---|---|---|---|
| S1 Figure (fig:joint-transfer) | `V3/figures/make_transfer_figures_v3.py` | `fig_appB_joint_transfer.pdf`, `tableH_joint_transfer.csv`, `transfer_stats.json` | [PW] |
| S2 Table S1 (tab:srmse-sweep) | `srmse_threshold/threshold_sweep.py` | `srmse_threshold/threshold_sweep.csv`, `per_case_srmse.csv`, `threshold_sweep.log` | [REPO] |
| S3.1 Table S2 (tab:layerwise_outcomes) | `V3/figures/make_paper_tables_v3.py` | `V3/figures/tables/tab_layerwise_outcomes.tex` | [REPO] |
| S3.2 Table S3, Figure (fig:l2-abnormality) | `make_paper_tables_v3.py`; `V3/ablation_l2/run_ensemble_table.py`, `make_L2_semantics_figure.py` | `tab_chiH_decomposition.tex`; `ensemble_results.csv`, `fig_L2_abnormality_vs_drift.pdf` | [REPO]; SRMSE_J [ENS] [PW] |
| S3.3 Table S4, Figure (fig:l3-stress) | `make_paper_tables_v3.py`; `V3/l3_stress_case.py`, `l3_counterfactual_telework.py`, `make_figures_v3.py` | `tab_layer3_diagnostics.tex`; `V3/l3_stress/`, `V3/l3_counterfactual/`, `figD_L3_stress.pdf` | [REPO]; curves [ENS] |
| S3.4 Table S5 (tab:wfh-age-tx) | `V3/ablation_l2/make_wfh_age_tables.py` (step 21 of `regen_after_g1fix.sh`) | `V3/ablation_l2/wfh_age_{TX,OH,CA}_2024.{tex,csv}` | [PW] [RAW] |
| S3.5 Figure (fig:lipschitz-transfer) | `make_transfer_figures_v3.py` | `fig_cor1_marginal_transfer.pdf`, `tableG_marginal_transfer.csv` | [PW] |
| S3.6 gate sweep, kappa ladders, B0 ladder, null reference | `V3/sensitivity/sensitivity_sweep.py`, `kappa_recheck.py`, `kappa_null_sim.py` | `V3/sensitivity/*.csv`, `kappa_recheck.md` | [REPO] |
| S3.6 Table S6 (tab:tau-ladder) | `tau_ladder/*` | `tau_ladder/tab_tau_ladder.tex`, `tau_*.csv`, `figS_tau_ladder_D.pdf` | runs [ENS] [PW]; table [REPO] |
| S3.7 Table S7 (tab:sigma-sensitivity), exact thresholds | `sigma_sensitivity/*` | `sigma_sensitivity/results/*` | Layer 2 [REPO]; Layer 3 / D [ENS] [PW] |
| S3.8 Table S8 (tab:seed-robustness) | `seed_robustness/*` | `seed_robustness/results/*` | [CKPT] [ENS] [PW]; table [REPO] |
| S3.8 TTVAE record reuse across seeds | `seed_robustness/verify/ttvae_records.py`, `ttvae_ca_records.py` | `seed_robustness/verify/ttvae_records.log`, `ttvae_ca_records.log` | [ENS] [RAW] |
| S3.9 Table S9 (tab:truncated-priors) | `truncated_priors/scripts/*` | `truncated_priors/certificate/*.csv`, `truncated_priors/diag/*.csv` | [ENS] [PW] |
| S4.1 household / group quarters | `household_gq/household_gq_share.py` | `household_gq/household_gq_share.log` | [RAW] |
| S4.1 income in constant dollars | `_pums_proc.py`; `income_deflation/code/relabel.py`, `extract_hh.py` | `income_deflation/results/income_shares_by_definition.csv` | [RAW] |
| S4.1 commute-mode coding (JWTRNS) | `jwtrns_coding/*.py` | `jwtrns_coding/results/*.csv` | [RAW] [ENS] [PW] |
| S4.1 splits and overlap rates (observed and expected under independent draws) | `iid_reference/code/reproduce_resample.py`, `verify_design.py`, `overlap_tv.py`; `iid_reference/independence/code/independence_test.py` | `iid_reference/out/*`, `iid_reference/results/overlap.csv`, `marginal_tv_fresh_vs_prod.csv`, `iid_reference/independence/out/*`, `iid_reference/logs/*` | [RAW] |
| S4.1 fresh test reference: Layer-1 statistics and base-year fidelity | `iid_reference/code/make_fresh_test.py`, `run_l1_iid.py` (uses `household_disjoint/code/rerun_l1_paper_disjoint.py`), `control_check_iid.py`, `fidelity_iid.py` | `iid_reference/results/{l1_compare,l1_marginal_attr_compare,control_reproduction,fidelity_*}.csv` | [RAW] [ENS] [CKPT] |
| S4.1 fresh test reference: valid count, selected priors, decisions | `iid_reference/code/compare_iid_g1fix.py`, `check_scenario_refs_g1fix.py` | `iid_reference/corrected/results/{certificate_L1_fresh,l1_headline,selected_priors}.csv`, `iid_reference/corrected/*.log`, `scenario_refs_check.json` | [RAW] [ENS] [PW] |
| S4.1 household-disjoint, IPW and placebo references | `household_disjoint/code/*` | `household_disjoint/results/*`, `refs/*.csv`, `corrected/` | [RAW] [ENS] [CKPT] |
| S4.1 hyper-parameter search (IF1, rank agreement) | `hpo/`, `if1_metric_check/if1_check.py`, `if1_leakfree.py`, `check/verify_if1.py` | `if1_metric_check/*.csv` | [RAW] [ENS] |
| S4.1 diagnostic rows | `hpo/fidelity_table.py`, certificate `--extras` | `main_experiments/audits_v3/PAPER_xstate_v3_extras/xstate_certificate_all.csv` | [REPO] |
| Table S10 (tab:hpo-budget-v3) | `python -m hpo.finalize --budget-table-only` (or `bash hpo/run_all.sh budget`) | `hpo/records/budget_table.csv` | [REPO] |
| Table S11 (tab:fidelity-v3) | `hpo/fidelity_table_tex.py` | `main_experiments/tables/tab_fidelity_v3.tex` | [REPO] |
| Table S11 caption, oracle floor (about 0.022, 0.015 to 0.030 across draws) | `iid_reference/code/fidelity_iid.py`, `household_disjoint/code/fidelity_disjoint.py` | `iid_reference/results/fidelity_yardsticks.csv` (rows `oracle_prodtest`, column `srmse_1_orig`), `household_disjoint/results/fidelity_oracle_floor.csv`; the caption text is a fixed string in `fidelity_table_tex.py` | [RAW] |
| Table S12 (tab:condition-source-v3) | inputs: `hpo/records/final/{semapop_vae_<ST>_train_mu,ttvae_<ST>_prior}/aggregate.json`, `PAPER_xstate_v3_extras/xstate_certificate_all.csv` (no generator) | - | [REPO] |
| S4.2 hard validity rules | `audit/layer1.py`, `main_experiments/l1_invalid_fix_shadow.py`, `truncated_priors/scripts/verify_ens_g1fix.py` | `truncated_priors/diag/reference_rule_violations.csv` | [RAW] for the reference check |
| Table S13 (tab:sigma-calibration) | upper block: defaults of `audit/layer2_historical_sigma.py` / `main_experiments/rerun_audit_new.py`; lower block: `sigma_calibration/sigma_lower_block.py` | `sigma_calibration/sigma_lower_block.csv` | [REPO] |

The authors also re-ran, from the ensembles and projected weights, the S4.1 fresh-reference and
household-disjoint collectors and the Table S10 budget step before release; all outputs matched the
shipped ones.

## Known limitations

- Some numbers are read from the shipped outputs and printed by no script: "SRMSE_J monotone in all 18
  pairs" and its 0.97-3.35 range (`ensemble_results.csv`), the 5/18/8 crossing counts of S3.2, the
  work-from-home share at tau* (tau runs), the commute term 4.43 of S3.7, and the HPO facts of S4.1
  (25 % / 9 %, trials #17/#16, extra trials, CTGAN pac/log-frequency; from the Optuna databases in
  `hpo/records/studies/`).
- Tables S10 and S12 were typed by hand from the shipped CSVs.
- `sigma_sensitivity/code/make_figE_postproj.py` writes its LaTeX table to the same file as step 09 of
  `regen_after_g1fix.sh` (`main_experiments/tables/tab_selected_prior_budget_v3.tex`). The authors ran
  step 09 after it; do the same.
- `seed_robustness/cmds/*.sh` pass `--sample-batch 32768` for TabDDPM. The audited CA/TX seed runs
  regenerated the ensembles at 4096 with `tabddpm_gen_shard.py` (see `g1fix_audit_*.sh`,
  `launch_*_gen.sh`).
- `iid_reference/` (`verify_design.py`, `make_fresh_ref.py`, the independence test) and
  `income_deflation/` (`extract_hh.py`, `raw_shares.py`) read the raw 2013 state files
  `data/pums13/{person,housing}_2013_<ST>_full.csv`, which no shipped script produces (available on
  request).
- The logs were written in the authors' working tree, so they show its folder names (for example
  `audits_v3_referee/tau` for `tau_ladder/`); `popaudit_paths.py` lists the mapping.
