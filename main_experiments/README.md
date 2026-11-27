# Main-text experiments

This folder holds the audit of the 81 prior-state-horizon cases (Sections 5-6 and Appendices A-B):
the audit drivers, the decision certificate, the per-case outputs, and the scripts and outputs behind
every main-text table and figure.

```
run_audit_v3.py                 runs Layers 1-3 for the 81 cases (--extras: diagnostic rows)
rerun_l1_paper.py, rerun_audit_new.py   Layer-1 and Layer-2/3 drivers called by run_audit_v3.py
l1_invalid_fix_shadow.py        independent re-implementation of the seven hard rules (check of audit/layer1.py)
decision_certificate_xstate_v3.py   gates, first failed gate, R_2^MA, certificate tables
audits_v3/
  layer1_paper_<prior>_<ST>/, layer2_<prior>_<ST>_<Y>/, layer3_paperW_<prior>_<ST>_<Y>/   per-case outputs
  PAPER_xstate_v3/              certificate, budget and decision tables, figures, sensitivity, ablations
  PAPER_xstate_v3_extras/       certificate of the diagnostic rows (App. B, Supplementary S4.1, Table S12)
  bn_pgmpy_compare/             BN backend comparison (diagnostic, not reported in the paper)
  regen_after_g1fix.sh          the authors' driver for every main-text artifact (steps 01-22)
  regen_from_release.sh         the steps that run from this repository alone
  regen_logs_g1fix/             logs of the authors' last full run (numbers printed by each step)
tables/                         Table 3 and Supplementary Table S11 (.tex), written here by their generators
figures/fig_framework_flow.tex  TikZ source of Figure A1
```

Labels: [REPO] runs from this repository alone; [PW] needs the Layer-3 projected weights and the processed
scenario references; [ENS] needs the generated ensembles or the r* cells7 inputs of Layer 3; [CKPT] needs
checkpoints; [RAW] needs raw or processed PUMS files. The large inputs are available on request (see the top-level README).

## Pipeline

| step | command | needs |
|---|---|---|
| 1. Data: PUMS processing, state slices, splits, personas | `bash data_prep/run_data_chain.sh` (see `data_prep/README.md`) | [RAW], GPU for personas |
| 2. Priors: search, 3-seed training, K = 50 ensembles | `bash hpo/run_all.sh search`, then `finalize` (see `hpo/README.md`) | GPU, data |
| 3. Audit Layers 1-3, 81 cases | `python main_experiments/run_audit_v3.py [--extras]` | [ENS] [RAW] |
| 4. Certificate | `cd main_experiments && python decision_certificate_xstate_v3.py [--extras]` | [REPO] |
| 5. Tables and figures | `bash main_experiments/audits_v3/regen_after_g1fix.sh` | [PW] [ENS] for some steps |
| 5'. Everything that runs from this repository | `bash main_experiments/audits_v3/regen_from_release.sh` | [REPO] |

Notes:

- **Audit (step 3).** `run_audit_v3.py --l23-only` re-runs Layer 2, the r* cells7 inputs and Layer 3 and
  reuses Layer 1. `audit/layer1.py` implements the seven hard rules of Supplementary S4.2, so a fresh Layer-1 run
  reproduces every value of the shipped `layer1_ensemble_summary.json` (the fresh file also has
  `rule_child_not_employed_std`, lacks the provenance key `invalid_rules_note`, and orders keys
  differently). `l1_invalid_fix_shadow.py` writes `audits_v3_l1fix/`, identical for all 51 pairs. In every
  `layer1_paper_*` folder the summary and per-realization CSVs were regenerated with the released
  `audit/layer1.py`; the certificate reads only the JSON. The primary BN row is audited under the tag `bn` (study
  `bn_pgmpy_<ST>`); the `--extras` run audits the same study again as `bn_pgmpy`, so `layer*_bn_*` and
  `layer*_bn_pgmpy_*` are identical. Both are shipped: the certificate reads the first, `kappa_null_sim.py`
  the second. `layer3_projected_weight_files.json` in each `layer3_paperW_*` folder lists the projected-weight
  files, which are not shipped ([PW]).
- **Order (step 5).** In `regen_after_g1fix.sh`, step 11 runs before step 08 (Figure 4(b) reads the sweep)
  and step 09 must run after step 08: `make_figures_v3.py` writes the pre-projection
  `tableE_decision_dA.csv` and `figE_decision.*`, and step 09 replaces them with the post-projection
  versions used in the paper. Without projected weights, use `regen_from_release.sh`, which keeps the
  shipped post-projection files. `bn_pgmpy_compare/compare_bn.py` (log 23) and
  `PAPER_xstate_v3/headline_numbers.py` (log 24) run outside the driver.
- **Release check.** The authors ran `regen_from_release.sh` on a clean copy of the repository. Every
  regenerated table, CSV, PNG and LaTeX file was byte-identical to the shipped one; the PDFs differed only
  in their creation date and `sensitivity_flips.csv` only in row order. From the ensembles and projected
  weights they also re-ran Layer 1 for all 51 pairs, Layers 2-3 for 18 cases and steps 09 and 12; all
  outputs matched.
- **Headline log.** Only the `NEW` sections of `regen_logs_g1fix/24_headline_dump.log` belong to the paper.
  The `OLD` sections compare with outputs before the correction of the reference data (JWTRNS crosswalk,
  constant-dollar income) and of D (category union) and cannot be reproduced here.
  `PAPER_xstate_v3/headline_numbers.py` prints the `NEW` part.

## Item -> script -> output

Paths are relative to `main_experiments/audits_v3/` unless they start with a top-level folder or are
files at the repository root. A bare file name refers to a script or file named with its full path in an
earlier row. "Step NN" refers to `regen_after_g1fix.sh`.

| manuscript item | script | output | needs |
|---|---|---|---|
| Sec. 5.2 PUMS processing, constant-2013-dollar income, JWTRNS crosswalk | `_pums_proc.py`, `data_prep/run_data_chain.sh` | `data/pums*/person_level_merged_compact_*.csv` | [RAW] |
| Sec. 5.2 Ohio 2024 income example (42.4 / 28.2 / 22.4 %) | `supplementary_experiments/income_deflation/check/raw_shares.py`, `income_deflation/code/relabel.py` | `income_deflation/check/raw_shares.{csv,log}`, `income_deflation/results/income_shares_by_definition.csv` | [RAW] |
| Sec. 5.2-5.3, App. B datasets and splits | `data_prep/build_state_slices.py`, `data_prep/pums_states/<ST>/sample_and_split.py` | `data/pums_states/<ST>/...` | [RAW] |
| Sec. 5.3, App. B priors, HPO, training, personas | `hpo/`, `training/train_*.py`, `data_prep/pums_states/<ST>/run_persona_*.py`, `run_emb_*.py` | `hpo/records/` | GPU |
| Sec. 5.3 common settings (alpha, kappa1, kappa2, tau, B0, sigma floor) | `main_experiments/decision_certificate_xstate_v3.py`, `main_experiments/rerun_audit_new.py` | constants | - |
| Sec. 6, all Layer 1-3 numbers | `main_experiments/run_audit_v3.py` | `layer1_paper_*`, `layer2_*`, `layer3_paperW_*` | [ENS] [RAW] |
| Decision certificate | `main_experiments/decision_certificate_xstate_v3.py` (steps 01/02) | `PAPER_xstate_v3/xstate_{certificate_all,valid_only,model_by_gate,per_model_robustness}.csv` | [REPO] |
| Sec. 6.1 counts (`tab_horizon_validity.tex` is an intermediate table that the manuscript does not print) | `PAPER_xstate_v3/figures/make_paper_tables_v3.py` (step 07) | `PAPER_xstate_v3/figures/tables/tab_horizon_validity.tex` (the script says "cells" where the manuscript says "cases") | [REPO] |
| Figure 1 (fig:admission) | `PAPER_xstate_v3/figures/make_figures_v3.py` (step 08) | `PAPER_xstate_v3/figures/pdf/figA_admission.pdf` | [REPO] |
| Sec. 6.1 CTGAN base-year SRMSE | `hpo/fidelity_table_tex.py` | `main_experiments/tables/tab_fidelity_v3.tex` | [REPO] |
| Sec. 6.2 Layer-1 rejections | certificate (`first_fail`, `L1_*`), `layer1_ensemble_summary.json` (`rule_*_mean`) | certificate | [REPO] |
| Sec. 6.2 attribute shares of chi_H | `make_paper_tables_v3.py` | `PAPER_xstate_v3/figures/tables/tab_chiH_decomposition.tex` | [REPO] |
| Figure 2 (fig:l2-commute) and WFH shares | `PAPER_xstate_v3/ablation_l2/make_L2_commute_figure.py` (step 20) | `PAPER_xstate_v3/figures/pdf/fig_L2_wfh_unprojectable.pdf`; numbers in `regen_logs_g1fix/20_commute_figure.log` | [PW] |
| Sec. 6.2 chi_H vs SRMSE_J | `PAPER_xstate_v3/ablation_l2/run_ensemble_table.py` (16), `make_L2_semantics_figure.py` (17) | `PAPER_xstate_v3/ablation_l2/ensemble_results.csv`, `PAPER_xstate_v3/figures/pdf/fig_L2_abnormality_vs_drift.pdf` | [PW] [ENS]; figure [REPO] |
| Sec. 6.2 Layer 3 of the valid cases, displacement, horizon contrast | `PAPER_xstate_v3/headline_numbers.py`, `PAPER_xstate_v3/sensitivity/kappa_recheck.py` (step 10) | `regen_logs_g1fix/24_headline_dump.log`, `PAPER_xstate_v3/sensitivity/kappa_recheck.md` | [REPO] |
| Sec. 6.2 stress projections of rejected priors | `make_paper_tables_v3.py` | `PAPER_xstate_v3/figures/tables/tab_layer3_diagnostics.tex` | [REPO] |
| Sec. 6.2 counterfactual stress test | `PAPER_xstate_v3/l3_counterfactual_telework.py` (05/06) | `PAPER_xstate_v3/l3_counterfactual/*/l3_counterfactual_curve.csv` | [ENS] [RAW] |
| Residual D on the category union (D = d_A,ref + mu_out/2, mu_out = projected mass on categories absent from the reference), R_2^MA > D | `PAPER_xstate_v3/figures/make_figE_postproj.py` (09) + `PAPER_xstate_v3/dA_union.py` | `PAPER_xstate_v3/figures/tables/tableE_decision_dA.csv` | [PW] |
| Figure 3 (fig:certificate) | `make_figures_v3.py` (08), then `make_figE_postproj.py` (09) | `PAPER_xstate_v3/figures/pdf/figE_decision.pdf` | [PW] |
| Table 3 (tab:selected-prior-budget), Sec. 6.3 | `make_figE_postproj.py` (09) | `main_experiments/tables/tab_selected_prior_budget_v3.tex`, `PAPER_xstate_v3/exp_ii_scenario_budget_post.csv` | [PW] |
| Sec. 6.3 v(11) | `PAPER_xstate_v3/rebuild_lever_data.py` (01b) | `PAPER_xstate_v3/exp_ii_vH.csv` | [REPO] |
| Sec. 6.3 marginal transfer | `PAPER_xstate_v3/figures/make_transfer_figures_v3.py` (12) | `PAPER_xstate_v3/figures/tables/{tableG,tableH}_*.csv`, `PAPER_xstate_v3/figures/tables/transfer_stats.json` | [PW] |
| Sec. 6.3 single shares | `headline_numbers.py` | `24_headline_dump.log` | [REPO] |
| Sec. 6.3 tau paragraph | `supplementary_experiments/tau_ladder/` | see Supplementary Table S6 | [ENS] [PW] |
| Sec. 6.4 kappa2 ladder, Figure 4(b), B0 ladder | `PAPER_xstate_v3/sensitivity/sensitivity_sweep.py` (11), `kappa_recheck.py` (10), `kappa_null_sim.py` (10b) | `PAPER_xstate_v3/sensitivity/*.csv`, `kappa_recheck.md` (Chinese headings; English copy `kappa_recheck_EN.md`) | [REPO] |
| Sec. 6.5, Table 4 (tab:baseline-counts) | `PAPER_xstate_v3/baselines/baseline_reproduction.py` (01d), `make_figures_v3.py` figF | `PAPER_xstate_v3/baselines/baseline_comparison.csv`, `PAPER_xstate_v3/figures/tables/tableF_comparison.csv` | [REPO] |
| Sec. 6.5 SRMSE threshold facts | `supplementary_experiments/srmse_threshold/threshold_sweep.py` | `srmse_threshold/threshold_sweep.{csv,log}` | [REPO] |
| Figure 4 (fig:baseline-comparison) | `make_figures_v3.py` (08, after 11) | `figures/pdf/figF_comparison_sensitivity.pdf` | [REPO] |
| Sec. 7 seed and sigma sensitivity | Supplementary S3.7, S3.8 | see `supplementary_experiments/README.md` | - |
| App. B, Table B1 taxi footnote | `supplementary_experiments/jwtrns_coding/verify_crosswalk.py` | `jwtrns_coding/results/footprint_taxi.csv` | [RAW] |
| App. B BN configuration | `hpo/spaces.py` (BDeu, equivalent sample size 66.6383, maximum in-degree 4), `training/train_bn_pgmpy.py` | `hpo/records/final/bn_pgmpy_<ST>/` | - |
| (diagnostic, not in the paper) BN backend comparison | certificate `--extras`, `bn_pgmpy_compare/compare_bn.py` | `PAPER_xstate_v3_extras/`, `bn_pgmpy_compare/` | [REPO]; D part [PW] |
| Figure A1 (fig:framework-flow) | `main_experiments/figures/fig_framework_flow.tex` (standalone TikZ; `pdflatex` gives `fig_framework_flow.pdf`) | - | - |
| Tables 1, 2, D1 | static | - | - |

## Known limitations

- Some numbers are read from the shipped outputs and printed by no script: the Appendix A allowance
  values (computed inline by `dkw_critical` in the certificate), the per-rule breakdown of the Layer-1
  rejections and the invalid-mass ratios (certificate and `layer1_ensemble_summary.json`), the commute
  constraint binding in the 2024 scenarios (`attribute_projected_losses_json`) and the observed category
  counts of Table B1.
- Table 4 was typed by hand from the shipped CSVs, and the caption of Table 3 was edited by hand after
  generation.
- Numbers marked [PW] can be checked against the shipped tables and logs but not regenerated without the
  projected weights.
- `kappa_recheck.py` prints Chinese strings; they are code and were left unchanged so that outputs stay
  identical.
