# Layer 1 against household-disjoint, IPW and placebo references (Supplementary S4.1)

Supplementary S4.1 states that removing the test records that share a source household with the
training split "would not give a valid reference, because it selectively discards large households,
which carry many resampled records". This folder holds the analysis behind that sentence. No prior was
retrained; only the Layer-1 reference was swapped.

**Question.** The 2013 base-year data are a PWGTP-proportional resample of PUMS persons, and the
train/validation/test split works on rows, so the Layer-1 reference (`testdata_pers.csv`) shares source
persons and households with the training split. Does the Layer-1 verdict depend on that overlap?

## What was run

| step | script | output |
|---|---|---|
| 1 references | `code/build_refs.py` | `refs/<ST>/testdata_hhdisjoint_trainval.csv` (a, primary), `testdata_hhdisjoint_train.csv` (b); `refs/summary.csv`, `refs/marginal_shift.csv` |
| 1c IPW | `code/build_ipw_ref.py` | `refs/<ST>/testdata_hhdisjoint_trainval_ipw.csv`: (a) post-stratified on household size n_h back to the test composition; `refs/ipw_*.csv` |
| 1e placebo | `code/build_placebo_ref.py` | `refs/<ST>/testdata_placebo_s{1,2}.csv`: the selection rule of (a) applied to a random set of 50,000 positions instead of the real train+val positions (same expected composition as (a), but keeps train-overlapping households); `refs/placebo_summary.csv` |
| 2 driver | `code/rerun_l1_paper_disjoint.py` (= `experiments/E_state/rerun_l1_paper.py` with only the reference path and output root settable, see `code/driver.diff`), `code/run_l1_disjoint.py` | `runs/<variant>/layer1_paper_<tag>_<ST>/` |
| 3 control | `run_l1_disjoint.py --variant orig` (all 27 pairs), `code/control_check.py` | `results/control_reproduction.csv` |
| 4 runs | variants `hhdisjoint_trainval`, `hhdisjoint_train`, `hhdisjoint_trainval_ipw`, `placebo_s1`, `placebo_s2` (27 pairs each) | `runs/<variant>/` |
| 5 compare | `code/compare.py` (gate = `decision_certificate_xstate_v3.layer1_test`, imported), `code/oracle_l1.py`, `code/overlap_contrast.py` | `results/l1_disjoint_compare.csv`, `l1_marginal_attr_compare.csv`, `l1_real_yardsticks.csv`, `overlap_contrast.csv` |
| 5' compare (current) | `../misc_g1fix/code/compare_l1_disjoint_g1fix.py` (= `compare.py` with D on the category union) | `../misc_g1fix/l1_disjoint/results/` (certificates, `l1_disjoint_headline.csv`), `../misc_g1fix/l1_disjoint/compare_g1fix.log` |
| 6 fidelity | `code/fidelity_disjoint.py` (`hpo/finalize.reference_metrics_for` with only the reference swapped; all 13 rows of Table S11 x 3 seeds) | `results/fidelity_per_seed.csv`, `fidelity_disjoint_compare.csv`, `fidelity_oracle_floor.csv` |

The reference files under `refs/<ST>/` and the run folders under `runs/` are record-level or large and
are not shipped (available on request); the summaries are. The first comparison was made before the
correction of D (category union) and of the 2018-2024 references; its certificate files and headline
were re-collected with `compare_l1_disjoint_g1fix.py`, and only the re-collected versions are shipped.
Layer-1 statistics themselves were not affected by that correction (`l1_disjoint_compare.csv` and
`l1_marginal_attr_compare.csv` are identical in both collections).

Gate (as in the certificate): an invalid mass above 0.005 rejects with no allowance. Otherwise a pair
passes when marg <= 0.05 - eta, dep <= 0.10 - eta and 1 - cov <= 0.30 - eta_supp, with
eta = sqrt(log(2u/alpha')/(2n)), alpha' = 0.05/3, u = 9 and n = K n_agents for marg/dep, and u = 1 and
n = n_agents for support. `*_refn` columns repeat the gate with n_agents = |reference variant|; they give
the same verdicts.

## Key results

- **Overlap.** In CA/OH/TX, 39.1/65.7/49.4 % of the test rows share their source household with the
  training split (train+val: 45.0/71.6/55.5 %) and 14.5/37.8/21.5 % share their source person. Reference
  sizes: (a) 181,537/18,209/94,648 of 330,109/64,112/212,756 and (b) 200,910/21,990/107,608 rows, both with
  0 shared households and persons. Validation rows share a household with training rows at similar rates,
  so the HPO validation split is not household-held-out either.
- **(a) is not representative.** A household survives only if none of its n_h resampled rows fell into
  train+val; the empirical survival rates match the probability prod_j (N-M-j)/(N-j), so the subset is
  biased towards small households. The real test split itself scores marg 0.055/0.081/0.125 (CA/TX/OH)
  against (a), so a perfect population model would fail the Layer-1 marginal gate on (a).
- **Control.** With the original reference, the driver reproduces the shipped Layer-1 outputs exactly in
  all 27 pairs (`results/control_reproduction.csv`): 18/27 pairs pass, 54/81 cases pass Layer 1 and
  23/81 cases are valid (`../misc_g1fix/l1_disjoint/results/l1_disjoint_headline.csv`, variant `orig`).
- **Layer-1 passes among the 18 passing pairs:** (a) 0, placebo_s1 0, placebo_s2 0, (b) 5, IPW 15 (fails:
  WGAN-GP-ZCR OH, FlowSyn OH, TabDDPM OH). The 9 failing pairs fail in every variant, because the invalid
  mass does not depend on the reference.
- **Overlap effect at matched composition, (a) minus mean(placebo)** (`results/overlap_contrast.csv`):
  marg -0.0003 on average, dep -0.0008, 1 - cov +0.0011, all within the placebo spread except a small
  record-reuse signature in 1 - cov for TTVAE (+0.007 to +0.009) and SemaPop-VAE+ (+0.001 to +0.003). The
  Layer-1 collapse on (a) is therefore a composition effect: placebo references collapse identically.
- **IPW** flips occur only in Ohio, where (a) has no households with n_h >= 12 and the reweighted
  reference stays shifted (the real test split scores marg 0.035 / dep 0.079 against it). Taken at face
  value, the valid count would drop from 23 to 21; none of the flipped pairs is a selected prior, so the
  selected priors and the deployment decisions are unchanged.
- **Fidelity (Table S11).** Recomputation against the shipped reference reproduces all 117 stored values.
  Against (a), the good priors' SRMSE rises roughly 3-10x, as does that of a real 10k test-split sample
  (`results/fidelity_oracle_floor.csv`: column `srmse_1_orig` is the oracle floor of the Table S11
  caption, 0.015 to 0.030 across draws, mean about 0.022).

## Caveats

- A household-disjoint reference cannot be made representative here without retraining, because the
  split itself is row-based. Ohio is the worst case. The placebo contrast is the identifying comparison.
- `layer1.py` implements the seven hard rules of Supplementary S4.2 (there is no
  `rule_student_age_reasonable` column).
