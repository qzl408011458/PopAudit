# Hyper-parameter search and training

Optuna search, final training and fidelity tables of the nine candidate priors and the diagnostic rows
(Section 5.3, Appendix B, Supplementary S4.1 and Tables S10-S12). The trainers are in `training/` and the
models in `model/`; `spaces.py` names the trainer of each model, and `run.py` and `finalize.py` run it as
`training/<trainer>` with the repository root as working directory.

```
bash hpo/run_all.sh search        # Optuna TPE (seed 42), median pruner, selection on the validation split
bash hpo/run_all.sh finalize      # 3-seed retrain, reference evaluation, K = 50 ensemble on seed 42
bash hpo/run_all.sh diagnostics   # rows outside the candidate set (App. B, S4.1, Table S12)
bash hpo/run_all.sh budget        # Table S10 budget table
python hpo/fidelity_table_tex.py  # Table S11 -> main_experiments/tables/tab_fidelity_v3.tex
python hpo/fidelity_table.py      # console version of the fidelity table, diagnostic rows included
```

`hpo/common.py` reads the state datasets from `$POPAUDIT_DATA/pums_states` and keeps studies, trials and
finalize runs under `$POPAUDIT_HPO` (default `hpo/records/`, where the authors' records are). Point
`POPAUDIT_HPO` elsewhere before a new search or finalize run to keep the shipped records intact. If you
obtain the generated ensembles on request, unpack them into `hpo/records/final/` or point
`POPAUDIT_ENSEMBLES` at their folder.

## Records (`hpo/records/`)

| path | contents |
|---|---|
| `studies/*.db` | Optuna study databases (user attributes sanitized) |
| `best/*.json` | best trial per study |
| `final/<study>/{aggregate.json,seed_runs.json,budget_row.csv}` | selected configuration, 3-seed fidelity, budget row |
| `final/<study>/<run>_final_s42/metrics.json` | seed-42 run metrics of the BN variants and the as-shipped CTGAN/TVAE rows |
| `budget_table.csv` | Table S10 (the LaTeX table was typed from it) |
| `finalize_commands.tsv` | exact trainer argv of every finalize run |

Notes:

- **SemaPop-VAE+ arm.** The audited SemaPop-VAE+ is the M-R arm: `--cond-mode train` resamples
  training-record embeddings. The argparse default of `train_semapop_vae_v2.py` is `--cond-mode mu` (the M-U
  arm, Table S12 only). `run_all.sh` and `finalize_commands.tsv` pass the audited setting explicitly.
- **Finalize settings.** `finalize_commands.tsv` covers all 81 primary prior-state-seed runs and the
  diagnostic rows; `$POPAUDIT_OUT/hpo_v3` there stands for the HPO output root. The commands (and the `cmd`
  field of `best/*.json`) name the trainers as they sat at the authors' repository root; in this release run
  them from the root as `python training/train_*.py ...`. The as-shipped CTGAN and
  TVAE runs are missing because their logs were overwritten; their configuration is in
  `final/{ctgan,tvae_ref}_defaults_<ST>/*_final_s42/metrics.json` (`config`). TabDDPM uses the trainer
  default `--sample-batch 4096`, which the finalize argv does not pass. For the reselected TabDDPM
  (`tabddpm_<ST>_valsrmse`), `aggregate.json` holds the configuration that was actually retrained; it
  differs from `best/tabddpm_<ST>.json`, which is the ML-efficacy optimum.
- **BN.** The BN configuration of App. B (BDeu, equivalent sample size 66.6383, maximum in-degree 4, set in
  `spaces.py`) was taken, as Supplementary S4.1 states, from an earlier validation-split search on a
  different data build, whose records are not part of this release. `train_bn_v2.py` is shipped because
  `spaces.py` and `train_bn_pgmpy.py` import it.
- **Internal labels.** Some code strings keep the authors' internal labels: the `notes` of the search-space
  definitions in `spaces.py`, the provenance strings in `utils/criteria.py`, `train_bn_v2.py` and `run.py`,
  argparse descriptions of some trainers, an assert message in `model/wgan_llm.py`, literal metrics keys,
  and the study attributes in `studies/*.db` (`search_space_notes`, `objective`), including decision dates
  and reviewer-item codes (such as "R2-1.2") of the review of an earlier paper on SemaPop, for which these
  baselines were first rebuilt. They are code or stored records and were left unchanged so that behavior
  and outputs stay identical. Comments and docstrings were cleaned; the enumerators F2-F8 left in some
  trainer docstrings are the authors' item numbers.
