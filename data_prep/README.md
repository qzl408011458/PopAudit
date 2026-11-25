# Data preparation

From the raw ACS PUMS files to everything the trainers and the audit read (Section 5.2, Appendix B).
Outputs go to `data/` below the repository root (`POPAUDIT_DATA`; a symlink is fine).

| step | command | needs |
|---|---|---|
| Download ACS 1-year PUMS 2009-2013, 2018, 2021, 2024 | `bash data_prep/download_pums.sh [RAW_DIR]` | internet, about 19 GB |
| 1. National compact person tables | `bash data_prep/run_data_chain.sh [RAW_DIR]` (step 1) | raw PUMS |
| 2. 2013 training base (PWGTP resample) | step 2 | |
| 3. State slices: 2013 base, history 2009-2013, references 2018/2021/2024 | step 3 (`build_state_slices.py --write`) | |
| 4. Column configs, train/validation/test split | step 4 (`pums_states/<ST>/sample_and_split.py`) | |
| 5. Persona texts (Qwen3-8B), reference symlink | step 5 (`pums_states/<ST>/run_persona_<ST>.py`, `merge_shards.py`) | GPU, 24 GB |
| 6. Persona embeddings | step 6 (`pums_states/<ST>/run_emb_<ST>.py`) | GPU, 24 GB |
| 7. Frozen evaluation protocol | step 7 (`utils/freeze_eval_config_v3.py` -> `configs/eval_v3_pums12*.yaml`) | 2013 slices |
| 8. GMM persona prior (diagnostic M-U arm only) | step 8 (`model/persona_prior.py`, writes `results/prior_v3/<ST>_gmm/`) | embeddings |

Notes:

- `_pums_proc.py` and `_gen_pub_personas_traintest.py` stay at the repository root, where `run_data_chain.sh`,
  the `run_persona_<ST>.py` scripts and `supplementary_experiments/income_deflation` use them.
- `_pums_proc.py` writes income classes in constant 2013 dollars (ADJINC x CPI-U; `--nominal-income`
  restores nominal dollars) and maps the ACS 2019+ JWTRNS codes onto the JWTR categories (chosen from the
  person-file header). The 2013 training base is a PWGTP-proportional resample with replacement
  (`--seed 42 --drop-wgtp`).
- Before release, the authors re-ran steps 1-4 read-only: the 2013 resample, all 27 state slices
  (`build_state_slices.py --check`), the splits (`dataset_ids.pkl`) and the three
  `configs/eval_v3_pums12*.yaml` files all reproduce byte for byte.
- The splits are 40,000 training and 10,000 validation records with seed 42; the rest is the test split
  (330,109 CA, 212,756 TX and 64,112 OH records). `referencedata_pers.csv` is a symlink to
  `testdata_pers.csv`. The OH/TX split scripts write `testdata_pers.csv` themselves; CA writes it in
  `run_persona_CA.py`. The OH/TX split scripts also copy `data/pums/cols_config.yaml`, a 9-column legacy
  config shipped as `data_prep/pums/cols_config.yaml`.
- Persona texts are not bit-reproducible: Qwen3-8B sampling (temperature 0.9, top-p 0.9, at most 512 new
  tokens) is not seeded, and the number of shards used for CA and TX was not recorded. The exact texts and
  embeddings used in the paper are available on request. The GMM persona prior of step 8 is used only by
  the diagnostic M-U row; its command line was reconstructed from the stored prior metadata (K = 64, full
  covariance).
