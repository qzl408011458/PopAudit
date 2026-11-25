#!/usr/bin/env bash
# Data chain of the paper, from the raw ACS PUMS files to everything the trainers and the audit read.
# Run from anywhere; paths are repository-relative.  Steps marked [GPU] need a CUDA GPU with >= 24 GB
# (Qwen3-8B).  Steps 1-4 were re-checked by the authors (byte-identical outputs, see README).
#   bash data_prep/run_data_chain.sh [RAW_DIR]       # default RAW_DIR = data/raw (see download_pums.sh)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${POPAUDIT_PYTHON:-python}
RAW=${1:-data/raw}
mkdir -p data/pums data/pums18 data/pums21 data/pums24 data/pums_historical

# 1. National compact person tables, unweighted (PWGTP kept); income classes in constant 2013 dollars
#    (ADJINC x CPI-U), JWTRNS -> JWTR crosswalk for ACS 2019+ (chosen from the person-file header).
for y in 2009 2010 2011 2012 2013; do yy=${y:2:2}
  $PY _pums_proc.py --unweighted --survey-year $y \
     --housing-files $RAW/$y/ss${yy}husa.csv $RAW/$y/ss${yy}husb.csv \
     --person-files  $RAW/$y/ss${yy}pusa.csv $RAW/$y/ss${yy}pusb.csv \
     --output data/pums_historical/person_level_merged_compact_pwgtp_$y.csv
done
for y in 2018 2021 2024; do yy=${y:2:2}
  $PY _pums_proc.py --unweighted --survey-year $y \
     --housing-files $RAW/$y/psam_husa.csv $RAW/$y/psam_husb.csv \
     --person-files  $RAW/$y/psam_pusa.csv $RAW/$y/psam_pusb.csv \
     --output data/pums$yy/person_level_merged_compact_pwgtp_$y.csv
done

# 2. 2013 training base: PWGTP-proportional resample with replacement, seed 42, WGTP dropped.
$PY _pums_proc.py --survey-year 2013 --drop-wgtp --seed 42 \
   --housing-files $RAW/2013/ss13husa.csv $RAW/2013/ss13husb.csv \
   --person-files  $RAW/2013/ss13pusa.csv $RAW/2013/ss13pusb.csv \
   --output data/pums/person_level_merged_compact_2013.csv

# 3. State slices: 2013 base, history 2009-2013, references 2018/2021/2024.
$PY data_prep/build_state_slices.py --write

# 4. Column configs, train/val/test split (seed 42; 40,000 / 10,000 / rest) and the reference symlink.
cp data_prep/pums/cols_config.yaml data/pums/cols_config.yaml          # 9-column legacy config, copied by OH/TX
for st in CA OH TX; do
  cp data_prep/pums_states/$st/cols_config_v3.yaml data/pums_states/$st/cols_config_v3.yaml
  $PY data_prep/pums_states/$st/sample_and_split.py
done
# OH/TX write testdata_pers.csv in sample_and_split.py; CA writes it in run_persona_CA.py (step 5).

# 5. [GPU] Persona texts (Qwen/Qwen3-8B; T=0.9, top_p=0.9, <=512 new tokens; NOT seeded, so the texts
#    are not bit-reproducible).  Optional sharding: --num_shards S --shard_id k, then merge_shards.py.
for st in CA OH TX; do
  $PY data_prep/pums_states/$st/run_persona_$st.py --device cuda:0
done
for st in CA OH TX; do
  ( cd data/pums_states/$st && ln -sf testdata_pers.csv referencedata_pers.csv )
done

# 6. [GPU] Persona embeddings: Qwen3-8B hidden states, mean of the last 4 layers, masked mean pooling,
#    LayerNorm; 4096-d -> embPers.pkl (train) and embPers_val.pkl (val).
for st in CA OH TX; do $PY data_prep/pums_states/$st/run_emb_$st.py --device cuda:0; done

# 7. Frozen evaluation protocol (category lists on the union of the three 2013 state slices).
#    The released configs/ already hold the output; re-running reproduces them byte for byte.
$PY utils/freeze_eval_config_v3.py

# 8. Persona prior for the DIAGNOSTIC prior-sampled SemaPop-VAE+ arm (M-U, Supplementary Table S12 only;
#    the audited M-R arm resamples training-record embeddings).  Arguments reconstructed from the stored
#    prior metadata (selected: K = 64, full covariance).
for st in CA OH TX; do
  $PY -m model.persona_prior --emb data/pums_states/$st/embPers.pkl --emb-val data/pums_states/$st/embPers_val.pkl \
     --out results/prior_v3/${st}_gmm --family gmm --n-components 64 --k-grid 32 64 128
done
echo "data chain complete"
