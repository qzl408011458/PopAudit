#!/usr/bin/env bash
# TTVAE OH/TX seeds 142/242: generation-only from stored checkpoints, sharded over realizations
# --- release path setup (popaudit_paths equivalents; override via environment, see README) ---
: "${POPAUDIT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${POPAUDIT_PYTHON:=python}"
# --- end release path setup ---
S=${POPAUDIT_ROOT}/supplementary_experiments/seed_robustness
cd ${POPAUDIT_ROOT}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
PY=${POPAUDIT_PYTHON}
job () { local g=$1 st=$2 seed=$3 nsh=$4
  for ((j=0;j<nsh;j++)); do rids=$(seq -s, $j $nsh 49)
    CUDA_VISIBLE_DEVICES=$g $PY $S/ttvae_gen_only.py $st $seed $S/final --rids $rids > $S/logs/ttvae_${st}_s${seed}_shard$j.log 2>&1 &
  done; }
job 1 OH 142 5; job 1 OH 242 5; job 0 TX 142 10; job 2 TX 242 10
# validation: rid 1 of the seed-42 model must equal the stored seed-42 file byte for byte
CUDA_VISIBLE_DEVICES=1 $PY $S/ttvae_gen_only.py OH 42 $S/test/ttshard --rids 1 > $S/test/ttshard.log 2>&1 &
wait; echo "[$(date +%T)] TTVAE shards done"
