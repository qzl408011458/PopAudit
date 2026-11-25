#!/usr/bin/env bash
# loosening check (tau_a > 1) for the 10 VALID 2024 cases, g1fix 2026-11-24; outputs runs_loose/
# --- release path setup (popaudit_paths equivalents; override via environment, see README) ---
: "${POPAUDIT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${POPAUDIT_PYTHON:=python}"
# --- end release path setup ---
cd ${POPAUDIT_ROOT}/supplementary_experiments/tau_ladder
P=${POPAUDIT_PYTHON}
TL="1.5 2.0 3.0 5.0 10.0"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for c in "semapop_vae CA" "ttvae CA" "bn CA" "tabddpm CA" "semapop_vae OH" "ttvae OH" "bn OH" "semapop_vae TX" "ttvae TX" "bn TX"; do
  set -- $c
  $P tau_loose.py --model $1 --state $2 --taus $TL > jobloose_$1_$2.log 2>&1 &
done
wait
echo ALLDONE > done_loose.flag
