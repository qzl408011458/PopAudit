#!/usr/bin/env bash
# g1fix re-run (2026-11-24): all 10 VALID 2024 cases of the new certificate, one common grid
# (= the old fine grid TF, a superset of the old coarse grid TR), so the selected prior can be tracked.
# --- release path setup (popaudit_paths equivalents; override via environment, see README) ---
: "${POPAUDIT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${POPAUDIT_PYTHON:=python}"
# --- end release path setup ---
cd ${POPAUDIT_ROOT}/supplementary_experiments/tau_ladder
P=${POPAUDIT_PYTHON}
TF="1.0 0.9 0.8 0.75 0.7 0.6 0.55 0.5 0.45 0.4 0.375 0.35 0.325 0.3 0.25 0.2 0.15 0.1 0.05 0.02 0.01 0.005 0.001"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for c in "semapop_vae CA" "ttvae CA" "bn CA" "tabddpm CA" "semapop_vae OH" "ttvae OH" "bn OH" "semapop_vae TX" "ttvae TX" "bn TX"; do
  set -- $c
  $P tau_ladder.py --model $1 --state $2 --taus $TF > job_$1_$2.log 2>&1 &
done
wait
echo ALLDONE > done.flag
