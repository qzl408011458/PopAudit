#!/usr/bin/env bash
# g1fix follow-up (2026-11-24): full Layer 1/2/3 audit of the regenerated TabDDPM TX seed ensembles (s142, s242;
# default --sample-batch 4096, launch_tx_gen.sh), with audit_seed.py exactly as g1fix_audit_ca.sh did for CA.
# The old (32768) audit dirs were copied to a backup outside this release and removed first.
# Loops until every cell (1 model x 2 seeds x 3 horizons) has a Layer-3 summary.
# --- release path setup (popaudit_paths equivalents; override via environment, see README) ---
: "${POPAUDIT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
: "${POPAUDIT_PYTHON:=python}"
: "${POPAUDIT_SUPP:=$POPAUDIT_ROOT/experiments/E_state/supplementary}"
# --- end release path setup ---
S=${POPAUDIT_SUPP}/seed
PY=${POPAUDIT_PYTHON}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
cd $S
while true; do
  n=0; for s in 142 242; do for y in 2018 2021 2024; do
    [ -f audits_s$s/layer3_paperW_tabddpm_TX_$y/layer3_summary.json ] && n=$((n+1)); done; done
  [ $n -ge 6 ] && break
  for s in 142 242; do $PY audit_seed.py --seed $s --only tabddpm_TX --jobs 3 --l1-jobs 1 >> logs/g1fix_audit_tx_s$s.log 2>&1 & done; wait
  sleep 60
done; echo "[$(date +%T)] TX audits done"
