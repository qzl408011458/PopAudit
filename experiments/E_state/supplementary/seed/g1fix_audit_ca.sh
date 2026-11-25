#!/usr/bin/env bash
# g1fix seed rerun (2026-11-24): full Layer 1/2/3 audit of the newly generated CA TabDDPM + TTVAE seed
# ensembles (s142, s242), with audit_seed.py exactly as for the other cells; loops until every cell
# (2 models x 2 seeds x 3 horizons) has a Layer-3 summary.  audit_seed.py only takes cells whose 50 files
# exist and are older than 90 s.
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
  n=0; for s in 142 242; do for m in ttvae tabddpm; do for y in 2018 2021 2024; do
    [ -f audits_s$s/layer3_paperW_${m}_CA_$y/layer3_summary.json ] && n=$((n+1)); done; done; done
  [ $n -ge 12 ] && break
  for s in 142 242; do $PY audit_seed.py --seed $s --only ttvae_CA,tabddpm_CA --jobs 1 --l1-jobs 1 >> logs/g1fix_audit_ca_s$s.log 2>&1 & done; wait
  sleep 60
done; echo "[$(date +%T)] CA audits done"
