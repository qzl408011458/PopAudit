#!/usr/bin/env bash
# g1fix seed rerun (2026-11-24): Layer 2 + cells7 + Layer 3 from scratch on the new references for the
# existing seed ensembles (Layer 1 reused: 2013 test data and L1 code unchanged).  The old L2/L3/cells7
# outputs were copied to a backup outside this release and are removed here first,
# so no file of the old references can survive (same as run_audit_v3.py --l23-only for production).
# --- release path setup (popaudit_paths equivalents; override via environment, see README) ---
: "${POPAUDIT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${POPAUDIT_PYTHON:=python}"
# --- end release path setup ---
S=${POPAUDIT_ROOT}/supplementary_experiments/seed_robustness
PY=${POPAUDIT_PYTHON}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
cd $S
for s in 42 142 242; do rm -rf audits_s$s/layer2_* audits_s$s/layer3_paperW_* audits_s$s/layer3_rstar_inputs_cells7_v2; done
( $PY audit_seed.py --seed 42 --only semapop_vae_TX,tabddpm_OH --jobs 2 --l1-jobs 1 > logs/g1fix_audit_s42_check.log 2>&1
  $PY audit_seed.py --seed 142 --jobs 2 --l1-jobs 1 > logs/g1fix_audit_s142.log 2>&1 ) &
$PY audit_seed.py --seed 242 --jobs 2 --l1-jobs 1 > logs/g1fix_audit_s242.log 2>&1 &
wait; echo "[$(date +%T)] existing-seed audits done"
