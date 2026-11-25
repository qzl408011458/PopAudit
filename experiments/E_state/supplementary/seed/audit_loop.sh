#!/usr/bin/env bash
# re-run the shadow audit for each seed until all 15 cells x 3 years are done
# --- release path setup (popaudit_paths equivalents; override via environment, see README) ---
: "${POPAUDIT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
: "${POPAUDIT_PYTHON:=python}"
: "${POPAUDIT_SUPP:=$POPAUDIT_ROOT/experiments/E_state/supplementary}"
# --- end release path setup ---
cd ${POPAUDIT_SUPP}/seed
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=${POPAUDIT_PYTHON}
seedloop () { local s=$1
  while true; do
    n=$(ls -d audits_s$s/layer3_paperW_*/ 2>/dev/null | while read d; do [ -f $d/layer3_summary.json ] && echo x; done | wc -l)
    l1=$(ls audits_s$s/layer1_paper_*/layer1_ensemble_summary.json 2>/dev/null | wc -l)
    [ "$n" -ge 45 ] && [ "$l1" -ge 15 ] && break
    $PY audit_seed.py --seed $s --jobs 6 --l1-jobs 3 >> logs/audit_s${s}.log 2>&1
    sleep 120
  done; echo "[$(date +%T)] seed $s audits complete"; }
seedloop 142 & seedloop 242 & wait
