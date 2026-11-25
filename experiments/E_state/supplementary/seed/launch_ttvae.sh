#!/usr/bin/env bash
# TTVAE needs cwd = repo root (relative data/ paths); 6 jobs, 2 per GPU
# --- release path setup (popaudit_paths equivalents; override via environment, see README) ---
: "${POPAUDIT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
: "${POPAUDIT_SUPP:=$POPAUDIT_ROOT/experiments/E_state/supplementary}"
# --- end release path setup ---
S=${POPAUDIT_SUPP}/seed
cd ${POPAUDIT_ROOT}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
run () { local g=$1 n=$2; echo "[$(date +%T)] START $n gpu$g"; CUDA_VISIBLE_DEVICES=$g bash $S/cmds/$n.sh > $S/logs/$n.log 2>&1; echo "[$(date +%T)] END $n rc=$?"; }
run 0 ttvae_CA_s142 & run 2 ttvae_CA_s242 & run 1 ttvae_OH_s142 & run 1 ttvae_OH_s242 & run 2 ttvae_TX_s142 & run 0 ttvae_TX_s242 &
wait; echo "[$(date +%T)] ALL TTVAE DONE"
