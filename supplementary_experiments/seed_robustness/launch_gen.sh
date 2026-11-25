#!/usr/bin/env bash
# seed-2/3 K=50 ensemble generation: 6 TabDDPM jobs (2 per GPU) + 6 lanes of SemaPop->TTVAE->WGAN
# --- release path setup (popaudit_paths equivalents; override via environment, see README) ---
: "${POPAUDIT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# --- end release path setup ---
cd ${POPAUDIT_ROOT}/supplementary_experiments/seed_robustness
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
run () { local g=$1 n=$2; echo "[$(date +%T)] START $n gpu$g"; CUDA_VISIBLE_DEVICES=$g bash cmds/$n.sh > logs/$n.log 2>&1; echo "[$(date +%T)] END $n rc=$?"; }
run 0 tabddpm_CA_valsrmse_s142 & run 0 tabddpm_TX_valsrmse_s242 &
run 2 tabddpm_CA_valsrmse_s242 & run 2 tabddpm_TX_valsrmse_s142 &
run 1 tabddpm_OH_valsrmse_s142 & run 1 tabddpm_OH_valsrmse_s242 &
lane () { local g=$1 st=$2 s=$3; run $g semapop_vae_${st}_train_s$s; run $g ttvae_${st}_s$s; run $g wgan_zcr_${st}_s$s; }
lane 0 CA 142 & lane 0 CA 242 & lane 1 OH 142 & lane 1 OH 242 & lane 2 TX 142 & lane 2 TX 242 &
wait; echo "[$(date +%T)] ALL GEN DONE"
