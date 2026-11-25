#!/usr/bin/env bash
# g1fix follow-up (2026-11-24): regenerate TabDDPM TX seeds 142/242 with the train_tabddpm_v2 DEFAULT
# --sample-batch 4096 (= production s42 TX finalize, the OH seed runs and the new CA seed runs).  The old
# ensembles (cmds/tabddpm_TX_valsrmse_s*.sh, --sample-batch 32768) were moved to
# a backup outside this release (same relative path).  Procedure identical to launch_ca_gen.sh:
# generation only from the stored checkpoints (final/.../tabddpm_last.pt), TARGS = the cmds argv with
# ' --sample-batch 32768' removed, tabddpm_gen_shard.py sharded over realization ids, ensemble-n 212756.
# 4 shards per seed (rids j, j+4, ...), seeds 142+242 on each of GPUs 0-3; GPU 3 also runs the
# seed-42 byte-for-byte check (production TX checkpoint, rid 1).
# --- release path setup (popaudit_paths equivalents; override via environment, see README) ---
: "${POPAUDIT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${POPAUDIT_PYTHON:=python}"
: "${POPAUDIT_ENSEMBLES:=${POPAUDIT_HPO:-$POPAUDIT_ROOT/hpo/records}/final}"
# --- end release path setup ---
S=${POPAUDIT_ROOT}/supplementary_experiments/seed_robustness
PY=${POPAUDIT_PYTHON}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
cd ${POPAUDIT_ROOT}
TARGS () {  # $1 seed  $2 out-root
  sed -e 's#^.*tabddpm_gen_wrapper.py ##' -e 's# --sample-batch 32768##' -e "s#--out-root [^ ]*#--out-root $2#" $S/cmds/tabddpm_TX_valsrmse_s$1.sh; }
tab_shard () { local g=$1 s=$2 j=$3; local rids=$(seq -s, $j 4 49)
  local sd=$S/final/tabddpm_TX_valsrmse/_shards/sh$j/tabddpm_TX_valsrmse_final_s$s; mkdir -p $sd
  ln -sf $S/final/tabddpm_TX_valsrmse/tabddpm_TX_valsrmse_final_s$s/tabddpm_last.pt $sd/tabddpm_last.pt
  CUDA_VISIBLE_DEVICES=$g $PY $S/tabddpm_gen_shard.py --rids $rids \
    --ens-dir $S/final/tabddpm_TX_valsrmse/tabddpm_TX_valsrmse_final_s$s/generated_ensembles -- \
    $(TARGS $s $S/final/tabddpm_TX_valsrmse/_shards/sh$j) > $S/logs/g1fix_tabddpm_TX_s${s}_sh$j.log 2>&1; }
TARGS 142 X | grep -q -- '--sample-batch' && { echo "sample-batch still present"; exit 1; }
for j in 0 1 2 3; do tab_shard $j 142 $j & tab_shard $j 242 $j & done
(
  # seed-42 check: same shard path, production TX checkpoint, rid 1 -> compare with the stored production file
  C=$S/test/tabddpm_TX_s42chk; mkdir -p $C/tabddpm_TX_valsrmse_final_s42
  ln -sf ${POPAUDIT_ENSEMBLES}/tabddpm_TX_valsrmse/tabddpm_TX_valsrmse_final_s42/tabddpm_last.pt \
    $C/tabddpm_TX_valsrmse_final_s42/tabddpm_last.pt
  A42=$(TARGS 142 $C | sed -e 's#--seed 142#--seed 42#' -e 's#--sample-seed 1376#--sample-seed 1276#' -e 's#final_s142#final_s42#')
  CUDA_VISIBLE_DEVICES=3 $PY $S/tabddpm_gen_shard.py --rids 1 --ens-dir $C/ens -- $A42 > $S/logs/g1fix_check_tabddpm_TX_s42.log 2>&1
  cmp $C/ens/final_generated_samples_1.csv \
      ${POPAUDIT_ENSEMBLES}/tabddpm_TX_valsrmse/tabddpm_TX_valsrmse_final_s42/generated_ensembles/final_generated_samples_1.csv \
    && echo "S42 TX rid1 BYTE-IDENTICAL" >> $S/logs/g1fix_check_tabddpm_TX_s42.log \
    || echo "S42 TX rid1 DIFFERS" >> $S/logs/g1fix_check_tabddpm_TX_s42.log
) &
wait; echo "[$(date +%T)] TX GEN DONE"
