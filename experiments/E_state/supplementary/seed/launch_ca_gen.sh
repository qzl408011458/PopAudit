#!/usr/bin/env bash
# g1fix seed rerun (2026-11-24): CA TabDDPM + TTVAE at seeds 142/242, GENERATION ONLY from the stored
# hpo_v3 checkpoints (no retraining), production sampler settings:
#   TabDDPM: train_tabddpm_v2 --load-final --ensemble --generate-only, DEFAULT --sample-batch 4096
#            (= production s42 CA finalize and the OH seed runs; the old cmds/tabddpm_CA_*.sh used 32768),
#            sharded over realization ids (tabddpm_gen_shard.py), 3 shards per seed on GPUs 0-2.
#   TTVAE:   ttvae_gen_only.py (shard mode = loop body of sample_ttvae_ensemble), batch 2048, triangular.
# GPU 3: seed-42 byte-for-byte checks (TabDDPM CA rid 1, TTVAE CA rid 1), then TTVAE CA s142 and s242.
# --- release path setup (popaudit_paths equivalents; override via environment, see README) ---
: "${POPAUDIT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
: "${POPAUDIT_PYTHON:=python}"
: "${POPAUDIT_SUPP:=$POPAUDIT_ROOT/experiments/E_state/supplementary}"
# --- end release path setup ---
S=${POPAUDIT_SUPP}/seed
PY=${POPAUDIT_PYTHON}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
cd ${POPAUDIT_ROOT}
TARGS () {  # $1 seed  $2 out-root
  sed -e 's#^.*tabddpm_gen_wrapper.py ##' -e 's# --sample-batch 32768##' -e "s#--out-root [^ ]*#--out-root $2#" $S/cmds/tabddpm_CA_valsrmse_s$1.sh; }
tab_shard () { local g=$1 s=$2 j=$3; local rids=$(seq -s, $j 3 49)
  CUDA_VISIBLE_DEVICES=$g $PY $S/tabddpm_gen_shard.py --rids $rids \
    --ens-dir $S/final/tabddpm_CA_valsrmse/tabddpm_CA_valsrmse_final_s$s/generated_ensembles -- \
    $(TARGS $s $S/final/tabddpm_CA_valsrmse/_shards/sh$j) > $S/logs/g1fix_tabddpm_CA_s${s}_sh$j.log 2>&1; }
for j in 0 1 2; do tab_shard $j 142 $j & tab_shard $j 242 $j & done
(
  # seed-42 checks: same shard path, production checkpoint, rid 1 (seed 142) -> compare with the stored file
  A42=$(TARGS 142 $S/test/tabddpm_CA_s42chk | sed -e 's#--seed 142#--seed 42#' -e 's#--sample-seed 1376#--sample-seed 1276#' -e 's#final_s142#final_s42#')
  CUDA_VISIBLE_DEVICES=3 $PY $S/tabddpm_gen_shard.py --rids 1 --ens-dir $S/test/tabddpm_CA_s42chk/ens -- $A42 > $S/logs/g1fix_check_tabddpm_CA_s42.log 2>&1
  CUDA_VISIBLE_DEVICES=3 $PY $S/ttvae_gen_only.py CA 42 $S/test/ttvae_CA_s42chk --rids 1 > $S/logs/g1fix_check_ttvae_CA_s42.log 2>&1
) & CHK=$!
CUDA_VISIBLE_DEVICES=3 $PY $S/ttvae_gen_only.py CA 142 $S/final --rids $(seq -s, 0 49) > $S/logs/g1fix_ttvae_CA_s142.log 2>&1 &
wait $CHK
CUDA_VISIBLE_DEVICES=3 $PY $S/ttvae_gen_only.py CA 242 $S/final --rids $(seq -s, 0 49) > $S/logs/g1fix_ttvae_CA_s242.log 2>&1 &
wait; echo "[$(date +%T)] CA GEN DONE"
