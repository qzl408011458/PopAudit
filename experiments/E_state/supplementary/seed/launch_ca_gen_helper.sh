#!/usr/bin/env bash
# GPU 3 helper (2026-11-24): once TTVAE CA s142/s242 are written, two extra TabDDPM CA shards (one per seed)
# take realization ids in DESCENDING order; tabddpm_gen_shard.py skips ids already written, so the
# main shards (ascending) and the helper meet in the middle.  Same argv/procedure as launch_ca_gen.sh.
# --- release path setup (popaudit_paths equivalents; override via environment, see README) ---
: "${POPAUDIT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
: "${POPAUDIT_PYTHON:=python}"
: "${POPAUDIT_SUPP:=$POPAUDIT_ROOT/experiments/E_state/supplementary}"
# --- end release path setup ---
S=${POPAUDIT_SUPP}/seed
PY=${POPAUDIT_PYTHON}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
cd ${POPAUDIT_ROOT}
until [ $(ls $S/final/ttvae_CA/ttvae_CA_final_s142/generated_ensembles/final_generated_samples_*.csv 2>/dev/null | wc -l) -ge 50 ] && \
      [ $(ls $S/final/ttvae_CA/ttvae_CA_final_s242/generated_ensembles/final_generated_samples_*.csv 2>/dev/null | wc -l) -ge 50 ]; do sleep 30; done
TARGS () { sed -e 's#^.*tabddpm_gen_wrapper.py ##' -e 's# --sample-batch 32768##' -e "s#--out-root [^ ]*#--out-root $2#" $S/cmds/tabddpm_CA_valsrmse_s$1.sh; }
for s in 142 242; do
  sd=$S/final/tabddpm_CA_valsrmse/_shards/sh3/tabddpm_CA_valsrmse_final_s$s; mkdir -p $sd
  ln -sf $S/final/tabddpm_CA_valsrmse/tabddpm_CA_valsrmse_final_s$s/tabddpm_last.pt $sd/tabddpm_last.pt
  CUDA_VISIBLE_DEVICES=3 $PY $S/tabddpm_gen_shard.py --rids $(seq -s, 49 -1 0) \
    --ens-dir $S/final/tabddpm_CA_valsrmse/tabddpm_CA_valsrmse_final_s$s/generated_ensembles -- \
    $(TARGS $s $S/final/tabddpm_CA_valsrmse/_shards/sh3) > $S/logs/g1fix_tabddpm_CA_s${s}_sh3.log 2>&1 &
done
wait; echo "[$(date +%T)] helper done"
