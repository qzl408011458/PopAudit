#!/usr/bin/env bash
# Hyper-parameter search and training of the nine candidate priors and the diagnostic rows (Section 5.3,
# Appendix B, Supplementary S4.1).  A cleaned summary of the authors' launch recipes; GPU ids, lane packing
# and log names are free.  hpo/common.py reads the state datasets from $POPAUDIT_DATA/pums_states and keeps
# studies, trials and finalize runs under $POPAUDIT_HPO (default results/hpo_v3, where the authors' records
# are); point POPAUDIT_HPO elsewhere before a new search or finalize run to keep those records intact.  The exact trainer argv of every finalize run (3 seeds per prior and
# state, plus --ensemble --ensemble-n |reference| on seed 42) is in results/hpo_v3/finalize_commands.tsv.
#   bash hpo/run_all.sh {search|finalize|diagnostics|budget}
set -u
cd "$(dirname "$0")/.."
PY=${POPAUDIT_PYTHON:-python}
G=${GPU:-0}
MODE=${1:-search}
case "$MODE" in
search)   # Optuna TPE (seed 42), median pruner; budgets allocated per search-space dimension (CA 2.5, TX/OH 1.5)
  for st in CA TX OH; do
    $PY -m hpo.run --model semapop_vae --state $st --gpu $G --cond-mode train
    for m in ttvae tabddpm wgan_zcr ctgan tvae_ref borysov; do $PY -m hpo.run --model $m --state $st --gpu $G; done
  done ;;
finalize) # 3-seed retrain + reference evaluation + K=50 ensemble on seed 42 (the audited studies)
  for st in CA TX OH; do
    $PY -m hpo.finalize --model semapop_vae --state $st --gpu $G --cond-mode train
    for m in ttvae wgan_zcr ctgan tvae_ref borysov flowsyn; do $PY -m hpo.finalize --model $m --state $st --gpu $G; done
    $PY -m hpo.finalize --model tabddpm --state $st --gpu $G --reselect-criterion val_srmse --out-tag valsrmse
    $PY -m hpo.finalize --model bn_pgmpy --state $st                  # no search (BDeu, ess 66.6383, <=4 parents, MLE)
  done ;;
diagnostics) # rows outside the candidate set (App. B, Supplementary S4.1 and Table S12)
  for st in CA TX OH; do
    $PY -m hpo.finalize --model tabddpm --state $st --gpu $G                 # authors' ML-efficacy criterion
    for m in ctgan_defaults tvae_ref_defaults borysov_defaults; do $PY -m hpo.finalize --model $m --state $st --gpu $G; done
    $PY -m hpo.finalize --model bn_pgmpy_bic --state $st
    $PY -m hpo.finalize --model semapop_vae --state $st --gpu $G --cond-mode train --sampling-arm mu --out-tag mu --seeds 42
    $PY -m hpo.finalize --model ttvae --state $st --gpu $G --cond-mode mu --sampling-arm prior --out-tag prior --seeds 42
  done ;;
budget)   # Supplementary Table S10 inputs (budget_table.csv) from the study databases
  $PY -m hpo.finalize --budget-table-only ;;
*) echo "usage: $0 {search|finalize|diagnostics|budget}"; exit 2 ;;
esac
