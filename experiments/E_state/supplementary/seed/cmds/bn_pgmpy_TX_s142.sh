# --- release path setup (popaudit_paths equivalents; override via environment, see README) ---
: "${POPAUDIT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
: "${POPAUDIT_PYTHON:=python}"
: "${POPAUDIT_SUPP:=$POPAUDIT_ROOT/experiments/E_state/supplementary}"
: "${POPAUDIT_DATA:=$POPAUDIT_ROOT/data}"
# --- end release path setup ---
${POPAUDIT_PYTHON} ${POPAUDIT_ROOT}/train_bn_pgmpy.py --data-dir ${POPAUDIT_DATA}/pums_states/TX --eval-config ${POPAUDIT_ROOT}/configs/eval_v3_pums12.yaml --seed 142 --sample-seed 1376 --out-root ${POPAUDIT_SUPP}/seed/final/bn_pgmpy_TX --tag bn_pgmpy --run-name bn_pgmpy_TX_final_s142 --n-samples 10000 --cols-config cols_config_v3.yaml --score bdeu --max-indegree 4 --ess 66.6383 --cpt mle --decode empirical --ensemble --ensemble-n 212756
