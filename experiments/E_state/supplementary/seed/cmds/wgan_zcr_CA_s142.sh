# --- release path setup (popaudit_paths equivalents; override via environment, see README) ---
: "${POPAUDIT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
: "${POPAUDIT_PYTHON:=python}"
: "${POPAUDIT_SUPP:=$POPAUDIT_ROOT/experiments/E_state/supplementary}"
: "${POPAUDIT_DATA:=$POPAUDIT_ROOT/data}"
# --- end release path setup ---
${POPAUDIT_PYTHON} ${POPAUDIT_ROOT}/train_wgan_zcr_v2.py --data-dir ${POPAUDIT_DATA}/pums_states/CA --eval-config ${POPAUDIT_ROOT}/configs/eval_v3_pums12.yaml --f1-weight 0.0 --seed 142 --sample-seed 1376 --device cuda:0 --out-root ${POPAUDIT_SUPP}/seed/final/wgan_zcr_CA --tag wgan_zcr_v2 --run-name wgan_zcr_CA_final_s142 --n-samples 10000 --select-on if1 --cols-config cols_config_v3.yaml --noise-dim 128 --g-hidden 256 256 256 --d-hidden 256 512 256 --dropout-g 0.0230188 --dropout-d 0.263592 --lambda-gp 10 --batch-size 256 --epochs 200 --n-critic 2 --lr-g 3.21015e-05 --lr-d 0.0017464 --gumbel-tau 0.633323 --bd-weight 0.0928787 --ad-weight 0.00163616 --zcr-cat-scale 1 --zcr-every 1 --ensemble --ensemble-n 330109
