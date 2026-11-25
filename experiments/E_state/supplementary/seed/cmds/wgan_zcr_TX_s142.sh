# --- release path setup (popaudit_paths equivalents; override via environment, see README) ---
: "${POPAUDIT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
: "${POPAUDIT_PYTHON:=python}"
: "${POPAUDIT_SUPP:=$POPAUDIT_ROOT/experiments/E_state/supplementary}"
: "${POPAUDIT_DATA:=$POPAUDIT_ROOT/data}"
# --- end release path setup ---
${POPAUDIT_PYTHON} ${POPAUDIT_ROOT}/train_wgan_zcr_v2.py --data-dir ${POPAUDIT_DATA}/pums_states/TX --eval-config ${POPAUDIT_ROOT}/configs/eval_v3_pums12.yaml --f1-weight 0.0 --seed 142 --sample-seed 1376 --device cuda:0 --out-root ${POPAUDIT_SUPP}/seed/final/wgan_zcr_TX --tag wgan_zcr_v2 --run-name wgan_zcr_TX_final_s142 --n-samples 10000 --select-on if1 --cols-config cols_config_v3.yaml --noise-dim 256 --g-hidden 256 256 256 --d-hidden 256 512 256 --dropout-g 0.0645093 --dropout-d 0.2744 --lambda-gp 10 --batch-size 256 --epochs 300 --n-critic 1 --lr-g 1.39869e-05 --lr-d 0.00189328 --gumbel-tau 0.625699 --bd-weight 0.0722506 --ad-weight 0.00329001 --zcr-cat-scale 1 --zcr-every 1 --ensemble --ensemble-n 212756
