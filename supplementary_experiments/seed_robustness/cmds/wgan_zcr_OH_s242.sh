# --- release path setup (popaudit_paths equivalents; override via environment, see README) ---
: "${POPAUDIT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${POPAUDIT_PYTHON:=python}"
: "${POPAUDIT_DATA:=$POPAUDIT_ROOT/data}"
# --- end release path setup ---
${POPAUDIT_PYTHON} ${POPAUDIT_ROOT}/training/train_wgan_zcr_v2.py --data-dir ${POPAUDIT_DATA}/pums_states/OH --eval-config ${POPAUDIT_ROOT}/configs/eval_v3_pums12.yaml --f1-weight 0.0 --seed 242 --sample-seed 1476 --device cuda:0 --out-root ${POPAUDIT_ROOT}/supplementary_experiments/seed_robustness/final/wgan_zcr_OH --tag wgan_zcr_v2 --run-name wgan_zcr_OH_final_s242 --n-samples 10000 --select-on if1 --cols-config cols_config_v3.yaml --noise-dim 256 --g-hidden 256 256 256 --d-hidden 256 512 256 --dropout-g 0.0645093 --dropout-d 0.2744 --lambda-gp 10 --batch-size 256 --epochs 300 --n-critic 1 --lr-g 1.39869e-05 --lr-d 0.00189328 --gumbel-tau 0.625699 --bd-weight 0.0722506 --ad-weight 0.00329001 --zcr-cat-scale 1 --zcr-every 1 --ensemble --ensemble-n 64112
