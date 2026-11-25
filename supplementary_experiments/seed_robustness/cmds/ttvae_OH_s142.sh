# --- release path setup (popaudit_paths equivalents; override via environment, see README) ---
: "${POPAUDIT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${POPAUDIT_PYTHON:=python}"
# --- end release path setup ---
${POPAUDIT_PYTHON} ${POPAUDIT_ROOT}/train_ttvae_CA.py OH --cols-config cols_config_v3.yaml --eval-config ${POPAUDIT_ROOT}/configs/eval_v3_pums12.yaml --f1-weight 0.0 --select-on val_srmse --seed 142 --sample-seed 1376 --device cuda:0 --out-root ${POPAUDIT_ROOT}/supplementary_experiments/seed_robustness/final/ttvae_OH --run-name ttvae_OH_final_s142 --n-samples 10000 --sample-method triangular --d-model 128 --nhead 4 --num-layers 1 --dim-feedforward 256 --dropout 0 --z-dim 192 --lr 0.000159305 --weight-decay 1.33112e-05 --batch-size 512 --epochs 100 --beta-mmd 0.23102 --patience 10 --ensemble --ensemble-n 64112
