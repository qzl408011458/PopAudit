# --- release path setup (popaudit_paths equivalents; override via environment, see README) ---
: "${POPAUDIT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${POPAUDIT_PYTHON:=python}"
# --- end release path setup ---
${POPAUDIT_PYTHON} ${POPAUDIT_ROOT}/training/train_ttvae_CA.py CA --cols-config cols_config_v3.yaml --eval-config ${POPAUDIT_ROOT}/configs/eval_v3_pums12.yaml --f1-weight 0.0 --select-on val_srmse --seed 142 --sample-seed 1376 --device cuda:0 --out-root ${POPAUDIT_ROOT}/supplementary_experiments/seed_robustness/final/ttvae_CA --run-name ttvae_CA_final_s142 --n-samples 10000 --sample-method triangular --d-model 64 --nhead 8 --num-layers 1 --dim-feedforward 256 --dropout 0.1 --z-dim 128 --lr 1.28607e-05 --weight-decay 1.09919e-05 --batch-size 256 --epochs 180 --beta-mmd 0.187599 --patience 12 --ensemble --ensemble-n 330109
