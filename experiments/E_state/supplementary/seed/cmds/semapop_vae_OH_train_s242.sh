# --- release path setup (popaudit_paths equivalents; override via environment, see README) ---
: "${POPAUDIT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
: "${POPAUDIT_PYTHON:=python}"
: "${POPAUDIT_SUPP:=$POPAUDIT_ROOT/experiments/E_state/supplementary}"
: "${POPAUDIT_DATA:=$POPAUDIT_ROOT/data}"
# --- end release path setup ---
${POPAUDIT_PYTHON} ${POPAUDIT_ROOT}/train_semapop_vae_v2.py --data-dir ${POPAUDIT_DATA}/pums_states/OH --eval-config ${POPAUDIT_ROOT}/configs/eval_v3_pums12.yaml --f1-weight 0.0 --seed 242 --sample-seed 1476 --device cuda:0 --out-root ${POPAUDIT_SUPP}/seed/final/semapop_vae_OH_train --tag semapop_vae_v2 --run-name semapop_vae_OH_train_final_s242 --n-samples 10000 --select-on val_srmse --cols-config cols_config_v3.yaml --emb-file embPers.pkl --emb-val-file embPers_val.pkl --cond-source persona --cond-mode train --latent-dim 64 --cond-dim 256 --enc-hidden 1024 --dec-hidden 1024 --enc-depth 2 --dec-depth 2 --adapter-hidden 512 --adapter-dropout 0.0530785 --dropout 0.11988 --lr 0.000845395 --weight-decay 0 --batch-size 512 --epochs 600 --beta 2.03325 --beta-warmup-epochs 0 --lambda-m 0.384298 --grad-clip 1 --ensemble --ensemble-n 64112 --generate-only
