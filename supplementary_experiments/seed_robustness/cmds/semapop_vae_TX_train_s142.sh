# --- release path setup (popaudit_paths equivalents; override via environment, see README) ---
: "${POPAUDIT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${POPAUDIT_PYTHON:=python}"
: "${POPAUDIT_DATA:=$POPAUDIT_ROOT/data}"
# --- end release path setup ---
${POPAUDIT_PYTHON} ${POPAUDIT_ROOT}/train_semapop_vae_v2.py --data-dir ${POPAUDIT_DATA}/pums_states/TX --eval-config ${POPAUDIT_ROOT}/configs/eval_v3_pums12.yaml --f1-weight 0.0 --seed 142 --sample-seed 1376 --device cuda:0 --out-root ${POPAUDIT_ROOT}/supplementary_experiments/seed_robustness/final/semapop_vae_TX_train --tag semapop_vae_v2 --run-name semapop_vae_TX_train_final_s142 --n-samples 10000 --select-on val_srmse --cols-config cols_config_v3.yaml --emb-file embPers.pkl --emb-val-file embPers_val.pkl --cond-source persona --cond-mode train --latent-dim 128 --cond-dim 256 --enc-hidden 512 --dec-hidden 512 --enc-depth 2 --dec-depth 2 --adapter-hidden 2048 --adapter-dropout 0.15936 --dropout 0.156597 --lr 0.000788632 --weight-decay 0 --batch-size 512 --epochs 650 --beta 2.4209 --beta-warmup-epochs 0 --lambda-m 0.884927 --grad-clip 1 --ensemble --ensemble-n 212756 --generate-only
