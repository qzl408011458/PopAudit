# --- release path setup (popaudit_paths equivalents; override via environment, see README) ---
: "${POPAUDIT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${POPAUDIT_PYTHON:=python}"
: "${POPAUDIT_DATA:=$POPAUDIT_ROOT/data}"
# --- end release path setup ---
${POPAUDIT_PYTHON} ${POPAUDIT_ROOT}/train_semapop_vae_v2.py --data-dir ${POPAUDIT_DATA}/pums_states/CA --eval-config ${POPAUDIT_ROOT}/configs/eval_v3_pums12.yaml --f1-weight 0.0 --seed 142 --sample-seed 1376 --device cuda:0 --out-root ${POPAUDIT_ROOT}/supplementary_experiments/seed_robustness/final/semapop_vae_CA_train --tag semapop_vae_v2 --run-name semapop_vae_CA_train_final_s142 --n-samples 10000 --select-on val_srmse --cols-config cols_config_v3.yaml --emb-file embPers.pkl --emb-val-file embPers_val.pkl --cond-source persona --cond-mode train --latent-dim 256 --cond-dim 256 --enc-hidden 512 --dec-hidden 512 --enc-depth 2 --dec-depth 2 --adapter-hidden 1024 --adapter-dropout 0.0221437 --dropout 0.178772 --lr 0.000722584 --weight-decay 0 --batch-size 512 --epochs 450 --beta 0.380064 --beta-warmup-epochs 0 --lambda-m 0.695287 --grad-clip 1 --ensemble --ensemble-n 330109 --generate-only
