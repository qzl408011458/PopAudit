# --- release path setup (popaudit_paths equivalents; override via environment, see README) ---
: "${POPAUDIT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
: "${POPAUDIT_PYTHON:=python}"
: "${POPAUDIT_SUPP:=$POPAUDIT_ROOT/experiments/E_state/supplementary}"
: "${POPAUDIT_DATA:=$POPAUDIT_ROOT/data}"
# --- end release path setup ---
${POPAUDIT_PYTHON} ${POPAUDIT_SUPP}/seed/tabddpm_gen_wrapper.py --data-dir ${POPAUDIT_DATA}/pums_states/CA --eval-config ${POPAUDIT_ROOT}/configs/eval_v3_pums12.yaml --f1-weight 0.0 --seed 242 --sample-seed 1476 --device cuda:0 --out-root ${POPAUDIT_SUPP}/seed/final/tabddpm_CA_valsrmse --tag tabddpm_v2 --run-name tabddpm_CA_valsrmse_final_s242 --n-samples 10000 --select-on ml_eff --cols-config cols_config_v3.yaml --hidden 1024 1024 1024 1024 --dim-t 128 --dropout 0.193431 --weight-decay 1.5726e-05 --batch-size 1024 --epochs 550 --lr 0.000204166 --grad-clip 1.22237 --timesteps 1000 --schedule cosine --parametrization x0 --gaussian-parametrization eps --gaussian-loss mse --time-sampler uniform --lambda-multi 0.859597 --lambda-gauss 0.300149 --num-transform standard --load-final --ensemble --ensemble-n 330109 --generate-only --sample-batch 32768
