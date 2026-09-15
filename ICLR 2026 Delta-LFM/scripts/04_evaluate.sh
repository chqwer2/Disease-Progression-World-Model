#!/usr/bin/env bash
set -eu; source "$(dirname "$0")/00_env.sh"
# Stage 4, evaluation standard (RECIPE.md): run for three consecutive epochs and report the mean.
# The noise scale is read from the checkpoint's args.json; do not pass --fm_flowinit_sigma.
EP="${1:-7}"; PART="${2:-adni}"; N="${3:-1000000}"   # N larger than the partition = every test pair
# default: one deterministic pass. STOCHASTIC=1 averages two antithetic samples of the flow variant.
SAMP=(--fm_det 1 --n_avg 1 --fm_antithetic 0)
[ "${STOCHASTIC:-0}" = "1" ] && SAMP=(--n_avg 2 --fm_antithetic 1)
python eval_fm.py \
  --config config/default/Step3_3D_FM_multidt15.yaml \
  --data_dir "$DATA_DIR" --latent_path "$WORK_DIR/latents" --temp_path "$WORK_DIR" \
  --aekl_ckpt "${AE_CKPT:?export AE_CKPT=\$WORK_DIR/ae/all-ae-<EP>-3D.pth}" \
  --res_scale "$RES_SCALE" --norm_mode std --cache_dir none \
  --fm_scheme std --fm_scale_norm 0 --fm_res_noise 1 \
  --fm_mask_lambda 5 --fm_mask_mode soft --fm_mask_src latent \
  --fm_hist_mode "prev1+prev2" --split_v3 1 --test_part "$PART" \
  --fm_ckpt "$WORK_DIR/fm_rectified/fm-unet-ep-${EP}.pth" \
  "${SAMP[@]}" --fm_seed 1234 --fm_sigma 0 --batch_size 1 --num_workers 1 \
  --n_eval "$N" --raw_out 1 --dump_pred "$WORK_DIR/pred_ep${EP}_${PART}" --tag "ep${EP}_${PART}"
