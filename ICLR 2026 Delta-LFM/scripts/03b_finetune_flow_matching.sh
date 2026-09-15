#!/usr/bin/env bash
set -eu; source "$(dirname "$0")/00_env.sh"
# Stage 6b. Fine-tune the base model for 10 epochs with the change-aware terms:
# energy mask and soft cF1. FM_BASE_CKPT is a checkpoint from 03_train_flow_matching.sh.
: "${SIGMA:?run scripts/02b_probe_sigma.sh and export SIGMA=<printed value>}"
# default: regress the change directly. STOCHASTIC=1 trains the flow-from-noise variant instead.
X0=(--fm_x0_pred 1); [ "${STOCHASTIC:-0}" = "1" ] && X0=(--fm_x0_pred 0)
python step3_train_flowmatching.py "${X0[@]}" \
  --config config/default/Step3_3D_FM_multidt15.yaml \
  --data_dir "$DATA_DIR" --latent_path "$WORK_DIR/latents" --temp_path "$WORK_DIR" \
  --aekl_ckpt "${AE_CKPT:?export AE_CKPT=\$WORK_DIR/ae/all-ae-<EP>-3D.pth}" --output_dir fm \
  --fm_init_ckpt "${FM_BASE_CKPT:?export FM_BASE_CKPT=\$WORK_DIR/fm_base_rectified/fm-unet-ep-<EP>.pth}" \
  --res_scale "$RES_SCALE" --norm_mode std --cache_dir none \
  --fm_scheme std --fm_scale_norm 0 --fm_res_noise 1 \
  --fm_mask_lambda 5 --fm_mask_mode soft --fm_mask_src latent \
  --fm_hist_mode "prev1+prev2" --split_v3 1 \
  --fm_rf1_w 0.0 --fm_cf1_w 0.5 --fm_dir_w 0.0 \
  --fm_flowinit_sigma "$SIGMA" \
  --batch_size 8 --lr 2.5e-5 --n_epochs 10
