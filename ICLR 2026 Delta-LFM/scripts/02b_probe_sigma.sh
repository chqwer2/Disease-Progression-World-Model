#!/usr/bin/env bash
set -eu; source "$(dirname "$0")/00_env.sh"
# Stage 2b. Measure --fm_flowinit_sigma on the training pairs of your latents; export it as SIGMA for stage 3.
python probe_flowinit_sigma.py \
  --config config/default/Step3_3D_FM_multidt15.yaml \
  --data_dir "$DATA_DIR" --latent_path "$WORK_DIR/latents" --temp_path "$WORK_DIR" \
  --res_scale "$RES_SCALE" --norm_mode std --cache_dir none \
  --fm_hist_mode "prev1+prev2" --split_v3 1
