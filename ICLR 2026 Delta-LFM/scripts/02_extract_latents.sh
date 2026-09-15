#!/usr/bin/env bash
set -eu; source "$(dirname "$0")/00_env.sh"
# Stage 2. Whole-brain latents on the 112^3 grid. AE_CKPT selects the autoencoder checkpoint.
python step2_extract_latents.py \
  --config config/default/Step2_3D_Extract.yaml \
  --data_dir "$DATA_DIR" --latent_path "$WORK_DIR/latents" \
  --output_dir "$WORK_DIR/latents" --temp_path "$WORK_DIR" \
  --aekl_ckpt "${AE_CKPT:?export AE_CKPT=\$WORK_DIR/ae/all-ae-<EP>-3D.pth}" \
  --res_scale "$RES_SCALE" --norm_mode std
