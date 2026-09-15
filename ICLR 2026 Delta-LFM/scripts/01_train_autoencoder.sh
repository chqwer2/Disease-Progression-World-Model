#!/usr/bin/env bash
set -eu; source "$(dirname "$0")/00_env.sh"
# Stage 1. Brain-patch training; checkpoints land in $WORK_DIR/ae (see RECIPE.md).
# ARCRANK=1 (recommended) adds the ArcRank + contrastive terms; ARCRANK=0 reproduces the released checkpoint.
ARC=()
if [ "${ARCRANK:-1}" = "1" ]; then
  ARC=(--use_contrastive --arc_delta_angle --angle_weight 0.35 --arc_order_w 0.2)
fi
python step1_v2_axes.py \
  --config config/default/Step1_3D_AE_Contastive.yaml \
  --data_dir "$DATA_DIR" --output_dir ae --temp_path "$WORK_DIR" \
  --norm_mode std --res_scale "$RES_SCALE" --crop_mode patch \
  --batch_size 1 --lr 1e-5 --num_workers 4 --warmup_epochs 0 \
  --aug_rigid 3 --aug_rot_deg 5 --adv_warmup 400 --adv_ramp 1200 \
  --ae_chgrec_w 0.3 --ae_dircosE_w 0.3 \
  --save_every 1 --eval_steps 250 --lin_max_batches 8 --eval_every 99 \
  --n_epochs 30 --steps_per_epoch 2000 "${ARC[@]}"
