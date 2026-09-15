#!/bin/bash
# Stage 4 — Mamba ControlNet (progression). No label leakage: conditions on baseline
# covariates + delta_t only. Needs the PAIRS csv + AE + diffusion checkpoints.
#SBATCH --job-name=mc_cnet
#SBATCH --partition=gpu          # EDIT: your cluster partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mc_cnet_%j.log
set -e
# ---- config (EDIT) ----
DATA=${DATA:-./data}
PAIRS_CSV=${PAIRS_CSV:-$DATA/pairs.csv}          # PAIRS csv (starting_/followup_)
AE_CKPT=${AE_CKPT:-./runs/ae/autoencoder-ep-8.pth}
DIFF_CKPT=${DIFF_CKPT:-./runs/diffusion/unet-ep-99.pth}
OUT=${OUT:-./runs/controlnet}
# -----------------------
cd "$(dirname "$0")/.."
source "${CONDA_BASE:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-progression}"
export ACCELERATE_MIXED_PRECISION=bf16
python -u scripts/training/train_control_mamba.py \
  --dataset_csv "$PAIRS_CSV" --cache_dir "$DATA/cache_cnet" --output_dir "$OUT" \
  --aekl_ckpt "$AE_CKPT" --diff_ckpt "$DIFF_CKPT" \
  --n_epochs 100 --batch_size 2 --lr 2.5e-5 --num_workers 8 --gpus 0
# NOTE: controlnet corr converges by ~ep20; more epochs do not help (data ceiling).
