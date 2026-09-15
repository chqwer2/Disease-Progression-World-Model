#!/bin/bash
# Stage 3 — Mamba latent diffusion (epsilon-prediction) on the extracted latents.
#SBATCH --job-name=mc_diff
#SBATCH --partition=gpu          # EDIT: your cluster partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/mc_diff_%j.log
set -e
# ---- config (EDIT) ----
DATA=${DATA:-./data}
DATA_CSV=${DATA_CSV:-$DATA/scans.csv}           # single-visit CSV
AE_CKPT=${AE_CKPT:-./runs/ae/autoencoder-ep-8.pth}
OUT=${OUT:-./runs/diffusion}
# -----------------------
cd "$(dirname "$0")/.."
source "${CONDA_BASE:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-progression}"
export ACCELERATE_MIXED_PRECISION=bf16
python -u scripts/training/train_diffusion_mamba.py \
  --dataset_csv "$DATA_CSV" --cache_dir "$DATA/cache_diff" --output_dir "$OUT" \
  --aekl_ckpt "$AE_CKPT" \
  --n_epochs 100 --batch_size 8 --lr 2.5e-5 --num_workers 8 --gpus 0
