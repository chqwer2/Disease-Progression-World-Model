#!/bin/bash
# Stage 2 — encode every scan to a latent (*_new_latent.npz, key 'data', (4,D,H,W)).
#SBATCH --job-name=mc_extract
#SBATCH --partition=gpu          # EDIT: your cluster partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/mc_extract_%j.log
set -e
# ---- config (EDIT) ----
DATA=${DATA:-./data}
DATA_CSV=${DATA_CSV:-$DATA/scans.csv}           # single-visit CSV (all scans)
AE_CKPT=${AE_CKPT:-./runs/ae/autoencoder-ep-8.pth} # from stage 1
# -----------------------
cd "$(dirname "$0")/.."
source "${CONDA_BASE:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-progression}"
export ACCELERATE_MIXED_PRECISION=bf16
python -u scripts/prepare/extract_latents.py \
  --dataset_csv "$DATA_CSV" --aekl_ckpt "$AE_CKPT" --gpus 0
