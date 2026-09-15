#!/bin/bash
# Stage 1 — train the MAISI autoencoder (single-image). ~res 1.5mm, latent (4,32,36,32).
#SBATCH --job-name=mc_ae
#SBATCH --partition=gpu          # EDIT: your cluster partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/mc_ae_%j.log
set -e
# ---- config (EDIT) ----
DATA=${DATA:-./data}
DATA_CSV=${DATA_CSV:-$DATA/scans.csv}           # single-visit CSV
OUT=${OUT:-./runs/ae}
PRETRAINED=                                            # optional MAISI init .pth
# -----------------------
cd "$(dirname "$0")/.."
source "${CONDA_BASE:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-progression}"
export ACCELERATE_MIXED_PRECISION=bf16
python -u scripts/training/train_autoencoder.py \
  --dataset_csv "$DATA_CSV" --cache_dir "$DATA/cache_ae" --output_dir "$OUT" \
  ${PRETRAINED:+--aekl_ckpt "$PRETRAINED"} \
  --n_epochs 50 --batch_size 2 --lr 1e-4 --num_workers 8 --gpus 0
