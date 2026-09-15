#!/bin/bash
# Inference — SDEdit + conservative blend (t_start=200, steps=100, alpha=0.10).
# Runs predict_followup.py on the first few held-out pairs (prints pred vs copy PSNR/SSIM).
#SBATCH --job-name=mc_predict
#SBATCH --partition=gpu          # EDIT: your cluster partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=logs/mc_predict_%j.log
set -e
cd "$(dirname "$0")/.."
source "${CONDA_BASE:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-progression}"
export ACCELERATE_MIXED_PRECISION=bf16
# eval a few held-out pairs (checkpoints/CSV are set at the top of predict_followup.py)
for i in 0 1 2 3 4; do python -u predict_followup.py --row $i; done
# single deployment call: predict a followup from a baseline latent + target interval:
#   python predict_followup.py --start_latent BASELINE_new_latent.npz --delta_t_years 2.0 --out pred.npy
