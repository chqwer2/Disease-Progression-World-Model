#!/bin/bash
# Evaluation — operating-point sweep (vs copy baseline) + official change metrics.
#SBATCH --job-name=mc_eval
#SBATCH --partition=gpu          # EDIT: your cluster partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/mc_eval_%j.log
set -e
# ---- config (EDIT paths inside the .py headers if different) ----
CNET_CKPT=${CNET_CKPT:-./runs/controlnet/cnet-ep-99.pth}
# -----------------------
cd "$(dirname "$0")/.."
source "${CONDA_BASE:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-progression}"
export ACCELERATE_MIXED_PRECISION=bf16
echo "########## operating-point sweep (t_start x steps x alpha vs copy) ##########"
T_STARTS=150,200,250 STEPS_LIST=25,50,100 ALPHAS=0.05,0.10,0.15 N=20 \
  python -u eval_final.py "$CNET_CKPT"
echo "########## official change metrics (detrended, direction-aware) ##########"
python -u eval_changemetric.py
