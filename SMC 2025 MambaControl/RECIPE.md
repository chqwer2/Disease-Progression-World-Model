# The verified recipe

Exact commands to reproduce the released model, in order. The narrative version with
context is in [README.md](README.md); this file is the command sequence.

Results are reported on the **ADNI-3 longitudinal cohort** — subjects with ≥ 2 imaging
sessions and complete demographics, split 7:1:2 **per subject** — trained on
2× NVIDIA V100 32 GB.

> **Scope.** This recipe reproduces the Mamba latent-diffusion + Mamba ControlNet
> backbone (28.37 dB PSNR / 92.01 SSIM). The Fourier anatomy-graph module that produces
> the paper's bolded row is **not part of this release**, so the 29.72 dB figure cannot
> be reproduced from this repository.

## At a glance

| Stage | Script | Produces | Input CSV |
|---|---|---|---|
| 0 | [`../AD_data_processing/`](../AD_data_processing/) + `scripts/prepare/prepare_csv.py` | per-visit (A) and pair (B) tables | — |
| 1 | `scripts/training/train_autoencoder.py` | MAISI autoencoder | A |
| 2 | `scripts/prepare/extract_latents.py` | `<scan>_new_latent.npz` per scan | A |
| 3 | `scripts/training/train_diffusion_mamba.py` | Mamba latent diffusion UNet | A |
| 4 | `scripts/training/train_control_mamba.py` | Mamba ControlNet | **B** |
| 5 | `predict_followup.py` / `eval_final.py` | predictions, scores | B |

Each stage has a SLURM launcher in `run/`. Edit the `# ---- config (EDIT) ----` block and
`sbatch` it, or run the underlying command below directly.

## Prerequisites

```bash
conda create -n mambacontrol python=3.10 -y && conda activate mambacontrol
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install causal-conv1d>=1.4.0 --no-build-isolation
pip install mamba-ssm>=2.2.2     --no-build-isolation

export AD_PROGRESSION_ROOT=/path/to/dataset   # prepended to relative image paths
export ACCELERATE_MIXED_PRECISION=bf16
```

Requires Linux, an NVIDIA GPU and a CUDA toolchain matching your PyTorch build — the
Mamba kernels are CUDA-only. Run every command from this directory so `mambacontrol/` is
importable. Sanity check:

```bash
python -c "import torch, monai, mamba_ssm; print(torch.__version__, torch.cuda.is_available())"
python -c "from mambacontrol import const; print(const.INPUT_SHAPE_AE)"   # (120, 144, 120)
```

---

## Stage 0 — cohort tables

Images must be `.nii.gz`, skull-stripped, 1.5 mm isotropic, rigidly registered to each
subject's baseline, cropped/padded to `(120, 144, 120)` and scaled to `[0, 1]`.

Cohort construction shared with Δ-LFM lives in
[`../AD_data_processing/`](../AD_data_processing/). From the resulting manifest, build the
two derived tables this project consumes:

```bash
python scripts/prepare/prepare_csv.py \
    --dataset_csv    /path/to/manifest.csv \
    --output_path    $WORK_DIR/csv \
    --coarse_regions /path/to/coarse_regions.csv
```

**A** = one row per visit with normalised regional volumes. **B** = baseline/follow-up
pairs with `starting_*` / `followup_*` columns. Δt is derived per pair as
`(followup_follow_up − starting_follow_up) / 12` years, and the 8-dim conditioning vector is

```
[followup_age, sex, starting_diagnosis, starting_cerebral_cortex,
 starting_hippocampus, starting_amygdala, starting_cerebral_white_matter,
 starting_lateral_ventricle]
```

## Stage 1 — autoencoder

```bash
python scripts/training/train_autoencoder.py \
    --dataset_csv $WORK_DIR/csv/A.csv --cache_dir $WORK_DIR/cache \
    --output_dir $WORK_DIR/ae \
    --n_epochs 50 --batch_size 2 --lr 1e-4 --num_workers 8 --gpus 0
```

**Check this before going further** — everything downstream is bounded by it:

```bash
python evaluate_ae.py --ckpt $WORK_DIR/ae/autoencoder-ep-49.pth \
    --csv $WORK_DIR/csv/A.csv --n 32 --out ae_eval.json
```

## Stage 2 — latents

```bash
python scripts/prepare/extract_latents.py \
    --dataset_csv $WORK_DIR/csv/A.csv \
    --aekl_ckpt   $WORK_DIR/ae/autoencoder-ep-49.pth --gpus 0
```

Writes `<scan>_new_latent.npz`, key `data`, shape `(4, 30, 36, 30)`. A per-dataset
`scale_factor = 1/std(latent)` is applied before the network and removed before decoding.

> **Latent shapes are not interchangeable.** `(4,30,36,30)` sibling latents (120³ FOV) and
> `(4,32,36,32)` latents (128³ FOV) come from the same weights but different crops; mixing
> them silently degrades everything.

## Stage 3 — Mamba latent diffusion

```bash
python scripts/training/train_diffusion_mamba.py \
    --dataset_csv $WORK_DIR/csv/A.csv --cache_dir $WORK_DIR/cache \
    --output_dir $WORK_DIR/diff --aekl_ckpt $WORK_DIR/ae/autoencoder-ep-49.pth \
    --n_epochs 100 --batch_size 8 --lr 2.5e-5 --gpus 0
```

## Stage 4 — Mamba ControlNet

Needs the **pairs** table:

```bash
python scripts/training/train_control_mamba.py \
    --dataset_csv $WORK_DIR/csv/B.csv --cache_dir $WORK_DIR/cache \
    --output_dir $WORK_DIR/cnet --aekl_ckpt $WORK_DIR/ae/autoencoder-ep-49.pth \
    --diff_ckpt  $WORK_DIR/diff/unet-ep-99.pth \
    --n_epochs 100 --batch_size 2 --lr 2.5e-5 --gpus 0
```

**Select the ControlNet checkpoint by correlation**, not by taking the last epoch: run
`eval_final.py` per checkpoint and keep the one that scores best.

Training uses AdamW (lr 1e-4 for the autoencoder, 2.5e-5 for diffusion and ControlNet),
weight decay 0.01, gradient clipping at norm 2.0, and bf16 mixed precision. Checkpoints are
written every epoch and only the most recent few are kept.

## Stage 5 — inference and evaluation

The default operating point is set at the top of `predict_followup.py`:
`t_start = 200`, `steps = 100`, and a conservative blend
`followup = baseline + α · (prediction − baseline)` with **α = 0.10**.

```bash
export MC_AE_CKPT=$WORK_DIR/ae/autoencoder-ep-49.pth
export MC_UNET_CKPT=$WORK_DIR/diff/unet-ep-99.pth
export MC_CNET_CKPT=$WORK_DIR/cnet/cnet-ep-99.pth

# single prediction from a baseline latent and a target interval
python predict_followup.py --start_latent BASELINE_new_latent.npz \
    --delta_t_years 2.0 --out pred.npy

# operating-point sweep against the copy floor
T_STARTS=150,200,250 STEPS_LIST=25,50,100 ALPHAS=0.05,0.10,0.15 N=20 \
    python eval_final.py $MC_CNET_CKPT

# direction-aware change metrics
python eval_changemetric.py

# full test pass with saved volumes and side-by-side images
python test.py --dataset_csv $WORK_DIR/csv/B.csv --cache_dir $WORK_DIR/cache \
    --output_dir $WORK_DIR/test --aekl_ckpt $MC_AE_CKPT \
    --diff_ckpt $MC_UNET_CKPT --cnet_ckpt $MC_CNET_CKPT \
    --save --save_dir test_results --gpus 0
```

---

## Evaluation rules

These change the numbers, so apply them consistently:

- **Report n ≥ 64 pairs.** Per-case scores on this task are very noisy and small-n
  rankings are not reliable.
- **Score on the native 120×144×120 grid.** `decode(latent)` lives on the autoencoder's
  FOV; scoring on a padded 128³ grid penalises a border the latent pipeline cannot
  produce. Voxel-space methods are FOV-insensitive, so this is a fairness fix, not a crop
  trick.
- **Copy is a strong floor.** The real follow-up-versus-baseline change is a small fraction
  of the intensity range, comparable to the autoencoder's reconstruction error, so copying
  the baseline scan already scores near-optimal PSNR and raw PSNR barely separates methods.
  The direction-aware change metrics from `eval_changemetric.py` are the informative lens.
  (They are defined formally in the "Metrics" section of Δ-LFM's
  [`README.md`](../ICLR%202026%20Delta-LFM/README.md#metrics).)

## Results

| | PSNR (dB) ↑ | SSIM (%) ↑ | hippocampus MAE ↓ |
|---|---|---|---|
| **MambaControl (this release: diffusion + control)** | **28.37 ± 1.70** | **92.01 ± 1.29** | **0.029 ± 0.030** |
| MambaControl (+ Fourier anatomy graph, *not released*) | 29.72 ± 1.04 | 93.60 ± 0.96 | 0.018 ± 0.014 |

215.5 M generator parameters. Regional MAE is also reported for amygdala, lateral
ventricles, thalamus and CSF; see Table I and the Section IV-B ablation in the paper.
These numbers are quoted from the paper, not regenerated from this repository.

## Variants

`train_residual_regression.py` / `resreg_eval.py` implement a non-generative ablation: a
UNet takes `[baseline_latent (4ch), Δt plane (1ch)]` plus the 8-dim covariates and
regresses the **latent residual** `followup_latent − baseline_latent` under an L1 loss,
with `prediction = decode(baseline_latent + net(...))`. It optimises the change directly
instead of sampling a plausible follow-up, trading image realism for a better-aligned
change field.

```bash
MC_LATENT_SUFFIX=_new_latent.npz MC_NOCACHE=1 python train_residual_regression.py \
    --dataset_csv $WORK_DIR/csv/B.csv --cache_dir $WORK_DIR/cache \
    --output_dir $WORK_DIR/resreg --aekl_ckpt $MC_AE_CKPT \
    --n_epochs 60 --batch_size 8 --gpus 0

MC_LATENT_SUFFIX=_new_latent.npz python resreg_eval.py \
    --ckpt $WORK_DIR/resreg/resreg-ep-59.pth --aekl_ckpt $MC_AE_CKPT --num 64 --gpu 0
```

## Reproducibility notes

- MetaTensor latents are stripped with `.as_tensor()` inside the training loop — without
  it, cross-attention allocates tens of GB.
- The `run/*.sh` launchers read `DATA`, `OUT`, `AE_CKPT`, `DIFF_CKPT`, `CNET_CKPT`,
  `CONDA_BASE` and `CONDA_ENV`, and need their `#SBATCH --partition` line edited for your
  cluster.
- Full variable reference: see [README.md](README.md#configuration).
