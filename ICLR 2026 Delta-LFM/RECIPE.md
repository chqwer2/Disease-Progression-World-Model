# Recipe

Exact commands to train and evaluate Δ-LFM, one component per section. `scripts/0*.sh` wrap the
same commands.

| Section | Script | Produces |
|---|---|---|
| [1. Setup](#1-setup) | `scripts/00_env.sh` | environment variables |
| [2. Data](#2-data) | `../AD_data_processing/run.sh` | pair and per-visit CSVs |
| [3. Autoencoder](#3-autoencoder) | `step1_v2_axes.py` | 3D autoencoder, 30 epochs (60 000 steps) |
| [4. Latents](#4-latents) | `step2_extract_latents.py` | one latent per visit |
| [5. Noise-scale probe](#5-noise-scale-probe) | `probe_flowinit_sigma.py` | `--fm_flowinit_sigma` for your latents |
| [6. Flow matching](#6-flow-matching) | `step3_train_flowmatching.py` | 30 epochs base + 10 epochs fine-tune, in one of two variants |
| [7. Evaluation](#7-evaluation) | `eval_fm.py` | scores and predictions |

---

## 1. Setup

```bash
pip install torch monai diffusers nibabel pandas scikit-image lpips

export DATA_DIR=/path/to/dataset       # contains Image/<subject>/<date>/t1.nii.gz and derived/*.csv
export WORK_DIR=/path/to/outputs       # checkpoints, latents and results
export DERIVED_DIR=$DATA_DIR/derived   # optional; defaults to $DATA_DIR/derived
```

Run every command from this directory. The YAML configs expand `${DATA_DIR}` and `${WORK_DIR}`.
`--output_dir` is relative to `--temp_path`. On machines with mixed GPU models set
`CUDA_DEVICE_ORDER=PCI_BUS_ID`; `--gpu` selects the device.

The MAISI VAE initialisation (`autoencoder_epoch273.pt`) is downloaded on first use.

## 2. Data

```bash
RAW_DATA_DIR=/path/to/raw/cohort OUT_DIR=$DERIVED_DIR bash ../AD_data_processing/run.sh
```

Builds the per-visit table (autoencoder, `dataset_csv` in `Step1_3D_AE_Contastive.yaml`) and the
(baseline, follow-up) pair table with prior-visit columns (flow matching, `dataset_csv` in
`Step3_3D_FM_multidt15.yaml`). Point the two config entries at your files. See
[`../AD_data_processing/README.md`](../AD_data_processing/README.md) for the schema.

## 3. Autoencoder

```bash
python step1_v2_axes.py \
  --config config/default/Step1_3D_AE_Contastive.yaml \
  --data_dir $DATA_DIR --temp_path $WORK_DIR --output_dir ae \
  --norm_mode std --res_scale 0.82 --crop_mode patch \
  --batch_size 1 --lr 1e-5 --num_workers 4 --warmup_epochs 0 \
  --aug_rigid 3 --aug_rot_deg 5 --adv_warmup 400 --adv_ramp 1200 \
  --ae_chgrec_w 0.3 --ae_dircosE_w 0.3 \
  --save_every 1 --eval_steps 250 --lin_max_batches 8 --eval_every 99 \
  --n_epochs 30 --steps_per_epoch 2000 \
  --use_contrastive --arc_delta_angle --angle_weight 0.35 --arc_order_w 0.2
```

Checkpoints are written every epoch as `$WORK_DIR/ae/all-ae-<EP>-3D.pth`. Select one on the
reconstruction and latent-trajectory metrics printed at each epoch, and pass it to the next sections
as `$AE_CKPT`.

| Flag | Role |
|---|---|
| `--crop_mode patch` | trains on native-resolution brain patches of the 1.5 mm volume |
| `--aug_rigid 3 --aug_rot_deg 5` | one rigid transform shared by the three visits of a triplet, so the change between visits is preserved |
| `--use_adv` (default on), `--adv_warmup 400 --adv_ramp 1200` | patch adversarial loss, switched on after 400 steps and ramped over 1200 |
| `--ae_chgrec_w 0.3` | change reconstruction: residual of the reconstructed change inside the change region (direction and amplitude) |
| `--ae_dircosE_w 0.3` | direction agreement of the reconstructed change |
| `--use_contrastive --arc_delta_angle --angle_weight 0.35` | recommended. ArcRank: orders each subject's visits angularly so consecutive latent steps align |
| `--arc_order_w 0.2` | recommended. scale-free time ordering along the trajectory chord |
| `--steps_per_epoch 2000 --n_epochs 30` | one epoch = 2000 optimiser steps, so 30 epochs = 60 000 steps; checkpoints are saved every epoch |

The flow-matching sections below were validated with an autoencoder trained without the last line of
the command (ArcRank and contrastive terms off; `ARCRANK=0 bash scripts/01_train_autoencoder.sh`).
The startup log prints the active terms, e.g.
`[switches] active traj terms = ['angle_weight', 'arc_order_w'] | use_contrastive=True`.

Triplets of consecutive visits are split 8:2 by patient with an md5 hash of the subject id.

## 4. Latents

```bash
python step2_extract_latents.py \
  --config config/default/Step2_3D_Extract.yaml \
  --data_dir $DATA_DIR --latent_path $WORK_DIR/latents --temp_path $WORK_DIR \
  --aekl_ckpt $AE_CKPT \
  --res_scale 0.82 --norm_mode std
```

Each visit is encoded from the whole brain: the canonical 128×144×128 volume at 1.5 mm is resized
with trilinear interpolation to 112×112×112 (`res_scale 0.82` rounds every axis to a multiple of 16),
giving a 4×28×28×28 latent. Section 7 inverts this resize for the image output.

## 5. Noise-scale probe

The flow starts from Gaussian noise with standard deviation `--fm_flowinit_sigma`. The value depends
on the autoencoder, the resolution and the cohort, so it is measured on the training pairs of your
latents:

```bash
python probe_flowinit_sigma.py \
  --config config/default/Step3_3D_FM_multidt15.yaml \
  --data_dir $DATA_DIR --latent_path $WORK_DIR/latents --temp_path $WORK_DIR \
  --res_scale 0.82 --norm_mode std --cache_dir none \
  --fm_hist_mode "prev1+prev2" --split_v3 1
```

It prints `std(z1 − z0)` and the line `--fm_flowinit_sigma <value>`. Pass that line to section 6.
Re-run the probe whenever the autoencoder, resolution or cohort changes.

## 6. Flow matching

**Two variants.** Both use the same two-stage schedule below and differ only in what the network
predicts and how inference draws the prediction:

| Variant | Training | Inference |
|---|---|---|
| **Direct change — default** (the paper's setting) | `--fm_x0_pred 1`: the network regresses the change δ = z₁ − z₀ itself, so the target variance does not diverge as t → 1 | `--fm_det 1 --n_avg 1`: the noise input is zeroed, so one deterministic pass gives the prediction and averaging has no effect |
| **Stochastic flow — optional** | `--fm_x0_pred 0`: the network predicts the velocity of a flow that starts from Gaussian noise | integrate the flow and average `--n_avg 2` antithetic samples (`z`, `−z`) |

The commands below are the default variant. For the stochastic one, set `--fm_x0_pred 0` in both
stages and evaluate with `--n_avg 2 --fm_antithetic 1` instead of `--fm_det 1 --n_avg 1`; everything
else is identical.

Two stages. The first learns the change field from the flow loss alone; the second adds the
change-aware terms and fine-tunes. These terms slow down how fast the model recovers the change
amplitude, so they are switched on only once the amplitude is in place:

| Term | What it constrains | Effect on amplitude early in training | Used |
|---|---|---|---|
| energy mask, `--fm_mask_lambda 5` | where the error is measured (spatial attention) | smallest | fine-tune stage |
| soft cF1, `--fm_cf1_w 0.5` | how much the predicted and true change overlap (differentiable F1) | moderate | fine-tune stage |
| direction, `--fm_dir_w` | which way the change points (cosine in channel space) | largest | off |

### 6a. Base stage — 30 epochs, flow loss only

```bash
python step3_train_flowmatching.py \
  --config config/default/Step3_3D_FM_multidt15.yaml \
  --data_dir $DATA_DIR --latent_path $WORK_DIR/latents --temp_path $WORK_DIR \
  --aekl_ckpt $AE_CKPT --output_dir fm_base \
  --res_scale 0.82 --norm_mode std --cache_dir none \
  --fm_scheme std --fm_scale_norm 0 --fm_res_noise 1 \
  --fm_hist_mode "prev1+prev2" --split_v3 1 \
  --fm_x0_pred 1 \
  --fm_mask_lambda 0 --fm_rf1_w 0.0 --fm_cf1_w 0.0 --fm_dir_w 0.0 \
  --fm_flowinit_sigma <value from section 5> \
  --batch_size 8 --lr 2.5e-5 --n_epochs 30
```

### 6b. Fine-tune — 10 epochs with the change-aware terms

```bash
python step3_train_flowmatching.py \
  --config config/default/Step3_3D_FM_multidt15.yaml \
  --data_dir $DATA_DIR --latent_path $WORK_DIR/latents --temp_path $WORK_DIR \
  --aekl_ckpt $AE_CKPT --output_dir fm \
  --fm_init_ckpt $WORK_DIR/fm_base_rectified/fm-unet-ep-<EP>.pth \
  --res_scale 0.82 --norm_mode std --cache_dir none \
  --fm_scheme std --fm_scale_norm 0 --fm_res_noise 1 \
  --fm_mask_lambda 5 --fm_mask_mode soft --fm_mask_src latent \
  --fm_hist_mode "prev1+prev2" --split_v3 1 \
  --fm_x0_pred 1 \
  --fm_rf1_w 0.0 --fm_cf1_w 0.5 --fm_dir_w 0.0 \
  --fm_flowinit_sigma <value from section 5> \
  --batch_size 8 --lr 2.5e-5 --n_epochs 10
```

Every flag other than the change-aware terms, the epoch count and `--fm_init_ckpt` is identical in
the two stages. Checkpoints and `args.json` are written to `$WORK_DIR/<output_dir>_rectified/`; evaluate the
fine-tuned stage.

| Flag | Role |
|---|---|
| `--split_v3 1` | every cohort is split 8:2 by patient (md5 of the subject id); each cohort has its own test partition |
| `--fm_res_noise 1` | Δ-Res-Flow: noise flows to the latent change δ = z₁ − z₀, with `z₀` concatenated as condition |
| `--fm_hist_mode prev1+prev2` | the two previous visits enter as extra input channels |
| `--fm_init_ckpt` | warm start: load the base-stage weights instead of training from scratch |
| `--fm_mask_lambda 5 --fm_mask_mode soft --fm_mask_src latent` | per-voxel loss weight 1 + 5·m, with m the normalised latent change \|z₁ − z₀\| |
| `--fm_cf1_w 0.5` | differentiable cF1 surrogate (`src/change_f1_soft.py`); off by default, on in the fine-tune stage only |
| `--fm_dir_w 0.0` | the change-direction term is off: of the change-aware terms it holds back the amplitude the most |
| `--fm_rf1_w 0.0` | the rF1 surrogate is off; rF1 increases with prediction amplitude |
| `--fm_x0_pred 1` | the network regresses δ directly (default variant); drop it for the stochastic flow |
| `--fm_flowinit_sigma` | noise scale from section 5 |

## 7. Evaluation

```bash
python eval_fm.py \
  --config config/default/Step3_3D_FM_multidt15.yaml \
  --data_dir $DATA_DIR --latent_path $WORK_DIR/latents --temp_path $WORK_DIR \
  --aekl_ckpt $AE_CKPT \
  --res_scale 0.82 --norm_mode std --cache_dir none \
  --fm_scheme std --fm_scale_norm 0 --fm_res_noise 1 \
  --fm_mask_lambda 5 --fm_mask_mode soft --fm_mask_src latent \
  --fm_hist_mode "prev1+prev2" --split_v3 1 --test_part adni \
  --fm_ckpt $WORK_DIR/fm_rectified/fm-unet-ep-<EP>.pth \
  --fm_det 1 --n_avg 1 --fm_seed 1234 --fm_sigma 0 --batch_size 1 \
  --n_eval <N> --raw_out 1 --dump_pred $WORK_DIR/pred_ep<EP> --tag ep<EP>
```

Results are written to `ae_runs/fm_eval/<tag>.json`; `--dump_pred` saves the volumes.

### Evaluation standard

1. **Test patients.** `--split_v3 1` with the same `--test_part` (`adni`, `aibl`, `oasis` or `all`) for
   every model. To score another method's case list, pass it through `FM_PAIR_KEYS` (one
   `subject__startdate__followdate` per line), keep only cases in this model's test partition and
   report how many remain.
2. **Sampling.** Default variant: `--fm_det 1 --n_avg 1`, one deterministic pass with the noise input
   zeroed. Stochastic variant: `--n_avg 2 --fm_antithetic 1`, two antithetic samples averaged, fixed
   seed. `--fm_flowinit_sigma` is read from the checkpoint's `args.json`; do not override it.
3. **Checkpoints.** Report the mean over three consecutive epochs rather than a single one, and
   state which epochs were used.
4. **Two output spaces, each scored with the same metric panel and its own copy baseline (predict no change).**
   - *Latent-decoded* (`pred` in the JSON): prediction, baseline and follow-up all decoded by the
     autoencoder.
   - *Original scan* (`raw_output` in the JSON, `--raw_out 1`): the image output of the method,

     $$\hat{x}_1^{\text{raw}} = x_0^{\text{raw}} + a\,\big(\mathrm{up}(\hat{x}_1^{\text{dec}}) - \mathrm{up}(x_0^{\text{dec}})\big),
     \qquad (a, b) = \arg\min \sum_{\Omega} \big(x_0^{\text{raw}} - a\,\mathrm{up}(x_0^{\text{dec}}) - b\big)^2$$

     where `up` is the trilinear resize from the 112³ model grid back to the 128×144×128 canonical
     grid (the inverse of section 4), $x^{\text{raw}}$ is the scan after 1.5 mm resampling, per-volume
     scaling to [0, 1] and padding to the canonical grid,
     $\Omega = \lbrace x_0^{\text{raw}} > 0.05 \rbrace \cup \lbrace x_1^{\text{raw}} > 0.05 \rbrace$,
     and the result is clipped to [0, 1]. Image detail comes from the baseline scan; the model
     contributes the change. Use this output for visual comparison with image-space methods.
5. **Metrics.** cF1 (`src/change_f1.py`) is the primary metric; definitions of all metrics are in the
   [README](README.md#metrics). PSNR and SSIM are reported only next to the copy baseline.

Each result JSON also records the sampling settings, the evaluated population (`eval_env`, including
the copy-baseline PSNR) and the checkpoint's training arguments. The δ dump under
`ae_runs/fm_eval/delta/<tag>/` refuses to overwrite a non-empty directory; use a new tag or
`FM_DUMP_OVERWRITE=1`.
