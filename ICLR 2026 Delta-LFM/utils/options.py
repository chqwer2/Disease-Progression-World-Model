import argparse
import os
import torch
from .utils_yaml import load_yaml



# Step 1: Create the full parser using YAML values as defaults
parser = argparse.ArgumentParser()

# ---------------- Dataset ----------------
parser.add_argument('--config', default="", type=str, help="Path to the YAML config file")
parser.add_argument('--num_workers', default=12, type=int, help='Number of workers for DataLoader')
parser.add_argument('--num_samples', default=2, type=int, help='Number of samples per CT scan')

# ---------------- Paths ----------------
parser.add_argument('--data_dir', default="", type=str, help="Root data directory (e.g., main_adni/data)")
parser.add_argument('--cache_dir', default="", type=str, help="Directory to store cached data")
parser.add_argument('--output_dir', default="", type=str, help="Directory to store outputs")
parser.add_argument('--temp_path', default="", type=str, help="Path to store latent representations")
parser.add_argument('--dataset_csv', default="", type=str,
                    help="Path to the dataset table; empty falls back to the dataset_csv entry in the YAML config")
parser.add_argument('--latent_path', default="", type=str, help="Path to store latent representations")

# ---------------- Model Checkpoints ----------------
parser.add_argument('--aekl_ckpt', default=None, type=str)
parser.add_argument('--disc_ckpt', default=None, type=str)
parser.add_argument('--diff_ckpt', default=None, type=str)
parser.add_argument('--cnet_ckpt', default=None, type=str)


# ---------------- GPU ----------------
parser.add_argument('--gpu', type=str, default='0', help='GPU index to use')
parser.add_argument('--dist', action='store_true', help='Use distributed training')
parser.add_argument('--DEBUG', action='store_true', help='Enable debug mode')
parser.add_argument('--extract_latent', action='store_true', help='Extract latent representations')

# ---------------- Training ----------------
parser.add_argument('--max_batch_size', default=1, type=int)
parser.add_argument('--n_epochs', default=5, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--lr', default=2.5e-5, type=float)

# ---------------- Diffusion ----------------
parser.add_argument('--num_train_timesteps', default=1000, type=int)
parser.add_argument("--use_vig", action="store_true")
parser.add_argument("--use_seg", action="store_true")
parser.add_argument("--use_t", action="store_true")
parser.add_argument("--use_standard_norm", action="store_true")
parser.add_argument("--num_classes", default=0, type=int)

# ---------------- Loss --------------------
parser.add_argument("--use_contrastive", action="store_true")
parser.add_argument("--crop_mode", default="resize", type=str, choices=["resize","patch"], help="resize=downsample the whole brain; patch=native-resolution brain patch (MAISI-style)")
parser.add_argument("--aug_rigid", default=0.0, type=float,
                    help="Translation magnitude (voxels) of the rigid augmentation shared by the three visits. 0=off. "
                         "Translation and rotation only: scaling and elastic warps would alter volume, which is the signal")
parser.add_argument("--aug_rot_deg", default=5.0, type=float,
                    help="Rotation magnitude (degrees) of --aug_rigid")
parser.add_argument("--traj_subspace_k", default=0, type=int,
                    help="Project the trajectory losses onto the top-K principal components of the population delta before "
                         "computing them. Reproducible patient-specific directions live in the leading components; the "
                         "remaining ones are dominated by noise. 0=off")
parser.add_argument("--ae_predz_w", default=0.0, type=float,
                    help="Direction-predictability loss 1-cos(delta_proj, W.z0_proj), where W is a closed-form ridge "
                         "regression that is not backpropagated through. Optimizes the downstream objective directly. "
                         "Requires --traj_subspace_k")
parser.add_argument("--ae_snr_w", default=0.0, type=float,
                    help="Delta signal-to-noise loss: a one-sided penalty on the amount by which the short-interval rate "
                         "of change exceeds the long-interval one, equivalent to pushing the intercept sigma in "
                         "||d||^2=sigma^2+(k*dt)^2 toward zero. The log ratio makes it scale-invariant, so collapse gains "
                         "nothing. 0=off")
parser.add_argument("--ae_snr_mindt", default=1.0, type=float,
                    help="ae_snr_w: skip samples where either visit interval is shorter than this (months); the log ratio "
                         "is unstable there")
parser.add_argument("--loss_std_frame", default=0, type=int,
                    help="1=compute the change losses (dircos/dircosE/dircosR/chgrec) in the per-volume standardized "
                         "frame. Raw [0,1] differences also carry global brightness and contrast drift, which adds "
                         "interval-independent noise to the change field. 0=original behaviour")
parser.add_argument("--norm_shared", default=0, type=int,
                    help="1=share the baseline mu/sigma across a subject's three visits when normalizing encoder inputs "
                         "(and correspondingly in _denorm_out, so reconstructions share the ground-truth frame). "
                         "Per-visit statistics differ, so per-volume normalization injects an artificial component into "
                         "the delta. 0=original behaviour")
parser.add_argument("--norm_mode", default="01", type=str, choices=["01","pm1","std"], help="encoder-input normalization: [0,1] / [-1,1] / standardize (losses stay in [0,1])")
parser.add_argument("--freeze_decoder", action="store_true", help="freeze decoder params (preserve pretrained change-fidelity / ae_recon ceiling; train encoder only)")
parser.add_argument("--steps_per_epoch", default=0, type=int, help="treat N optimizer steps as one epoch (0=off, use the full loader). Shortens the epoch-end probe/checkpoint cycle on full-res runs.")
parser.add_argument("--save_every", default=5, type=int, help="checkpoint every N epochs")


# ---------------- Mamba / ZigMa ----------------
parser.add_argument('--embed_dim', default=768, type=int, help='Embedding dimension for ZigMa model')
parser.add_argument('--depth', default=24, type=int, help='Depth of ZigMa model')
parser.add_argument('--patch_size', default=2, type=int, help='Patch size for ZigMa model')
parser.add_argument('--scan_type', default='zigzagN8', type=str, help='Scan type: zigzagN8, hilbertN8, etc.')
parser.add_argument('--use_pe', default=2, type=int, help='Positional embedding type: 0=none, 1=fixed, 2=learnable')

parser.add_argument('--d_context', default=1, type=int, help='ZigMa context length')

# ---------------- Task and Channels ----------------
parser.add_argument("--in_channels", default=3, type=int, help="Model input channel count")
parser.add_argument("--cond_channels", default=64, type=int, help="Conditioning channel count")
parser.add_argument("--seg_channels", default=64, type=int, help="Segmentation channel count")
parser.add_argument("--contrastive_channel", default=0, type=int, help="Contrastive feature channel count")

# ---------------- Misc ----------------
parser.add_argument("--comment", default="", type=str, help="Optional comment or tag for this run")
parser.add_argument("--project", action="store_true", help="Enable project mode")

# ---------------- AE recipe / contrastive-loss knobs -------
parser.add_argument("--angle_weight", default=0.0, type=float, help="ArcRank angle (direction) loss weight")
parser.add_argument("--arc_delta_angle", action="store_true", help="ArcRank angle on trajectory deltas (straightness) instead of spatial U")
parser.add_argument("--arc_diversity", action="store_true", help="penalize cross-patient direction alignment (keep dir_diversity high)")
parser.add_argument("--w_div", default=0.0, type=float, help="diversity (anti-convergence) weight")
parser.add_argument("--arc_channels", default=0, type=int, help="apply traj terms to first-k latent channels only (0=all)")
parser.add_argument("--arc_curv", default=0.0, type=float, help="curvature (2nd-difference) straightness weight")
parser.add_argument("--arc_svd", default=0.0, type=float, help="trajectory collinearity (1-line_r2) weight, computed on the subspace")
parser.add_argument("--arc_whiten", action="store_true", help="whitening (decorrelate) diversity instead of pairwise cosine")
parser.add_argument("--arc_offline", action="store_true", help="sign-pinned SVD straightness (off-line motion fraction + forward-sign hinge)")
parser.add_argument("--w_sign", default=1.0, type=float, help="forward-sign hinge weight inside the off-line term")
parser.add_argument("--arc_biomarker", default=0.0, type=float, help="disease-biomarker anchor weight (align latent axis to ventricle-hippocampus severity)")
parser.add_argument("--arc_decode", default=0.0, type=float, help="train-time decode-along-line consistency weight (decode interp latent -> real mid visit)")
parser.add_argument("--arc_change_w", default=0.0, type=float, help="change-fidelity: L1(decode(z3)-decode(z1) denormed, real v3-v1) — bind latent movement to real image change")
parser.add_argument("--arc_change_corr_w", default=0.0, type=float, help="correlation change-fidelity: maximize per-sample brain-masked Pearson(decode(z3)-decode(z1), real v3-v1) — targets ceil_pcc directly")
parser.add_argument("--arc_decode_fixed", default=0.0, type=float, help="decode-consistency in the denormalized frame: L1(denorm(decode(0.5(z1+z3))), real v2)")
parser.add_argument("--arc_decode_tw", action="store_true", help="arc_decode_fixed uses the REAL time fraction a=(t1-t0)/(t2-t0) instead of hard-coded 0.5 (visit spacing is uneven)")
parser.add_argument("--arc_decode_sg", action="store_true", help="stop-gradient the latents in arc_decode_fixed (train decoder only, do not distort encoder geometry)")
parser.add_argument("--arc_order_w", default=0.0, type=float, help="scale-free time-ordering along the trajectory chord (orders latent by time WITHOUT magnitude inflation; replaces magnitude-rnc)")
parser.add_argument("--order_margin", default=0.2, type=float, help="min fraction-of-chord each consecutive step must advance for --arc_order_w")
parser.add_argument("--arc_dec_ckpt", default=1, type=int,
                    help="Use gradient checkpointing for the decode-consistency losses (interpolation/extrapolation): activations are recomputed in the backward pass rather than stored. 1=on (default; required to avoid OOM at full resolution), 0=off")
parser.add_argument("--arc_angle_span", default=0, type=int,
                    help="Compute the direction-consistency loss over the full span, cos(z2-z1,z3-z1), instead of adjacent segments, cos(z2-z1,z3-z2). Adjacent segments cover a shorter interval and share z2, whose noise enters the two segments with opposite sign and biases the cosine downwards")
parser.add_argument("--traj_min_dt", default=0.0, type=float,
                    help="Trajectory losses (angle/extrap) sample only triplets whose two intervals both exceed this many months. Short-interval directions are almost entirely noise, but raising the threshold also discards triplets. Gates the trajectory losses only; reconstruction, perceptual and adversarial terms are unaffected")
parser.add_argument("--arc_extrap_dtw", default=0.0, type=float,
                    help="Delta-t weighting exponent p for the extrapolation loss: w proportional to (t2-t1)^p, normalized to mean 1. Noise is interval-independent while signal scales with delta t, so short-interval directions are almost entirely noise. 0=unweighted (original behaviour)")
parser.add_argument("--arc_extrap_lat_w", default=0.0, type=float,
                    help="Latent-space extrapolation consistency, bypassing the decoder: L=||z3-(z2+beta(z2-z1))||^2/||z3-z2||^2 with the denominator detached, beta set by the true visit intervals, and out-of-range samples skipped. Unlike --arc_decode_extrap_w, it does not force the decoder to render out-of-distribution latents")
parser.add_argument("--arc_decode_extrap_w", default=0.0, type=float, help="separate (gentler) weight for extrapolation decode-consistency")
parser.add_argument("--arc_decode_extrap", action="store_true", help="also add extrapolation decode consistency (predict future visit)")
parser.add_argument("--arc_vsmooth", default=0.0, type=float, help="velocity-smoothness weight (penalize abrupt speed change)")
parser.add_argument("--arc_isotropy", default=0.0, type=float, help="delta-z isotropy prior weight (whiten raw steps, helps flow matching)")
parser.add_argument("--arc_div_hinge", default=0.0, type=float, help="diversity hinge margin (0=off; penalize only pairs closer than margin)")
parser.add_argument("--arc_id_invariance", action="store_true", help="force free (non-progression) channels constant across a patient's visits")
parser.add_argument("--w_id", default=0.0, type=float, help="identity-invariance weight")
parser.add_argument("--val_ckpt", default="", type=str, help="AE checkpoint for decode-along-line validation")
parser.add_argument("--val_max_patients", default=0, type=int, help="cap gathered patients for a fast subset change_pcc (0=all)")
parser.add_argument("--val_out", default="", type=str, help="where to write the decode-along-line json")
parser.add_argument("--no_adv", action="store_true",
                    help="Disable the adversarial loss. Adversarial is on by default: L1 + perceptual + adversarial form the base trio")
parser.add_argument("--no_ssim", action="store_true", default=True,
                    help="Disable the SSIM loss (off by default). Using SSIM requires --no_ssim False together with --use_ssim")
parser.add_argument("--adv_energy_w", default=0.0, type=float,
                    help="Per-patch energy weighting of the adversarial loss: w=1+lam*E with E=rank(|v1-v3|). The default of 0 reproduces the uniform whole-image average used by MAISI; values above 0 enable the weighting")
parser.add_argument("--adv_warmup", default=0, type=int, help="hold adversarial off for N steps, then ramp")
parser.add_argument("--adv_ramp", default=1000, type=int, help="steps to ramp adversarial weight to full")
parser.add_argument("--rnc_autoscale", default=0.0, type=float,
                    help="Rescale rnc at every step to R x rec (detached). 0=off. "
                         "A fixed weight grows increasingly aggressive as rec falls; this pins the relative strength instead")
parser.add_argument("--traj_center_pop", default=0, type=int,
                    help="Subtract the running population-mean drift before computing the trajectory losses (predz/G4). "
                         "Without it all three losses admit the same degenerate solution: send every subject's delta in one direction")
parser.add_argument("--seed", default=0, type=int,
                    help="Random seed. Repeating a recipe under a different seed measures the run-to-run noise floor, "
                         "which is the reference for any comparison between configurations")
parser.add_argument("--rnc_weight",   default=0.0,  type=float, help="ArcRank rank (magnitude monotone) loss weight")
parser.add_argument("--rank_margin",  default=0.1,   type=float, help="margin m in max(0, m-(Sigma_j-Sigma_i))")
parser.add_argument("--angle_mode",   default="infonce", type=str, choices=["infonce", "l1"],
                    help="direction-consistency form: CLIP InfoNCE (default) or paper's L1 sum|Ui-Uj|")
parser.add_argument("--rank_mode",    default="magnitude", type=str, choices=["magnitude", "displacement"],
                    help="rank term: paper magnitude-monotone (default) or legacy displacement")
parser.add_argument("--dataloader",   default="ad_progression", type=str, help="which dataset loader to use")
parser.add_argument("--eval_every",   default=2, type=int, help="run the linearity probe every N epochs")
parser.add_argument("--eval_batch_size", default=8, type=int, help="batch size for the linearity/visit loader")
parser.add_argument("--lin_max_batches", default=0, type=int, help="cap linearity-probe batches (0=all)")

# ---- reconstruction anchor weights (defaults follow MAISI) ----
parser.add_argument("--perceptual_weight", default=0.3, type=float, help="LPIPS weight (MAISI: 0.3)")
parser.add_argument("--adv_weight",        default=0.1, type=float, help="PatchGAN weight (MAISI: 0.1)")
parser.add_argument("--kl_weight",         default=1e-7, type=float, help="KL weight (MAISI: 1e-7)")
parser.add_argument("--use_lpips", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--use_adv",   action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--use_ssim",  action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--use_kl",    action=argparse.BooleanOptionalAction, default=True)

# ---- order-based trajectory-SVD loss ----
parser.add_argument("--traj", action="store_true", help="enable order-based trajectory loss (full latent, per patient)")
parser.add_argument("--traj_terms", default="T1,C1,C3", type=str, help="comma subset of T1,T3,C1,C3,LINE")
parser.add_argument("--w_t1",   default=0.02, type=float, help="collinearity (1-line_r2) weight")
parser.add_argument("--w_t3",   default=0.02, type=float, help="spectral-gap weight")
parser.add_argument("--w_c1",   default=0.01, type=float, help="order-monotone hinge weight")
parser.add_argument("--w_c3",   default=0.01, type=float, help="step-direction weight")
parser.add_argument("--w_line", default=0.02, type=float, help="non-SVD rank-regression control weight")
parser.add_argument("--w_move", default=0.01, type=float, help="min-motion (anti-collapse) weight")
parser.add_argument("--move_floor", default=0.0, type=float, help="min per-step ||dz|| required (MOVE)")
parser.add_argument("--traj_margin",   default=0.1, type=float, help="min normalized order gap (C1)")
parser.add_argument("--warmup_epochs", default=3, type=int, help="ramp trajectory weight over first N epochs")

# ---- fast screening ----
parser.add_argument("--res", default=0, type=int, help="cube resolution override, e.g. 48; 0=use config image_size")
parser.add_argument("--res_scale", default=0.0, type=float, help="aspect-preserving proportional downsample of canonical (128,144,128); e.g. 0.875 -> (112,128,112) ~1.7mm near-native. 0=off. Takes precedence over --res.")
parser.add_argument("--fm_tail_w", default="", type=str, help="'K,w_in', e.g. '8,0.05': downweight the error component lying in the first K principal components of delta to w_in and keep the tail at 1.0. The leading components dominate the norm of delta, so a plain L2 spends most of its budget there while the tail carries the patient-specific pattern. Empty=off (original behaviour)")
parser.add_argument("--train_cohort", default="", type=str, help="Train on these cohorts only (adni / adni+aibl / all); the test split is unchanged. Empty=use all (original behaviour). Changes the training cohort only, unlike --split_v2, which changes the split itself")
parser.add_argument("--fm_wd", default=1e-6, type=float, help="AdamW weight decay. The 1e-6 default is the original, near-unregularized behaviour; raise it (e.g. 1e-2) to reduce the train/test gap")
parser.add_argument("--fm_pair_cap", default=0, type=int, help="Maximum number of pairs contributed by any one patient (train split only). 0=unlimited (original behaviour). Subjects with many visits contribute disproportionately many pairs, so a cap evens out their weight")
parser.add_argument("--keep_ckpt", default=0, type=int, help="Keep the most recent N checkpoints. 0=keep all (default), which is required if the best epoch is picked afterwards from the external per-epoch autoeval JSON. Pass a small value (e.g. 3) to save disk space")
parser.add_argument("--fm_rate_pow", default=1.0, type=float, help="Fractional exponent of the realtime scheme: target = (x1-x0)/dt**p. ||delta|| grows sublinearly with the interval, so p below 1 makes the target more nearly independent of delta t. 1.0=original realtime")
parser.add_argument("--split_v3", default=0, type=int, help="Split each cohort 8:2 by patient; --test_part selects all/adni/aibl/oasis for the test set. Complementary to split_v2, which is the zero-shot setting")
parser.add_argument("--split_v2", default=0, type=int, help="Patient-disjoint split: train on 80%% of ADNI patients; --test_part selects adni (in-distribution), cross (AIBL+OASIS) or both. 0=the older row-wise 80/20 split")
parser.add_argument("--test_part", default="adni", type=str, help="Test partition under split_v2: adni | cross | both")
parser.add_argument("--fm_gbar", default="", type=str, help="Population-prior residualization: the training target becomes delta - gbar, where gbar is the mean change computed on the training set (npz), added back at inference. The model then learns only the individual deviation instead of spending capacity on the population mean. Empty=off")
parser.add_argument("--fm_pathc_w", default=0.0, type=float, help="Weight of the path-consistency loss, which forces delta_hat(p1->x1) to approximately equal (x0-p1) + delta_hat(x0->x1), constraining temporal additivity using real prior visits. 0=off (default)")
parser.add_argument("--fm_det", default=0, type=int, help="Deterministic trajectory: zero the flow's noise input and roll out along physical time from 0 to the full change. Removes sampling variance, so repeated evaluations of one case are identical. 0=off (default, stochastic sampling)")
parser.add_argument("--fm_scheme", default="std", type=str, choices=["std","realtime"], help="flow-matching time scheme: std [0,1] velocity x1-x0, or realtime velocity (x1-x0)/Dt")
parser.add_argument("--fm_sigma", default=0.0, type=float, help="Brownian-bridge noise sigma in compute_xt (0=deterministic rectified)")
parser.add_argument("--fm_noise_schedule", default="bridge", type=str, choices=["bridge","cosine","earlyhigh","latehigh","flat"], help="t-dependent noise envelope for the sigma term: bridge=sqrt(t(1-t)), cosine=sin(pi t), earlyhigh/latehigh peak in first/second half, flat=plateau")
parser.add_argument("--fm_scale_norm", default=1, type=int, help="1=per-sample scale-normalized loss (default), 0=plain velocity-MSE so /Dt scheme diff is preserved")
parser.add_argument("--fm_res_noise", default=0, type=int, help="1=Δ-Res-Flow: flow from NOISE->delta(x1-x0), x0 concatenated as condition (8-in/4-out); removes copy bias. 0=standard x0->x1 flow")
parser.add_argument("--fm_x0_cond", default=0, type=int, help="1=x0-start flow with x0 as explicit concat condition (8-in/4-out); breaks copy collapse while keeping x0-start sampling")
parser.add_argument("--fm_x0_start_noise", default=0.0, type=float, help="noisy-x0-start stochastic interpolant sigma: xt=x0+t*delta+sigma*(1-t)*z (t=0 noisy anchor breaks copy, t=1=x1); 8-in/4-out. 0=off. try 0.3-1.0")
parser.add_argument("--fm_cfg_dropout", default=0.0, type=float, help="CFG: prob of dropping age_gap cond to null idx=15 during training (num_class_embeds->16). ~0.15")
parser.add_argument("--fm_cond_drop", default=0.0, type=float, help="CFG on the CONTINUOUS cond_vec (patient covariates+dt): probability of dropping cond_vec to null during training. Enables guidance at eval via --fm_cond_w")
parser.add_argument("--fm_cond_w", default=0.0, type=float, help="CFG guidance weight at eval on cond_vec: v = v_null + w*(v_cond - v_null). Amplifies the PATIENT-SPECIFIC direction (not a blunt delta rescale). 0=off")
parser.add_argument("--fm_ema", default=0.0, type=float, help="EMA decay for the flow weights (e.g. 0.999). Standard in diffusion/FM; improves sample quality. 0=off")
parser.add_argument("--fm_cfg_w", default=0.0, type=float, help="CFG guidance weight at eval: v=v_null+w*(v_cond-v_null). 0=off, try 2-3")
parser.add_argument("--fm_grav_out", default=0.0, type=float, help="energy-gravity CONFINE: penalize |delta_hat| OUTSIDE the change mask (suppress spurious change). try 0.5-2")
parser.add_argument("--fm_grav_in", default=0.0, type=float, help="energy-gravity ATTRACT: push change-mass fraction INTO the mask, bounded (1-frac_in). try 0.2-1")
parser.add_argument("--fm_dt_tau", default=0.0, type=float, help="Delta-t loss weighting tau: per-sample weight w=Dt/(Dt+tau), down-weights short-gap low-SNR pairs (replaces /Dt). 0=off (uniform)")
parser.add_argument("--fm_histw_lambda", default=0.0, type=float, help="Weight the loss per voxel by the individual deviation of the historical energy: w *= 1+lambda*|C_ind|_norm, focusing on where this patient departs from the population")
parser.add_argument("--fm_xattn_mode", default="none", type=str, help="Cross-attention injection: history is encoded as a token sequence and written into the features through cross-attention (additive or generative)")
parser.add_argument("--fm_xattn_ntok", default=64, type=int)
parser.add_argument("--fm_timewarp", default="none", type=str, help="Time warping: per-voxel effective time t*s(x), with s given by the historical energy, so fast-changing regions advance further along the flow")
parser.add_argument("--fm_timewarp_src", default="ind", type=str, help="last|rate|ind: which etraj channel drives the time warp. ind=individual deviation (recommended), last=cumulative energy (largely shared across the population), rate=time derivative of the energy")
parser.add_argument("--fm_timewarp_w", default=0.5, type=float)
parser.add_argument("--fm_spade_mode", default="none", type=str, help="Components of the multi-scale SPADE spatial modulation: at each resolution level the history maps predict per-voxel gamma/beta that modulate the features")
parser.add_argument("--fm_flowinit", default="none", type=str, help="Flow initialization: linext draws z~N(v1*dt_target, sigma^2), changing the flow's task from generating the whole delta to correcting a linear extrapolation")
parser.add_argument("--fm_flowinit_sigma", default=1.0, type=float)
parser.add_argument("--fm_flowinit_w", default=1.0, type=float)
parser.add_argument("--fm_gain_mode", default="none", type=str, help="Components routed through the spatial-modulation path, kept separate from the concat path: geometric prior visits enter as features via concat, energy enters as weights via modulation")
parser.add_argument("--fm_gain_norm", type=int, default=0, help="Constrain the gain field to a mean-1 multiplicative redistribution, so the total predicted change is preserved and only its spatial distribution changes")
parser.add_argument("--fm_lat_ew", type=float, default=0.0, help="Reweight the latent change field by energy before decoding, preserving the total. Latent-space counterpart of the image-space reweighting")
parser.add_argument("--fm_hist_inject", default="concat", type=str, help="concat|cond|both|gain|cnet|cnetft - concat: concatenated into the backbone input; cond: spatially pooled and added to the time embedding to modulate every resblock")
parser.add_argument("--fm_flowinit_ecov", default=0.0, type=float, help="Energy-shaped noise covariance: z~N(mu, sigma^2(1+lam*E)^2), with E the normalized E_ind channel of etraj. 0=off")
parser.add_argument("--fm_ecov_src", default="ekind", type=str, help="Which energy map ecov uses: ekind (E_ind, the individual deviation) | ek (E_last)")
parser.add_argument("--fm_energy_wavg", default=0.0, type=float, help="Temperature with which energy scores and weights the n_avg candidates")
parser.add_argument("--fm_eimg_xattn", default=0, type=int, help="Native-resolution VFIN energy -> learned encoder -> cross-attention tokens, with no fixed pooling")
parser.add_argument("--fm_eimg_ntok", default=64, type=int, help="Number of tokens (must be a perfect cube)")
parser.add_argument("--fm_vfin_recw", default=0.0, type=float, help="Weight the fullrec image L1 by native-resolution VFIN energy: w=1+lam*E, with no downsampling. Pooling the energy map blurs the change region, so it is kept at native resolution")
parser.add_argument("--fm_elossw", type=float, default=0.0, help="Strength L of the per-voxel latent-energy loss weighting: w=(1-L)+L*2*E")
parser.add_argument("--fm_elossw_src", type=str, default="vslope", help="Which energy component drives the loss weighting")
parser.add_argument("--fm_ephi_w", type=float, default=0.0, help="Deployable energy suppression: loss += w*mean(|dpred|*(1-E_rank)), a one-sided penalty on predicted change where past energy was low. It is not a reparameterization, so the model cannot bypass it")
parser.add_argument("--ae_change_w", type=float, default=0.0, help="Autoencoder change-preservation loss L1((R_j-R_i),(I_j-I_i)); its raw scale is comparable to the reconstruction loss, so lam of order 1 balances the two")
parser.add_argument("--ae_change_pairs", type=int, default=1, help="How many temporal pairings the change-preservation loss uses (1=only the widest span (0,2), which saves memory; 3=all)")
parser.add_argument("--ae_erec_w", type=float, default=0.0, help="Energy-weighted reconstruction loss: w=1+lam*E with E the rank-normalized |I0-I2|, spending autoencoder capacity where change actually occurs")
parser.add_argument("--ae_nuis_w", type=float, default=0.0, help="Acquisition-nuisance invariance: ||E(T(x))-E(x)||^2/var(z), with T a sub-voxel shift plus bias field plus noise. Small acquisition perturbations can move the latent as far as real progression does, so the encoder is asked to ignore them")
parser.add_argument("--ae_nuis_crop", type=int, default=48, help="Size of the random cubic block on which nuisance invariance is applied (the encoder is fully convolutional); the whole brain would run out of memory")
parser.add_argument("--ae_nuis_shift", type=float, default=1.0, help="Maximum sub-voxel shift of the nuisance transform (voxels), applied by interpolation")
parser.add_argument("--ae_nuis_bias", type=float, default=0.07, help="Multiplicative bias-field magnitude of the nuisance transform")
parser.add_argument("--ae_nuis_noise", type=float, default=0.01, help="Gaussian noise sigma of the nuisance transform")
parser.add_argument("--ae_dircos_w", type=float, default=0.0, help="Directional-cosine change loss: mean-removed 1-cos(reconstructed change field, true change field). This is the directional component of ceil_pcc; it ignores amplitude and targets cF1 directly")
parser.add_argument("--ae_dircosR_w", type=float, default=0.0, help="Region-restricted directional cosine, computed only inside the change region R={|true change|>0.25*P99} so that it matches the R used by cF1")
parser.add_argument("--fm_chan_norm", type=int, default=0, help="Per-channel normalization of the flow-matching loss: the delta channels differ widely in standard deviation while carrying comparable information, so without normalization the highest-variance channel dominates the gradient")
parser.add_argument("--fm_band_loss", default="", type=str, help="Per-band weights for the band-whitened loss, e.g. 1,1,3,1,0.5. Each band is normalized by its own scale so every band contributes a relative error, which differs fundamentally from --fm_tail_w, which weights without whitening")
parser.add_argument("--fm_band_basis", default="", type=str, help="Band basis npz containing the principal components and the band edges")
parser.add_argument("--fm_band_l2", default=0, type=int, help="0=L1 (the same norm as the baseline), 1=L2 after whitening (Mahalanobis distance)")
parser.add_argument("--fm_band_mix", default=1.0, type=float, help="loss=(1-lambda)*voxel loss + lambda*band loss")
parser.add_argument("--fm_band_wout", default=0.0, type=float, help="Weight of the residual lying outside the span of the band basis, which is essentially unpredictable noise. Default 0")
parser.add_argument("--fm_residual_ridge", default="", type=str, help="Residual flow matching: given a ridge-regression npz, the target becomes delta - W.z0, so the model learns only what the linear term cannot capture")
parser.add_argument("--ae_chgrec_q", default=0.90, type=float, help="Quantile defining the change region: the top (1-q) fraction of |delta_gt| within the brain. Higher q gives a tighter region; a threshold relative to P99 instead selects most of the brain")
parser.add_argument("--ae_sep_w", default=0.0, type=float,
                    help="Hinge on a lower bound for visit separability, relu(m - ||z_{t+1}-z_t||/||z_t||), which directly prevents the per-visit latents from collapsing to a point. The ratio makes it scale-free, detaching the denominator raises only the numerator, and the hinge zeroes the gradient once the margin is met so it cannot be traded for amplitude")
parser.add_argument("--ae_sep_margin", default=0.02, type=float,
                    help="The margin m used above. Calibrate it slightly above the ||z_{t+1}-z_t||/||z_t|| ratio that the diagnostic prints for the current model")
parser.add_argument("--ae_axes_lr_mult", default=100.0, type=float,
                    help="Learning-rate multiplier, relative to args.lr, for the ae_axes_k mode matrix E and the coefficient head. These heads are trained from scratch and need a faster rate than the backbone")
parser.add_argument("--ae_coef_ema", default=0.01, type=float,
                    help="EMA momentum for the population mean in the ae_axes_k coefficient-predictability term. With batch_size=1 an in-batch mean subtraction degenerates, so a cross-step EMA is required")
parser.add_argument("--ae_dircosE_pow", default=0.0, type=float,
                    help="Soft-weight exponent for dircosE: w=(E/meanE)^pow clipped to [0,5]. 0=binary mask (original behaviour). Energy is a continuous map, and binarizing it discards amplitude information")
parser.add_argument("--ae_dircosE_w", default=0.0, type=float,
                    help="Directional-cosine loss whose region is defined by the precomputed energy map (v2+v3). Same as --ae_dircosR_w except for the region source: the energy map instead of a quantile of |delta_gt|, so the region does not share its definition with the cF1 threshold")
parser.add_argument("--ae_chgrec_w", default=0.0, type=float, help="Change-reconstruction loss: the relative residual |delta_rec-delta_gt|/|delta_gt| inside the change region, which directly optimizes the autoencoder's change ceiling. Unlike dircos, which constrains direction only, this includes amplitude")
parser.add_argument("--ae_axes_k", default=0, type=int, help="v2: dimensionality k of the progression subspace (0=off, reducing to v1)")
parser.add_argument("--ae_axes_w", default=0.3, type=float, help="v2: weight of the subspace-concentration term")
parser.add_argument("--ae_coef_w", default=0.2, type=float, help="v2: weight of the coefficient-predictability term")
parser.add_argument("--fm_tail_ridge", default="", type=str, help="Inference-time tail replacement: path to the z0 ridge-regression npz. The flow-matching subspace component is kept and the tail is replaced by the ridge prediction")
parser.add_argument("--fm_tail_ridge_w", default=1.0, type=float, help="Tail-replacement weight: 1=replace fully, 0=keep, intermediate values blend")
parser.add_argument("--fm_tail_ridge_full", default=0, type=int, help="1=predict the entire delta with ridge regression instead of the flow, routed through the same eval path so the protocol stays identical")
parser.add_argument("--fm_tail_ridge_mm", default=0, type=int, help="1=amplitude matching: keep the flow-matching tail norm and replace only the direction, decoupling the comparison from amplitude effects")
parser.add_argument("--eval_strat", type=int, default=0, help="Interleave the test set in proportion to cohort size, so that taking the first N is automatically a stratified sample. 0=off (original ordering)")
parser.add_argument("--fm_cnet_w", type=float, default=1.0, help="Gain applied to the ControlNet residual at inference: w>1 amplifies the energy bypass, w=0 reduces to the energy-free baseline used as a control")
parser.add_argument("--fm_cnet_hist", default="none", type=str, help="Component combination for the ControlNet condition, decoupled from --fm_hist_mode, so the backbone can take history by concat while the ControlNet is fed energy separately")
parser.add_argument("--fm_cnet_shuffle", type=int, default=0, help="Control: roll the ControlNet energy condition along the batch dimension so each sample receives another patient's energy. Architecture, parameters and training budget are unchanged; only the information is destroyed")
parser.add_argument("--fm_prev2_fill", default="zero", type=str, help="zero|const: when prev2 is missing, use v2=0 (default) or fall back to v2<-v1 (constant velocity, zero acceleration)")
parser.add_argument("--fm_hist_mode", default="none", type=str, help="none|base|prev|both|energy|base_energy|prev_energy|both_energy — which history to concat into the backbone input")
parser.add_argument("--fm_hist_drop", default=0.0, type=float, help="randomly drop the history channels during training (CFG-style null), so present/absent history are learned as distinct modes")
parser.add_argument("--fm_hist_concat", default=0, type=int, help="concat multi-timepoint history (v1/acc/meta, 12ch) into the BACKBONE input instead of a frozen-backbone ControlNet; new conv_in channels zero-init")
parser.add_argument("--fm_mani_w", default=0.0, type=float, help="manifold-consistency: penalize ||enc(dec(x0+delta))-(x0+delta)||^2/||.||^2 on the change crop; keeps predicted latents on the AE manifold")
parser.add_argument("--fm_croprec_mode", default="abs", type=str, help="abs|change: change compares decoded CHANGE instead of absolute image")
parser.add_argument("--fm_croprec_src", default="latent", type=str, help="latent|energy: crop centering signal")
parser.add_argument("--fm_comp_w", default=0.0, type=float, help="temporal composition consistency: delta(p1->x1) == delta(p1->x0)+delta(x0->x1); scale-invariant relative residual")
parser.add_argument("--fm_blockdir_w", default=0.0, type=float, help="block-wise directional loss weight (magnitude-invariant placement loss)")
parser.add_argument("--fm_blockdir_k", default=4, type=int, help="block edge length in latent voxels")
parser.add_argument("--fm_driftsub_w", default=0.0, type=float, help="drift-subtracted residual MSE weight; needs derived/drift_W.npz")
parser.add_argument("--fm_mask_lambda", default=0.0, type=float, help="energy-mask voxel weighting: per-voxel weight w=1+lambda*energy_mask, focuses loss on real-change regions. 0=off. Needs --energy_path")
parser.add_argument("--fm_mask_mode", default="soft", type=str, choices=["soft","hard","pure"], help="energy-mask weighting mode: soft=1+lambda*energy; hard=1+lambda*(energy>thr) binary change-region; pure=energy+floor (loss ~only in change region)")
parser.add_argument("--fm_mask_src", default="energy", type=str, help="source of the change mask: energy(precomputed image-space map) | latent(|x1-x0| in latent space, matching the loss domain)")
parser.add_argument("--fm_x0_pred", default=0, type=int, help="x0-prediction: regress delta directly instead of velocity delta-z; removes the exploding target variance as t->1")
parser.add_argument("--fm_mask_thr", default=0.05, type=float, help="threshold on normalized energy for hard mask change-region")
parser.add_argument("--fm_delta_train_scale", default=1.0, type=float, help="scale the delta target in res-flow, so the amplitude is learned at training time instead of rescaled at test time")
parser.add_argument("--fm_init_ckpt", default="", type=str, help="warm-start: load a pretrained fm-unet .pth to fine-tune from (fast iteration)")
parser.add_argument("--fm_fullrec_w", default=0.0, type=float, help="FULL-image recon at small bs: decode first N full imgs/step (grad) vs GT (no grad), L1. batch-subsample avoids OOM. try 0.1-0.5")
parser.add_argument("--fm_fullrec_nb", default=2, type=int, help="num samples/step to full-decode for --fm_fullrec_w (memory knob; 1-2)")
parser.add_argument("--fm_fullrec_mask", default=0.0, type=float, help="energy-mask weighting for --fm_fullrec_w image L1 (mask upsampled to image res). soft: w=1+lambda*mask; pure: mask-only. Stops background dominating -> less copy-reward. try 4-10")
parser.add_argument("--fm_fullrec_mask_mode", default="soft", type=str, help="soft (1+lambda*mask keep global) | pure (mask-only change region) for --fm_fullrec_mask")
parser.add_argument("--fm_history", default=0, type=int, help="ISOLATED patient-history conditioning: encode the previous scan (prior visit) into cross-attention context tokens to steer generation. 0=off (identical to base res-flow). Needs a triples CSV with prior_image_path + has_prior.")
parser.add_argument("--fm_history_ntok", default=64, type=int, help="number of history context tokens from HistoryEncoder (perfect cube: 27/64/125). Default 64.")
parser.add_argument("--fm_energy_cond", default="none", type=str, help="condition the flow on a change-ENERGY field as an extra input channel [xt,x0,energy]. none | oracle (followup_energy=WHERE change will be; LEAK, ceiling probe) | past (starting_energy=where patient changed prior->starting; deployable). Raw soft [0,1] field.")
parser.add_argument("--fm_t_schedule", default="uniform", type=str, help="flow-matching t sampling: uniform | logitnorm (SD3: t=sigmoid(N(0,1)), focuses mid-timesteps)")
parser.add_argument("--fm_t_power", default=3.0, type=float, help="power for --fm_t_schedule late: t=u**(1/p); p>1 concentrates training on t->1")
parser.add_argument("--fm_asym_w", default=1.0, type=float, help="asymmetric magnitude penalty. Samples whose predicted |delta| is SMALLER than the true |delta| get this loss multiplier (rF1 penalizes under-magnitude more than over-magnitude). 1.0=off, try 2.0")
parser.add_argument("--fm_x0_drop", default=0.0, type=float, help="randomly zero the x0 condition channels during training. Forces the net to read x_t (which carries z) instead of predicting from x0 alone, which collapses to a deterministic mean. try 0.1-0.3")
parser.add_argument("--fm_wta_k", default=0, type=int, help="winner-take-all with K hypotheses (K noise draws, loss only on the closest). The minimiser becomes the set of conditional modes rather than the mean, so it cannot blur. try 2-4; costs K-1 extra forwards")
parser.add_argument("--fm_sub_w", default=0.0, type=float, help="subspace constraint: match the per-mode batch std of the predicted delta PCA coefficients to the real ones. Penalises shrinkage in the modes that carry most of the change energy; low-dim distribution matching, no discriminator. try 0.1-0.5")
parser.add_argument("--fm_sub_k", default=40, type=int, help="number of PCA modes for --fm_sub_w")
parser.add_argument("--fm_sub_mode", default="persample", type=str, help="persample = per-sample per-mode |coef| matching (cannot be satisfied by inflating batch variance); batchstd = batch-std matching")
parser.add_argument("--fm_train_tdt", default=0.0, type=float, help="train t only on [0,T_i] with T_i = clamp(dt_years/ref,0.15,1) — matches the T(dt) truncated sampler (ref years, e.g. 6.0). 0=off")
parser.add_argument("--fm_div_w", default=0.0, type=float, help="diversity regularizer — two noise draws per step, penalize their predictions being too similar. Fights collapse to the conditional mean. Costs 1 extra forward. try 0.05-0.2")
parser.add_argument("--fm_magcal_src", default="energy", type=str, help="region source for --fm_magcal_w. latent = |x1-x0| in latent space (same domain as the loss); energy = precomputed image-space map")
parser.add_argument("--fm_t_min", default=0.0, type=float, help="restrict training t to [t_min,1]; focuses capacity on the late path that sets delta sharpness")
parser.add_argument("--fm_controlnet", default=0, type=int, help="ControlNet-conditioned scan history: freeze the pretrained res-flow backbone, train a control branch that ingests the PREVIOUS scan (prior latent, 4ch) and injects residuals at every scale. Strong spatial conditioning. Needs triples CSV.")
parser.add_argument("--fm_cnet_cond", default="scan", type=str, help="ControlNet condition: scan=prior latent (4ch) | traj=[prior, velocity=x0-prior] (8ch, targets change RATE/magnitude)")
parser.add_argument("--fm_magcal_w", default=0.0, type=float, help="region magnitude calibration: match ||delta_hat||->||delta_gt|| in the change region, so the amplitude is learned at training time instead of rescaled at test time. try 0.3-1")
parser.add_argument("--fm_adv_w", default=0.0, type=float, help="adversarial/distribution-matching loss on the CHANGE FIELD delta: a patch discriminator separates predicted vs real delta. Directly penalizes blurry/conservative (mean-collapsed) predictions. try 0.05-0.2")
parser.add_argument("--fm_adv_warmup", default=200, type=int, help="steps before adversarial loss ramps in (D needs to be non-random first)")
parser.add_argument("--fm_croprec_w", default=0.0, type=float, help="crop-decode recon: L1 image-space loss on a fixed change-region crop (decode only the crop -> no OOM). try 0.3-1")
parser.add_argument("--fm_recon_w", default=0.0, type=float, help="weight of masked image-space recon aux loss (decode predicted x1 vs real followup). 0=off")
parser.add_argument("--fm_magw", default=0.0, type=float, help="magnitude-weighted FM loss, per-voxel weight 1+magw*|x1-x0|_norm, focuses capacity on true-change voxels. try 2-5")
parser.add_argument("--fm_region_w", default=0.0, type=float, help="soft-Dice region-overlap loss between predicted-change and GT-change regions (rF1 surrogate). try 0.2-1")
parser.add_argument("--fm_cf1_w", default=0.0, type=float, help="differentiable cF1 surrogate loss (same maths as src/change_f1_soft.py): continuous weights replace the hard threshold, so the loss aligns directly with the cF1 evaluation metric")
parser.add_argument("--fm_cf1_tau", default=0.25, type=float, help="soft-threshold factor of the cF1 loss, tau = frac * P99(|delta_gt|). Smaller lets more true-change voxels into the gradient (pushes recall). The evaluation uses a hard threshold of 0.25")
parser.add_argument("--fm_dir_w", default=0.0, type=float, help="change-direction agreement loss, |Δgt|-weighted (1-cos(δ̂,δgt)); raises direction agreement in change region -> rF1 ceiling. try 1-3")
parser.add_argument("--fm_rf1_w", default=0.0, type=float, help="differentiable region-F1 (rF1) surrogate loss: dir-aware soft-region-F1 with P99 threshold. try 0.3-1")
parser.add_argument("--fm_phi_w", default=0.0, type=float, help="potential-field transport, penalize |delta_hat|*Phi (Phi=smooth dist-potential 0-in-region), pulls change toward change-region. try 0.1-0.5")
parser.add_argument("--fm_regmag_w", default=0.0, type=float, help="region magnitude match |E_pred-E_gt| anti-collapse. try 0.1-0.5")
parser.add_argument("--energy_path", default="", type=str, help="dir of precomputed latent-grid energy maps (parallels latent_path); enables followup_energy in loader")
parser.add_argument("--eval_steps", default=0, type=int, help="run eval every N optimizer steps (0=epoch-end only)")
parser.add_argument("--min_timepoints", default=3, type=int, help="minimum distinct timepoints per subject for autoencoder triplets; 2 pads 2-visit subjects to (v0, v1, v1), which is only safe for a pure-reconstruction autoencoder")

# Parse CLI args
args = parser.parse_args()


yaml_config = load_yaml(args.config)  # returns a dict
# default_config = load_yaml(yaml_config["default"])




# ---------------- Override with YAML config ----------------
temp_path = args.temp_path if args.temp_path else yaml_config.get("temp_path", "")

for key, value in yaml_config.items():
    if key == "default":
        continue  # Skip "default" key itself

    if key in ["latent_path", "output_dir", "cache_dir"]:
        value = f"{temp_path}/{value}"

    if hasattr(args, key):
        current = getattr(args, key)
        if current in [None, "", 0, False] and value not in [None, "", 0, False]:
            setattr(args, key, value)
    else:
        setattr(args, key, value)  # Add missing keys if needed

# ---------------- CLI resolution override (fast screening) ----------------
# --res 48 forces image_size = [48,48,48], overriding the YAML, for faster train+test.
if getattr(args, "res_scale", 0.0) and float(args.res_scale) > 0:
    _canon = (128, 144, 128)
    args.image_size = [max(16, int(round(c * float(args.res_scale) / 16.0)) * 16) for c in _canon]
elif getattr(args, "res", 0):
    args.image_size = [int(args.res)] * 3






