"""Δ-LFM Step-3 flow-matching EVALUATION harness.

Generates followup latents from starting latents on the held-out TEST split
(last 20% of pairs), decodes via the AE, and scores the predicted followup
against the real followup with the change-sensitive metric suite:
  ΔF1 / Recall / Precision (src.delta_f1),
  CHANGE_MAE / DICE / PCC (utils.utils_metric.compute_change_metrics),
  Δ-RMAE (whole-brain relative), PSNR, SSIM.

Scheme-aware sampler (mirrors training exactly):
  * std      : net predicts u=x1-x0        -> integrate z += v*ds          over s in [0,1]
  * realtime : net predicts u=(x1-x0)/Dt   -> integrate z += v*ds*Dt       (Dt per-sample = follow_up_gap/10)
Both feed timesteps = path fraction s in [0,1] (matches training's `timesteps=t`).

A COPY baseline (pred = input, i.e. no-change) is scored alongside as a reference
floor: it earns high PSNR/SSIM (anatomy dominates) but ~0 ΔF1 (predicts no change).

Runs on a single GPU (validation only).
"""
import os, sys, json, argparse
import numpy as np
import torch
import src.flow as _flowmod
import torch.nn.functional as F

# ---- pull the eval-only args off argv BEFORE utils.options parses it ----
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--fm_ckpt", type=str, required=True)
_pre.add_argument("--fm_det", type=int, default=0, help="deterministic trajectory: noise is zeroed, so n_avg and antithetic sampling have no effect")
_pre.add_argument("--fm_rollout_years", type=float, default=0.0, help="physical-time rollout: call the model repeatedly in steps of R years, using each step's predicted latent as the next starting point. The denoising axis is unchanged (still the full [0,1]); only delta t is split across calls. 0=off")
_pre.add_argument("--eval_on", type=str, default="test", help="which split to evaluate on: test (default, unchanged behaviour) | train. Used to inspect the train/test gap")
_pre.add_argument("--fm_rate_pow", type=float, default=1.0, help="the realtime scheme divides by dt**p; must match the value used at training time")
_pre.add_argument("--fm_gbar", type=str, default="", help="population-prior npz, paired with --fm_gbar on the training side")
_pre.add_argument("--fm_pca_basis", type=str, default="", help="delta principal-component basis npz, fitted on the training split only")
_pre.add_argument("--fm_pca_k", type=int, default=0, help="project onto the first K principal components; 0=off")
_pre.add_argument("--fm_seed", type=int, default=-1, help="fix the sampling noise so that different deltas and inference settings under one checkpoint share the same noise realization, making per-case comparisons paired. -1=unseeded")
_pre.add_argument("--fm_hist_ckpt", type=str, default="", help="history encoder ckpt (fm-hist-ep-N); auto-derived from --fm_ckpt if empty")
_pre.add_argument("--fm_postmask_thr", type=float, default=-1.0, help="INSTANT oracle-localization ceiling probe: at eval, zero predicted delta OUTSIDE the true change region (|z1-z0| normalized > thr). No training. -1=off.")
_pre.add_argument("--fm_noise_temp", type=float, default=1.0, help="temperature on the INITIAL noise: z ~ N(0, tau^2 I). Unlike --fm_delta_scale (which rescales the output and its error alike), this travels the map the model actually learned, yielding a different plausible future rather than an inflated copy of the same one")
_pre.add_argument("--fm_t_end_dt", type=float, default=0.0, help="PER-SAMPLE endpoint tied to the TARGET TIME (paper formulation): T_i = clamp(dt_years_i / ref, 0.15, 1.0) with ref = this value (e.g. 6.0 years). Longer follow-up integrates further along the path. 0 = off")
_pre.add_argument("--fm_t_end", type=float, default=1.0, help="integrate only over [0,T] then JUMP to the endpoint via delta_hat = z_T + (1-T)*v(z_T,T). Skips the t->1 region, where the velocity target is hardest to fit. 1.0 = normal full integration")
_pre.add_argument("--fm_step_power", type=float, default=1.0, help="sampling step schedule: s=u**(1/p); p>1 concentrates steps near t=1 where the velocity field changes fastest")
_pre.add_argument("--fm_cnet_ckpt", type=str, default="", help="controlnet ckpt (fm-cnet-ep-N); auto-derived from --fm_ckpt if empty")
_pre.add_argument("--n_eval", type=int, default=40)
_pre.add_argument("--eval_bs", type=int, default=2)
_pre.add_argument("--infer_steps", type=int, default=50)
_pre.add_argument("--n_avg", type=int, default=1)
_pre.add_argument("--fm_antithetic", type=int, default=0, help="antithetic sampling: draw the n_avg samples in pairs of z and -z, so the perpendicular noise component cancels to first order (independent sampling only reduces it as 1/sqrt(n)). 0=off")
_pre.add_argument("--fm_subspace", type=str, default="")
_pre.add_argument("--fm_delta_scale", type=float, default=1.0)
_pre.add_argument("--fm_hist_concat", type=int, default=0, help="history channels concatenated into backbone input (20ch)")
_pre.add_argument("--fm_hist_mode", type=str, default="none", help="components joined by +: first/prev1/prev2/energy/traj")
_pre.add_argument("--fm_hist_inject", type=str, default="concat", help="concat|cond|both|gain|cnet|cnetft")
_pre.add_argument("--fm_gain_mode", type=str, default="none", help="components routed through the spatial-modulation path")
_pre.add_argument("--fm_prev2_fill", type=str, default="zero", help="zero|const: when prev2 is missing, use v2=0 or v2<-v1 (constant velocity)")
_pre.add_argument("--fm_flowinit", type=str, default="none")
_pre.add_argument("--fm_flowinit_sigma", type=float, default=None, help="flow-initialization noise scale. Must equal the training value; if omitted it is read from the checkpoint's args.json (falls back to 1.0 with a warning)")
_pre.add_argument("--fm_flowinit_w", type=float, default=1.0)
_pre.add_argument("--fm_mag_norm", type=str, default="", help="decouple amplitude from direction: oracle uses the true ||delta|| (upper bound), pred uses the regression-predicted ||delta||. The direction still comes from the flow-matching model")
_pre.add_argument("--fm_cnet_hist", type=str, default="none")
_pre.add_argument("--fm_delta_pow", type=float, default=1.0)
_pre.add_argument("--fm_postmask_src", type=str, default="oracle", help="postmask source: oracle | ek (cumulative E_last) | ekind (population mean removed) | arate (a_E, the energy growth rate) | egrow (change added in the last year) | eunion (union across timepoints) | energy")
_pre.add_argument("--fm_postmask_mode", type=str, default="quantile", help="quantile: keep the top (1-thr) fraction of voxels by mask value, which is insensitive to outliers | absmax: divide by the maximum and compare against the threshold (sensitive to outliers)")
_pre.add_argument("--fm_postmask_fixed", type=int, default=0, help="control: use the energy map of the first case for every case, i.e. another subject's map. A result equal to the individualized map indicates the gain comes from fixed anatomical weighting rather than individualization")
_pre.add_argument("--fm_postmask_soft", type=float, default=0.0, help="soft-mask strength alpha: z *= (1-a + a*m). A hard mask zeroes low-energy regions and discards their information, while a soft mask only downweights them. 0=hard mask")
_pre.add_argument("--raw_out", type=int, default=0, help="1: also produce and score the image-space output in the original scan space (see README, Evaluation standard): pred_raw = x0_raw + a*(up(pred) - up(x0)), with a, b fitted x0_raw ~ a*up(x0) + b inside the brain")
_pre.add_argument("--raw_canonical", type=str, default="128,144,128", help="canonical raw grid (voxels at 1.5 mm) that the latent grid was resized from")
_pre.add_argument("--dump_pred", type=str, default="", help="write (x0, prediction, ground truth) images to this directory for eval_panel.py to compute the full metric panel")
_pre.add_argument("--fm_tail_ridge", type=str, default="", help="inference-time tail replacement: the z0 ridge-regression npz. Must be declared on the pre-parser above so it is visible before the model is built")
_pre.add_argument("--fm_tail_ridge_w", type=float, default=1.0, help="tail-replacement weight: 1=replace fully")
_pre.add_argument("--fm_tail_ridge_mm", type=int, default=0, help="1=amplitude matching (preserve the norm and replace direction only)")
_pre.add_argument("--fm_cascade", type=str, default="", help="coarse-to-fine cascade: predict the entire delta from the cascade npz, routed through the same eval path so the metric protocol matches the other arms exactly")
_pre.add_argument("--fm_residual_add", type=str, default="", help="residual flow-matching inference: add the linear term W.z0 back to the model output. Required whenever --fm_residual_ridge was used during training. Must be declared on the pre-parser above")
_pre.add_argument("--fm_tail_ridge_full", type=int, default=0, help="1=predict the entire delta with ridge regression (the pure-regression arm)")
_pre.add_argument("--tag", type=str, default="cfg")
_pre.add_argument("--fm_energy_wavg", type=float, default=0.0, help="score and weight the n_avg candidates by energy (temperature; 0=off, reducing to equal weights)")
_eval_args, _rest = _pre.parse_known_args()
sys.argv = [sys.argv[0]] + _rest


def _train_args_of(ckpt):
    """Training arguments saved next to a checkpoint (args.json), or None."""
    for _p in (os.path.join(os.path.dirname(str(ckpt or "")), "args.json"),
               os.path.join("ae_runs", os.path.basename(os.path.dirname(str(ckpt or ""))).replace("_rectified", ""), "args.json")):
        if os.path.exists(_p):
            return json.load(open(_p))
    return None


if _eval_args.fm_flowinit_sigma is None:
    _ta = _train_args_of(getattr(_eval_args, "fm_ckpt", None)) or {}
    if _ta.get("fm_flowinit_sigma") is not None:
        _eval_args.fm_flowinit_sigma = float(_ta["fm_flowinit_sigma"])
        print("[eval] fm_flowinit_sigma=%.4f read from the checkpoint's training arguments" % _eval_args.fm_flowinit_sigma, flush=True)
    else:
        _eval_args.fm_flowinit_sigma = 1.0
        print("[eval] WARNING: fm_flowinit_sigma not given and not found in args.json; using 1.0. "
              "A mismatch with the training value changes the sampled amplitude.", flush=True)

from utils.options import args
# Flags from the pre-parser (_pre) are copied onto args, otherwise they are not visible when the model is built
# and the input-channel count can disagree with the checkpoint (conv_in shape mismatch on load)
for _k, _v in vars(_eval_args).items():
    if _v is not None and (not hasattr(args, _k) or _k in ('fm_hist_concat', 'fm_hist_mode', 'fm_hist_inject', 'fm_gain_mode', 'fm_flowinit', 'fm_flowinit_w', 'fm_flowinit_sigma', 'fm_spade_mode', 'fm_timewarp', 'fm_xattn_mode', 'fm_prev2_fill', 'fm_cnet_hist')):
        setattr(args, _k, _v)
from utils import import_from_dotted_path
from utils.utils_metric import compute_change_metrics
from src.change_f1 import change_f1
from src.delta_f1 import delta_f1_metrics, region_f1_metrics

# Derived-artifact root (population priors, PCA bases, energy maps).
DERIVED_DIR = os.environ.get("DERIVED_DIR",
                             os.path.join(os.environ.get("DATA_DIR", "."), "derived"))


DEVICE = "cuda"


def _seed_all(sd):
    """Fix the sampling noise so that different deltas and inference settings under one checkpoint share the same noise realization, making per-case comparisons paired.
    sd < 0 does not seed."""
    if sd is None or int(sd) < 0:
        return
    import random as _rd
    sd = int(sd)
    _rd.seed(sd); np.random.seed(sd)
    torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)
torch.backends.cudnn.benchmark = True


from collections import OrderedDict

# ===== continuous conditioning vector =====
# context[:, :8] = [starting_age, sex, starting_diagnosis, cortex, hippo, amygdala, wm, ventricle]
# sex is dropped when the column is constant, and any followup_diagnosis-based difference is
# excluded because it leaks the target; dt is kept CONTINUOUS rather than discretized.
_COND_IDX = [0, 1, 2, 3, 4, 5, 6, 7]       # age, sex (0/1), diagnosis, 5 volumes
_COND_STATS = {}
def _cond_stats(args):
    global _COND_STATS
    if _COND_STATS: return _COND_STATS
    import pandas as pd, numpy as _np
    d = pd.read_csv(args.dataset_csv)
    if "sex" not in d.columns or d["sex"].nunique() <= 1: d["sex"] = 0.5
    tr = d.iloc[:int(0.8 * len(d))]
    cols = ["starting_age", "sex", "starting_diagnosis", "starting_cerebral_cortex", "starting_hippocampus",
            "starting_amygdala", "starting_cerebral_white_matter", "starting_lateral_ventricle"]
    V = tr[cols].fillna(0).to_numpy(dtype="float32")
    dt = ((tr["followup_age"] - tr["starting_age"]) * 100).to_numpy(dtype="float32")
    F = _np.concatenate([dt[:, None], _np.log1p(_np.clip(dt, 0, None))[:, None], V], 1)
    _COND_STATS = {"mean": F.mean(0), "std": F.std(0) + 1e-6}
    return _COND_STATS

def build_cond_vec(batch, context, DEVICE, args):
    """-> (B, 10) standardized [dt, log1p(dt), age, diagnosis, 5 volumes]"""
    import torch as _t
    st = _cond_stats(args)
    cov = context[:, 0, :8][:, _COND_IDX]                                   # (B,7)
    dt = ((batch["followup_age"] - batch["starting_age"]).to(DEVICE).float() * 100.0).view(-1, 1)
    f = _t.cat([dt, _t.log1p(dt.clamp_min(0)), cov.to(DEVICE).float()], 1)  # (B,9)
    mu = _t.tensor(st["mean"], device=DEVICE).view(1, -1)
    sd = _t.tensor(st["std"], device=DEVICE).view(1, -1)
    return ((f - mu) / sd).float()



def build_multi_cond(batch, x0, scale_factor, DEVICE):
    """Multi-timepoint injected control signal (16 channels):
       [x0 | v1=(x0-p1)/dt1 | a=(v1-v2) | dt1,dt2,has1,has2 broadcast maps]
       v and a are the first and second derivatives of the individual trajectory; once the population
       drift is removed, that is where the patient-specific signal is concentrated.
       Channels for missing history are zeroed, and the validity-bit channels tell the network which ones."""
    import torch as _t
    B = x0.shape[0]; sp = x0.shape[2:]
    p1 = batch["prior_latent"].to(DEVICE).float() * scale_factor
    p2 = batch.get("prior2_latent", batch["prior_latent"]).to(DEVICE).float() * scale_factor
    h1 = batch.get("has_prior", _t.ones(B)).to(DEVICE).float().view(B,1,1,1,1)
    h2 = batch.get("has_prior2", _t.zeros(B)).to(DEVICE).float().view(B,1,1,1,1)
    fu0 = batch["starting_follow_up"].to(DEVICE).float().view(B,1,1,1,1)
    f1  = batch.get("prior_follow_up",  batch["starting_follow_up"]).to(DEVICE).float().view(B,1,1,1,1)
    f2  = batch.get("prior2_follow_up", batch["starting_follow_up"]).to(DEVICE).float().view(B,1,1,1,1)
    d1 = ((fu0 - f1) / 12.0).clamp_min(0.25)      # years
    d2 = ((f1  - f2) / 12.0).clamp_min(0.25)
    v1 = ((x0 - p1) / d1) * h1                     # individual velocity (4ch)
    v2 = ((p1 - p2) / d2) * h2
    acc = (v1 - v2) * h2                           # individual acceleration (4ch)
    o = _t.ones(B,1,*sp, device=DEVICE, dtype=x0.dtype)
    meta = _t.cat([o*(d1*h1), o*(d2*h2), o*h1, o*h2], dim=1)   # 4ch
    return _t.cat([x0, v1, acc, meta], dim=1)                  # 16ch


_COMP_CH = {"first": 6, "prev1": 6, "prev2": 6, "energy": 4, "traj": 4, "accel": 6,
            "trajc": 13, "etraj": 6, "ietraj": 6, "pdrop": 3,
            "ek": 1, "ekind": 1, "ek2": 2, "egrid": 5, "vfin": 2, "vslope": 1, "anat": 4}   # ietraj/pdrop are image-space (native resolution) and cannot be represented in the latent
_LEGACY = {"none": "", "base": "first", "prev": "prev1", "both": "prev1+prev2",
           "energy": "energy", "base_energy": "first+energy",
           "prev_energy": "prev1+energy", "both_energy": "prev1+prev2+energy",
           "all": "traj", "all_energy": "traj+energy", "all_both": "traj+prev1+prev2"}
def _comps(mode):
    m = str(mode)
    m = _LEGACY.get(m, m)
    return [c for c in m.split("+") if c]
def hist_channels(mode):
    return sum(_COMP_CH.get(c, 0) for c in _comps(mode))

def build_hist_cond(batch, x0, scale_factor, DEVICE, mode):
    """Composable history conditions. Any of first/prev1/prev2/energy/traj can be joined with +.
       All components follow the same code path, so the only variable is which components are used."""
    import torch as _t
    B = x0.shape[0]; sp = x0.shape[2:]
    one = _t.ones(B, 1, *sp, device=DEVICE, dtype=x0.dtype)
    def _s(k, d=None):
        v = batch.get(k, d)
        return v.to(DEVICE).float().view(B, 1, 1, 1, 1) if v is not None else None
    def _ones5(): return _t.ones(B, 1, 1, 1, 1, device=DEVICE)
    fu0 = _s("starting_follow_up")
    _FILL = batch.get("_prev2_fill", "zero")
    parts = []
    for c in _comps(mode):
        if c == "first":
            zf = batch["first_latent"].to(DEVICE).float() * scale_factor
            hf = _s("has_first");  hf = _ones5() if hf is None else hf
            ff = _s("first_follow_up", batch["starting_follow_up"])
            T = ((fu0 - ff) / 12.0).clamp_min(0.25)
            parts += [((x0 - zf) / T) * hf, one * (T * hf), one * hf]
        elif c == "prev1":
            p1 = batch["prior_latent"].to(DEVICE).float() * scale_factor
            h1 = _s("has_prior");  h1 = _ones5() if h1 is None else h1
            f1 = _s("prior_follow_up", batch["starting_follow_up"])
            d1 = ((fu0 - f1) / 12.0).clamp_min(0.25)
            parts += [((x0 - p1) / d1) * h1, one * (d1 * h1), one * h1]
        elif c == "accel":
            # Acceleration v1-v2: the flow-initialization prior already uses v1 (first order), and what a linear
            # extrapolation lacks is exactly the second-order term. Supplying only what the prior lacks
            # avoids duplicating information the prior already carries (as prev1 would).
            p1 = batch["prior_latent"].to(DEVICE).float() * scale_factor
            p2 = batch.get("prior2_latent", batch["prior_latent"]).to(DEVICE).float() * scale_factor
            h1 = _s("has_prior");  h1 = _ones5() if h1 is None else h1
            h2 = _s("has_prior2"); h2 = h2 if h2 is not None else _t.zeros(B,1,1,1,1, device=DEVICE)
            f1 = _s("prior_follow_up", batch["starting_follow_up"])
            f2 = _s("prior2_follow_up", batch["starting_follow_up"])
            d1 = ((fu0 - f1) / 12.0).clamp_min(0.25); d2 = ((f1 - f2) / 12.0).clamp_min(0.25)
            _v1 = ((x0 - p1) / d1) * h1; _v2 = ((p1 - p2) / d2) * h2
            parts += [(_v1 - _v2) * h2, one * (d2 * h2), one * h2]
        elif c == "prev2":
            p1 = batch["prior_latent"].to(DEVICE).float() * scale_factor
            p2 = batch.get("prior2_latent", batch["prior_latent"]).to(DEVICE).float() * scale_factor
            h2 = _s("has_prior2"); h2 = h2 if h2 is not None else _t.zeros(B,1,1,1,1, device=DEVICE)
            f1 = _s("prior_follow_up", batch["starting_follow_up"])
            f2 = _s("prior2_follow_up", batch["starting_follow_up"])
            d2 = ((f1 - f2) / 12.0).clamp_min(0.25)
            _v2r = (p1 - p2) / d2
            if str(_FILL) == "const":
                # When p2 is missing, v2 <- v1 (zero acceleration, i.e. constant velocity) rather than v2=0, which would imply the subject was previously static with very large acceleration
                d1c = ((fu0 - f1) / 12.0).clamp_min(0.25)
                _v1r = (x0 - p1) / d1c
                _v2r = _v2r * h2 + _v1r * (1.0 - h2)
                parts += [_v2r, one * (d2 * h2 + d1c * (1.0 - h2)), one * h2]
            else:
                parts += [_v2r * h2, one * (d2 * h2), one * h2]
        elif c == "energy":
            parts += [batch["cum_energy"].to(DEVICE).float()]
        elif c == "traj":
            parts += [batch["traj_fit"].to(DEVICE).float()]
        elif c == "trajc":
            parts += [batch["trajc"].to(DEVICE).float()]      # 4x[a,b,r]+cov, channels not compressed
        elif c == "etraj":
            parts += [batch["etraj"].to(DEVICE).float()]
        elif c == "vfin":
            # Final energy form (pooled to the latent): [magnitude term |D_K| x edge, temporal-consistency term C2 x edge]
            # The two channels are supplied separately rather than pre-summed, so the network learns their weighting
            parts += [batch["vfin"].to(DEVICE).float()]
        elif c == "vslope":
            # Per-voxel time slope |a| x edge (the intercept b absorbs the patient's intrinsic anatomical
            # pattern). This branch must stay in sync with the training side, otherwise the ControlNet
            # condition is None under --fm_cnet_hist vslope.
            if not globals().get("_VSLOPE_EVAL_SHOWN"):
                globals()["_VSLOPE_EVAL_SHOWN"] = 1
                print("[vslope] component connected on the eval side", flush=True)
            parts += [batch["vslope"].to(DEVICE).float()]
        elif c == "anat":
            # Anatomical anchor: energy indicates where change occurs and x0 indicates what tissue is there.
            # Without anatomical information, high energy cannot distinguish real atrophy from registration artifacts or tissue boundaries.
            if not globals().get("_ANAT_SHOWN"):
                globals()["_ANAT_SHOWN"] = 1
                print("[anat] anatomical anchor added to the ControlNet condition (x0, 4ch)", flush=True)
            parts += [x0]
        elif c == "egrid":
            # Energy sequence resampled onto a fixed time grid [E(t0), E(-1y), E(-2y), E(-3y), cov]; no fitting, so the sequence shape is preserved
            parts += [batch["egrid"].to(DEVICE).float()]
        elif c in ("ek", "ekind", "ek2"):
            _et = batch.get("etraj", None)
            if _et is None:
                _sl = torch.zeros((x0.shape[0], _COMP_CH[c]) + tuple(x0.shape[2:]),
                                  device=DEVICE, dtype=torch.float32)
            else:
                _et = _et.to(DEVICE).float()
                _idx = {"ek": [0], "ekind": [4], "ek2": [0, 4]}[c]
                _sl = _et[:, _idx]
            parts += [_sl]
        elif c == "ietraj":
            parts += [batch["ietraj"].to(DEVICE).float()]   # native-resolution energy sequence
        elif c == "pdrop":
            parts += [batch["pdrop"].to(DEVICE).float()]    # local decorrelation only (not a per-voxel statistic)      # [E_last,a_E,b_E,r_E,E_ind,cov]
    return _t.cat(parts, dim=1) if parts else None


def _build_gain_net(in_ch, DEVICE):
    """History -> per-voxel gain field g = 1 + f(hist), with the final layer zero-initialized so the starting point g=1 matches the baseline.
       The gain field expresses directly where this patient should change more or less than the population average, and g itself visualizes as an individualized progression-rate map."""
    import torch.nn as _nn
    _last = _nn.Conv3d(32, 4, 3, padding=1)
    _nn.init.zeros_(_last.weight); _nn.init.zeros_(_last.bias)
    return _nn.Sequential(_nn.Conv3d(in_ch, 32, 3, padding=1), _nn.SiLU(), _last).to(DEVICE)


def flow_init_mu(batch, x0, scale_factor, DEVICE, mode, dt_years):
    """Flow-initialization prior mu: start the flow from the linear extrapolation given by the history rather than from pure noise.
       z ~ N(mu, sigma^2 I),  mu = v1 * dt_target  (v1 = (x0-p1)/dt1, the recent velocity)
       The task of the flow becomes correcting the error of a linear extrapolation rather than generating the whole delta from noise.
       A diffusion model requires a standard Gaussian prior; this degree of freedom is specific to flow matching."""
    import torch as _t
    if str(mode) == "none": return None
    B = x0.shape[0]
    p1 = batch["prior_latent"].to(DEVICE).float() * scale_factor
    h1 = batch.get("has_prior", None)
    h1 = (h1.to(DEVICE).float().view(B,1,1,1,1) if h1 is not None else _t.ones(B,1,1,1,1, device=DEVICE))
    fu0 = batch["starting_follow_up"].to(DEVICE).float().view(B,1,1,1,1)
    f1 = batch.get("prior_follow_up", batch["starting_follow_up"]).to(DEVICE).float().view(B,1,1,1,1)
    d1 = ((fu0 - f1) / 12.0).clamp_min(0.25)
    v1 = ((x0 - p1) / d1) * h1                      # recent velocity (per year)
    return v1 * dt_years.view(B,1,1,1,1)            # linear extrapolation to the target time


def remove_module_prefix(sd):
    out = OrderedDict()
    for k, v in sd.items():
        out[k[7:] if k.startswith("module.") else k] = v
    return out


def psnr(a, b, data_range=None):
    # decoded images live in per-volume normalized space (~[-1,+3]); derive the
    # data_range from the target rather than assuming [0,1], else PSNR is crushed.
    if data_range is None:
        data_range = float(b.max() - b.min()) or 1.0
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


def ssim3d(a, b, dr):
    """3D SSIM via the MONAI Gaussian sliding window (win=11, sigma=1.5).
    A single global window (one mean/var/cov for the whole volume) would, on nearly identical images,
      reduce to a monotone function of MSE and could not see structural change at the scale of the change itself."""
    import numpy as _np, torch as _t
    from monai.metrics import SSIMMetric as _SM
    _c = globals().setdefault("_SSIM3D_CACHE", {})
    k = round(float(dr), 4)
    if k not in _c:
        _c[k] = _SM(spatial_dims=3, data_range=float(dr), kernel_type="gaussian",
                    win_size=11, kernel_sigma=1.5)
    ta = a if _t.is_tensor(a) else _t.from_numpy(_np.ascontiguousarray(a))
    tb = b if _t.is_tensor(b) else _t.from_numpy(_np.ascontiguousarray(b))
    ta = ta.float(); tb = tb.float()
    while ta.dim() < 5: ta = ta.unsqueeze(0)
    while tb.dim() < 5: tb = tb.unsqueeze(0)
    return float(_c[k](ta, tb).mean())


def drmae(pred, target, input, brain):
    dgt = (target - input)[brain]
    dgn = (pred - input)[brain]
    return float(np.abs(dgt - dgn).sum() / (np.abs(dgt).sum() + 1e-6))


_MAGREP = []
_SPLITREP = []
_BANDREP = []
_SAMPS = []
_SAMPSTAT = []
flow_init_batch = None
flow_init_dt = None

@torch.no_grad()
def sample(flowmatching, autoencoder, x0, context, class_cond, scale_factor,
           scheme, dt_vec, steps, _gain_cond=None, res_noise=False, x0_cond=False, x0_start_noise=0.0, cfg_w=0.0, n_avg=1, subspace=None, delta_scale=1.0, energy=None, postmask=None, controlnet=None, cnet_prior=None, cond_vec=None, hist_cat=None, return_latent=False):
    """x0: raw starting latent [B,4,...]. Returns decoded followup-pred image [B,1,...] on CPU."""
    x0s = x0 * scale_factor
    _p = float(getattr(_eval_args, "fm_step_power", 1.0))
    _tend = float(getattr(_eval_args, "fm_t_end", 1.0))
    s_steps = torch.linspace(0.0, 1.0, steps + 1, device=DEVICE)
    if _p != 1.0: s_steps = s_steps ** (1.0 / _p)   # p>1 -> denser near t=1
    if _tend != 1.0: s_steps = s_steps * _tend      # integrate only up to T
    _tref = float(getattr(_eval_args, "fm_t_end_dt", 0.0))
    _Tvec = None
    if _tref > 0 and dt_vec is not None:
        # dt_vec = follow_up interval / 10 (months/10) -> years = dt_vec*10/12
        _yrs = dt_vec.float() * 10.0 / 12.0
        _Tvec = (_yrs / _tref).clamp(0.15, 1.0).view(-1, 1, 1, 1, 1)   # (B,1,1,1,1)
    if res_noise:
        # Δ-Res-Flow: integrate NOISE -> delta, conditioned on x0 (concat); x1 = x0 + delta
        _acc = None
        _zprev = None
        for _r in range(max(1, int(n_avg))):                         # 0a: multi-sample averaging in latent
            if int(getattr(_eval_args, "fm_antithetic", 0)) and (_r % 2 == 1) and _zprev is not None:
                z = -_zprev            # antithetic: paired with the previous draw, so the perpendicular perturbation cancels to first order
            else:
                if int(getattr(_eval_args, "fm_det", 0)):
                    z = torch.zeros_like(x0s)    # deterministic rollout: no noise, so n_avg and antithetic sampling have no effect
                else:
                    z = torch.randn_like(x0s) * float(getattr(_eval_args, "fm_noise_temp", 1.0)) * float(getattr(_eval_args, "fm_flowinit_sigma", 1.0))
                _zprev = z.clone()     # clone because z may be modified in place below
            _Z0 = z.clone()
            _GFIELD = None
            if getattr(flowmatching, "gain_net", None) is not None and _gain_cond is not None:
                _GFIELD = 1.0 + flowmatching.gain_net(_gain_cond.float())
                if int(getattr(args, 'fm_gain_norm', 0)):
                    # Total-preserving constraint (must match training): g <- g/mean(g), which redistributes without changing the total magnitude
                    _rd = tuple(range(1, _GFIELD.ndim))
                    _GFIELD = _GFIELD / _GFIELD.mean(dim=_rd, keepdim=True).clamp_min(1e-6)
                    if not globals().get('_GNORM_EVAL'):
                        globals()['_GNORM_EVAL']=1
                        print('[gain_norm] total-preserving constraint enabled on the eval side', flush=True)
            if str(getattr(_eval_args, "fm_flowinit", "none")) != "none" and flow_init_batch is not None:
                _m0 = flow_init_mu(flow_init_batch, x0s, 1.0, DEVICE, str(_eval_args.fm_flowinit), flow_init_dt)
                if _m0 is not None: z = z + _m0 * float(getattr(_eval_args, "fm_flowinit_w", 1.0))
            for i in range(steps):
                s = s_steps[i]; ds = (s_steps[i + 1] - s); s_b = s.expand(z.shape[0])
                if _Tvec is not None:
                    s_b = (s * _Tvec.view(-1)); ds = ds * _Tvec        # per-sample endpoint T_i
                _mi = torch.cat([z, x0s] + ([energy] if energy is not None else []) + ([hist_cat] if hist_cat is not None else []), dim=1).float()
                if cfg_w > 0:
                    _vc = flowmatching(x=_mi, timesteps=s_b, context=context, class_labels=class_cond)
                    _vn = flowmatching(x=_mi, timesteps=s_b, context=context, class_labels=torch.full_like(class_cond, 15))
                    v = _vn + cfg_w * (_vc - _vn)
                elif float(getattr(args, "fm_cond_w", 0.0)) > 0 and cond_vec is not None:
                    _w = float(args.fm_cond_w)
                    _vc = flowmatching(x=_mi, timesteps=s_b, context=context, class_labels=class_cond, cond_vec=cond_vec)
                    _vn = flowmatching(x=_mi, timesteps=s_b, context=context, class_labels=class_cond, cond_vec=torch.zeros_like(cond_vec))
                    v = _vn + _w * (_vc - _vn)
                elif controlnet is not None:
                    _dh, _mh = controlnet(x=_mi, timesteps=s_b, controlnet_cond=cnet_prior, context=context, class_labels=class_cond)
                    _cw = float(getattr(args, "fm_cnet_w", 1.0))
                    if _cw != 1.0:
                        _dh = [h * _cw for h in _dh] if _dh is not None else _dh
                        _mh = _mh * _cw if _mh is not None else _mh
                        if not globals().get("_CNETW_SHOWN"):
                            globals()["_CNETW_SHOWN"] = 1
                            print(f"[cnet_w] ControlNet residual scaling w={_cw}", flush=True)
                    v = flowmatching(x=_mi, timesteps=s_b, context=context, class_labels=class_cond, cond_vec=cond_vec, down_block_additional_residuals=_dh, mid_block_additional_residual=_mh)
                else:
                    v = flowmatching(x=_mi, timesteps=s_b, context=context, class_labels=class_cond, cond_vec=cond_vec)
                if int(getattr(args, "fm_x0_pred", 0)):
                    # The model outputs delta_hat; the implied noise is recovered as z_hat=(xt-t*d)/(1-t), giving the equivalent velocity v=d-z_hat
                    _om = (1.0 - s).clamp_min(1e-3)
                    v = v - (z - s * v) / _om
                if _GFIELD is not None:
                    v = _GFIELD * (v + _Z0) - _Z0        # same form as training: delta' = g*delta
                z = z + v * ds                                       # z: noise -> delta
            if _Tvec is not None:
                _sTv = (s_steps[-1] * _Tvec.view(-1))
                _miT = torch.cat([z, x0s] + ([energy] if energy is not None else []) + ([hist_cat] if hist_cat is not None else []), dim=1).float()
                _vT = flowmatching(x=_miT, timesteps=_sTv, context=context, class_labels=class_cond, cond_vec=cond_vec)
                z = z + (1.0 - _Tvec) * _vT
            elif _tend != 1.0:
                # Terminal jump: take one step to the endpoint using the velocity at T, bypassing the unreliable region as t approaches 1
                _sT = s_steps[-1]
                _miT = torch.cat([z, x0s] + ([energy] if energy is not None else []) + ([hist_cat] if hist_cat is not None else []), dim=1).float()
                _vT = flowmatching(x=_miT, timesteps=_sT.expand(z.shape[0]), context=context,
                                   class_labels=class_cond, cond_vec=cond_vec)
                z = z + (1.0 - _sT) * _vT
            _mn = str(getattr(_eval_args, "fm_mag_norm", "") or "")
            if _mn:
                _gtl = globals().get('_GT_LATENT', None)
                _tgt = None
                if _mn == "oracle" and _gtl is not None:
                    _tgt = (_gtl - x0s).flatten(1).norm(dim=1)        # true ||delta|| (upper bound)
                elif _mn == "pred":
                    _mp = globals().get('_MAG_PRED', None)
                    if isinstance(_mp, tuple) and _mp[0] == "rel":
                        _tgt = z.flatten(1).norm(dim=1) * _mp[1]      # relative calibration
                    else:
                        _tgt = _mp
                if _tgt is not None:
                    _cur = z.flatten(1).norm(dim=1).clamp_min(1e-8)
                    z = z * (_tgt / _cur).view(-1, *([1] * (z.dim() - 1)))
            # ===== (CASCADE) coarse-to-fine cascade prediction, replacing delta entirely =====
            # Mirrors run_edges in build_cascade_v2.py exactly. It runs through the same eval code path,
            # so the metric protocol matches the flow-matching and pure-ridge arms exactly, which is what makes them comparable.
            _cs_npz = str(getattr(_eval_args, "fm_cascade", "") or getattr(args, "fm_cascade", "") or "")
            if _cs_npz and not os.path.exists(_cs_npz):
                raise FileNotFoundError("[cascade] model file not found: %s" % _cs_npz)
            if _cs_npz:
                _dvc = z.device
                if "_CASC" not in globals():
                    import numpy as _np7
                    _z7 = _np7.load(_cs_npz)
                    _C = {k: torch.from_numpy(_z7[k]).float().to(_dvc)
                          for k in ("PC", "d_mu", "z_mu", "Pz", "fs", "Xfit")}
                    _C["edges"] = [int(v) for v in _z7["edges"]]
                    _C["nb"] = len(_C["edges"]) - 1
                    _C["bands"] = []
                    for _i in range(_C["nb"]):
                        _kd = str(_z7["b%d_kind" % _i][0])
                        _d = {"kind": _kd, "s": float(_z7["b%d_s" % _i]),
                              "alpha": float(_z7["b%d_alpha" % _i]), "r": float(_z7["b%d_r" % _i])}
                        if _kd == "lin":
                            _d["W"] = torch.from_numpy(_z7["b%d_W" % _i]).float().to(_dvc)
                        else:
                            _d["al"] = torch.from_numpy(_z7["b%d_alpha_dual" % _i]).float().to(_dvc)
                            _d["g"] = float(_z7["b%d_gamma" % _i])
                        _C["bands"].append(_d)
                    # Rebuild the per-stage appended features recursively on the fitting set (the bundle stores only the initial Xfit)
                    _Xf = _C["Xfit"]
                    for _d in _C["bands"]:
                        _d["Xfit_i"] = _Xf                       # fitting features used by the kernel method at this stage
                        if _d["kind"] == "lin":
                            _pf = torch.cat([_Xf, torch.ones(_Xf.shape[0], 1, device=_dvc)], 1) @ _d["W"]
                        else:
                            _sq = (_Xf.pow(2).sum(1)[:, None] + _Xf.pow(2).sum(1)[None, :]
                                   - 2 * _Xf @ _Xf.t()).clamp_min(0)
                            _pf = torch.exp(-_d["g"] * _sq) @ _d["al"]
                        _Xf = torch.cat([_Xf, (_pf / _pf.norm(dim=1, keepdim=True).clamp_min(1e-6))[:, :8]], 1)
                    globals()["_CASC"] = _C
                    print("[cascade] loaded %s | %d stages edges=%s alpha=%s"
                          % (_cs_npz, _C["nb"], _C["edges"],
                             [round(_b["alpha"], 2) for _b in _C["bands"]]), flush=True)
                _C = globals()["_CASC"]
                _x0c = (x0s / scale_factor).flatten(1)
                _X = ((_x0c - _C["z_mu"]) @ _C["Pz"].t()) / _C["fs"]
                _out = torch.zeros(_X.shape[0], _C["edges"][-1], device=_dvc)
                for _i, _d in enumerate(_C["bands"]):
                    _lo, _hi = _C["edges"][_i], _C["edges"][_i + 1]
                    _Xf_i = _d["Xfit_i"]
                    if _d["kind"] == "lin":
                        _pp = torch.cat([_X, torch.ones(_X.shape[0], 1, device=_dvc)], 1) @ _d["W"]
                    else:
                        _sq = (_X.pow(2).sum(1)[:, None] + _Xf_i.pow(2).sum(1)[None, :]
                               - 2 * _X @ _Xf_i.t()).clamp_min(0)
                        _pp = torch.exp(-_d["g"] * _sq) @ _d["al"]
                    _out[:, _lo:_hi] = _d["s"] * _pp
                    _X = torch.cat([_X, (_pp / _pp.norm(dim=1, keepdim=True).clamp_min(1e-6))[:, :8]], 1)
                _dc = (_C["d_mu"] + _out @ _C["PC"]) * scale_factor
                if not globals().get("_CASC_SHOWN"):
                    globals()["_CASC_SHOWN"] = 1
                    print("[cascade] active | ||FM delta||=%.3f -> ||cascade delta||=%.3f"
                          % (z.flatten(1).norm(dim=1).mean().item(),
                             _dc.norm(dim=1).mean().item()), flush=True)
                    # Protocol-alignment diagnostic: the cascade and ridge models are both fitted on the raw npz latents.
                    #   If ||z0|| here differs from ||z0|| read directly from the npz with numpy, the latents fed by eval
                    #   have been transformed further, and both models would be operating outside their training distribution.
                    print("[cascade-DIAG] ||x0s||=%.3f ||x0s/sf||=%.3f sf=%.5f"
                          % (x0s.flatten(1).norm(dim=1).mean().item(),
                             _x0c.norm(dim=1).mean().item(), float(scale_factor)), flush=True)
                z = _dc.reshape_as(z)

            # ===== (RESIDUAL-ADD) inference path for residual flow matching: delta_hat = FM(residual) + W.z0 =====
            # step3_residual_fm.py changes the training target to delta - W.z0, so the linear term must be added back at inference.
            # The formula matches the training side exactly (fitted on unscaled latents, so divide by scale_factor first and multiply the result back).
            # The linear part comes entirely from the ridge regression and the model only has to supply the
            #       residual, so a zero residual contribution reduces exactly to pure ridge regression.
            _ra_npz = str(getattr(_eval_args, "fm_residual_add", "") or getattr(args, "fm_residual_add", "") or "")
            if _ra_npz and not os.path.exists(_ra_npz):
                raise FileNotFoundError("[residual_add] model file not found: %s" % _ra_npz)
            if _ra_npz:
                if "_RADD" not in globals():
                    import numpy as _np8
                    _z8 = _np8.load(_ra_npz)
                    globals()["_RADD"] = {k: torch.from_numpy(_z8[k]).float()
                                          for k in ("B", "Pz", "z_mu", "fs", "Wd", "d_mu")}
                    print("[residual_add] loaded %s" % _ra_npz, flush=True)
                _RA = globals()["_RADD"]; _dv8 = z.device
                _x0r8 = (x0s / scale_factor).flatten(1)
                _f8 = ((_x0r8 - _RA["z_mu"].to(_dv8)) @ _RA["Pz"].to(_dv8).t()) / _RA["fs"].to(_dv8)
                _f18 = torch.cat([_f8, torch.ones(_f8.shape[0], 1, device=_dv8)], dim=1)
                _lin8 = (_RA["d_mu"].to(_dv8) + (_f18 @ _RA["B"].to(_dv8)) @ _RA["Wd"].to(_dv8)) * scale_factor
                if not globals().get("_RADD_SHOWN"):
                    globals()["_RADD_SHOWN"] = 1
                    print("[residual_add] active | ||FM residual||=%.3f ||W.z0||=%.3f -> ||total||=%.3f"
                          % (z.flatten(1).norm(dim=1).mean().item(),
                             _lin8.norm(dim=1).mean().item(),
                             (z.flatten(1) + _lin8).norm(dim=1).mean().item()), flush=True)
                z = (z.flatten(1) + _lin8).reshape_as(z)

            # ===== (RIDGE-TAIL) replace the flow-matching tail with the z0 ridge-regression tail =====
            # The flow-matching model recovers the dominant subspace of the delta well but its tail direction
            # is weak, whereas a ridge regression on z0 predicts the tail better. This keeps the subspace
            # component from the model and takes the tail from the ridge regression, with no retraining.
            # The ridge regression is fitted on unscaled latents, whereas z and x0s here have been multiplied by scale_factor, so they must be divided back first.
            _rp_npz = str(getattr(_eval_args, "fm_tail_ridge", "") or getattr(args, "fm_tail_ridge", "") or "")
            if _rp_npz and not os.path.exists(_rp_npz):
                raise FileNotFoundError("[tail_ridge] model file not found: %s" % _rp_npz)
            if _rp_npz:
                try:
                    if "_RIDGE" not in globals():
                        import numpy as _np3
                        _zz = _np3.load(_rp_npz)
                        globals()["_RIDGE"] = {k: torch.from_numpy(_zz[k]).float()
                                               for k in ("B", "Pz", "z_mu", "fs", "Wd", "d_mu", "V")}
                        print("[tail_ridge] loaded %s" % _rp_npz, flush=True)
                    _R = globals()["_RIDGE"]
                    _dev = z.device
                    _B_, _Pz, _zmu, _fs = _R["B"].to(_dev), _R["Pz"].to(_dev), _R["z_mu"].to(_dev), _R["fs"].to(_dev)
                    _Wd, _dmu, _Vt = _R["Wd"].to(_dev), _R["d_mu"].to(_dev), _R["V"].to(_dev)
                    _x0raw = (x0s / scale_factor).flatten(1)                  # (B, D) unscaled
                    _F = ((_x0raw - _zmu) @ _Pz.t()) / _fs
                    _F1 = torch.cat([_F, torch.ones(_F.shape[0], 1, device=_dev)], dim=1)
                    _rraw = _dmu + (_F1 @ _B_) @ _Wd                          # ridge-regression delta_hat (unscaled)
                    _r = _rraw * scale_factor                                 # back to the same scale as z
                    _zf = z.flatten(1)
                    _zin = (_zf @ _Vt.t()) @ _Vt                              # subspace component of the model output (retained)
                    _rt = _r - (_r @ _Vt.t()) @ _Vt                           # ridge-regression tail
                    _zt = _zf - _zin                                          # model tail (the part being replaced)
                    if int(getattr(_eval_args, "fm_tail_ridge_mm", None) if getattr(_eval_args, "fm_tail_ridge_mm", None) is not None else getattr(args, "fm_tail_ridge_mm", 0)):
                        # Amplitude matching: preserve the norm of the model tail and replace direction only, which decouples the comparison from amplitude
                        _rt = _rt * (_zt.norm(dim=1, keepdim=True) / _rt.norm(dim=1, keepdim=True).clamp_min(1e-9))
                    _w = float(getattr(_eval_args, "fm_tail_ridge_w", None) if getattr(_eval_args, "fm_tail_ridge_w", None) is not None else getattr(args, "fm_tail_ridge_w", 1.0))
                    if int(getattr(_eval_args, "fm_tail_ridge_full", None) if getattr(_eval_args, "fm_tail_ridge_full", None) is not None else getattr(args, "fm_tail_ridge_full", 0)):
                        # Pure-ridge arm: the entire delta comes from the ridge regression and the model output is unused.
                        # It runs through the same eval code path, so the metric protocol matches the other arms exactly.
                        z = _r.reshape_as(z)
                    else:
                        z = (_zin + (1.0 - _w) * _zt + _w * _rt).reshape_as(z)
                    if not globals().get("_RIDGE_SHOWN"):
                        globals()["_RIDGE_SHOWN"] = 1
                        print("[tail_ridge] w=%.2f mm=%d | ||FM tail||=%.3f ||ridge tail||=%.3f"
                              % (_w, int(getattr(_eval_args, "fm_tail_ridge_mm", None) if getattr(_eval_args, "fm_tail_ridge_mm", None) is not None else getattr(args, "fm_tail_ridge_mm", 0)),
                                 _zt.norm(dim=1).mean().item(), _rt.norm(dim=1).mean().item()), flush=True)
                except Exception as _e:
                    print("[tail_ridge] failed, skipping: %r" % (_e,), flush=True)
            # Delta export (FM_DUMP_DELTA=<dir>): stores delta in UNSCALED latent units, matching the cascade npz,
            #   so mixture weights and amplitude grids can be swept offline without rerunning GPU inference.
            #   Stored before delta_scale is applied, so the file holds the raw model output.
            # Enabled by default: without it, only the summary statistics are kept and any reuse requires a rerun.
            _dd = os.environ.get("FM_DUMP_DELTA", "") or \
                  os.path.join("ae_runs", "fm_eval", "delta", str(_eval_args.tag or "untagged"))
            if _dd and _dd.lower() != "off":
                os.makedirs(_dd, exist_ok=True)
                # Overwrite guard: if a tag substitution silently fails, several runs write into the same directory and destroy an earlier dump.
                #   The run aborts if delta_*.npy already exists; overwriting requires FM_DUMP_OVERWRITE=1.
                if not globals().get("_DUMP_GUARD_OK"):
                    globals()["_DUMP_GUARD_OK"] = 1
                    import glob as _glob
                    _ex = _glob.glob(os.path.join(_dd, "delta_*.npy"))
                    if _ex and os.environ.get("FM_DUMP_OVERWRITE", "") != "1":
                        raise SystemExit(
                            "[dump-guard] %s already contains %d delta_*.npy files; refusing to overwrite.\n"
                            "  Use a different --tag, or set FM_DUMP_OVERWRITE=1 explicitly." % (_dd, len(_ex)))
                globals()["_DDN"] = globals().get("_DDN", 0)
                _arr = (z / scale_factor).detach().float().cpu().numpy()
                for _b in range(_arr.shape[0]):
                    np.save(os.path.join(_dd, "delta_%04d.npy" % globals()["_DDN"]), _arr[_b])
                    globals()["_DDN"] += 1
            if delta_scale != 1.0:                                   # rF1: amplify predicted change
                z = z * delta_scale
            if postmask is not None:                                 # output-side spatial mask
                _pmf, _thr = postmask
                z = z * (_pmf if _thr < 0 else (_pmf > _thr).float())
            if subspace is not None:                                 # M-B: snap delta onto (mean+subspace)
                _V, _mu = subspace
                _draw = (z / scale_factor).reshape(z.shape[0], -1)
                _dc = _draw - _mu
                z = (_mu + (_dc @ _V) @ _V.t()).reshape_as(z) * scale_factor
            try:
                _gtl = globals().get('_GT_LATENT', None)
                if _gtl is not None:
                    _dg = (_gtl - x0s).flatten(1); _dp = z.flatten(1)
                    _MAGREP.append((( _dp.norm(dim=1)/(_dg.norm(dim=1)+1e-9) ).mean().item(),
                                    torch.nn.functional.cosine_similarity(_dp,_dg,dim=1).mean().item(),
                                    ((_dp*_dg).sum(1)/(_dp*_dp).sum(1).clamp_min(1e-9)).mean().item()))
                    # ---- (SPLIT) decompose cos into the dominant subspace and the tail ----
                    # Almost all of the delta norm lies in the leading principal components, while the
                    #       change-sensitive metrics respond mainly to the tail, so a high overall cos can come
                    #       entirely from the uninformative subspace.
                    # Print-only, wrapped in try, and does not affect any metric.
                    try:
                        if "_TAILV_E" not in globals():
                            import numpy as _np2
                            _zz = _np2.load(os.path.join(DERIVED_DIR, "pca_delta_basis.npz"))
                            globals()["_TAILV_E"] = torch.from_numpy(_zz["comps"][:8]).float()
                        _V = globals()["_TAILV_E"].to(_dp.device)
                        _pin = (_dp @ _V.t()) @ _V; _ptl = _dp - _pin
                        _gin = (_dg @ _V.t()) @ _V; _gtl2 = _dg - _gin
                        _cs = torch.nn.functional.cosine_similarity
                        # ---- (BANDS) multi-band profile: locate where the model and the ridge regression cross over ----
                        try:
                            if "_BANDV" not in globals():
                                import numpy as _np4
                                _zb = _np4.load(os.path.join(DERIVED_DIR, "pca_delta_bands.npz"))
                                globals()["_BANDV"] = torch.from_numpy(_zb["comps"]).float()
                                globals()["_BANDEDGE"] = [int(x) for x in _zb["edges"]]
                                globals()["_BANDNAMES"] = [str(x) for x in _zb["names"]]
                            _BV = globals()["_BANDV"].to(_dp.device)
                            _ed = globals()["_BANDEDGE"]
                            _row = []
                            for _j in range(len(_ed) - 1):
                                _Bj = _BV[_ed[_j]:_ed[_j + 1]]
                                _pb = (_dp @ _Bj.t()) @ _Bj
                                _gb = (_dg @ _Bj.t()) @ _Bj
                                _row.append(_cs(_pb, _gb, dim=1).mean().item())
                            # final band: the residual outside every basis
                            _pr = _dp - (_dp @ _BV.t()) @ _BV
                            _gr = _dg - (_dg @ _BV.t()) @ _BV
                            _row.append(_cs(_pr, _gr, dim=1).mean().item())
                            _BANDREP.append(_row)
                        except Exception: pass
                        _SPLITREP.append((_cs(_pin,_gin,dim=1).mean().item(),
                                          _cs(_ptl,_gtl2,dim=1).mean().item(),
                                          (_gin.norm(dim=1)/_dg.norm(dim=1).clamp_min(1e-9)).mean().item(),
                                          (_pin.norm(dim=1)/_dp.norm(dim=1).clamp_min(1e-9)).mean().item()))
                    except Exception: pass
            except Exception as _e: pass
            if str(getattr(args, "fm_scheme", "std")) != "std":
                # Paired with delta/delta_t^p on the training side: sampling yields a rate, and multiplying by delta_t^p restores the displacement
                _rp = float(getattr(_eval_args, "fm_rate_pow", 1.0))
                z = z * dt_vec.view(-1, *([1] * (z.dim() - 1))).clamp_min(1e-3).pow(_rp)
            _x1 = x0s + z                                            # x1 = x0 + delta
            if float(getattr(_eval_args, "fm_energy_wavg", 0.0)) > 0:
                globals().setdefault("_EWCAND", []).append(_x1.clone())
            if os.environ.get("FM_SAMPDIAG"):
                _SAMPS.append(z.detach().flatten(1).clone())
            _acc = _x1 if _acc is None else _acc + _x1
        _ewt = float(getattr(_eval_args, "fm_energy_wavg", 0.0))
        _cands = globals().pop("_EWCAND", None)
        if _ewt > 0 and _cands and len(_cands) > 1:
            # Score the candidates by energy: decode each one and compare the spatial agreement of its implied change with the energy map
            import numpy as _npw
            _em = globals().get("_EWMAP", None)          # (B,1,D,H,W) image-resolution energy
            if _em is not None:
                _x0img = autoencoder.decode((x0s / scale_factor).to(DEVICE)).float()
                _sc = []
                for _c in _cands:
                    _ci = autoencoder.decode((_c / scale_factor).to(DEVICE)).float()
                    _dv = (_ci - _x0img).abs()
                    _e = _em
                    if _e.shape[2:] != _dv.shape[2:]:
                        _e = torch.nn.functional.interpolate(_e, size=_dv.shape[2:], mode="trilinear", align_corners=False)
                    _br = (_x0img > 0.05)
                    _a = _dv[_br].flatten(); _b = _e.expand_as(_dv)[_br].flatten()
                    _a = _a - _a.mean(); _b = _b - _b.mean()
                    _sc.append((_a * _b).sum() / (_a.norm() * _b.norm() + 1e-8))
                _w = torch.softmax(torch.stack(_sc) / max(_ewt, 1e-6), dim=0)
                x1s = sum(w * c for w, c in zip(_w, _cands))
                if globals().get("_EWN", 0) < 3:
                    globals()["_EWN"] = globals().get("_EWN", 0) + 1
                    print(f"[energy-wavg] candidate scores {[f'{float(x):.3f}' for x in _sc]} -> weights {[f'{float(x):.3f}' for x in _w]}", flush=True)
            else:
                x1s = _acc / max(1, int(n_avg))
        else:
            x1s = _acc / max(1, int(n_avg))
        _flowmod._RATE_POW = float(getattr(_eval_args, "fm_rate_pow", 1.0))
        # ---- add the population prior back (--fm_gbar), paired with residualization on the training side ----
        _gbf = str(getattr(_eval_args, "fm_gbar", "") or "")
        if _gbf:
            if "_GBAR" not in globals():
                _gz = np.load(_gbf)
                globals()["_GBAR"] = torch.from_numpy(_gz["mean"]).float()
                print("[gbar] population prior loaded and added back at inference", flush=True)
            _gb = globals()["_GBAR"].to(x1s.device).reshape(1, *x1s.shape[1:]).to(x1s.dtype)
            x1s = x1s + _gb * scale_factor
        # ---- PCA subspace projection (--fm_pca_basis / --fm_pca_k) ----
        _pcaf = str(getattr(_eval_args, "fm_pca_basis", "") or "")
        _pcak = int(getattr(_eval_args, "fm_pca_k", 0))
        if _pcaf and _pcak > 0:
            if "_PCA_B" not in globals():
                _zz = np.load(_pcaf)
                globals()["_PCA_B"] = (torch.from_numpy(_zz["mean"]).float().to(x1s.device),
                                       torch.from_numpy(_zz["comps"]).float().to(x1s.device))
                print("[pca] basis loaded K_max=%d" % globals()["_PCA_B"][1].shape[0], flush=True)
            _pmu, _pV = globals()["_PCA_B"]
            _pV = _pV[:_pcak]
            _dd = ((x1s - x0s) / scale_factor).reshape(x1s.shape[0], -1).float()
            _cc = (_dd - _pmu) @ _pV.T                       # project onto the K-dimensional coefficients
            _dp = (_cc @ _pV + _pmu).reshape(x1s.shape)      # map back to the latent (linear)
            x1s = x0s + _dp.to(x1s.dtype) * scale_factor
        _gp = float(getattr(_eval_args, "fm_delta_pow", 1.0))
        if _gp != 1.0:
            # Norm-preserving power transform: changes only the shape of the change field, not the total amplitude (decoupled from delta_scale)
            # gamma>1 sharpens (amplitude concentrates in strong-change regions, raising precision); gamma<1 flattens (raising recall)
            _d = x1s - x0s
            _n0 = _d.flatten(1).norm(dim=1).clamp_min(1e-9)
            _d = _d.sign() * _d.abs().pow(_gp)
            _n1 = _d.flatten(1).norm(dim=1).clamp_min(1e-9)
            _d = _d * (_n0 / _n1).view(-1, *([1] * (_d.dim() - 1)))
            x1s = x0s + _d
        if os.environ.get("FM_SAMPDIAG") and len(_SAMPS) >= 2:
            import torch.nn.functional as _Fd
            _S = torch.stack(_SAMPS, 0)                       # (N,B,D)
            _mean = _S.mean(0)                                # (B,D) averaged delta
            _n_i = _S.norm(dim=2)                             # (N,B) norm of each sample
            _n_m = _mean.norm(dim=1)                          # (B,)
            _N = _S.shape[0]
            _cos_pair = []
            for _i in range(_N):
                for _j in range(_i + 1, _N):
                    _cos_pair.append(_Fd.cosine_similarity(_S[_i], _S[_j], dim=1))
            _cp = torch.stack(_cos_pair, 0).mean().item() if _cos_pair else float("nan")
            _gtl = globals().get("_GT_LATENT", None)
            _ci = _cm = float("nan")
            if _gtl is not None:
                _dg = (_gtl - x0s).flatten(1)
                _ci = torch.stack([_Fd.cosine_similarity(_S[_k], _dg, dim=1) for _k in range(_N)], 0).mean().item()
                _cm = _Fd.cosine_similarity(_mean, _dg, dim=1).mean().item()
            _SAMPSTAT.append((_n_i.mean().item(), _n_m.mean().item(), _cp, _ci, _cm))
            import numpy as _np
            _A = _np.array(_SAMPSTAT)
            print(f"[SAMPDIAG] n={len(_SAMPSTAT)} | single-sample ||delta||={_A[:,0].mean():.4f} averaged ||delta||={_A[:,1].mean():.4f} "
                  f"ratio={_A[:,1].mean()/max(_A[:,0].mean(),1e-9):.3f} | pairwise cos between samples={_A[:,2].mean():.3f} | "
                  f"cos(single sample,gt)={_A[:,3].mean():.4f} -> cos(mean,gt)={_A[:,4].mean():.4f}", flush=True)
            _SAMPS.clear()
        _lew = float(getattr(args, "fm_lat_ew", 0.0))
        _LE = globals().get("_LATEW_E", None)
        if _lew > 0.0 and _LE is not None:
            # Total-preserving reweighting in latent space (before decoding), the strict control for the image-space version
            _el = _LE if _LE.ndim == x1s.ndim else _LE.unsqueeze(1)
            _el = _el[:, :1].to(x1s.device).float()
            _B = _el.shape[0]; _fl = _el.reshape(_B, -1)
            _rk = _fl.argsort(dim=1).argsort(dim=1).float() / max(_fl.shape[1] - 1, 1)
            _w = ((1.0 - _lew) + _lew * 2.0 * _rk).reshape(_el.shape)
            x1s = x0s + (x1s - x0s) * _w
            if not globals().get("_LATEW_SHOWN"):
                globals()["_LATEW_SHOWN"] = 1
                print(f"[lat_ew] total-preserving latent-space reweighting lam={_lew} w range [{float(_w.min()):.3f},{float(_w.max()):.3f}]", flush=True)
        if return_latent:
            return (x1s / scale_factor)     # rollout: return the latent to use as the next starting point
        _pred = autoencoder.decode((x1s / scale_factor).to(DEVICE)).float()
        _im = globals().get("_IMGMASK", None)
        if _im is not None and _im[0] is not None and _im[0].numel() > 8:
            # Tighten the change field at image resolution: delta = pred - dec(x0), masked by a quantile of the precomputed energy map
            _x0img = autoencoder.decode((x0s / scale_factor).to(DEVICE)).float()
            _d = _pred - _x0img
            _E = _im[0]
            while _E.dim() < _pred.dim(): _E = _E.unsqueeze(1)
            if _E.shape[2:] != _pred.shape[2:]:
                _E = torch.nn.functional.interpolate(_E, size=_pred.shape[2:], mode="trilinear", align_corners=False)
            _br = (_x0img > 0.05)                                   # threshold within the brain only
            _fl = _E.flatten(1); _bf = _br.flatten(1)
            _qs = []
            for _b in range(_E.shape[0]):
                _v = _fl[_b][_bf[_b]]
                _qs.append(torch.quantile(_v.float(), _im[1]) if _v.numel() > 10 else _fl[_b].min())
            _q = torch.stack(_qs).view(-1, *([1] * (_E.dim() - 1)))
            _pred = _x0img + _d * ((_E > _q).float() * _br.float())
        return _pred.cpu()
    if x0_start_noise > 0:
        # noisy-x0-start: begin at x0+sigma*z, integrate drift -> x1 (x0 fixed concat anchor)
        z = x0s + x0_start_noise * torch.randn_like(x0s)
        for i in range(steps):
            s_ = s_steps[i]; ds = (s_steps[i + 1] - s_); s_b = s_.expand(z.shape[0])
            _mi = torch.cat([z, x0s], dim=1).float()
            if cfg_w > 0:
                _vc = flowmatching(x=_mi, timesteps=s_b, context=context, class_labels=class_cond)
                _vn = flowmatching(x=_mi, timesteps=s_b, context=context, class_labels=torch.full_like(class_cond, 15))
                v = _vn + cfg_w * (_vc - _vn)
            else:
                v = flowmatching(x=_mi, timesteps=s_b, context=context, class_labels=class_cond)
            z = z + v * ds
        return autoencoder.decode((z / scale_factor).to(DEVICE)).float().cpu()
    if x0_cond:
        # x0-start flow with x0 as fixed concat anchor; integrate x0 -> x1
        z = x0s
        for i in range(steps):
            s_ = s_steps[i]; ds = (s_steps[i + 1] - s_); s_b = s_.expand(z.shape[0])
            _mi = torch.cat([z, x0s], dim=1).float()
            if cfg_w > 0:
                _vc = flowmatching(x=_mi, timesteps=s_b, context=context, class_labels=class_cond)
                _vn = flowmatching(x=_mi, timesteps=s_b, context=context, class_labels=torch.full_like(class_cond, 15))
                v = _vn + cfg_w * (_vc - _vn)
            else:
                v = flowmatching(x=_mi, timesteps=s_b, context=context, class_labels=class_cond)
            if scheme == "realtime":
                step = v * ds * dt_vec.view(-1, *([1] * (v.dim() - 1)))
            else:
                step = v * ds
            z = z + step
        return autoencoder.decode((z / scale_factor).to(DEVICE)).float().cpu()
    z = x0s
    for i in range(steps):
        s = s_steps[i]
        ds = (s_steps[i + 1] - s)
        s_b = s.expand(z.shape[0])
        v = flowmatching(x=z.float(), timesteps=s_b, context=context, class_labels=class_cond)
        if scheme == "realtime":
            step = v * ds * dt_vec.view(-1, *([1] * (v.dim() - 1)))   # z += v*ds*Dt
        else:
            step = v * ds                                            # z += v*ds
        z = z + step
    z = z / scale_factor
    return autoencoder.decode(z.to(DEVICE)).float().cpu()


def main():
    print(f"[eval:{_eval_args.tag}] scheme={args.fm_scheme} ckpt={_eval_args.fm_ckpt}")

    # ---------- models ----------
    autoencoder = import_from_dotted_path(args.autoencoder)(args).to(DEVICE).float()
    w = remove_module_prefix(torch.load(args.aekl_ckpt, map_location="cpu"))
    autoencoder.load_state_dict(w); autoencoder.eval()
    for p in autoencoder.parameters(): p.requires_grad = False

    _EHM = str(getattr(args, 'fm_hist_mode', 'none'))
    _EHCH = hist_channels(_EHM) if _EHM != 'none' else ((16 if int(getattr(args, 'fm_hist_concat', 0)) >= 2 else 12) if int(getattr(args, 'fm_hist_concat', 0)) else 0)
    _EINJ = str(getattr(args, 'fm_hist_inject', 'concat'))
    _ECDIM = 10 + (2 * _EHCH if (_EHM != 'none' and _EINJ in ('cond','both')) else 0)
    _EGM = str(getattr(args, 'fm_gain_mode', 'none'))
    _res_noise = bool(int(getattr(args, "fm_res_noise", 0)))
    _x0_cond = bool(int(getattr(args, "fm_x0_cond", 0)))
    _x0_sn = float(getattr(args, "fm_x0_start_noise", 0.0))
    _cfg_w = float(getattr(args, "fm_cfg_w", 0.0)); _nce_eval = 16 if _cfg_w > 0 else 15
    _echan_eval = 1 if str(getattr(args, "fm_energy_cond", "none")) != "none" else 0
    if _res_noise or _x0_cond or _x0_sn > 0:
        flowmatching = import_from_dotted_path(args.flowmatching)(args, in_channels=8 + _echan_eval + (_EHCH if _EINJ in ('concat','both') else 0), use_image=False, out_channels=4, num_class_embeds=_nce_eval, cond_dim=_ECDIM).to(DEVICE)
    else:
        flowmatching = import_from_dotted_path(args.flowmatching)(args, in_channels=4, use_image=False, num_class_embeds=_nce_eval).to(DEVICE)
    fw = remove_module_prefix(torch.load(_eval_args.fm_ckpt, map_location="cpu"))
    _gm_e = str(getattr(args, "fm_gain_mode", "none"))
    if _gm_e != "none" or str(getattr(args, "fm_hist_inject", "concat")) == "gain":
        _gc = hist_channels(_gm_e) if _gm_e != "none" else hist_channels(_EHM)
        flowmatching.gain_net = _build_gain_net(_gc, DEVICE)
        print(f"[gain-eval] spatial-modulation network {_gc}ch built, source={_gm_e if _gm_e != 'none' else _EHM}", flush=True)
    _r = flowmatching.load_state_dict(fw, strict=False)   # older checkpoints have no cond_mlp, which stays zero-initialized (an unconditional injection, matching the original behaviour)
    if _r.missing_keys: print(f"[eval] missing (expected for pre-cond ckpts): {len(_r.missing_keys)}")
    flowmatching.eval()
    for p in flowmatching.parameters(): p.requires_grad = False
    controlnet = None
    if _EINJ in ('cnet','cnetft'): args.fm_controlnet = 1
    if int(getattr(args, "fm_controlnet", 0)):
        from src.model3D.networks import init_large_controlnet
        _cm = str(getattr(args, "fm_cnet_cond", "scan"))
        _CNH = str(getattr(args, "fm_cnet_hist", "none"))
        _cnc = hist_channels(_CNH) if _CNH != "none" else (hist_channels(_EHM) if _EINJ in ("cnet", "cnetft") else (16 if _cm == "multi" else (13 if _cm == "hist13" else (5 if _cm == "energy" else (8 if _cm == "traj" else 4)))))
        _inch_e = 8 + (hist_channels(_EHM) if (_EHM != "none" and _EINJ in ("concat", "both")) else 0)
        controlnet = init_large_controlnet(None, in_channels=_inch_e, conditioning_embedding_in_channels=_cnc, cross_attention_dim=10).to(DEVICE)
        _cc = _eval_args.fm_cnet_ckpt or _eval_args.fm_ckpt.replace("fm-unet-ep", "fm-cnet-ep").replace("snap-ep", "snap-cnet-ep")
        controlnet.load_state_dict(remove_module_prefix(torch.load(_cc, map_location="cpu"))); controlnet.eval()
        for p in controlnet.parameters(): p.requires_grad = False
        print(f"[eval] ControlNet ON, loaded {_cc}")
    history_encoder = None
    if int(getattr(args, "fm_history", 0)):
        from src.model3D.history_encoder import HistoryEncoder
        history_encoder = HistoryEncoder(in_ch=4, ctx_dim=10, n_tokens=int(getattr(args, "fm_history_ntok", 64))).to(DEVICE)
        _hist_ckpt = _eval_args.fm_hist_ckpt or _eval_args.fm_ckpt.replace("fm-unet-ep", "fm-hist-ep").replace("snap-ep", "snap-hist-ep")
        _hw = remove_module_prefix(torch.load(_hist_ckpt, map_location="cpu"))
        history_encoder.load_state_dict(_hw); history_encoder.eval()
        for p in history_encoder.parameters(): p.requires_grad = False
        print(f"[eval] history ON, loaded {_hist_ckpt}")

    # ---------- data ----------
    from dataset.oasis_dataset_3D_pair_latent import get_brain_dataset
    train_loader, train_ds, _ = get_brain_dataset(args, mode="train", with_seg=False)
    # scale_factor: identical recipe to training (1/std of first 10 train starting-latents)
    zl = torch.stack([train_ds[i]["starting_latent"] for i in range(10)], dim=0)
    scale_factor = (1.0 / torch.std(zl)).item()
    print(f"[eval] scale_factor={scale_factor:.6f}")
    _sub = None
    if getattr(_eval_args, 'fm_subspace', ''):
        _dz = np.load(_eval_args.fm_subspace)
        _sub = (torch.from_numpy(_dz['V']).float().to(DEVICE), torch.from_numpy(_dz['mean']).float().to(DEVICE))
        print(f"[eval] M-B subspace loaded: V{tuple(_sub[0].shape)}")

    # decoded-latent space: GT = AE.decode(real followup latent). Isolates flow-matching
    # quality from AE reconstruction error (pred is decoded the same way). with_image left
    # off (CSV image paths are relative and unused by training).
    _SPLIT = os.environ.get("FM_EVAL_SPLIT", "test")   # set to train when dumping the residual fitting set
    if _SPLIT != "test": print("[eval] split =", _SPLIT, flush=True)
    test_loader, test_ds, _ = get_brain_dataset(args, mode=_SPLIT, with_seg=False)

    # ---------- eval loop ----------
    _seed_all(getattr(_eval_args, "fm_seed", -1))
    agg = {"pred": [], "copy": []}
    _RAW = int(getattr(_eval_args, "raw_out", 0) or 0)
    if _RAW:
        agg.update({"pred_raw": [], "copy_raw": []})
        from monai import transforms as _mt
        _RAW_CANON = tuple(int(v) for v in str(_eval_args.raw_canonical).split(","))
        _raw_prep = _mt.Compose([_mt.LoadImage(image_only=True), _mt.EnsureChannelFirst(),
                                 _mt.Spacing(pixdim=1.5, mode="bilinear"), _mt.ScaleIntensity(minv=0, maxv=1),
                                 _mt.ResizeWithPadOrCrop(spatial_size=_RAW_CANON, mode="constant")])

        def _raw_up(v):
            # inverse of the latent pipeline's trilinear resize (align_corners=False) to the model grid
            return F.interpolate(torch.from_numpy(np.ascontiguousarray(v)).float()[None, None], size=_RAW_CANON,
                                 mode="trilinear", align_corners=False)[0, 0].numpy().astype(np.float64)

        def _raw_path(p):
            p = str(p)
            if os.path.isabs(p):
                return p
            for _root in (str(args.data_dir), os.path.join(str(args.data_dir), "Final")):
                if os.path.exists(os.path.join(_root, p)):
                    return os.path.join(_root, p)
            return os.path.join(str(args.data_dir), p)
    n_done = 0
    # --eval_on train: reuse the train_loader built above, which is otherwise only used to compute scale_factor
    _eval_loader = train_loader if str(getattr(_eval_args, "eval_on", "test")) == "train" else test_loader
    print("[eval] eval_on=%s" % str(getattr(_eval_args, "eval_on", "test")), flush=True)
    for batch in _eval_loader:
        # optional fixed pair list, one "subject__startdate__followdate" per line (e.g. another method's case list)
        if os.environ.get("FM_PAIR_KEYS", ""):
            if "_PAIR_KEYS" not in globals():
                globals()["_PAIR_KEYS"] = set(l.strip() for l in open(os.environ["FM_PAIR_KEYS"]) if l.strip())
            _k_sp = batch["starting_image_path"]; _k_fp = batch["followup_image_path"]
            _k_sp = _k_sp[0] if isinstance(_k_sp, (list, tuple)) else _k_sp
            _k_fp = _k_fp[0] if isinstance(_k_fp, (list, tuple)) else _k_fp
            if "/".join(str(_k_sp).split("/")[-3:-1]).replace("/", "__") + "__" + str(_k_fp).split("/")[-2] not in globals()["_PAIR_KEYS"]:
                continue
        z0 = batch["starting_latent"].to(DEVICE).float()
        z1 = batch["followup_latent"].to(DEVICE).float()
        starting_age = batch["starting_age"].to(DEVICE).unsqueeze(1)
        followup_age = batch["followup_age"].to(DEVICE).unsqueeze(1)
        starting_dia = batch["starting_diagnosis"].to(DEVICE).unsqueeze(1)
        followup_dia = batch["followup_diagnosis"].to(DEVICE).unsqueeze(1)
        context = batch["context"].to(DEVICE).squeeze(1)
        context = torch.cat([context, followup_age - starting_age,
                             followup_dia - starting_dia], dim=1).float().unsqueeze(1)
        if history_encoder is not None:
            _prior = batch["prior_latent"].to(DEVICE).float() * scale_factor
            _hp = batch["has_prior"].to(DEVICE).float() if "has_prior" in batch else None
            context = torch.cat([context, history_encoder(_prior, _hp)], dim=1)
        age_gap = ((batch["followup_age"] - batch["starting_age"]).to(DEVICE).float() * 100).long()
        dt_vec = (batch["followup_follow_up"].to(DEVICE).float()
                  - batch["starting_follow_up"].to(DEVICE).float()) / 10.0

        _energy = None
        _ecm = str(getattr(args, "fm_energy_cond", "none"))
        if _ecm != "none":
            if _ecm == "latoracle":
                _en = (z1 - z0).abs().mean(1, keepdim=True); _energy = _en / (_en.amax(dim=(2, 3, 4), keepdim=True) + 1e-6)
            elif _ecm == "latpast":
                _pr = batch["prior_latent"].to(DEVICE).float(); _en = (z0 - _pr).abs().mean(1, keepdim=True); _energy = _en / (_en.amax(dim=(2, 3, 4), keepdim=True) + 1e-6)
                if "has_prior" in batch: _energy = _energy * batch["has_prior"].to(DEVICE).float().view(-1, 1, 1, 1, 1)
            else:
                _ek = "followup_energy" if _ecm == "oracle" else "starting_energy"
                _energy = batch[_ek].to(DEVICE).float()
        _postmask = None
        if _eval_args.fm_postmask_thr >= 0.0:
            _pms = str(getattr(_eval_args, "fm_postmask_src", "oracle"))
            if _pms == "oracle":
                _pm = (z1 - z0).abs().mean(1, keepdim=True) * scale_factor   # true change region (upper bound)
            elif _pms in ("ek", "ekind", "arate") and "etraj" in batch:
                # ek=E_last (cumulative energy) | ekind=E_ind (population mean removed) | arate=a_E (energy growth rate)
                # Value of multiple timepoints: a large accumulated change is not the same as an ongoing one. High arate means the region is still progressing.
                _ci = {"ek": 0, "arate": 1, "ekind": 4}[_pms]
                _pm = batch["etraj"].to(DEVICE).float()[:, _ci:_ci+1].abs()
            elif _pms in ("egrow", "eunion") and "egrid" in batch:
                # egrid = [E(t0), E(-1y), E(-2y), E(-3y), cov], a fixed time grid with no fitting
                _eg = batch["egrid"].to(DEVICE).float()
                if _pms == "egrow":
                    _pm = (_eg[:, 0:1] - _eg[:, 1:2]).clamp_min(0.0)          # change added over the most recent year
                else:
                    _pm = _eg[:, 0:4].amax(dim=1, keepdim=True)               # union across timepoints
            elif _pms == "etrue":
                # IMGMASK: the image-resolution energy map applied as a post-decode mask; nothing happens on the latent side
                _et = batch.get("etrue", None)
                def _tot(v):
                    # The batch supplies path strings (the MONAI transform did not run), so the npz is loaded directly by path
                    import numpy as _np, os as _os
                    if v is None: return None
                    if isinstance(v, (list, tuple)):
                        _arrs = []
                        for _x in v:
                            _x = _x if isinstance(_x, str) else str(_x)
                            if _os.path.exists(_x):
                                _d = _np.load(_x, allow_pickle=True)
                                _a = _d["data"].astype("float32")
                                if int(_d["valid"]) == 0: _a = None      # no history means no energy
                            else: _a = None
                            _arrs.append(_a)
                        if all(a is None for a in _arrs): return None
                        _sh = next(a.shape for a in _arrs if a is not None)
                        _arrs = [(_np.ones(_sh, "float32") if a is None else a) for a in _arrs]  # missing = keep everything
                        v = _np.stack(_arrs)
                    elif isinstance(v, str):
                        if not _os.path.exists(v): return None
                        _d = _np.load(v, allow_pickle=True)
                        if int(_d["valid"]) == 0: return None
                        v = _d["data"].astype("float32")[None]
                    if isinstance(v, _np.ndarray): v = torch.from_numpy(v)
                    v = v.to(DEVICE).float()
                    while v.dim() < 5: v = v.unsqueeze(1)
                    return v
                globals()["_IMGMASK"] = (_tot(_et), float(_eval_args.fm_postmask_thr))
                _pm = None
            elif _pms == "pop":
                # Population mean future-change map: a constant map estimated on the training split, used as a non-individualized reference
                import numpy as _npp
                _pf = os.path.join(DERIVED_DIR, "pop_change_map.npy")
                _pmap = torch.from_numpy(_npp.load(_pf)).to(DEVICE).float()[None, None]
                _pm = _pmap.expand(z0.shape[0], 1, *_pmap.shape[2:]).contiguous()
            elif _pms == "v1":
                # The patient's most recent change rate |x0-p1|
                _p1 = batch.get("prior_latent", None)
                if _p1 is None: _pm = (z1 - z0).abs().mean(1, keepdim=True) * scale_factor
                else: _pm = (z0 - _p1.to(DEVICE).float()).abs().mean(1, keepdim=True) * scale_factor
            elif _pms == "popv1":
                # population prior x individual rate (both used)
                import numpy as _npp
                _pmap = torch.from_numpy(_npp.load(os.path.join(DERIVED_DIR, "pop_change_map.npy"))).to(DEVICE).float()[None, None]
                _pmap = _pmap.expand(z0.shape[0], 1, *_pmap.shape[2:])
                _p1 = batch.get("prior_latent", None)
                _v = (z0 - _p1.to(DEVICE).float()).abs().mean(1, keepdim=True) if _p1 is not None else torch.ones_like(_pmap)
                _pm = (_pmap / (_pmap.amax(dim=(2,3,4), keepdim=True)+1e-6)) * (_v / (_v.amax(dim=(2,3,4), keepdim=True)+1e-6))
            elif _pms == "energy" and "starting_energy" in batch:
                _pm = batch["starting_energy"].to(DEVICE).float()[:, :1]
            else:
                _pm = (z1 - z0).abs().mean(1, keepdim=True) * scale_factor
                print(f"[postmask] source {_pms} unavailable, falling back to oracle", flush=True)
            if _pm is None:
                _postmask = None                      # etrue: uses the post-decode image-space mask, leaving the latent side untouched
            elif str(getattr(_eval_args, "fm_postmask_mode", "quantile")) == "quantile":
                # Quantile threshold: keep the top (1-thr) fraction of voxels. Normalizing by the maximum is dominated by outlier voxels,
                # so even a nominally permissive threshold can remove almost all predicted change in practice.
                # Shuffled control: always use the map of the first case, i.e. another subject's map. Any spatial modulation can recover
                #   error by increasing amplitude, so this control is required to separate individualization from fixed anatomical weighting.
                if int(getattr(_eval_args, "fm_postmask_fixed", 0)):
                    if "_PMFIX" not in globals():
                        globals()["_PMFIX"] = _pm[:1].detach().clone()
                        print("[postmask] control active: using the energy map of the first case for every case", flush=True)
                    _pm = globals()["_PMFIX"].expand_as(_pm).contiguous()
                _flat = _pm.flatten(1)
                _q = torch.quantile(_flat.float(), float(_eval_args.fm_postmask_thr), dim=1)
                _hard = (_pm > _q.view(-1, *([1] * (_pm.dim() - 1)))).float()
                _al = float(getattr(_eval_args, "fm_postmask_soft", 0.0))
                if _al > 0.0:
                    _pmn = _pm / (torch.quantile(_flat.float(), 0.99, dim=1)
                                  .view(-1, *([1] * (_pm.dim() - 1))) + 1e-6)
                    _pm = (1.0 - _al) + _al * _pmn.clamp(0.0, 1.0)
                    _postmask = (_pm, -1.0)          # thr=-1 means scale by weight only
                else:
                    _postmask = (_hard, 0.5)
            else:
                _pm = _pm / (_pm.amax(dim=(2, 3, 4), keepdim=True) + 1e-6)
                _postmask = (_pm, float(_eval_args.fm_postmask_thr))
        if float(getattr(_eval_args, "fm_energy_wavg", 0.0)) > 0:
            import numpy as _npw2, os as _osw
            _pv = batch.get("vfin_img", None)
            _lst = []
            if _pv is not None:
                _pv = _pv if isinstance(_pv, (list, tuple)) else [_pv]
                for _q in _pv:
                    try:
                        _d = _npw2.load(str(_q), allow_pickle=True); _arr = _d["data"].astype("float32")
                        if "valid" in _d.files and int(_d["valid"]) == 0: _arr = None
                    except Exception: _arr = None
                    _lst.append(_arr)
            if _lst and all(x is not None for x in _lst):
                globals()["_EWMAP"] = torch.from_numpy(_npw2.stack(_lst)).unsqueeze(1).to(DEVICE).float()
            else:
                globals()["_EWMAP"] = None
        if str(getattr(_eval_args, "fm_mag_norm", "") or "") == "pred":
            # Per-patient amplitude calibration from a ||delta|| regressor fitted on the training split (delta t plus the energy dynamics E_dot, with E_hat = E_dot x future delta t)
            import json as _js, os as _os
            if "_MAGPRED_TBL" not in globals():
                _f = os.path.join(DERIVED_DIR, "mag_pred.json")
                globals()["_MAGPRED_TBL"] = _js.load(open(_f)) if _os.path.exists(_f) else {}
                print(f"[mag_norm=pred] loaded {len(globals()['_MAGPRED_TBL'])} predictions", flush=True)
            _tb = globals()["_MAGPRED_TBL"]
            _ks = batch.get("starting_image_path", None)
            _vals = []
            if _ks is not None:
                _ks = _ks if isinstance(_ks, (list, tuple)) else [_ks]
                for _k in _ks:
                    _k = str(_k); _kk = "/".join(_k.split("/")[-3:-1])
                    _vals.append(_tb.get(_kk, None))
            if _vals and all(v is not None for v in _vals):
                # Relative calibration: the regressor is fitted on raw latents while evaluation runs under --norm_mode std,
                #   so the absolute amplitude is scale-mismatched. Only the relative deviation is used:
                #   ||delta_hat|| <- current ||delta_hat|| x (pred_i / mean pred). The global scale is still set by the model and delta_scale,
                #   and energy only determines whether this patient progresses faster or slower than average, which is the quantity of interest and is immune to the scale mismatch.
                if "_MAGPRED_MEAN" not in globals():
                    import numpy as _np2
                    globals()["_MAGPRED_MEAN"] = float(_np2.mean(list(_tb.values()))) if _tb else 1.0
                    print(f"[mag_norm=pred] relative calibration, mean prediction {globals()['_MAGPRED_MEAN']:.2f}", flush=True)
                _rel = torch.tensor(_vals, device=DEVICE).float() / max(globals()["_MAGPRED_MEAN"], 1e-6)
                globals()["_MAG_PRED"] = ("rel", _rel)
            else:
                globals()["_MAG_PRED"] = None
        _cnet_prior = None
        if controlnet is not None:
            _cmf = str(getattr(args, "fm_cnet_cond", "scan"))
            _cnh_e = str(getattr(args, "fm_cnet_hist", "none"))
            if _cnh_e != "none":
                _cnet_prior = build_hist_cond(batch, z0 * scale_factor, scale_factor, DEVICE, _cnh_e)
            elif _cmf == "hist13":
                _cnet_prior = batch["hist_cond"].to(DEVICE).float() * scale_factor
            elif _EINJ in ("cnet", "cnetft"):
                _cnet_prior = build_hist_cond(batch, z0 * scale_factor, scale_factor, DEVICE, _EHM)
            elif _cmf == "multi":
                _cnet_prior = build_multi_cond(batch, z0 * scale_factor, scale_factor, DEVICE)
            elif _cmf == "energy":
                _cnet_prior = torch.cat([z0 * scale_factor, batch["starting_energy"].to(DEVICE).float()], dim=1)
            else:
                _pr = batch["prior_latent"].to(DEVICE).float() * scale_factor
                _cnet_prior = torch.cat([_pr, (z0 * scale_factor) - _pr], dim=1) if _cmf == "traj" else _pr
        _cv = build_cond_vec(batch, context, DEVICE, args)
        if float(getattr(args, "fm_lat_ew", 0.0)) > 0.0:
            globals()["_LATEW_E"] = batch["vslope"].to(DEVICE).float() if "vslope" in batch else None
            if globals()["_LATEW_E"] is None:
                raise RuntimeError("--fm_lat_ew>0 but the batch has no vslope: the dataset did not load energy, and silently falling back is not permitted")
        globals()['flow_init_batch'] = batch
        globals()['flow_init_dt'] = ((batch['followup_follow_up'] - batch['starting_follow_up']).to(DEVICE).float() / 12.0)
        try: globals()['_GT_LATENT'] = batch['followup_latent'].to(DEVICE).float()*scale_factor
        except Exception: globals()['_GT_LATENT'] = None
        batch['_prev2_fill'] = str(getattr(args, 'fm_prev2_fill', 'zero'))
        _hm_e = str(getattr(args, 'fm_hist_mode', 'none'))
        _inj_e = str(getattr(args, 'fm_hist_inject', 'concat'))
        if _hm_e != 'none':
            _hfull = build_hist_cond(batch, z0 * scale_factor, scale_factor, DEVICE, _hm_e)
            _hcat = _hfull if _inj_e in ('concat', 'both') else None
            if _inj_e in ('cond', 'both'):
                _cv = torch.cat([_cv, torch.cat([_hfull.mean(dim=(2,3,4)), _hfull.std(dim=(2,3,4))], dim=1).to(_cv.dtype)], dim=1)
        else:
            _hcat = build_multi_cond(batch, z0 * scale_factor, scale_factor, DEVICE)[:, 4:] if int(getattr(args, 'fm_hist_concat', 0)) else None
        if _hcat is not None and int(getattr(args, 'fm_hist_concat', 0)) >= 2 and 'cum_energy' in batch:
            _hcat = torch.cat([_hcat, batch['cum_energy'].to(DEVICE).float()], dim=1)
        _gcond = build_hist_cond(batch, z0 * scale_factor, scale_factor, DEVICE, str(getattr(args, 'fm_gain_mode', 'none'))) if str(getattr(args, 'fm_gain_mode', 'none')) != 'none' else None
        _roll = float(getattr(_eval_args, "fm_rollout_years", 0.0))
        _dty_r = float((globals()['flow_init_dt']).mean().item()) if globals().get('flow_init_dt') is not None else 0.0
        _K = int(max(1, round(_dty_r / _roll))) if (_roll > 0 and _dty_r > 0) else 1
        if _K > 1:
            # Physical-time rollout: the denoising axis still runs the full [0,1] each step; only delta t is split across K calls
            _b = dict(batch)
            _sa, _fa = batch["starting_age"], batch["followup_age"]
            _s0, _f0 = batch["starting_follow_up"], batch["followup_follow_up"]
            _z_cur = z0
            _prev_lat, _prev_fu = batch["prior_latent"], batch.get("prior_follow_up", _s0)
            for _k in range(1, _K + 1):
                _r0, _r1 = (_k - 1) / _K, _k / _K
                _b["starting_age"] = _sa + (_fa - _sa) * _r0
                _b["followup_age"] = _sa + (_fa - _sa) * _r1
                _b["starting_follow_up"] = _s0 + (_f0 - _s0) * _r0
                _b["followup_follow_up"] = _s0 + (_f0 - _s0) * _r1
                _b["prior_latent"], _b["prior_follow_up"] = _prev_lat, _prev_fu
                if _k >= 2:                      # prev2 falls back to the real baseline scan, the only genuine history available
                    _b["prior2_latent"], _b["prior2_follow_up"] = batch["prior_latent"], batch.get("prior_follow_up", _s0)
                _cvk = build_cond_vec(_b, context, DEVICE, args)
                _hk = build_hist_cond(_b, _z_cur * scale_factor, scale_factor, DEVICE, _hm_e) if _hm_e != 'none' else _hcat
                _z_nx = sample(flowmatching, autoencoder, _z_cur, context, age_gap, scale_factor,
                               args.fm_scheme, dt_vec / _K, _eval_args.infer_steps, _gain_cond=_gcond, res_noise=_res_noise, x0_cond=_x0_cond, x0_start_noise=_x0_sn, cfg_w=_cfg_w, n_avg=_eval_args.n_avg, subspace=_sub, delta_scale=_eval_args.fm_delta_scale, energy=_energy, postmask=_postmask, controlnet=controlnet, cnet_prior=_cnet_prior, cond_vec=_cvk, hist_cat=_hk, return_latent=True)
                _prev_lat, _prev_fu = _z_cur, _b["starting_follow_up"]
                _z_cur = _z_nx
            with torch.no_grad():
                x_pred = autoencoder.decode(_z_cur.to(DEVICE)).float().cpu()
        else:
            x_pred = sample(flowmatching, autoencoder, z0, context, age_gap, scale_factor,
                            args.fm_scheme, dt_vec, _eval_args.infer_steps, _gain_cond=_gcond, res_noise=_res_noise, x0_cond=_x0_cond, x0_start_noise=_x0_sn, cfg_w=_cfg_w, n_avg=_eval_args.n_avg, subspace=_sub, delta_scale=_eval_args.fm_delta_scale, energy=_energy, postmask=_postmask, controlnet=controlnet, cnet_prior=_cnet_prior, cond_vec=_cv, hist_cat=_hcat)
        with torch.no_grad():
            x_in = autoencoder.decode(z0).float().cpu()      # baseline (decoded starting latent)
            x_tg = autoencoder.decode(z1).float().cpu()      # GT followup (decoded real latent)

        B = x_pred.shape[0]
        for b in range(B):
            pr = x_pred[b, 0].numpy().astype(np.float64)
            ip = x_in[b, 0].numpy().astype(np.float64)
            tg = x_tg[b, 0].numpy().astype(np.float64)
            if _RAW:
                _rs = batch["starting_image_path"]; _rf = batch["followup_image_path"]
                _rs = _rs[b] if isinstance(_rs, (list, tuple)) else _rs
                _rf = _rf[b] if isinstance(_rf, (list, tuple)) else _rf
                xr = _raw_prep(_raw_path(_rs)).squeeze().numpy().astype(np.float64)
                gr = _raw_prep(_raw_path(_rf)).squeeze().numpy().astype(np.float64)
                ip_u, pr_u = _raw_up(ip), _raw_up(pr)
                brain_r = (xr > 0.05) | (gr > 0.05)
                _A = np.stack([ip_u[brain_r], np.ones(int(brain_r.sum()))], 1)
                _ra, _rb = np.linalg.lstsq(_A, xr[brain_r], rcond=None)[0]
                prr = np.clip(xr + _ra * (pr_u - ip_u), 0.0, 1.0)
            if str(getattr(_eval_args, "dump_pred", "") or ""):
                import os as _os
                _dd = _eval_args.dump_pred; _os.makedirs(_dd, exist_ok=True)
                globals()["_DUMPN"] = globals().get("_DUMPN", 0) + 1
                _sp = batch.get("starting_image_path", None)
                _fp = batch.get("followup_image_path", None)
                _key = "case%04d" % globals()["_DUMPN"]
                if _sp is not None:
                    _t = _sp[b] if isinstance(_sp, (list, tuple)) else _sp
                    _key = "/".join(str(_t).split("/")[-3:-1]).replace("/", "__")
                    # Include the follow-up date in the key: keying on the start date alone lets multiple
                    #   follow-ups of the same baseline overwrite each other, silently reducing the dump size
                    if _fp is not None:
                        _u = _fp[b] if isinstance(_fp, (list, tuple)) else _fp
                        _key = _key + "__" + str(_u).split("/")[-2]
                _extra = dict(x0_raw=xr.astype("float16"), pred_raw=prr.astype("float16"),
                              gt_raw=gr.astype("float16"), raw_fit=np.array([_ra, _rb])) if _RAW else {}
                np.savez_compressed(_os.path.join(_dd, _key + ".npz"),
                                    x0=ip.astype("float16"), pred=pr.astype("float16"),
                                    gt=tg.astype("float16"), **_extra)
            brain = (ip > 0.05) | (tg > 0.05)
            if brain.sum() == 0:
                brain = np.ones_like(ip, dtype=bool)
            mask = torch.from_numpy(brain.astype(np.float32))[None, None]

            dr = float(tg.max() - tg.min()) or 1.0   # target data-range for PSNR/SSIM
            _sp0 = batch.get("starting_image_path", None); _sp1 = batch.get("followup_image_path", None)
            def _one(v, b_):
                if v is None: return ""
                try: return str(v[b_])
                except Exception: return str(v)
            _pid = _one(_sp0, b) + "->" + _one(_sp1, b)
            if globals().get("_PSNRDIAG_N", 0) < 3:
                globals()["_PSNRDIAG_N"] = globals().get("_PSNRDIAG_N", 0) + 1
                # Per-case diagnostic for cross-checking against a direct numpy decode: with the same formula,
                #   target and latent path, a difference means the two are not using the same cases or latents.
                print("[PSNR-DIAG] pid=%s | ‖z0‖=%.3f | dr=%.4f | tg[%.3f,%.3f] | copyPSNR=%.4f"
                      % (_pid.split("/")[-3] if "/" in _pid else _pid,
                         float(np.sqrt((ip ** 2).sum())) * 0 + float(z0[b].flatten().norm().item()),
                         dr, float(tg.min()), float(tg.max()), psnr(ip, tg, dr)), flush=True)
            def _score(pred_img, tgt, inp, brain_m, data_range):
                """Full metric panel; identical for the latent-decoded and the original-scan output."""
                mask_t = torch.from_numpy(brain_m.astype(np.float32))[None, None]
                df = delta_f1_metrics(pred_img, tgt, inp, mask=brain_m)
                rf = region_f1_metrics(pred_img, tgt, inp, mask=brain_m)
                cm = compute_change_metrics(
                    torch.from_numpy(pred_img)[None, None].float(),
                    torch.from_numpy(tgt)[None, None].float(),
                    torch.from_numpy(inp)[None, None].float(),
                    mask_t)
                cm = {k: (float(v.item()) if torch.is_tensor(v) else float(v)) for k, v in cm.items()}
                _cf = change_f1(pred_img, tgt, inp, mask=brain_m)
                return {
                    "pid": _pid,
                    "cF1": _cf["cF1"], "cRecall": _cf["cRecall"], "cPrecision": _cf["cPrecision"],
                    "dF1": df["dF1"], "Recall": df["Recall"], "Precision": df["Precision"],
                    "rF1": rf["rF1"], "rRecall": rf["rRecall"], "rPrecision": rf["rPrecision"],
                    "CHANGE_MAE": cm.get("CHANGE_MAE", float("nan")),
                    "CHANGE_DICE": cm.get("CHANGE_DICE", float("nan")),
                    "CHANGE_PCC": cm.get("CHANGE_PCC", float("nan")),
                    "DRMAE": drmae(pred_img, tgt, inp, brain_m),
                    "PSNR": psnr(pred_img, tgt, data_range), "SSIM": ssim3d(pred_img, tgt, data_range),
                }

            for name, pred_img in (("pred", pr), ("copy", ip)):
                agg[name].append(_score(pred_img, tg, ip, brain, dr))
            if _RAW:
                _drr = float(gr.max() - gr.min()) or 1.0
                for name, pred_img in (("pred_raw", prr), ("copy_raw", xr)):
                    _row = _score(pred_img, gr, xr, brain_r, _drr)
                    _row.update({"fit_a": float(_ra), "fit_b": float(_rb)})
                    agg[name].append(_row)
            n_done += 1
        print(f"[eval] {n_done}/{_eval_args.n_eval} pairs")
        if n_done >= _eval_args.n_eval:
            break

    # Map delta_XXXX.npy back to cases; without this the training-split dump cannot be paired with latents afterwards
    _ddp = os.environ.get("FM_DUMP_DELTA", "") or \
           os.path.join("ae_runs", "fm_eval", "delta", str(_eval_args.tag or "untagged"))
    if _ddp and _ddp.lower() != "off" and agg.get("pred"):
        with open(os.path.join(_ddp, "pids.txt"), "w") as _f:
            for _r in agg["pred"]: _f.write(_r["pid"] + "\n")
        print("[eval] wrote pids.txt (%d rows) -> %s" % (len(agg["pred"]), _ddp), flush=True)

    def summarize(rows):
        keys = [k for k in rows[0].keys() if k != "pid"]
        return {k: round(float(np.nanmean([r[k] for r in rows])), 4) for k in keys}

    out = {"tag": _eval_args.tag, "scheme": args.fm_scheme, "fm_scale_norm": int(getattr(args, "fm_scale_norm", 1)),
           "fm_sigma": float(getattr(args, "fm_sigma", 0.0)), "ckpt": _eval_args.fm_ckpt,
           # The sampling settings below must be recorded with every result, otherwise the operating point of a result file cannot be recovered and the delta curve is not reproducible.
           "fm_delta_scale": float(getattr(_eval_args, "fm_delta_scale", 1.0)),
           "n_avg": int(getattr(_eval_args, "n_avg", 1)),
           "fm_antithetic": int(getattr(_eval_args, "fm_antithetic", 0)),
           "fm_seed": int(getattr(_eval_args, "fm_seed", -1)),
           "fm_det": int(getattr(_eval_args, "fm_det", 0)),
           "fm_pca_k": int(getattr(_eval_args, "fm_pca_k", 0)),
           "fm_flowinit_sigma": float(_eval_args.fm_flowinit_sigma),
           "n_pairs": n_done, "pred": summarize(agg["pred"]), "copy_baseline": summarize(agg["copy"]),
           # What was evaluated, not only how it was sampled: cF1 and the copy-baseline PSNR both depend on
           # the pair population and on the autoencoder that decodes it.
           "eval_env": {
               "config":       str(getattr(args, "config", "") or ""),
               "dataset_csv":  str(getattr(args, "dataset_csv", "") or ""),
               "latent_path":  str(getattr(args, "latent_path", "") or ""),
               "aekl_ckpt":    str(getattr(args, "aekl_ckpt", "") or ""),
               "res_scale":    float(getattr(args, "res_scale", 0.0) or 0.0),
               "norm_mode":    str(getattr(args, "norm_mode", "") or ""),
               "fm_res_noise": int(getattr(args, "fm_res_noise", 0) or 0),
               "fm_hist_mode": str(getattr(args, "fm_hist_mode", "none") or "none"),
               "split_v3":     int(getattr(args, "split_v3", 0) or 0),
               "split_v2":     int(getattr(args, "split_v2", 0) or 0),
               "test_part":    str(getattr(args, "test_part", "") or ""),
               "eval_strat":   int(getattr(args, "eval_strat", 0) or 0),
               "pair_keys":    os.environ.get("FM_PAIR_KEYS", ""),
               "copy_psnr_fingerprint": round(float(summarize(agg["copy"]).get("PSNR", float("nan"))), 4),
           },
           # Write the training arguments of the checkpoint into the results, so each eval result documents its own recipe
           "train_args": _train_args_of(_eval_args.fm_ckpt)}
    if _RAW and agg["pred_raw"]:
        out["raw_output"] = {
            "definition": "pred_raw = x0_raw + a*(up(pred) - up(x0)); a,b = lstsq(x0_raw ~ a*up(x0)+b) on brain; "
                          "up = trilinear resize of the model grid to the canonical raw grid (align_corners=False); "
                          "raw = Spacing 1.5 mm + ScaleIntensity(0,1) + ResizeWithPadOrCrop(canonical); pred_raw clipped to [0,1]; "
                          "brain = x0_raw>0.05 | gt_raw>0.05",
            "pred": summarize(agg["pred_raw"]), "copy_baseline": summarize(agg["copy_raw"])}
    if os.environ.get("FM_DUMP_PAIRS", "0") == "1":
        out["per_pair"] = {"pred": agg["pred"], "copy": agg["copy"]}
    print("=" * 70)
    print(json.dumps(out, indent=2))
    print("=" * 70)
    os.makedirs("ae_runs/fm_eval", exist_ok=True)
    with open(f"ae_runs/fm_eval/{_eval_args.tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval] wrote ae_runs/fm_eval/{_eval_args.tag}.json")


if __name__ == "__main__":
    main()


if _MAGREP:
    import numpy as _np
    _a=_np.array(_MAGREP)
    print(f"[LATENT-MAG] ||delta_hat||/||delta_gt|| = {_a[:,0].mean():.3f} | cos = {_a[:,1].mean():.3f} | optimal scaling = {_a[:,2].mean():.3f}  (n_batch={len(_a)})", flush=True)

if _BANDREP:
    import numpy as _np
    _bb = _np.array(_BANDREP)          # (n_batch, n_band)
    _names = globals().get("_BANDNAMES", [])
    print("[LATENT-BANDS] per-band directional accuracy cos(pred_band, gt_band)  (n_batch=%d)" % len(_bb), flush=True)
    for _i, _nm in enumerate(_names):
        print("    %-14s cos = %.4f" % (_nm, _bb[:, _i].mean()), flush=True)
    print("    Reading: compare against the corresponding ridge-regression profile; bands beyond the crossover point should use the ridge regression.", flush=True)

if _SPLITREP:
    import numpy as _np
    _b=_np.array(_SPLITREP)
    print(f"[LATENT-SPLIT] cos_in (first 8 principal components) = {_b[:,0].mean():.4f} | cos_tail = {_b[:,1].mean():.4f}"
          f" | ground-truth norm share in = {_b[:,2].mean():.4f} | predicted norm share in = {_b[:,3].mean():.4f}  (n_batch={len(_b)})", flush=True)
    print("  Reading: cos_tail much lower than cos_in means the model has learned only the large, low-value components and not the tail that determines the metric.", flush=True)
