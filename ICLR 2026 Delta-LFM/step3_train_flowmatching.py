import os, gc, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils import args
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import src.flow as _flowmod
import torch.nn.functional as F
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from monai.utils import set_determinism
from torch import nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from accelerate import Accelerator
from PIL import Image

from src.model3D import utils_usage as utils  
from utils import args, import_from_dotted_path, utils_metric

from src.flow import compute_ut, compute_xt

from accelerate import DistributedDataParallelKwargs
from accelerate.utils import DistributedDataParallelKwargs
import imageio.v2 as imageio

# Derived-artifact root (population priors, PCA bases, energy maps).
DERIVED_DIR = os.environ.get("DERIVED_DIR",
                             os.path.join(os.environ.get("DATA_DIR", "."), "derived"))


# Set the desired behavior
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

# Initialize accelerator with custom DDP config
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

DEVICE = accelerator.device


set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



args.output_dir = f"{args.temp_path}/{args.output_dir}"

os.makedirs(args.cache_dir,  exist_ok=True)
os.makedirs(args.output_dir, exist_ok=True)



spatial_size = (96, 96, 96)


def prepare_latent(latents, mask, ratio=4):
    size = [s // ratio for s in spatial_size]

    # BG mask
    shrink_mask = F.interpolate(
        (mask <= 0).float(),
        size=size,
        mode='trilinear',
        align_corners=False
    )

    return latents * shrink_mask, shrink_mask
      

# Linear
@torch.no_grad()
def sample_using_flowmatching(
        autoencoder: nn.Module,
        diffusion: nn.Module,
        x0, x1, context, class_cond,
        device: str,
        scale_factor: int = 1,
        num_training_steps: int = 1000,
        num_inference_steps: int = 50,
        schedule: str = 'scaled_linear_beta',
        beta_start: float = 0.0015,
        beta_end: float = 0.0205,
        verbose: bool = True
) -> torch.Tensor:
    """
    Sampling random brain MRIs that follow the covariates in `context`.

    Args:
        autoencoder (nn.Module): the KL autoencoder
        diffusion (nn.Module): the UNet
        context (torch.Tensor): the covariates
        device (str): the device ('cuda' or 'cpu')
        scale_factor (int, optional): the scale factor (see Rombach et Al, 2021). Defaults to 1.
        num_training_steps (int, optional): T parameter. Defaults to 1000.
        num_inference_steps (int, optional): reduced T for DDIM sampling. Defaults to 50.
        schedule (str, optional): noise schedule. Defaults to 'scaled_linear_beta'.
        beta_start (float, optional): noise starting level. Defaults to 0.0015.
        beta_end (float, optional): noise ending level. Defaults to 0.0205.
        verbose (bool, optional): print progression bar. Defaults to True.
    Returns:
        torch.Tensor: the inferred follow-up MRI
    """
    flowmatching.eval()

    z = x0

    t_steps = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device)
    progress_bar = tqdm(range(num_inference_steps), desc="Sampling", disable=not verbose)


    for i in progress_bar:
        t  = t_steps[i]
        dt = t_steps[i+1] - t

        t_batch = t.expand(z.shape[0])  # shape [B]
        with accelerator.autocast():
            v = flowmatching(x=z.float(), timesteps=t_batch, context=context, class_labels=class_cond)

        z = z + v * dt  # Euler forward

    # decode the latent
    z = z / scale_factor
    x = autoencoder.decode(z.to(device)).cpu() 

    return x





# Linear
@torch.no_grad()
def sample_using_flowmatching_heun(
        autoencoder: nn.Module,
        diffusion: nn.Module,
        x0, x1, context, class_cond,
        device: str,
        scale_factor: int = 1,
        num_training_steps: int = 1000,
        num_inference_steps: int = 50,
        schedule: str = 'scaled_linear_beta',
        beta_start: float = 0.0015,
        beta_end: float = 0.0205,
        verbose: bool = True
) -> torch.Tensor:
    """
    Sampling random brain MRIs that follow the covariates in `context`.

    Args:
        autoencoder (nn.Module): the KL autoencoder
        diffusion (nn.Module): the UNet
        context (torch.Tensor): the covariates
        device (str): the device ('cuda' or 'cpu')
        scale_factor (int, optional): the scale factor (see Rombach et Al, 2021). Defaults to 1.
        num_training_steps (int, optional): T parameter. Defaults to 1000.
        num_inference_steps (int, optional): reduced T for DDIM sampling. Defaults to 50.
        schedule (str, optional): noise schedule. Defaults to 'scaled_linear_beta'.
        beta_start (float, optional): noise starting level. Defaults to 0.0015.
        beta_end (float, optional): noise ending level. Defaults to 0.0205.
        verbose (bool, optional): print progression bar. Defaults to True.
    Returns:
        torch.Tensor: the inferred follow-up MRI
    """

    flowmatching.eval()
   
    z = x0.clone()
    t_steps = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device)
    for i in range(num_inference_steps):
        t  = t_steps[i]
        tp = t_steps[i+1]
        dt = tp - t

        t_b = t.expand(z.shape[0])
        with accelerator.autocast():
            v_t = flowmatching(x=z.float(), timesteps=t_b, context=context, class_labels=class_cond)

        z_pred = z + v_t * dt  # predictor

        tp_b = tp.expand(z.shape[0])
        with accelerator.autocast():
            v_tp = flowmatching(x=z_pred.float(), timesteps=tp_b, context=context, class_labels=class_cond)

        z = z + 0.5 * (v_t + v_tp) * dt  # corrector

    x = autoencoder.decode((z / scale_factor).to(device)).cpu()
    
    return x




mask_key          = "starting_seg"
file_key          = "starting_file"
broken_latent_key = "starting_latent"
latent_key        = "followup_latent"


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



_DRIFT_W = {}
def _drift_mu(batch, context, DEVICE, args):
    """Population drift mu_hat(delta_t, covariates), from the closed-form ridge regression in drift_W.npz."""
    import numpy as _np, torch as _t, os as _os
    global _DRIFT_W
    if not _DRIFT_W:
        p = os.path.join(DERIVED_DIR, "drift_W.npz")
        if not _os.path.exists(p):
            _DRIFT_W = {"none": True}; return None
        d = _np.load(p)
        _DRIFT_W = {"W": _t.tensor(d["W"]), "mu": _t.tensor(d["mu"]), "sd": _t.tensor(d["sd"])}
    if _DRIFT_W.get("none"): return None
    cov = context[:, 0, :8].to(DEVICE).float()
    dt = ((batch["followup_age"] - batch["starting_age"]).to(DEVICE).float() * 100.0).view(-1, 1)
    f = _t.cat([dt, _t.log1p(dt.clamp_min(0)), cov], 1)
    f = (f - _DRIFT_W["mu"].to(DEVICE)) / _DRIFT_W["sd"].to(DEVICE)
    f = _t.cat([f, _t.ones(f.shape[0], 1, device=DEVICE)], 1)
    out = f @ _DRIFT_W["W"].to(DEVICE).float()
    return out.view(-1, 4, 28, 32, 28)



_COMP_CH = {"first": 6, "prev1": 6, "prev2": 6, "energy": 4, "traj": 4, "accel": 6,
            "trajc": 13, "etraj": 6, "ietraj": 6, "pdrop": 3,
            "ek": 1, "ekind": 1, "ek2": 2, "egrid": 5, "vfin": 2, "vslope": 1, "anat": 4}   # ietraj/pdrop are image-space (native resolution) and cannot be represented in the latent; trajc fits per-channel trajectories; etraj fits a per-timepoint energy sequence
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
            # E_K = the energy map of the most recent scan pair (etraj channel 0)
            # ekind = E_K minus the population mean (channel 4, the individual deviation); ek2 supplies both
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


def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = v
    return new_state_dict


def save_image(path, x):
    """
    Save a single image to the specified path.
    """

    x = np.clip(x, 0, 1)
    x = (x * 255).astype(np.uint8)

    # Save the image
    imageio.imwrite(path, x)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def save_image(path, image_array):
    """Convert np array to uint8 image and save"""
    image_array = np.clip(image_array, 0, 1)  # normalize for safety
    image_array = (image_array * 255).astype(np.uint8)
    Image.fromarray(image_array).save(path)

def get_middle_slices(volume):  # [C, D, H, W]
    d, h, w = volume.shape[1:]
    axial = volume[:, d // 2, :, :]     # shape: [C, H, W]
    coronal = volume[:, :, h // 2, :]   # shape: [C, D, W]
    sagittal = volume[:, :, :, w // 2]  # shape: [C, D, H]
    return [axial, coronal, sagittal]

def save_grid_image_by_plane(image_np, recon_broken, recon_np, save_root, b, epoch, modality_names=None):
    image_channels = image_np[b].shape[0]
    row_labels = modality_names[:image_channels] if modality_names else [f"Mod{i}" for i in range(image_channels)]
    row_labels += ["Mask"]
    col_labels = ["Input", "Broken", "Recon"]
    total_rows = image_channels + 1
    total_cols = 3  # input, broken, recon
    plane_names = ["Axial", "Coronal", "Sagittal"]

    image_views  = get_middle_slices(image_np[b])
    broken_views = get_middle_slices(recon_broken[b])
    recon_views  = get_middle_slices(recon_np[b])


    for p, plane in enumerate(plane_names):
        fig, axes = plt.subplots(total_rows, total_cols, figsize=(total_cols * 2, total_rows * 2))

        for r in range(image_channels):
            # Input, Broken, Recon for modality r
            axes[r, 0].imshow(image_views[p][r], cmap="gray")
            axes[r, 1].imshow(broken_views[p][r], cmap="gray")
            axes[r, 2].imshow(recon_views[p][r], cmap="gray")
            for c in range(total_cols):
                axes[r, c].axis("off")
            axes[r, 0].set_ylabel(row_labels[r], fontsize=12)

        # Last row: only show mask
        axes[image_channels, 1].axis("off")
        axes[image_channels, 2].axis("off")
        axes[image_channels, 0].axis("off")
        axes[image_channels, 0].set_ylabel("Mask", fontsize=12)

        # Column headers
        for c in range(total_cols):
            axes[0, c].set_title(col_labels[c], fontsize=12)

        fig.suptitle(f"Sample {b} - {plane} View", fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(save_root, f"{epoch}_sample{b}_{plane}.jpg")
        plt.savefig(save_path, dpi=150)
        plt.close()


def images_to_tensorboard(
        batch,
        writer,
        epoch,
        mode,
        autoencoder,
        diffusion,
        scale_factor,
        modality_names = ["T1c", "T1n", "T2w", "T2f"],
        use_heun=False,
):
    """
    Visualize the generation on tensorboard
    """
    if int(getattr(args, "fm_res_noise", 0)) or int(getattr(args, "fm_x0_cond", 0)) or float(getattr(args, "fm_x0_start_noise", 0.0)) > 0:
        return   # viz sampler is 4ch; res_noise/x0_cond/x0_start_noise model is 8ch -> skip viz

    x0 = batch[broken_latent_key].to(DEVICE).clone() * scale_factor
    x1 = batch[latent_key].to(DEVICE).clone() * scale_factor

    # print("inputs_latents = ", inputs_latents.shape, "context = ", context.shape)

    ae = autoencoder.module if hasattr(autoencoder, "module") else autoencoder

    context      = batch['context'].to(DEVICE).squeeze(1)  # [B, ContextDim], ContextDim = 8
    starting_age = batch['starting_age'].to(DEVICE).unsqueeze(1)  # [B, 1]
    followup_age = batch['followup_age'].to(DEVICE).unsqueeze(1)   # [B, 1]
    starting_dia = batch['starting_diagnosis'].to(DEVICE).unsqueeze(1)  # [B, 1]
    followup_dia = batch['followup_diagnosis'].to(DEVICE).unsqueeze(1)   # [B, 1]

    age_gap = (batch["followup_age"] - batch["starting_age"]).to(DEVICE).float()  # [B, 1]
    age_gap = (age_gap * 100).long()

    context = torch.concatenate([context, 
                                 followup_age-starting_age,
                                 followup_dia-starting_dia], dim=1).float().unsqueeze(1)  

    with torch.no_grad(), accelerator.autocast():
        sample_func = sample_using_flowmatching_heun if use_heun else sample_using_flowmatching

        image = sample_func(
            autoencoder=ae,
            diffusion=diffusion,
            x0=x0, #inputs_latents,
            x1=x1,
            context=context,
            class_cond=age_gap,
            num_inference_steps=100,
            device=DEVICE,
            scale_factor=scale_factor
        )

        recon_origin = ae.decode(x0 / scale_factor).cpu().numpy()  # [B, 1, H, W], decode_stage_2_outputs
        recon_broken = ae.decode(x1 / scale_factor).cpu().numpy()  # [B, 1, H, W]


    image_np     = recon_origin  #.cpu().numpy()  # [B, 1, H, W]
    recon_np     = image.cpu().numpy()  #.max(axis=1, keepdims=True)  # [B, 3, H, W] -> [B, 1, H, W]

    save_root = "./oasis_fm_samples"

    if use_heun:
        save_root += "_heun"

    os.makedirs(save_root, exist_ok=True)

    # for b in range(image_np.shape[0]):
   
    modality_names = [m.upper() for m in modality_names]

    from statistics import mean

    psnr_fake_vs_target_all,  ssim_fake_vs_target_all = [], []
    psnr_input_vs_target_all, ssim_input_vs_target_all = [], []
    psnr_input_vs_fake_all,   ssim_input_vs_fake_all = [], []

    # ------------------------------------------------
    # - image_np     = ground truth (decoded real followup latent)
    # - recon_np     = prediction (decoded sampled latent)
    # - recon_broken = input (decoded starting latent)
    # ------------------------------------------------

    # Fake → Target  (reconstruction vs ground truth)
    psnr_val = utils_metric.psnr_3d(recon_np, image_np)
    ssim_val = utils_metric.ssim_3d(recon_np, image_np)
    psnr_fake_vs_target_all.append(psnr_val)
    ssim_fake_vs_target_all.append(ssim_val)

    # Input → Target  (corrupted vs ground truth)
    psnr_val = utils_metric.psnr_3d(recon_broken, image_np)
    ssim_val = utils_metric.ssim_3d(recon_broken, image_np)
    psnr_input_vs_target_all.append(psnr_val)
    ssim_input_vs_target_all.append(ssim_val)

    # Input → Fake  (corrupted vs reconstruction)
    psnr_val = utils_metric.psnr_3d(recon_broken, recon_np)
    ssim_val = utils_metric.ssim_3d(recon_broken, recon_np)
    psnr_input_vs_fake_all.append(psnr_val)
    ssim_input_vs_fake_all.append(ssim_val)

    # ------------------------------------------------
    # Print summary
    # ------------------------------------------------
    print(f"""
       [Validation Summary over Epoch {epoch}]
       -------------------------------------------------------
       | Pair              |   PSNR (↑)     |   SSIM (↑)      |
       -------------------------------------------------------
       | Fake → Target     | {np.mean(psnr_fake_vs_target_all):.3f}      | {np.mean(ssim_fake_vs_target_all):.3f}
       | Input → Target    | {np.mean(psnr_input_vs_target_all):.3f}      | {np.mean(ssim_input_vs_target_all):.3f}
       | Input → Fake      | {np.mean(psnr_input_vs_fake_all):.3f}      | {np.mean(ssim_input_vs_fake_all):.3f}
       -------------------------------------------------------
    """)



    print("image save to:", save_root)
    for b in range(min(image_np.shape[0], 3)):
        # print("image_np[b] =", image_np[b].shape, recon_broken[b].shape, recon_np[b].shape, context_np[b].shape)
        save_grid_image_by_plane(image_np, recon_broken, recon_np, 
                                 save_root, b, epoch,
                                 modality_names=modality_names[:image_np[b].shape[0]])



class CustomPerceptualLoss(nn.Module):
    def __init__(self, in_channels=3,):
        super().__init__()

        from monai.networks.nets import resnet10, resnet18, resnet34
        import torch.nn as nn

        # Load pretrained DenseNet121 on MedNIST (6-class)
        feature_model = resnet18(spatial_dims=3, n_input_channels=1, 
                                    pretrained=True, feed_forward=False,       
                                    shortcut_type='A',
                                    bias_downsample=True)

        # Adjust the model in_channels if necessary
        if in_channels != 1:
            # Step 2: Modify the first conv layer to accept more channels
            old_conv = feature_model.conv1  #stem[0]
            new_conv = nn.Conv3d(
                in_channels=in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )

            # Step 3: Copy pretrained weights
            with torch.no_grad():
                if in_channels == 1:
                    new_conv.weight.copy_(old_conv.weight)
                elif in_channels < 4:
                    # Repeat weights from the 1-channel model
                    new_conv.weight.copy_(old_conv.weight.repeat(1, in_channels, 1, 1, 1) / in_channels)
                else:
                    # Use mean across input dim if in_channels is large
                    new_conv.weight.copy_(
                        old_conv.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1, 1)
                    )

            # Replace the conv layer
            feature_model.conv1 = new_conv



        feature_model.eval()
        for p in feature_model.parameters():
            p.requires_grad = False


        # print("feature_model = ", feature_model)

        self.feature_extractor = nn.Sequential(
            feature_model.conv1,
            feature_model.bn1,
            feature_model.act,
            feature_model.maxpool,
            feature_model.layer1,
            feature_model.layer2
        )

        self.criterion = nn.L1Loss()

    def forward(self, input, target):
        # Get features from both inputs
        input_feats  = self.feature_extractor(input).detach()
        target_feats = self.feature_extractor(target)
        return self.criterion(input_feats, target_feats)



if __name__ == '__main__':

    use_rectified = True

    if use_rectified:
        args.output_dir += "_rectified"

    print("output_dir=", args.output_dir)
    # Record the training arguments so the recipe remains recoverable from the checkpoint directory alone
    try:
        import json as _js, os as _os
        _ad = str(args.output_dir)          # already contains _rectified at this point, i.e. the checkpoint directory
        _os.makedirs(_ad, exist_ok=True)
        with open(_os.path.join(_ad, "args.json"), "w") as _f:
            _js.dump({k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                      for k, v in vars(args).items()}, _f, indent=2, ensure_ascii=False)
        print("[_ARGSAVE] training arguments written to %s/args.json" % _ad, flush=True)
    except Exception as _e:
        print("[_ARGSAVE] failed to store arguments (training is unaffected): %r" % (_e,), flush=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------- Define Dataloader ----------------
    num_train_timesteps = 1000
    
    from dataset.oasis_dataset_3D_pair_latent import get_brain_dataset

    train_loader, ds, id_to_label = get_brain_dataset(args, mode="train", with_seg=False)
    test_loader, test_ds, _       = get_brain_dataset(args, mode="test", with_seg=False)


    print("Setting up Autoencoder model...")
    autoencoder_func = import_from_dotted_path(args.autoencoder)
    autoencoder      = autoencoder_func(args).to(DEVICE).float()

    try:
        weight = torch.load(args.aekl_ckpt)
        weight = remove_module_prefix(weight)

        autoencoder.load_state_dict(weight)
        print("Successful load: ", args.aekl_ckpt)
    except FileNotFoundError:
        print(f"File {args.aekl_ckpt} not found, using random initialization for autoencoder.")
        

    autoencoder.to(DEVICE)
    autoencoder.eval()  # Important for inference
    for p in autoencoder.parameters():
        p.requires_grad = False



    try:
        perceptual_loss_fn = CustomPerceptualLoss(in_channels=4)  # in_channels=4
    except Exception as _pe:
        import torch.nn as _nn; perceptual_loss_fn = _nn.Identity()  # offline: MedicalNet unavailable; perceptual is UNUSED in the loss (commented out)
        print("[fm] perceptual loss disabled (offline):", _pe)

    flowmatching_func = import_from_dotted_path(args.flowmatching)
    _res_noise = int(getattr(args, "fm_res_noise", 0))  # Δ-Res-Flow: noise->delta, x0 as concat condition (8-in/4-out)
    _x0_cond = int(getattr(args, "fm_x0_cond", 0))       # x0-start flow, x0 as explicit concat condition (8-in/4-out)
    _x0_sn = float(getattr(args, "fm_x0_start_noise", 0.0))  # noisy-x0-start (stochastic interpolant), 8-in/4-out
    _nce = 16 if float(getattr(args, "fm_cfg_dropout", 0.0)) > 0 else 15  # CFG: extra null class slot idx=15
    _echan = 1 if str(getattr(args, "fm_energy_cond", "none")) != "none" else 0  # energy field as extra input channel
    if _res_noise or _x0_cond or _x0_sn > 0:
        _hm = str(getattr(args, "fm_hist_mode", "none"))
        _hc = int(getattr(args, "fm_hist_concat", 0))
        _hinj = str(getattr(args, "fm_hist_inject", "concat"))
        _hch = hist_channels(_hm) if _hm != "none" else 0
        _hcat = (_hch if _hinj in ("concat", "both") else 0) if _hm != "none" else ((16 if _hc >= 2 else 12) if _hc else 0)
        _cdim = 10 + (2 * _hch if (_hm != "none" and _hinj in ("cond", "both")) else 0)   # pooled history -> time embedding
        _spm = str(getattr(args, "fm_spade_mode", "none"))
        _sdim = hist_channels(_spm) if _spm != "none" else 0   # input channels of the multi-scale SPADE spatial modulation
        _gm = str(getattr(args, "fm_gain_mode", "none"))     # components routed through the spatial-modulation path, separate from the concat path
        _gch = hist_channels(_gm) if _gm != "none" else _hch
        _use_gain = (("gain" in _hinj) or _gm != "none") and (_hm != "none" or _gm != "none")
        if _use_gain and _hinj == "gain" and _gm == "none": _hcat = 0
        flowmatching = flowmatching_func(args, in_channels=8 + _echan + _hcat, use_image=False, out_channels=4, num_class_embeds=_nce, cond_dim=_cdim, spade_dim=_sdim).to(DEVICE)
    else:
        flowmatching = flowmatching_func(args, in_channels=4, use_image=False, num_class_embeds=_nce).to(DEVICE)   # , use_image=False)
    if _res_noise and _use_gain:
        flowmatching.gain_net = _build_gain_net(_gch, DEVICE)   # attached as an attribute so it enters the state_dict
        _gsrc = _gm if _gm != "none" else _hm
        print(f"[gain] spatial-modulation network {_gch}ch -> 4ch, zero-initialized (g=1); source={_gsrc}", flush=True)
    _init = str(getattr(args, "fm_init_ckpt", ""))
    if _init:
        _sd = torch.load(_init, map_location=DEVICE)
        if isinstance(_sd, dict) and "state_dict" in _sd: _sd = _sd["state_dict"]
        if not isinstance(_sd, dict): _sd = _sd.state_dict()
        if _echan and isinstance(_sd, dict) and "conv_in.conv.weight" in _sd and _sd["conv_in.conv.weight"].shape[1] == 8:
            _mw = flowmatching.state_dict()["conv_in.conv.weight"].clone()
            _mw[:, :8] = _sd["conv_in.conv.weight"].to(_mw.device, _mw.dtype); _mw[:, 8:] = 0  # energy channel starts inert -> exact warm-start
            _sd["conv_in.conv.weight"] = _mw
        # Input-channel expansion (8->20): new channel weights are zeroed, so the warm-start is bit-for-bit equivalent to the original model
        for _k in list(_sd.keys()):
            if _k.endswith('cond_mlp.0.weight'):
                # cond_vec expansion (10 -> 10+2*hist_ch): new dimensions are zeroed, so the warm-start is equivalent
                _wc = _sd[_k]
                _tc = flowmatching.cond_mlp[0].weight
                if _wc.shape[1] < _tc.shape[1]:
                    _nc = torch.zeros_like(_tc)
                    _nc[:, :_wc.shape[1]] = _wc.to(_nc.dtype)
                    _sd[_k] = _nc
                    print(f'[hist_inject] cond_mlp {_wc.shape[1]} -> {_tc.shape[1]} dim, new dimensions zero-initialized', flush=True)
            if _k.endswith('conv_in.conv.weight'):
                _w = _sd[_k]
                _tgt = flowmatching.conv_in.conv.weight
                if _w.shape[1] < _tgt.shape[1]:
                    _new = torch.zeros_like(_tgt)
                    _new[:, :_w.shape[1]] = _w.to(_new.dtype)
                    _sd[_k] = _new
                    print(f'[hist_concat] conv_in {_w.shape[1]} -> {_tgt.shape[1]} ch, new channels zero-initialized', flush=True)

        _r = flowmatching.load_state_dict(_sd, strict=False)
        print(f"[warm-start] loaded {_init} missing={len(_r.missing_keys)} unexpected={len(_r.unexpected_keys)}", flush=True)


    controlnet = None
    _hinj_c = str(getattr(args, "fm_hist_inject", "concat"))
    if int(getattr(args, "fm_controlnet", 0)) or _hinj_c in ("cnet", "cnetft"):
        from src.model3D.networks import init_large_controlnet
        _cm = str(getattr(args, "fm_cnet_cond", "scan"))
        if _hinj_c in ("cnet", "cnetft"):
            _cnc = hist_channels(str(getattr(args, "fm_hist_mode", "none")))   # the same components used by concat
        else:
            _cnh0 = str(getattr(args, "fm_cnet_hist", "none"))
            if _cnh0 != "none":
                _cnc = hist_channels(_cnh0)   # the ControlNet condition is independent of the backbone concat input
            else:
                _cnc = 16 if _cm == "multi" else (13 if _cm == "hist13" else (5 if _cm == "energy" else (8 if _cm == "traj" else 4)))
        _hm_c = str(getattr(args, "fm_hist_mode", "none"))
        _inch_c = 8 + (hist_channels(_hm_c) if (_hm_c != "none" and _hinj_c in ("concat", "both")) else 0)
        controlnet = init_large_controlnet(None, in_channels=_inch_c, conditioning_embedding_in_channels=_cnc, cross_attention_dim=10).to(DEVICE)
        if _hinj_c == "cnetft":
            print(f"[cnetft] ControlNet on, backbone trainable (isolates the injection mechanism so it is strictly comparable with concat), cond {_cnc}ch", flush=True)
        else:
            for _p in flowmatching.parameters(): _p.requires_grad = False   # standard ControlNet: the backbone is frozen
            flowmatching.eval()
            print(f"[cnet] ControlNet on, backbone frozen (standard usage), cond {_cnc}ch", flush=True)
    eimg_encoder = None
    if int(getattr(args, "fm_eimg_xattn", 0)):
        from src.model3D.history_encoder import HistoryEncoder
        eimg_encoder = HistoryEncoder(in_ch=1, ctx_dim=10,
                                      n_tokens=int(getattr(args, "fm_eimg_ntok", 64))).to(DEVICE)
        print(f"[eimg-xattn] native-resolution energy -> learned encoder -> {int(getattr(args,'fm_eimg_ntok',64))} tokens (no fixed pooling)", flush=True)
    history_encoder = None
    _xam = str(getattr(args, "fm_xattn_mode", "none"))
    if _xam != "none":
        from src.model3D.history_encoder import HistoryEncoder
        history_encoder = HistoryEncoder(in_ch=hist_channels(_xam), ctx_dim=10,
                                         n_tokens=int(getattr(args, "fm_xattn_ntok", 64))).to(DEVICE)
        print(f"[xattn] cross-attention history encoder: {hist_channels(_xam)}ch -> {int(getattr(args, 'fm_xattn_ntok', 64))} tokens x 10dim", flush=True)
    if int(getattr(args, "fm_history", 0)):
        from src.model3D.history_encoder import HistoryEncoder
        history_encoder = HistoryEncoder(in_ch=4, ctx_dim=10, n_tokens=int(getattr(args, "fm_history_ntok", 64))).to(DEVICE)  # ctx_dim=10=len(CONDITIONING_VARIABLES)+2
        print(f"[fm_history] HistoryEncoder ON n_tokens={int(getattr(args, 'fm_history_ntok', 64))}", flush=True)
    if controlnet is not None and str(getattr(args, "fm_hist_inject", "concat")) == "cnetft":
        _fm_params = list(controlnet.parameters()) + list(flowmatching.parameters())
    elif controlnet is not None:
        _fm_params = list(controlnet.parameters())
    else:
        _fm_params = list(flowmatching.parameters()) + (list(history_encoder.parameters()) if history_encoder is not None else [])
        if eimg_encoder is not None: _fm_params = _fm_params + list(eimg_encoder.parameters())
    _wd = float(getattr(args, "fm_wd", 1e-6))   # 1e-6 is effectively unregularized; raise it to regularize
    optimizer = torch.optim.AdamW(_fm_params, lr=args.lr, weight_decay=_wd)  # AdamW
    print("[reg] weight_decay=%g" % _wd, flush=True)
    _subw = float(getattr(args, "fm_sub_w", 0.0))
    _subV = None
    if _subw > 0.0:
        _pc = np.load(os.path.join(DERIVED_DIR, "change_pca_K64.npz"))
        _kk = int(getattr(args, "fm_sub_k", 40))
        _subV = torch.tensor(_pc["V"][:, :_kk], device=DEVICE, dtype=torch.float32)   # (D,K)
        _subMu = torch.tensor(_pc["mean"], device=DEVICE, dtype=torch.float32)
        print(f"[fm_sub] subspace constraint on K={_kk} w={_subw}", flush=True)
    _ema_d = float(getattr(args, "fm_ema", 0.0))
    ema_model = None
    if _ema_d > 0.0:
        import copy as _copy
        ema_model = _copy.deepcopy(flowmatching).eval()
        for _p in ema_model.parameters(): _p.requires_grad = False
        print(f"[fm_ema] EMA on decay={_ema_d}", flush=True)
    # ---- adversarial (distribution matching) on the change field ----
    _advw = float(getattr(args, "fm_adv_w", 0.0))
    disc = disc_opt = adv_fn = None
    if _advw > 0.0:
        from src.model3D.networks import init_patch_discriminator
        from monai.losses import PatchAdversarialLoss
        disc = init_patch_discriminator(None, spatial_dims=3, in_channels=4, num_layers_d=3).to(DEVICE)
        disc_opt = torch.optim.AdamW(disc.parameters(), lr=args.lr, weight_decay=1e-6)
        adv_fn = PatchAdversarialLoss(criterion="least_squares")
        print(f"[fm_adv] discriminator ON w={_advw} warmup={int(getattr(args,'fm_adv_warmup',200))}", flush=True)

    # with torch.no_grad(), accelerator.autocast():
    #     z = data_loader.dataset[0:5][latent_key]
    a = train_loader.dataset[0][broken_latent_key]

    with torch.no_grad(), accelerator.autocast():
        z_list = [train_loader.dataset[i][broken_latent_key] for i in range(10)]
        z = torch.stack(z_list, dim=0)  # Stack into a single tensor


    scale_factor = 1 / torch.std(z)  # normalizes the latents to unit std, estimated on the first few training latents
    print(f"Scaling factor set to {scale_factor}")

    autoencoder, optimizer, train_loader, flowmatching, perceptual_loss_fn = accelerator.prepare(
        autoencoder, optimizer, train_loader, flowmatching, perceptual_loss_fn, 
        # ddp_kwargs={"find_unused_parameters": True}
    )
    if history_encoder is not None:
        history_encoder = accelerator.prepare(history_encoder)
    if controlnet is not None:
        controlnet = accelerator.prepare(controlnet)

    writer   = SummaryWriter()
    global_counter = {'train': 0}  # , 'valid': 0 }
    loaders  = {'train': train_loader}  # , # 'valid': valid_loader }
    datasets = {'train': train_loader.dataset}  # , 'valid': validset }

    ae = autoencoder.module if hasattr(autoencoder, "module") else autoencoder
    gradient_accumulation_steps = args.grad_accum_steps if hasattr(args, 'grad_accum_steps') else 4  # for example


    for epoch in range(args.n_epochs):

        for mode in loaders.keys():
            loader = loaders[mode]
            flowmatching.train() if mode == 'train' else flowmatching.eval()
            epoch_loss = 0
            epoch_mse_ps = 0
            epoch_cos_ps = 0
            epoch_recon = 0

            progress_bar = tqdm(enumerate(loader), total=len(loader))
            progress_bar.set_description(f"{mode.upper()} Epoch {epoch}")

            for step, batch in progress_bar:
                if args.DEBUG and step >= 10:
                    print(f"[DEBUG] Step {step}: {batch[broken_latent_key].shape}")
                    break

                # Use to be  context: context tensor (N, 1, ContextDim).
                B = batch[latent_key].shape[0]
                _pc_loss = None   # the path-consistency loss is only assigned in the res_noise branch; this default prevents a NameError on other branches
                _flowmod._RATE_POW = float(getattr(args, "fm_rate_pow", 1.0))

                eps = 1e-3
                _ts = str(getattr(args, "fm_t_schedule", "uniform"))
                if _ts == "logitnorm":
                    t = torch.sigmoid(torch.randn(B, device=DEVICE)).clamp(eps, 1 - eps)
                elif _ts == "late":
                    # concentrate t near 1: that segment sets the delta's final sharpness/magnitude
                    _p = float(getattr(args, "fm_t_power", 3.0))
                    t = torch.rand(B, device=DEVICE).pow(1.0 / _p).clamp(eps, 1 - eps)
                else:
                    t = torch.rand(B, device=DEVICE, dtype=torch.float32).clamp(eps, 1 - eps)
                _ttdt = float(getattr(args, "fm_train_tdt", 0.0))
                if _ttdt > 0.0:
                    _yrs = (batch["followup_age"] - batch["starting_age"]).to(DEVICE).float() * 100.0
                    t = t * (_yrs / _ttdt).clamp(0.15, 1.0)      # train on [0,T_i] only, matching the truncated sampler
                _tmin = float(getattr(args, "fm_t_min", 0.0))
                if _tmin > 0.0:
                    t = _tmin + (1.0 - _tmin) * t     # train on the [t_min,1] segment only


                context      = batch['context'].to(DEVICE).squeeze(1)  # [B, ContextDim], ContextDim = 8
                starting_age = batch['starting_age'].to(DEVICE).unsqueeze(1)  # [B, 1]
                followup_age = batch['followup_age'].to(DEVICE).unsqueeze(1)   # [B, 1]
                starting_dia = batch['starting_diagnosis'].to(DEVICE).unsqueeze(1)  # [B, 1]
                followup_dia = batch['followup_diagnosis'].to(DEVICE).unsqueeze(1)   # [B, 1]

                age_gap = (batch["followup_age"] - batch["starting_age"]).to(DEVICE).float()  # [B, 1]
                age_gap = (age_gap * 100).long()


                context = torch.concatenate([context, 
                                             followup_age-starting_age,
                                             followup_dia-starting_dia], dim=1).float().unsqueeze(1)  
                if history_encoder is not None and int(getattr(args, "fm_history", 0)):
                    _prior = batch["prior_latent"].to(DEVICE).float() * scale_factor
                    _hp = batch["has_prior"].to(DEVICE).float() if "has_prior" in batch else None
                    context = torch.cat([context, history_encoder(_prior, _hp)], dim=1)  # (B, 1+N, 10)

                x0        = batch[broken_latent_key].to(DEVICE).clone() * scale_factor
                x1        = batch[latent_key].to(DEVICE).clone() * scale_factor


                _adv_cache = None
                use_rectified = True
                # with autocast(device_type='cuda',enabled=True):
                with accelerator.autocast():
                    if mode == 'train': optimizer.zero_grad(set_to_none=True)

                    # x0 -> x1
                    sigma_min = float(getattr(args, "fm_sigma", 0.0))  # --fm_sigma: 0=deterministic, >0=Brownian
                    _dt = (batch["followup_follow_up"].to(DEVICE).float() - batch["starting_follow_up"].to(DEVICE).float()) / 10.0  # follow_up months/10 (1mo=0.1)
                    _grav_noise = None   # additive noise so delta_hat = pred + _grav_noise (energy-gravity loss)
                    if int(getattr(args, "fm_res_noise", 0)):
                        # Δ-Res-Flow: flow from NOISE z -> delta(=x1-x0), x0 is the condition (concat).
                        # Removes the copy component entirely -> all capacity on the CHANGE.
                        delta = (x1 - x0) * float(getattr(args, "fm_delta_train_scale", 1.0))  # bake-in δ: flow to β·delta
                        # ---- rate parameterization (--fm_scheme != std): divide the target by delta_t^p, leaving the denoising axis unchanged ----
                        # This branch builds its own target, so --fm_scheme has to be applied here explicitly.
                        if str(getattr(args, "fm_scheme", "std")) != "std":
                            _rp = float(getattr(args, "fm_rate_pow", 1.0))
                            delta = delta / _dt.view(-1, *([1] * (delta.dim() - 1))).clamp_min(1e-3).pow(_rp)
                        z = (torch.zeros_like(delta) if int(getattr(args, "fm_det", 0))
                             else torch.randn_like(delta) * float(getattr(args, "fm_flowinit_sigma", 1.0)))
                        _ecl = float(getattr(args, "fm_flowinit_ecov", 0.0))
                        if _ecl > 0.0 and "etraj" in batch:
                            # Energy-shaped covariance: give more stochastic freedom where change is expected (modulates uncertainty, not the signal)
                            _eci = 4 if str(getattr(args, "fm_ecov_src", "ekind")) == "ekind" else 0
                            _ecm = batch["etraj"].to(DEVICE).float()[:, _eci:_eci+1].abs()
                            _ecm = _ecm / (_ecm.amax(dim=(2, 3, 4), keepdim=True) + 1e-6)
                            z = z * (1.0 + _ecl * _ecm)
                        _fim = str(getattr(args, "fm_flowinit", "none"))
                        if _fim != "none":
                            _dty = ((batch["followup_follow_up"] - batch["starting_follow_up"]).to(DEVICE).float() / 12.0)
                            _mu0 = flow_init_mu(batch, x0, scale_factor, DEVICE, _fim, _dty)
                            if _mu0 is not None:
                                z = z + _mu0 * float(getattr(args, "fm_flowinit_w", 1.0))
                        _tt = t.view(-1, *([1] * (delta.dim() - 1)))
                        # ===== time warping: per-voxel effective time t*s(x) =====
                        # s(x) = 1 + w*(hist_energy/mean - 1), clamp[0.5,2]
                        # Regions that changed quickly in the past advance further along the flow. No additional parameters, so it is directly deployable.
                        # Motivation: hippocampal and entorhinal atrophy rates exceed the whole-brain mean, and the distribution differs per patient.
                        _twm = str(getattr(args, "fm_timewarp", "none"))
                        if _twm != "none":
                            _tw_src = batch.get("etraj", None)
                            if _tw_src is not None:
                                # Channel selection matters: etraj = [E_last, a_E, b_E, r_E, E_ind, cov]
                                #   E_last (ch0): dominated by a population-shared anatomical prior, so the warp is nearly identical across subjects and not individualized
                                #   E_ind  (ch4): the individual deviation after removing the population component, the channel carrying patient-specific information
                                #   a_E    (ch1): the time derivative of energy (progression rate)
                                _tws = str(getattr(args, "fm_timewarp_src", "ind"))
                                _ci = {"last": 0, "rate": 1, "ind": 4}.get(_tws, 4)
                                _e = _tw_src.to(DEVICE).float()[:, _ci:_ci+1].abs()
                                _em2 = _e.mean(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
                                _sx = (1.0 + float(getattr(args, "fm_timewarp_w", 0.5)) * (_e / _em2 - 1.0)).clamp(0.5, 2.0)
                                _tt = (_tt * _sx).clamp(0.0, 1.0)
                        # ---- population-prior residualization (--fm_gbar): learn the individual deviation only ----
                        _gbf = str(getattr(args, "fm_gbar", "") or "")
                        if _gbf:
                            if "_GBAR" not in globals():
                                _gz = np.load(_gbf)
                                globals()["_GBAR"] = torch.from_numpy(_gz["mean"]).float()
                                print("[gbar] population prior loaded; the training target is now delta - gbar", flush=True)
                            _gb = globals()["_GBAR"].to(delta.device).reshape(1, *delta.shape[1:])
                            delta = delta - _gb * scale_factor
                        xt = (1.0 - _tt) * z + _tt * delta          # base -> delta path
                        if int(getattr(args, "fm_x0_pred", 0)):
                            ut = delta                               # x0-pred: regress delta directly, so the target variance does not diverge as t approaches 1
                        else:
                            ut = delta - z                           # velocity (const along linear path)
                        _x0d = float(getattr(args, "fm_x0_drop", 0.0))
                        _x0c = x0
                        if _x0d > 0.0 and mode == 'train':
                            _km = (torch.rand(x0.shape[0], 1, 1, 1, 1, device=x0.device) >= _x0d).float()
                            _x0c = x0 * _km          # when x0 is dropped, the network can only draw information from xt, which contains z
                        model_in = torch.cat([xt, _x0c], dim=1)     # 8ch: [xt(delta-space), x0(condition)]
                        _hmf = str(getattr(args, "fm_hist_mode", "none"))
                        _hpool = None; _gfield = None
                        if _hmf != "none" or str(getattr(args, "fm_gain_mode", "none")) != "none":
                            batch["_prev2_fill"] = str(getattr(args, "fm_prev2_fill", "zero"))
                            _hcnd = build_hist_cond(batch, x0, scale_factor, DEVICE, _hmf) if _hmf != "none" else None
                            _hd2 = float(getattr(args, "fm_hist_drop", 0.0))
                            if _hd2 > 0.0 and mode == 'train' and _hcnd is not None:
                                _km3 = (torch.rand(_hcnd.shape[0], 1, 1, 1, 1, device=_hcnd.device) >= _hd2).float()
                                _hcnd = _hcnd * _km3
                            _inj = str(getattr(args, "fm_hist_inject", "concat"))
                            if _inj in ("concat", "both") and _hcnd is not None:
                                model_in = torch.cat([model_in, _hcnd], dim=1)
                            # cnet/cnetft: history goes through the ControlNet side path and does not enter the backbone input
                            _gmf = str(getattr(args, "fm_gain_mode", "none"))
                            if getattr(flowmatching, "gain_net", None) is not None and ("gain" in _inj or _gmf != "none"):
                                # Separate injection: geometry (prev visits) enters as features via concat; energy enters as weights via spatial modulation
                                _gin = build_hist_cond(batch, x0, scale_factor, DEVICE, _gmf) if _gmf != "none" else _hcnd
                                _gfield = 1.0 + flowmatching.gain_net(_gin.float())
                                if int(getattr(args, 'fm_gain_norm', 0)):
                                    # Total-preserving constraint: g <- g / mean(g), normalized per sample over the spatial and channel dimensions, so the total magnitude is unchanged and only redistributed
                                    _rd = tuple(range(1, _gfield.ndim))
                                    _gm = _gfield.mean(dim=_rd, keepdim=True).clamp_min(1e-6)
                                    _gfield = _gfield / _gm
                                    if not globals().get('_GNORM_SHOWN'):
                                        globals()['_GNORM_SHOWN']=1
                                        print('[gain_norm] gain field constrained to a mean-1 multiplicative redistribution (total preserved)', flush=True)

                            if _inj in ("cond", "both"):
                                # Spatial pooling -> global vector -> added to cond_vec -> modulates every resblock
                                # History carries both where change occurs, which suits concat, and how fast the subject declines overall, which suits global modulation
                                _hpool = torch.cat([_hcnd.mean(dim=(2, 3, 4)), _hcnd.std(dim=(2, 3, 4))], dim=1)
                        elif int(getattr(args, "fm_hist_concat", 0)):
                            # History enters the backbone input directly rather than through the ControlNet side path, so the signal is present from the first convolution and the whole network can reorganize around it
                            _mc = build_multi_cond(batch, x0, scale_factor, DEVICE)   # 16ch, the first 4 of which are x0
                            _hd = float(getattr(args, "fm_hist_drop", 0.0))
                            if _hd > 0.0 and mode == 'train':
                                # History dropout (the CFG null-token idea): some samples have no history at all (all channels zero),
                                # and "zero" is numerically indistinguishable from "history present but no change", so the two modes interfere.
                                # Dropping history at random forces the network to learn them as two distinct modes rather than one blended mode.
                                _km2 = (torch.rand(_mc.shape[0], 1, 1, 1, 1, device=_mc.device) >= _hd).float()
                                _mc = _mc * _km2
                            model_in = torch.cat([model_in, _mc[:, 4:]], dim=1)       # +12ch
                            if int(getattr(args, "fm_hist_concat", 0)) >= 2 and "cum_energy" in batch:
                                # Cumulative change-field condition: [C_cum, C_rate, C_ind (the individual residual), coverage]
                                model_in = torch.cat([model_in, batch["cum_energy"].to(DEVICE).float()], dim=1)
                        # x0-prediction regresses delta directly, so the prediction must not be shifted by the noise
                        _grav_noise = None if int(getattr(args, "fm_x0_pred", 0)) else z
                    elif float(getattr(args, "fm_x0_start_noise", 0.0)) > 0:
                        # Noisy-x0-start (stochastic interpolant): xt = x0 + t*delta + sigma*(1-t)*z.
                        # t=0 = x0+sigma*z (noisy anchor breaks copy attractor); t=1 = x1. x0 also concat condition.
                        _sn = float(getattr(args, "fm_x0_start_noise", 0.0))
                        delta = x1 - x0
                        z = torch.randn_like(delta)
                        _tt = t.view(-1, *([1] * (delta.dim() - 1)))
                        xt = x0 + _tt * delta + _sn * (1.0 - _tt) * z
                        ut = delta - _sn * z
                        model_in = torch.cat([xt, x0], dim=1)
                        _grav_noise = _sn * z
                    elif int(getattr(args, "fm_x0_cond", 0)):
                        # x0-start flow (x0->x1) but x0 given as EXPLICIT concat condition.
                        # Model computes change relative to the anchor -> breaks copy collapse.
                        xt = compute_xt(x0=x0, x1=x1, t=t, sigma_min=sigma_min,
                                        noise_schedule=getattr(args, "fm_noise_schedule", "bridge"))
                        ut = compute_ut(x0=x0, x1=x1, t=t, dt=(_dt if getattr(args, "fm_scheme", "std") == "realtime" else None))
                        model_in = torch.cat([xt, x0], dim=1)        # 8ch: [xt, x0(anchor)]
                    else:
                        xt = compute_xt(x0=x0, x1=x1, t=t,  sigma_min=sigma_min,
                                        noise_schedule=getattr(args, "fm_noise_schedule", "bridge"))
                        ut = compute_ut(x0=x0, x1=x1, t=t, dt=(_dt if getattr(args, "fm_scheme", "std") == "realtime" else None))
                        model_in = xt

                    if float(getattr(args, "fm_cfg_dropout", 0.0)) > 0:  # CFG: drop cond to null idx=15
                        _dp = torch.rand(age_gap.shape[0], device=age_gap.device) < float(getattr(args, "fm_cfg_dropout", 0.0))
                        age_gap = torch.where(_dp, torch.full_like(age_gap, 15), age_gap)
                    _ecmode = str(getattr(args, "fm_energy_cond", "none"))
                    if _ecmode != "none":
                        if _ecmode == "latoracle":
                            _en = (x1 - x0).abs().mean(1, keepdim=True)                 # TRUE latent-change oracle |x1-x0|
                            _en = _en / (_en.amax(dim=(2, 3, 4), keepdim=True) + 1e-6)
                        elif _ecmode == "latpast":
                            _pr = batch["prior_latent"].to(DEVICE).float() * scale_factor
                            _en = (x0 - _pr).abs().mean(1, keepdim=True)                 # deployable past latent-change |x0-prior|
                            _en = _en / (_en.amax(dim=(2, 3, 4), keepdim=True) + 1e-6)
                            if "has_prior" in batch: _en = _en * batch["has_prior"].to(DEVICE).float().view(-1, 1, 1, 1, 1)
                        else:
                            _ek = "followup_energy" if _ecmode == "oracle" else "starting_energy"
                            _en = batch[_ek].to(DEVICE).float()
                        model_in = torch.cat([model_in, _en], dim=1)                     # [xt,x0,energy]
                    if controlnet is not None:
                        _cmf = str(getattr(args, "fm_cnet_cond", "scan"))
                        _cnh = str(getattr(args, "fm_cnet_hist", "none"))
                        if _cnh != "none":
                            _ccond = build_hist_cond(batch, x0, scale_factor, DEVICE, _cnh)
                        elif str(getattr(args, "fm_hist_inject", "concat")) in ("cnet", "cnetft"):
                            _ccond = build_hist_cond(batch, x0, scale_factor, DEVICE, str(getattr(args, "fm_hist_mode", "none")))
                        elif _cmf == "multi":
                            _ccond = build_multi_cond(batch, x0, scale_factor, DEVICE)
                        elif _cmf == "hist13":
                            _ccond = batch["hist_cond"].to(DEVICE).float() * scale_factor   # full-history 13ch temporal stats
                        elif _cmf == "energy":
                            _ccond = torch.cat([x0, batch["starting_energy"].to(DEVICE).float()], dim=1)   # baseline latent + past-change energy field (5ch)
                        else:
                            _prior = batch["prior_latent"].to(DEVICE).float() * scale_factor
                            _ccond = torch.cat([_prior, x0 - _prior], dim=1) if _cmf == "traj" else _prior
                        if int(getattr(args, 'fm_cnet_shuffle', 0)) and _ccond.size(0) > 1:
                            _ccond = _ccond.roll(1, dims=0)     # control: substitutes another patient's energy from the batch
                            if not globals().get('_CSHUF_SHOWN'):
                                globals()['_CSHUF_SHOWN']=1
                                print('[cnet_shuffle] control enabled: the ControlNet condition is rolled along the batch dimension (information destroyed; architecture, parameters and training budget unchanged)', flush=True)
                        _cvc = build_cond_vec(batch, context, DEVICE, args)
                        _dh, _mh = controlnet(x=model_in, timesteps=t, controlnet_cond=_ccond, context=context, class_labels=age_gap)
                        pred = flowmatching(x=model_in, timesteps=t, context=context, class_labels=age_gap, cond_vec=_cvc, down_block_additional_residuals=_dh, mid_block_additional_residual=_mh)
                    else:
                        _cv = build_cond_vec(batch, context, DEVICE, args)
                        if globals().get("_hpool") is not None:
                            _cv = torch.cat([_cv, _hpool.to(_cv.dtype)], dim=1)
                        _cd = float(getattr(args, "fm_cond_drop", 0.0))
                        if _cd > 0.0 and mode == 'train':   # CFG: randomly drop patient conditioning -> learn uncond branch
                            _keep = (torch.rand(_cv.shape[0], 1, device=_cv.device) >= _cd).float()
                            _cv = _cv * _keep
                        _xamf = str(getattr(args, "fm_xattn_mode", "none"))
                        if _xamf != "none" and history_encoder is not None:
                            # History -> token sequence -> appended to context -> written into the features by cross-attention
                            # This is an additive/generative injection, of the same kind as the concat path, rather than multiplicative gating
                            _xin = build_hist_cond(batch, x0, scale_factor, DEVICE, _xamf)
                            _hp2 = batch.get("has_prior", None)
                            _hp2 = _hp2.to(DEVICE).float() if _hp2 is not None else None
                            context = torch.cat([context, history_encoder(_xin, _hp2)], dim=1)
                        if eimg_encoder is not None and "vfin_img" in batch:
                            import numpy as _npe
                            _pe = batch["vfin_img"]; _pe = _pe if isinstance(_pe,(list,tuple)) else [_pe]
                            _es = []
                            for _q in _pe:
                                try:
                                    _d = _npe.load(str(_q), allow_pickle=True)
                                    _a = _d["data"].astype("float32")
                                    if "valid" in _d.files and int(_d["valid"]) == 0: _a = _npe.zeros_like(_a)
                                except Exception: _a = None
                                _es.append(_a)
                            if _es and any(e is not None for e in _es):
                                _sh1 = next(e.shape for e in _es if e is not None)
                                _es = [(_npe.zeros(_sh1, "float32") if e is None else e) for e in _es]      # missing -> all zeros (the encoder handles the null case)
                                _et = torch.from_numpy(_npe.stack(_es)).unsqueeze(1).to(DEVICE).float()
                                context = torch.cat([context, eimg_encoder(_et)], dim=1)
                        _spmf = str(getattr(args, "fm_spade_mode", "none"))
                        _spmap = build_hist_cond(batch, x0, scale_factor, DEVICE, _spmf) if _spmf != "none" else None
                        pred = flowmatching(x=model_in, timesteps=t, context=context, class_labels=age_gap, cond_vec=_cv, spade_map=_spmap)
                        # ---- path consistency (--fm_pathc_w): delta_hat(p1->x1) approx (x0-p1) + delta_hat(x0->x1) ----
                        _pcw = float(getattr(args, "fm_pathc_w", 0.0))
                        _pc_loss = None
                        if _pcw > 0.0 and mode == "train" and "prior_latent" in batch:
                            _p1 = batch["prior_latent"].to(DEVICE).float() * scale_factor
                            _h1 = batch.get("has_prior", None)
                            _h1 = (torch.ones(x0.shape[0], device=DEVICE) if _h1 is None
                                   else _h1.to(DEVICE).float()).view(-1, 1, 1, 1, 1)
                            _past = (x0 - _p1).detach()          # the known past change, not an estimate
                            _d2 = (delta + _past).detach()       # the true total change from p1 to x1
                            # Independent noise: sharing it with the main path would cancel the noise terms and reduce the constraint to an identity
                            _z2 = torch.randn_like(_d2) * float(getattr(args, "fm_flowinit_sigma", 1.0))
                            if int(getattr(args, "fm_det", 0)):
                                _z2 = torch.zeros_like(_d2)
                            _xt2 = (1.0 - _tt) * _z2 + _tt * _d2
                            # The history channels are zeroed here; the exact approach would reconstruct prior2->p1, which requires reordering the batch.
                            #   Cost: the two predictions run under different conditioning regimes, which weakens the constraint in the conservative direction.
                            _mi2 = torch.cat([_xt2, _p1], dim=1)
                            if model_in.shape[1] > _mi2.shape[1]:
                                _pad = torch.zeros(_mi2.shape[0], model_in.shape[1] - _mi2.shape[1],
                                                   *_mi2.shape[2:], device=_mi2.device, dtype=_mi2.dtype)
                                _mi2 = torch.cat([_mi2, _pad], dim=1)
                            _pred2 = flowmatching(x=_mi2, timesteps=t, context=context,
                                                  class_labels=age_gap, cond_vec=_cv, spade_map=_spmap)
                            _dh1 = pred + (0.0 if int(getattr(args, "fm_x0_pred", 0)) else z)
                            _dh2 = _pred2 + (0.0 if int(getattr(args, "fm_x0_pred", 0)) else _z2)
                            _res = _dh2 - (_past + _dh1)
                            _den = (_d2 ** 2).mean(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)
                            _pc_loss = (((_res ** 2) / _den) * _h1).mean()




                    if globals().get("_grav_noise") is not None and globals().get("_gfield") is not None:
                        # delta' = g*(pred+z), so the equivalent velocity is pred' = g*(pred+z) - z
                        pred = _gfield * (pred + _grav_noise) - _grav_noise
                    if use_rectified:
                        use_mse = False

                        # ----- SNR weighting (rectified path) -----
                        eps = 1e-3
                        t_clamped = t.clamp(eps, 1 - eps)

                        alpha = 1.0 - t_clamped          # (B,)
                        sigma = sigma_min + (1.0 - sigma_min) * t_clamped   # t_clamped                # (B,)
                        # snr   = (alpha**2) / (sigma**2 + 1e-8)   # large at early t, small at late t
                        snr   = (alpha) / (sigma + 1e-8)   # large at early t, small at late t

                        gamma = 5.0                      # tune 2–10
                        w = torch.minimum(snr, torch.full_like(snr, gamma)) / (snr + 1e-8)  # early down-weighting
                        w = w.detach().view(-1, *([1] * (pred.ndim - 1)))                   # (B,1,1,1[,1])

                        # ----- Scale normalization -----
                        # per-sample norm over non-batch dims
                        if int(getattr(args, "fm_scale_norm", 1)):
                            reduce_dims = tuple(range(1, ut.ndim))
                            norm_factor = ut.pow(2).mean(dim=reduce_dims, keepdim=True).sqrt().detach() + 1e-6
                        else:
                            norm_factor = 1.0  # plain velocity-MSE -> /Dt realtime scheme diff is preserved

                        pred_s = pred / norm_factor
                        ut_s   = ut   / norm_factor

                        # ----- Per-sample losses (no reduction), then weight -----
                        if use_mse:
                            mse_map   = (pred_s - ut_s).pow(2)                      # (B, ...)
                        else:
                            mse_map   = torch.abs(pred_s - ut_s)

                        # ----- (TAIL) downweight the dominant subspace so the loss moves to the components that carry information -----
                        # Almost all of the delta norm sits in a few leading principal components, while the
                        # change-sensitive metrics respond mainly to the tail, so an unweighted loss spends most
                        # of its budget on components that barely affect the metric.
                        _tw = str(getattr(args, "fm_tail_w", "") or "")
                        if _tw:
                            _k, _win = _tw.split(",")
                            _k = int(_k); _win = float(_win)
                            if "_TAILV" not in globals():
                                _z = np.load(os.path.join(DERIVED_DIR, "pca_delta_basis.npz"))
                                globals()["_TAILV"] = torch.from_numpy(_z["comps"][:_k]).float()
                                print("[tail_w] K=%d w_in=%.3f principal-component basis loaded from %s"
                                      % (_k, _win, tuple(globals()["_TAILV"].shape)), flush=True)
                            _V = globals()["_TAILV"].to(pred_s.device)               # (K, D)
                            _e = (pred_s - ut_s).reshape(pred_s.shape[0], -1)        # (B, D)
                            _ein = (_e @ _V.t()) @ _V                                # projection onto the dominant subspace
                            _et = _e - _ein                                          # tail residual
                            # The norm must match the unweighted branch above (L1 when use_mse=False); squaring here
                            #   would shift the loss scale by an order of magnitude and so change the effective
                            #   learning rate, and w_in=1 would no longer be a no-op.
                            if use_mse:
                                _m = _win * _ein.pow(2) + _et.pow(2)
                            else:
                                _m = _win * _ein.abs() + _et.abs()
                            mse_map = _m.reshape_as(pred_s)

                        # ----- (CHN) per-channel normalization, giving the 4 latent channels equal weight -----
                        if int(getattr(args, 'fm_chan_norm', 0)) and mse_map.ndim >= 5:
                            _cr = tuple([0] + list(range(2, mse_map.ndim)))
                            _cs = ut_s.pow(2).mean(dim=_cr, keepdim=True).sqrt().detach() + 1e-6
                            mse_map = mse_map / _cs
                            if not globals().get('_CHNORM_SHOWN'):
                                globals()['_CHNORM_SHOWN'] = 1
                                print('[chan_norm] per-channel normalization active, channel scales %s' % [round(float(x),4) for x in _cs.flatten()], flush=True)

                        # ----- (2) energy-mask voxel weighting: focus loss on real-change regions -----
                        # ===== individualized loss weighting using the individual deviation of the historical energy =====
                        # Basis: the population mean energy map explains most of the historical energy; the individual
                        #       deviation left after removing it is not predictable from the covariates.
                        # Where a patient departs from the population is therefore patient-specific information unavailable from covariates, and worth emphasizing in the loss.
                        _hwl = float(getattr(args, "fm_histw_lambda", 0.0))
                        _histw = None
                        if _hwl > 0.0:
                            _hk = "eratio" if "eratio" in batch else ("etraj" if "etraj" in batch else None)
                            if _hk is not None:
                                _hv = batch[_hk].to(DEVICE).float()
                                _ci = _hv[:, 0:1] if _hk == "eratio" else _hv[:, 4:5]   # eratio: log ratio; etraj: E_ind
                                _ci = _ci.abs()
                                _histw = 1.0 + _hwl * (_ci / (_ci.amax(dim=(2, 3, 4), keepdim=True) + 1e-6))
                        mask_lambda = float(getattr(args, "fm_mask_lambda", 0.0))
                        mask_mode = getattr(args, "fm_mask_mode", "soft")
                        mask_thr = float(getattr(args, "fm_mask_thr", 0.05))
                        if mask_lambda > 0.0 and (str(getattr(args, "fm_mask_src", "energy")) == "latent" or "followup_energy" in batch):
                            if str(getattr(args, "fm_mask_src", "energy")) == "latent":
                                # latent-space true change |x1-x0|: an oracle mask defined directly in the latent, unlike the image-space energy map
                                _emr = (x1 - x0).abs().mean(1, keepdim=True)
                                _em = _emr / (_emr.amax(dim=(2, 3, 4), keepdim=True) + 1e-6)
                            else:
                                _em  = batch["followup_energy"].to(DEVICE).float()       # (B,1,D,H,W) in [0,1]
                            if mask_mode == "hard":
                                _wv = 1.0 + mask_lambda * (_em > mask_thr).float()   # binary change-region focus
                            elif mask_mode == "pure":
                                _wv = _em + 0.02                                      # loss ~only in change region (tiny floor)
                            else:  # soft
                                _wv = 1.0 + mask_lambda * _em                        # graded (B,1,D,H,W)
                            if _histw is not None:
                                _wv = _wv * _histw          # combine the individualized weight: true change region x where this patient departs from the population
                            _num = (_wv * mse_map).reshape(mse_map.size(0), -1).sum(1)
                            _den = _wv.expand_as(mse_map).reshape(mse_map.size(0), -1).sum(1).clamp_min(1e-6)
                            mse_ps = _num / _den                                     # (B,) mask-weighted
                        elif _histw is not None:
                            _em = None
                            _num = (_histw * mse_map).reshape(mse_map.size(0), -1).sum(1)
                            _den = _histw.expand_as(mse_map).reshape(mse_map.size(0), -1).sum(1).clamp_min(1e-6)
                            mse_ps = _num / _den
                        else:
                            _em = None
                            mse_ps = mse_map.view(mse_map.size(0), -1).mean(1)       # (B,)

                        # ----- (A) asymmetric amplitude penalty: samples that underestimate |delta| are penalized twice, matching the double penalty rF1 applies to underestimation -----
                        _asym = float(getattr(args, "fm_asym_w", 1.0))
                        if _asym != 1.0:
                            _dha = pred if _grav_noise is None else (pred + _grav_noise)
                            _nh = _dha.flatten(1).norm(dim=1)
                            _ng = (x1 - x0).flatten(1).norm(dim=1)
                            _aw = torch.where(_nh < _ng, torch.full_like(_nh, _asym), torch.ones_like(_nh))
                            mse_ps = mse_ps * _aw
                        # ----- (L1) magnitude-weighted loss: upweight voxels with large TRUE change |x1-x0| -----
                        _magw = float(getattr(args, "fm_magw", 0.0))
                        if _magw > 0.0:
                            _dm = (x1 - x0).abs()
                            _rdm = tuple(range(1, _dm.ndim))
                            _dmn = _dm / (_dm.amax(dim=_rdm, keepdim=True) + 1e-6)     # per-sample [0,1]
                            _wm = 1.0 + _magw * _dmn
                            _n1 = (_wm * mse_map).reshape(mse_map.size(0), -1).sum(1)
                            _d1 = _wm.expand_as(mse_map).reshape(mse_map.size(0), -1).sum(1).clamp_min(1e-6)
                            mse_ps = _n1 / _d1

                        # ----- (EW) per-voxel energy loss weighting: the in-training form of the offline energy reweighting -----
                        _elw = float(getattr(args, "fm_elossw", 0.0))
                        if _elw > 0.0 and (getattr(args, "fm_elossw_src", "vslope") not in batch):
                            raise RuntimeError(f"--fm_elossw={_elw} but the batch has no '{getattr(args,'fm_elossw_src','vslope')}': the dataset did not load energy, and silently falling back is not permitted")
                        if _elw > 0.0 and (getattr(args, "fm_elossw_src", "vslope") in batch):
                            _ev = batch[getattr(args, "fm_elossw_src", "vslope")].to(mse_map.device).float()
                            if _ev.ndim == mse_map.ndim and _ev.shape[2:] == mse_map.shape[2:]:
                                _re = tuple(range(1, _ev.ndim))
                                _en = _ev / (_ev.amax(dim=_re, keepdim=True) + 1e-6)      # per-sample [0,1]
                                _we = (1.0 - _elw) + _elw * 2.0 * _en                     # mean approximately 1
                                _ne = (_we * mse_map).reshape(mse_map.size(0), -1).sum(1)
                                _de = _we.expand_as(mse_map).reshape(mse_map.size(0), -1).sum(1).clamp_min(1e-6)
                                mse_ps = _ne / _de
                                if not globals().get("_ELW_SHOWN"):
                                    globals()["_ELW_SHOWN"]=1
                                    print(f"[elossw] active L={_elw} src={getattr(args,'fm_elossw_src','vslope')} "
                                          f"E range [{float(_en.min()):.3f},{float(_en.max()):.3f}] w range [{float(_we.min()):.3f},{float(_we.max()):.3f}]", flush=True)

                        # cosine: 1 - cos(pred, ut) on normalized, scaled tensors
                        p_norm = F.normalize(pred_s, dim=1)                     # handles eps internally
                        u_norm = F.normalize(ut_s,   dim=1)
                        cos_map = 1 - (p_norm * u_norm).sum(dim=1, keepdim=True)  # (B,1, H, W) or (B,1, ...)
                        cos_ps  = cos_map.view(cos_map.size(0), -1).mean(1)       # (B,)

                        # ----- Combine with same weighting -----
                        lambda_cos = 0.25  # start small (0.2–0.3 often best)
                        # ----- (1) Delta-t loss weighting: down-weight short-gap low-SNR pairs (replaces /Dt) -----
                        _tau = float(getattr(args, "fm_dt_tau", 0.0))
                        if _tau > 0.0:
                            w_dt = (_dt / (_dt + _tau)).detach().reshape(-1)         # (B,) in (0,1): far->~1, near->small
                        else:
                            w_dt = torch.ones_like(mse_ps)
                        loss = (w.view(-1) * w_dt * (mse_ps )).mean()  # + lambda_cos * cos_ps
                        if _pc_loss is not None:
                            loss = loss + _pcw * _pc_loss
                        # ----- (WTA) multi-hypothesis winner-take-all: the optimum becomes the set of modes rather than the mean -----
                        _wk = int(getattr(args, "fm_wta_k", 0))
                        if _wk >= 2 and mode == 'train' and int(getattr(args, "fm_res_noise", 0)):
                            _errs = [((pred + z) - delta).flatten(1).pow(2).mean(1)]      # first hypothesis (reused)
                            for _j in range(_wk - 1):
                                _zj = torch.randn_like(delta)
                                _xtj = (1.0 - _tt) * _zj + _tt * delta
                                _pj = flowmatching(x=torch.cat([_xtj, _x0c], dim=1), timesteps=t,
                                                   context=context, class_labels=age_gap, cond_vec=_cv)
                                _errs.append(((_pj + _zj) - delta).flatten(1).pow(2).mean(1))
                            _best = torch.stack(_errs, 0).min(0).values                    # penalize only the closest one
                            loss = loss + _best.mean()
                        # ----- (SUB) subspace constraint: match the per-mode batch std, penalizing shrinkage on the dominant modes directly -----
                        if _subV is not None:
                            _dhs = (pred if _grav_noise is None else (pred + _grav_noise)).flatten(1)
                            _dgs = (x1 - x0).flatten(1)
                            _ch = (_dhs - _subMu) @ _subV          # (B,K) predicted coefficients
                            _ct = (_dgs - _subMu) @ _subV          # (B,K) true coefficients
                            if str(getattr(args, "fm_sub_mode", "persample")) == "batchstd":
                                if _ch.shape[0] > 1:
                                    _sh = _ch.std(0); _st = _ct.std(0)
                                    loss = loss + _subw * ((_sh - _st).abs() / (_st.abs() + 1e-6)).mean()
                            else:
                                # Per-sample, per-mode amplitude matching, so a uniform increase across the batch does not satisfy it and every patient must match
                                _sc = _ct.abs().mean(0, keepdim=True) + 1e-6         # normalize each mode by its own scale
                                loss = loss + _subw * ((_ch.abs() - _ct.abs()).abs() / _sc).mean()
                        # ----- (B) diversity regularizer: run a second forward pass with different noise and penalize overly similar outputs (counters mean collapse) -----
                        _divw = float(getattr(args, "fm_div_w", 0.0))
                        if _divw > 0.0 and mode == 'train' and int(getattr(args, "fm_res_noise", 0)):
                            _z2 = torch.randn_like(delta)
                            _xt2 = (1.0 - _tt) * _z2 + _tt * delta
                            _p2 = flowmatching(x=torch.cat([_xt2, x0], dim=1), timesteps=t, context=context,
                                               class_labels=age_gap, cond_vec=_cv)
                            _d1 = (pred + z).flatten(1); _d2 = (_p2 + _z2).flatten(1)
                            _rel = (_d1 - _d2).norm(dim=1) / (_d1.norm(dim=1) + 1e-6)   # relative difference
                            loss = loss + _divw * torch.relu(0.5 - _rel).mean()          # penalized when too similar

                        # ----- (4) energy GRAVITY: known change region as ATTRACTION prior on delta_hat (not just loss weight) -----
                        _grav_out = float(getattr(args, "fm_grav_out", 0.0))  # confine: suppress change OUTSIDE mask
                        _grav_in  = float(getattr(args, "fm_grav_in", 0.0))   # attract: concentrate change mass INSIDE mask
                        if (_grav_out > 0.0 or _grav_in > 0.0) and "followup_energy" in batch:
                            _emg = _em if _em is not None else batch["followup_energy"].to(DEVICE).float()  # load energy even if mask_lambda=0
                            _dh  = pred if _grav_noise is None else (pred + _grav_noise)   # implied change field delta_hat (latent)
                            _mag = _dh.abs()
                            if _grav_out > 0.0:
                                loss = loss + _grav_out * (_mag * (1.0 - _emg)).mean()     # penalize change outside mask
                            if _grav_in > 0.0:
                                _frac_in = (_mag * _emg).sum() / (_mag.sum() + 1e-6)       # fraction of change mass inside mask
                                loss = loss + _grav_in * (1.0 - _frac_in)                  # pull mass into mask (bounded [0,1])

                        # ----- (L2/L4) region losses on implied change field delta_hat -----
                        _region_w = float(getattr(args, "fm_region_w", 0.0))
                        _regmag_w = float(getattr(args, "fm_regmag_w", 0.0))
                        if _region_w > 0.0 or _regmag_w > 0.0:
                            _dhat = pred if _grav_noise is None else (pred + _grav_noise)
                            _dgt  = (x1 - x0)
                            _mh  = _dhat.abs().mean(dim=1, keepdim=True)   # (B,1,...) channel-collapsed |change|
                            _mgt = _dgt.abs().mean(dim=1, keepdim=True)
                            _rdc = tuple(range(1, _mh.ndim))
                            if _regmag_w > 0.0:   # L4: match total change energy per sample (anti-collapse)
                                _eh = _mh.reshape(_mh.size(0), -1).pow(2).sum(1).add(1e-8).sqrt()
                                _eg = _mgt.reshape(_mgt.size(0), -1).pow(2).sum(1).add(1e-8).sqrt()
                                loss = loss + _regmag_w * (_eh - _eg).abs().mean()
                            if _region_w > 0.0:   # L2: soft-Dice overlap of predicted vs GT change region
                                _gh = _mh  / (_mh.amax(dim=_rdc, keepdim=True) + 1e-6)
                                _gg = _mgt / (_mgt.amax(dim=_rdc, keepdim=True) + 1e-6)
                                _ph = torch.sigmoid((_gh - 0.25) / 0.1)
                                _pg = torch.sigmoid((_gg - 0.25) / 0.1)
                                _inter = (_ph * _pg).reshape(_ph.size(0), -1).sum(1)
                                _uni   = _ph.reshape(_ph.size(0), -1).sum(1) + _pg.reshape(_pg.size(0), -1).sum(1)
                                _dice  = (2.0 * _inter + 1.0) / (_uni + 1.0)
                                loss = loss + _region_w * (1.0 - _dice).mean()

                        # ----- (EPHI) deployable energy suppression: a one-sided penalty on predicted change where past energy was low -----
                        _epw = float(getattr(args, "fm_ephi_w", 0.0))
                        if _epw > 0.0:
                            if "vslope" not in batch:
                                raise RuntimeError("--fm_ephi_w>0 but the batch has no vslope: the dataset did not load energy, and silently falling back is not permitted")
                            _ev2 = batch["vslope"].to(pred.device).float()
                            _dh4 = pred if _grav_noise is None else (pred + _grav_noise)
                            if _ev2.shape[2:] == _dh4.shape[2:]:
                                _B2 = _ev2.shape[0]; _f2 = _ev2.reshape(_B2, -1)
                                _r2 = _f2.argsort(dim=1).argsort(dim=1).float() / max(_f2.shape[1] - 1, 1)
                                _sup = (1.0 - _r2).reshape(_ev2.shape)     # low energy -> strong suppression
                                loss = loss + _epw * (_dh4.abs() * _sup).mean()
                                if not globals().get("_EPHI_SHOWN"):
                                    globals()["_EPHI_SHOWN"] = 1
                                    print(f"[ephi] energy suppression term active w={_epw} suppression field range [{float(_sup.min()):.3f},{float(_sup.max()):.3f}]", flush=True)

                        # ----- (L3) potential-field transport: penalize |delta_hat|*Phi, Phi=smooth dist-potential (pull change toward region) -----
                        _phi_w = float(getattr(args, "fm_phi_w", 0.0))
                        if _phi_w > 0.0 and "followup_energy" in batch:
                            _emp = _em if _em is not None else batch["followup_energy"].to(DEVICE).float()
                            _eb = _emp
                            for _ in range(4):
                                _eb = F.avg_pool3d(_eb, kernel_size=5, stride=1, padding=2)   # diffuse mask outward -> graded field
                            _eb = _eb / (_eb.amax(dim=tuple(range(1, _eb.ndim)), keepdim=True) + 1e-6)
                            _phi = 1.0 - _eb                                                    # 0 in change-region, ->1 far away
                            _dh3 = pred if _grav_noise is None else (pred + _grav_noise)
                            loss = loss + _phi_w * (_dh3.abs() * _phi).mean()

                        # ----- (rF1) differentiable region-F1 surrogate: dir-aware (per-voxel cosine), P99 threshold -----
                        _rf1_w = float(getattr(args, "fm_rf1_w", 0.0))
                        if _rf1_w > 0.0:
                            _dh6 = pred if _grav_noise is None else (pred + _grav_noise)   # (B,C,...) implied change
                            _dg6 = (x1 - x0)
                            _B6 = _dh6.size(0)
                            _mg = torch.sqrt((_dg6 ** 2).sum(1) + 1e-12).reshape(_B6, -1)   # (B,N) GT |change|
                            _mp = torch.sqrt((_dh6 ** 2).sum(1) + 1e-12).reshape(_B6, -1)   # (B,N) pred |change|
                            _mgd = _mg.detach(); _tau = (0.25 * torch.quantile(_mgd, 0.99, dim=1, keepdim=True)).clamp_min(1e-6)  # 0.25*P99 = match rF1 metric
                            _sh = 0.5 * _tau
                            _sgt = torch.sigmoid((_mg - _tau) / _sh)                          # soft GT region
                            _spr = torch.sigmoid((_mp - _tau) / _sh)                          # soft pred region
                            _cos = ((_dh6 * _dg6).sum(1).reshape(_B6, -1)) / (_mp * _mg + 1e-6)  # per-voxel dir agreement
                            _dir = _cos.clamp_min(0.0)
                            _tp = _sgt * _spr * _dir
                            _rR = _tp.sum(1) / (_sgt.sum(1) + 1e-6)
                            _rP = _tp.sum(1) / (_spr.sum(1) + 1e-6)
                            _rf1 = 2.0 * _rP * _rR / (_rP + _rR + 1e-6)
                            loss = loss + _rf1_w * (1.0 - _rf1).mean()

                        # ----- (cF1) differentiable change-F1 surrogate (same maths as src/change_f1_soft.py) -----
                        # continuous weights g = |d| / (|d| + tau) replace the hard threshold -> differentiable everywhere;
                        # amplitude error is penalised symmetrically, so the loss cannot be lowered by rescaling
                        _cf1_w = float(getattr(args, "fm_cf1_w", 0.0))
                        if _cf1_w > 0.0:
                            _dhc = pred if _grav_noise is None else (pred + _grav_noise)
                            _dgc = (x1 - x0)
                            _Bc = _dhc.size(0)
                            _a = _dgc.reshape(_Bc, -1)                       # ground-truth delta
                            _b = _dhc.reshape(_Bc, -1)                       # predicted delta
                            _tf = float(getattr(args, "fm_cf1_tau", 0.25))
                            _tauc = (_tf * torch.quantile(_a.abs().detach(), 0.99, dim=1, keepdim=True)).clamp_min(1e-6)
                            _g = _a.abs() / (_a.abs() + _tauc)
                            _pw = _b.abs() / (_b.abs() + _tauc)
                            _rec = ((_g * _a * _b).sum(1) / ((_g * _a * _a).sum(1) + 1e-9)).clamp(0, 1)
                            _pre = ((_pw * _a * _b).sum(1) / ((_pw * _b * _b).sum(1) + 1e-9)).clamp(0, 1)
                            _cf1 = 2.0 * _pre * _rec / (_pre + _rec + 1e-9)
                            loss = loss + _cf1_w * (1.0 - _cf1).mean()
                            if not globals().get("_CF1_SHOWN"):
                                globals()["_CF1_SHOWN"] = 1
                                print("[cf1] active w=%g (differentiable soft-cF1 surrogate)" % _cf1_w, flush=True)

                        # ----- (DIR) change-direction agreement, weighted by GT |change| (targets the direction term of rF1) -----
                        _dir_w = float(getattr(args, "fm_dir_w", 0.0))
                        if _dir_w > 0.0:
                            _dhd = pred if _grav_noise is None else (pred + _grav_noise)
                            _dgd = (x1 - x0)
                            _wv = torch.sqrt((_dgd ** 2).sum(1, keepdim=True) + 1e-12)         # (B,1,...) GT change mag
                            _nh = torch.sqrt((_dhd ** 2).sum(1, keepdim=True) + 1e-12)
                            _cosd = (_dhd * _dgd).sum(1, keepdim=True) / (_wv * _nh + 1e-6)     # per-voxel cos
                            _ld = (_wv * (1.0 - _cosd)).sum() / (_wv.sum() + 1e-6)              # |Δgt|-weighted
                            loss = loss + _dir_w * _ld

                        # ----- (fullrec) FULL-image recon at small bs (decode N=2 full imgs/step; avoids OOM via batch-subsample not crop) -----
                        _fullrec_w = float(getattr(args, "fm_fullrec_w", 0.0))
                        if _fullrec_w > 0.0 and mode == 'train':
                            _aef = autoencoder.module if hasattr(autoencoder, "module") else autoencoder
                            _nbf = int(min(int(getattr(args, "fm_fullrec_nb", 2)), pred.shape[0]))
                            _dhf = pred if _grav_noise is None else (pred + _grav_noise)   # δ̂ implied change (res-flow: pred+z)
                            _x1pf = (x0[:_nbf] + _dhf[:_nbf]) / scale_factor      # x0+δ̂ ≈ x1 (raw)
                            _x1gf = (x1[:_nbf]) / scale_factor                    # GT followup latent (raw)
                            _impf = _aef.decode(_x1pf)                            # FULL image decode (grad -> latent)
                            with torch.no_grad():
                                _imgf = _aef.decode(_x1gf)                        # FULL GT image (no grad)
                            _vrw = float(getattr(args, "fm_vfin_recw", 0.0))
                            _vwf = None
                            if _vrw > 0.0 and "vfin_img" in batch:
                                # native-resolution VFIN weighting, with no downsampling
                                import numpy as _npv
                                _ps = batch["vfin_img"]
                                _ps = _ps if isinstance(_ps, (list, tuple)) else [_ps]
                                _ws = []
                                for _q in _ps[:_nbf]:
                                    try:
                                        _d = _npv.load(str(_q), allow_pickle=True)
                                        _a = _d["data"].astype("float32")
                                        if "valid" in _d.files and int(_d["valid"]) == 0: _a = _npv.zeros_like(_a)
                                    except Exception: _a = None
                                    _ws.append(_a)
                                if _ws and any(w is not None for w in _ws):
                                    _sh0 = next(w.shape for w in _ws if w is not None)
                                    _ws = [(_npv.zeros(_sh0, "float32") if w is None else w) for w in _ws]  # missing -> neutral (weight 1)
                                    _vwf = torch.from_numpy(_npv.stack(_ws)).unsqueeze(1).to(DEVICE).float()
                                    if _vwf.shape[2:] != _impf.shape[2:]:
                                        _vwf = F.interpolate(_vwf, size=_impf.shape[2:], mode="trilinear", align_corners=False)
                                    _vwf = _vwf / (_vwf.amax(dim=(2,3,4), keepdim=True) + 1e-6)
                                    _vwf = 1.0 + _vrw * _vwf
                            if _vwf is not None:
                                _errf = (_impf - _imgf).abs()
                                loss = loss + _fullrec_w * ((_errf * _vwf).sum() / (_vwf.sum() + 1e-6))
                                if global_counter[mode] % 300 == 0:
                                    print(f"[vfin-recw] lam={_vrw} w range [{_vwf.min():.2f},{_vwf.max():.2f}]", flush=True)
                            else:
                              _frmw = float(getattr(args, "fm_fullrec_mask", 0.0))
                              if _frmw > 0.0 and "followup_energy" in batch:
                                  _emf = batch["followup_energy"][:_nbf].to(DEVICE).float()          # (nb,1,d,h,w) latent-res [0,1]
                                  _emf = F.interpolate(_emf, size=_impf.shape[2:], mode="trilinear", align_corners=False)
                                  if str(getattr(args, "fm_fullrec_mask_mode", "soft")) == "pure":
                                      _wf = _emf + 1e-3                                              # change-region only (kills background copy-reward)
                                  else:
                                      _wf = 1.0 + _frmw * _emf                                       # soft: keep global + emphasize change
                                  _errf = (_impf - _imgf).abs()
                                  loss = loss + _fullrec_w * ((_errf * _wf).sum() / (_wf.sum() + 1e-6))
                              else:
                                  loss = loss + _fullrec_w * F.l1_loss(_impf, _imgf)

                        # ----- (magcal) region magnitude calibration: match ||delta_hat|| -> ||delta_gt|| in change region (bakes in δ-scale) -----
                        # ---- (adv) distribution matching on delta: penalize blurry/mean-collapsed change ----
                        if disc is not None and global_counter[mode] > int(getattr(args, "fm_adv_warmup", 200)):
                            _dha = pred if _grav_noise is None else (pred + _grav_noise)   # implied delta_hat
                            _dgt = (x1 - x0)
                            _lg = adv_fn(disc(_dha.float())[-1], target_is_real=True, for_discriminator=False)
                            loss = loss + _advw * _lg
                            _adv_cache = (_dha.detach(), _dgt.detach())   # the discriminator update is deferred until after the generator step, to avoid in-place graph corruption
                        # ===== block-wise direction loss: penalizes misplaced change only and is fully invariant to amplitude =====
                        # A global cosine is dominated by the whole-brain change pattern and says little about where the
                        # change is placed. Normalizing each block separately makes the gradient act on direction only,
                        # which avoids rewarding overshoot.
                        _bdw = float(getattr(args, "fm_blockdir_w", 0.0))
                        if _bdw > 0.0:
                            _dhb = pred if _grav_noise is None else (pred + _grav_noise)   # implied delta_hat
                            _dgb = (x1 - x0)
                            _k = int(getattr(args, "fm_blockdir_k", 4))                    # block edge length
                            _B, _C, _D, _H, _W = _dgb.shape
                            _Dp, _Hp, _Wp = (_D // _k) * _k, (_H // _k) * _k, (_W // _k) * _k
                            _a = _dhb[:, :, :_Dp, :_Hp, :_Wp].reshape(_B, _C, _Dp // _k, _k, _Hp // _k, _k, _Wp // _k, _k)
                            _b = _dgb[:, :, :_Dp, :_Hp, :_Wp].reshape(_B, _C, _Dp // _k, _k, _Hp // _k, _k, _Wp // _k, _k)
                            _a = _a.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(_B, -1, _C * _k * _k * _k)
                            _b = _b.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(_B, -1, _C * _k * _k * _k)
                            _cosb = torch.nn.functional.cosine_similarity(_a.float(), _b.float(), dim=2)   # (B, nblock)
                            _wb = _b.float().norm(dim=2)                                   # weighted by the true change energy
                            _wb = _wb / (_wb.sum(1, keepdim=True) + 1e-8)
                            loss = loss + _bdw * (_wb * (1.0 - _cosb)).sum(1).mean()

                        # ===== temporal composition consistency =====
                        # delta(p1->x1) must equal delta(p1->x0) + delta(x0->x1).
                        # Training treats each pair as an independent sample, so different intervals of the same patient may contradict each other.
                        # This constraint acts on direction and additive structure only, not amplitude, so it does not trade amplitude for rF1. It is self-supervised and disease-agnostic.
                        _cw = float(getattr(args, "fm_comp_w", 0.0))
                        if _cw > 0.0 and mode == 'train' and "prior_latent" in batch and int(getattr(args, "fm_res_noise", 0)):
                            _hp = batch.get("has_prior", None)
                            _hp = (_hp.to(DEVICE).float().view(-1) if _hp is not None else torch.ones(x0.shape[0], device=DEVICE))
                            if _hp.sum() > 0:
                                _p1 = batch["prior_latent"].to(DEVICE).float() * scale_factor
                                _cvb = build_cond_vec(batch, context, DEVICE, args)
                                # The three intervals share one noise draw, so the implied deltas can be added and compared directly
                                _zc = torch.randn_like(x0)
                                _tc = torch.rand(x0.shape[0], device=DEVICE).clamp(1e-3, 1 - 1e-3)
                                _tv = _tc.view(-1, *([1] * (x0.dim() - 1)))
                                def _idelta(_anchor, _tgt_guess):
                                    _xt = (1.0 - _tv) * _zc + _tv * _tgt_guess
                                    _mi = torch.cat([_xt, _anchor], dim=1)
                                    _pv = flowmatching(x=_mi, timesteps=_tc, context=context,
                                                       class_labels=age_gap, cond_vec=_cvb)
                                    return _pv + _zc                      # implied delta_hat
                                _da = _idelta(_p1, x0 - _p1)              # p1 -> x0
                                _db = _idelta(x0,  x1 - x0)               # x0 -> x1
                                _dc = _idelta(_p1, x1 - _p1)              # p1 -> x1 (should equal _da + _db)
                                _res = (_dc - (_da + _db)).flatten(1)
                                _den = _dc.flatten(1).norm(dim=1).clamp_min(1e-6)
                                _cl = (_res.norm(dim=1) / _den)           # relative residual, so it is scale-invariant
                                _comp_term = _cw * ((_cl * _hp).sum() / _hp.sum().clamp_min(1.0))
                                loss = loss + _comp_term
                                # Activation print: verifies the term is actually contributing, since a parameter excluded from the gradient
                                #   produces a silent no-op. The loss must be non-zero and the has_prior coverage must be reasonable.
                                if not globals().get("_COMPW_SHOWN"):
                                    globals()["_COMPW_SHOWN"] = 1
                                    print("[comp_w] active w=%.3f | has_prior in this batch %d/%d | mean relative residual %.4f | loss term %.5f"
                                          % (_cw, int(_hp.sum().item()), _hp.numel(),
                                             float((_cl * _hp).sum().item() / max(_hp.sum().item(), 1.0)),
                                             float(_comp_term.item())), flush=True)

                        # ===== population-drift removal, so the loss measures the individualized deviation only =====
                        # Population drift is largely predictable from a handful of covariates, so it contributes
                        # direction without contributing much energy yet still dominates the gradient. Removing it
                        # spends all capacity on the individual residual.
                        _dsw = float(getattr(args, "fm_driftsub_w", 0.0))
                        if _dsw > 0.0:
                            _mh = _drift_mu(batch, context, DEVICE, args)                  # (B,4,D,H,W) predicted population drift
                            if _mh is not None:
                                _dhr = (pred if _grav_noise is None else (pred + _grav_noise)) - _mh
                                _dgr = (x1 - x0) - _mh
                                loss = loss + _dsw * torch.nn.functional.mse_loss(_dhr.float(), _dgr.float())

                        _magcal_w = float(getattr(args, "fm_magcal_w", 0.0))
                        if _magcal_w > 0.0:
                            _dhm = pred if _grav_noise is None else (pred + _grav_noise)
                            _dgm = (x1 - x0)
                            if str(getattr(args, "fm_magcal_src", "energy")) == "latent":
                                _mr = (x1 - x0).abs().mean(1, keepdim=True)
                                _mskm = _mr / (_mr.amax(dim=(2, 3, 4), keepdim=True) + 1e-6)
                            elif "followup_energy" in batch:
                                _mskm = batch["followup_energy"].to(DEVICE).float()
                            else:
                                _mskm = (_dgm.abs().mean(1, keepdim=True) > _dgm.abs().mean()).float()
                            _rdm = tuple(range(1, _dhm.ndim))
                            _Eh = torch.sqrt((_mskm * _dhm.pow(2)).sum(_rdm) + 1e-8)   # (B,) pred change energy in region
                            _Eg = torch.sqrt((_mskm * _dgm.pow(2)).sum(_rdm) + 1e-8)   # (B,) GT change energy in region
                            loss = loss + _magcal_w * ((_Eh - _Eg).abs() / (_Eg + 1e-6)).mean()

                        # ===== manifold consistency: pull x0+delta_hat back toward the latent region the autoencoder has seen =====
                        # Measured as the round-trip residual ||enc(dec(z))-z||/||z||: extra latent-space losses can
                        # push the predicted latent off the autoencoder's manifold, buying a higher latent cosine that
                        # the decoder then pays back in image space. This term penalizes that directly.
                        _maniw = float(getattr(args, "fm_mani_w", 0.0))
                        if _maniw > 0.0 and mode == 'train':
                            _aem = autoencoder.module if hasattr(autoencoder, "module") else autoencoder
                            _dhm2 = pred if _grav_noise is None else (pred + _grav_noise)
                            _zp = (x0 + _dhm2) / scale_factor                 # predicted follow-up latent (raw)
                            _Bm, _Cm, _Dm, _Hm, _Wm = _zp.shape
                            _csm = int(min(16, _Dm, _Hm, _Wm))
                            _wcm = (x1 - x0).abs().mean(1)                    # true latent change (center crop)
                            _crops = []
                            for _b in range(_Bm):
                                _wv = _wcm[_b]
                                _gr = torch.nonzero(_wv > _wv.mean(), as_tuple=False).float()
                                _ct = _gr.mean(0) if _gr.numel() > 0 else _wv.new_tensor([_Dm/2., _Hm/2., _Wm/2.])
                                _zz = int(min(max(int(_ct[0]) - _csm // 2, 0), _Dm - _csm))
                                _yy = int(min(max(int(_ct[1]) - _csm // 2, 0), _Hm - _csm))
                                _xx = int(min(max(int(_ct[2]) - _csm // 2, 0), _Wm - _csm))
                                _crops.append(_zp[_b:_b+1, :, _zz:_zz+_csm, _yy:_yy+_csm, _xx:_xx+_csm])
                            _zc = torch.cat(_crops, 0)
                            _rec = _aem.encode(_aem.decode(_zc))              # the gradient passes through the full decode-encode path
                            _rec = _rec[0] if isinstance(_rec, (tuple, list)) else _rec
                            _mo2 = max(1, _zc.shape[-1] // 8)
                            _a2 = _rec[..., _mo2:-_mo2, _mo2:-_mo2, _mo2:-_mo2]
                            _b2 = _zc[..., _mo2:-_mo2, _mo2:-_mo2, _mo2:-_mo2]
                            _rd = tuple(range(1, _a2.ndim))
                            _lm = ((_a2 - _b2).pow(2).sum(_rd) / (_b2.pow(2).sum(_rd) + 1e-8)).mean()
                            loss = loss + _maniw * _lm
                            if global_counter[mode] % 200 == 0:
                                print(f"[mani] roundtrip={_lm.item():.4f} (x{_maniw} -> {_maniw*_lm.item():.4f} vs loss {loss.item():.4f})", flush=True)

                        # ----- (crop-recon) image-space recon on CHANGE-REGION crop only (avoids full-decode OOM) -----
                        _croprec_w = float(getattr(args, "fm_croprec_w", 0.0))
                        if _croprec_w > 0.0 and mode == 'train':
                            _aec = autoencoder.module if hasattr(autoencoder, "module") else autoencoder
                            _dhc = pred if _grav_noise is None else (pred + _grav_noise)   # δ̂ implied change
                            _x1p = (x0 + _dhc) / scale_factor                 # x0+δ̂ ≈ x1 (raw)
                            _x1g = (x1 / scale_factor)                        # GT followup latent (raw)
                            _Bc, _Cc, _Dc, _Hc, _Wc = _x1p.shape
                            _cs = int(min(16, _Dc, _Hc, _Wc))                 # latent crop size (incl margin)
                            if str(getattr(args, "fm_croprec_src", "latent")) == "latent":
                                _wc = (x1 - x0).abs().mean(1)          # oracle: the true latent change
                            elif "followup_energy" in batch:
                                _wc = batch["followup_energy"].to(DEVICE).float()[:, 0]
                            else:
                                _wc = (x1 - x0).abs().mean(1)
                            _cps = []; _cgs = []; _crd = []
                            for _b in range(_Bc):
                                _wv = _wc[_b]
                                if float(_wv.sum()) > 1e-6:
                                    _gr = torch.nonzero(_wv > _wv.mean(), as_tuple=False).float()
                                    _ct = _gr.mean(0) if _gr.numel() > 0 else _wv.new_tensor([_Dc/2., _Hc/2., _Wc/2.])
                                else:
                                    _ct = _wv.new_tensor([_Dc/2., _Hc/2., _Wc/2.])
                                _z0 = int(min(max(int(_ct[0]) - _cs // 2, 0), _Dc - _cs))
                                _y0 = int(min(max(int(_ct[1]) - _cs // 2, 0), _Hc - _cs))
                                _x0c = int(min(max(int(_ct[2]) - _cs // 2, 0), _Wc - _cs))
                                _crd.append((_z0, _y0, _x0c))
                                _cps.append(_x1p[_b:_b+1, :, _z0:_z0+_cs, _y0:_y0+_cs, _x0c:_x0c+_cs])
                                _cgs.append(_x1g[_b:_b+1, :, _z0:_z0+_cs, _y0:_y0+_cs, _x0c:_x0c+_cs])
                            _cp = torch.cat(_cps, 0); _cg = torch.cat(_cgs, 0)
                            _imp = _aec.decode(_cp)                            # decode pred crop (grad)
                            with torch.no_grad():
                                _img = _aec.decode(_cg)                        # decode GT crop (no grad)
                            _mo = max(1, _imp.shape[-1] // 8)                  # drop ~12.5% border (decoder-context margin)
                            _imp = _imp[..., _mo:-_mo, _mo:-_mo, _mo:-_mo]
                            _img = _img[..., _mo:-_mo, _mo:-_mo, _mo:-_mo]
                            if str(getattr(args, "fm_croprec_mode", "abs")) == "change":
                                # Compare the decoded change only: dec(x0+delta_hat)-dec(x0) against dec(x1)-dec(x0).
                                # In absolute images the unchanged portion dominates and would dilute the change signal.
                                _c0s = []
                                _x0g = (x0 / scale_factor)
                                for _b in range(_Bc):
                                    _c0s.append(_x0g[_b:_b+1, :, _crd[_b][0]:_crd[_b][0]+_cs,
                                                     _crd[_b][1]:_crd[_b][1]+_cs, _crd[_b][2]:_crd[_b][2]+_cs])
                                with torch.no_grad():
                                    _im0 = _aec.decode(torch.cat(_c0s, 0))[..., _mo:-_mo, _mo:-_mo, _mo:-_mo]
                                _lcr = F.l1_loss(_imp - _im0, _img - _im0)
                                loss = loss + _croprec_w * _lcr
                                if global_counter[mode] % 200 == 0:
                                    print(f"[croprec-change] L1={_lcr.item():.5f} (x{_croprec_w} -> {_croprec_w*_lcr.item():.5f} vs loss {loss.item():.4f})", flush=True)
                            else:
                                loss = loss + _croprec_w * F.l1_loss(_imp, _img)

                        # ----- (3) masked image-space recon aux: decode predicted followup vs real -----
                        _recon_w = float(getattr(args, "fm_recon_w", 0.0))
                        if _recon_w > 0.0 and mode == 'train':
                            _ae = autoencoder.module if hasattr(autoencoder, "module") else autoencoder
                            _dhr2 = pred if _grav_noise is None else (pred + _grav_noise)
                            x1_pred = (x0 + _dhr2) / scale_factor                    # delta-pred: x1 = x0 + delta_hat
                            with torch.no_grad():
                                img_true = _ae.decode(x1 / scale_factor)
                            img_pred = _ae.decode(x1_pred)                           # grad flows to pred
                            _rmap = torch.abs(img_pred - img_true)                   # (B,1,H,W,D) normalized space
                            if _em is not None:
                                _emi  = F.interpolate(_em, size=img_true.shape[-3:], mode='trilinear', align_corners=False)
                                _wvi  = 1.0 + mask_lambda * _emi
                                _recon = (_wvi * _rmap).sum() / _wvi.expand_as(_rmap).sum().clamp_min(1e-6)
                            else:
                                _recon = _rmap.mean()
                            loss = loss + _recon_w * _recon
                            epoch_recon += _recon.item()

                        epoch_mse_ps += mse_ps.mean().item()
                        epoch_cos_ps += cos_ps.mean().item()

                    else:
                        loss = F.l1_loss(pred, ut)

          

                if mode == 'train':
                    # Accumulated Loss
                    loss = loss / gradient_accumulation_steps  # normalize loss
                    accelerator.backward(loss)

                    if (step + 1) % gradient_accumulation_steps == 0 or (step + 1 == len(loader)):
                        optimizer.step()
                        optimizer.zero_grad()
                        if ema_model is not None:
                            with torch.no_grad():
                                _m = flowmatching.module if hasattr(flowmatching, "module") else flowmatching
                                for _pe, _pm in zip(ema_model.parameters(), _m.parameters()):
                                    _pe.mul_(_ema_d).add_(_pm.detach(), alpha=1.0 - _ema_d)
                                for _be, _bm in zip(ema_model.buffers(), _m.buffers()):
                                    _be.copy_(_bm)
                    # ---- discriminator update (after G step; uses detached tensors) ----
                    if disc is not None and '_adv_cache' in dir() and _adv_cache is not None:
                        _dh_d, _dg_d = _adv_cache
                        disc_opt.zero_grad(set_to_none=True)
                        _ld = 0.5 * (adv_fn(disc(_dg_d.float())[-1], target_is_real=True, for_discriminator=True)
                                   + adv_fn(disc(_dh_d.float())[-1], target_is_real=False, for_discriminator=True))
                        _ld.backward(); disc_opt.step()
                        _adv_cache = None


                epoch_loss += loss.item()

                if use_rectified:
                    progress_bar.set_postfix({
                        "Step": step,
                        "Mag" : epoch_mse_ps / (step + 1),
                        "Cos" : epoch_cos_ps / (step + 1),
                        "Loss": epoch_loss / (step + 1),
                        "Rec" : epoch_recon / (step + 1),
                        "w"   : w.mean().item() ,
                        # "Percept": perceptual_loss.item(),
                    })

                else:
                    progress_bar.set_postfix({
                        "Step": step,
                        "Loss": epoch_loss / (step + 1),
                        # "Percept": perceptual_loss.item(),
                    })

                global_counter[mode] += 1

            # end of epoch
            epoch_loss = epoch_loss / len(loader)
            writer.add_scalar(f'{mode}/epoch-mse', epoch_loss, epoch)

            # visualize results
            images_to_tensorboard(
                batch=batch,
                writer=writer,
                epoch=epoch,
                mode=mode,
                autoencoder=autoencoder,
                diffusion=flowmatching,
                scale_factor=scale_factor,
                modality_names=["t1c"]
            )

            images_to_tensorboard(
                batch=batch,
                writer=writer,
                epoch=epoch,
                mode=mode,
                autoencoder=autoencoder,
                diffusion=flowmatching,
                scale_factor=scale_factor,
                modality_names=["t1c"],
                use_heun=True
            )




        # save the model                
        savepath = os.path.join(args.output_dir, f'fm-unet-ep-{epoch}.pth')
        # torch.save(flowmatching.state_dict(), savepath)
        

        if accelerator.is_main_process:
            _sv = ema_model if ema_model is not None else flowmatching
            accelerator.save(_sv.state_dict(), savepath)
            if controlnet is not None:
                _cn = controlnet.module if hasattr(controlnet, "module") else controlnet
                accelerator.save(_cn.state_dict(), os.path.join(args.output_dir, f'fm-cnet-ep-{epoch}.pth'))
                try: os.remove(os.path.join(args.output_dir, f'fm-cnet-ep-{epoch - 1}.pth'))
                except FileNotFoundError: pass
            if history_encoder is not None:
                _he = history_encoder.module if hasattr(history_encoder, "module") else history_encoder
                accelerator.save(_he.state_dict(), os.path.join(args.output_dir, f'fm-hist-ep-{epoch}.pth'))
                pass   # deletion is handled by the --keep_ckpt logic below
            # ---- checkpoint retention: keep every epoch by default and let the external per-epoch
            # eval pick the best; --keep_ckpt N keeps only the last N so peak checkpoints are not lost.
            _keepn = int(getattr(args, 'keep_ckpt', 0))
            savepath = os.path.join(args.output_dir, f'fm-unet-ep-{epoch}.pth')
            if _keepn > 0:
                _old = epoch - _keepn
                if _old >= 0:
                    for _pat in (f'fm-unet-ep-{_old}.pth', f'fm-hist-ep-{_old}.pth', f'fm-cnet-ep-{_old}.pth'):
                        try:
                            os.remove(os.path.join(args.output_dir, _pat))
                        except FileNotFoundError:
                            pass

        print("Saving models to: ", savepath)
        print(f"Scaling factor set to {scale_factor}")

        gc.collect()
        torch.cuda.empty_cache()

        accelerator.wait_for_everyone()
