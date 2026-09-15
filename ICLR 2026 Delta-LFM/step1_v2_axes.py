import os, gc, sys

# self-contained: resolve src/ utils/ dataset/ from THIS folder, not the parent repo
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from utils import args

if 'LOCAL_RANK' not in os.environ:  # single-proc: honor --gpu; under accelerate multi-proc let it assign devices
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch.nn.functional as F
import warnings
import numpy as np
import torch
# avoid "received 0 items of ancdata" — use file_system sharing for many DataLoader workers
import torch.multiprocessing as _torch_mp
try:
    _torch_mp.set_sharing_strategy('file_system')
except Exception:
    pass
from tqdm import tqdm
from monai.utils import set_determinism

from torch.nn import L1Loss
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from monai.losses import PerceptualLoss, PatchAdversarialLoss
from src.partial_fc_v2 import PartialFC_V2

from src.model2D import utils_usage
from utils.utils_image import standardize_images, min_max_normalize
from utils import import_from_dotted_path, utils_metric
from src.model3D import (
    KLDivergenceLoss, init_patch_discriminator, GradientAccumulation
)
from utils.utils_image import save_image, pad_to_shape
from src.contrastive_losses import CombinedMarginLoss, RnCLoss, ArcFace, AngleLoss, DistCrossEntropy, \
    monotonicity_triplet_loss
from src.trajectory_losses import linear_trajectory_loss as traj_loss_fn
import accelerate
from accelerate import Accelerator
from collections import OrderedDict
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as ms_ssim

# Derived-artifact root (population priors, PCA bases, energy maps).
DERIVED_DIR = os.environ.get("DERIVED_DIR",
                             os.path.join(os.environ.get("DATA_DIR", "."), "derived"))



warnings.filterwarnings("ignore")
set_determinism(seed=int(getattr(args, 'seed', 0)))   # configurable via --seed
_NORM_SHARED = [0]     # must be defined before this point (the module executes top to bottom)
_NORM_SHARED[0] = int(getattr(args, "norm_shared", 0))
if _NORM_SHARED[0]:
    print("[norm_shared] enabled: the three visits of a subject share the baseline mu/sigma", flush=True)


save_epoch = max(1, int(getattr(args, "save_every", 5) or 5))
adv_weight        = getattr(args, "adv_weight", 0.1)          # MAISI-faithful default
perceptual_weight = getattr(args, "perceptual_weight", 0.3)
kl_weight         = getattr(args, "kl_weight", 1e-7)

accelerator = Accelerator()
DEVICE = accelerator.device

use_contrastive = args.use_contrastive
use_new_dataset = True
use_lpips       = getattr(args, "use_lpips", True)   # MAISI: on (perceptual 0.3)
use_kl          = getattr(args, "use_kl", True)
# Base losses: L1 + perceptual + adversarial are on by default; SSIM is off by default.
#   --use_adv / --use_ssim are still accepted for backward compatibility, but the effective switches are --no_adv / --no_ssim.
use_adv         = (not getattr(args, "no_adv", False))
use_ssim        = (getattr(args, "use_ssim", False) and not getattr(args, "no_ssim", True))


image_root = "./image_result/oasis_step1_ae_train_contrastive"
args.output_dir = f"{args.temp_path}/{args.output_dir}"

if use_ssim:
    args.output_dir += "_ssim"

print("Output directory:", args.output_dir)
print("Save   directory:", image_root)

os.makedirs(image_root, exist_ok=True)
os.makedirs(args.output_dir, exist_ok=True)



# ===== precomputed energy maps (used by --ae_dircosE_w). Available at training time because the change locations of training pairs are known annotations =====
_ENERGY_CACHE = {}
_ENERGY_DIR = os.path.join(DERIVED_DIR, "energy_full")
def _load_energy(path, out_shape):
    """Load the energy_full.nii.gz matching an image path and resample it to out_shape; returns None if unavailable."""
    import os as _os
    k = "/".join(str(path).split("/")[-3:-1])
    if k in _ENERGY_CACHE:
        return _ENERGY_CACHE[k]
    p = _os.path.join(_ENERGY_DIR, k, "energy_full.nii.gz")
    v = None
    if _os.path.exists(p):
        try:
            import nibabel as _nib, numpy as _np
            from scipy.ndimage import zoom as _zoom
            a = _np.asarray(_nib.load(p).dataobj).astype("float32")
            if a.max() > 0:
                if tuple(a.shape) != tuple(out_shape):
                    a = _zoom(a, [out_shape[i] / a.shape[i] for i in range(3)], order=1)
                v = _np.maximum(a, 0.0)
        except Exception:
            v = None
    if len(_ENERGY_CACHE) < 4000:
        _ENERGY_CACHE[k] = v
    return v


def clean_model_loading(path, device="cpu"):
    """
    Loads a model checkpoint and removes 'module.' prefixes from keys
    (for DataParallel-trained models). Also clears GPU/CPU cache to help 
    prevent memory leaks before loading.
    """
    gc.collect()
    torch.cuda.empty_cache()

    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)  # works if it's inside or not

    # Remove "module." prefix from keys
    state_dict = OrderedDict(
        (k.replace("module.", "", 1), v) for k, v in state_dict.items()
    )

    return state_dict




def _norm_in(x, mode, share=False, params=None):
    """Normalize the ENCODER input to a chosen range; returns (x_normed, params).
    Losses/eval stay in [0,1] via _denorm_out, so the only variable is what the
    encoder sees. '01'=[0,1] (identity), 'pm1'=[-1,1], 'std'=zero-mean unit-std.

    When share=True, x must be a stacked triplet [v1|v2|v3] ([3B,...]): the three visits
    share the mu/sigma of baseline v1. Computing them per volume makes each visit's
    normalization differ, which injects an artificial component into the inter-visit delta
    that can rival the biological signal. When params is not None the given statistics are
    reused, which aligns a perturbed view with the original."""
    if mode == "pm1":
        return x * 2.0 - 1.0, None
    if mode == "std":
        if params is not None:
            m, s = params
        else:
            m = x.mean(dim=(-1, -2, -3), keepdim=True)
            s = x.std(dim=(-1, -2, -3), keepdim=True).clamp_min(1e-5)
            if share:
                assert x.shape[0] % 3 == 0, \
                    "norm_shared requires a stacked triplet, received batch=%d" % x.shape[0]
                _b = x.shape[0] // 3
                m = m[:_b].repeat(3, 1, 1, 1, 1)
                s = s[:_b].repeat(3, 1, 1, 1, 1)
                if not globals().get("_NSH_SHOWN"):
                    globals()["_NSH_SHOWN"] = 1
                    print("[norm_shared] three visits share the baseline mu/sigma "
                          "(batch=%d, B=%d)" % (x.shape[0], _b), flush=True)
        return (x - m) / s, (m, s)
    return x, None


def _denorm_out(y, mode, params):
    if mode == "pm1":
        return (y + 1.0) / 2.0
    if mode == "std":
        m, s = params
        return y * s + m
    return y


_SUB = {"zbuf": [], "dbuf": [], "Vz": None, "Vd": None, "W": None, "n": 0, "dmean": None}


def _sub_update(z0, dl, K, every=200, cap=256, lam=1.0):
    """Maintain the top-K principal-component bases of the population delta and z0, together

    Rationale for the subspace: reproducible patient-specific directions live in the leading
    principal components; over the full latent dimensionality the signal is a small fraction
    of the norm and the gradient is dominated by noise. The buffers and the SVD run on CPU
    under no_grad, and the basis and W are constants with respect to the encoder (no
    backpropagation)."""
    import torch as _t
    _SUB["zbuf"].append(z0.detach().float().cpu())
    _SUB["dbuf"].append(dl.detach().float().cpu())
    # running population mean and collapse monitor: a high population share means every delta points the same way, which is a degenerate solution
    if len(_SUB["dbuf"]) >= 8:
        _Db = _t.cat(_SUB["dbuf"], 0)
        _dm = _Db.mean(0, keepdim=True)
        _SUB["dmean"] = _dm
        if (_SUB["n"] % 200) == 0:
            _pop = float((_dm ** 2).sum() / _Db.pow(2).sum(1).mean().clamp_min(1e-9))
            print("[collapse monitor] population share |delta_bar|^2/E|delta|^2 = %.4f "
                  "(higher means more uniform motion) | buffer %d" % (_pop, _Db.shape[0]), flush=True)
    if len(_SUB["zbuf"]) > cap:
        _SUB["zbuf"].pop(0); _SUB["dbuf"].pop(0)
    _SUB["n"] += 1
    if len(_SUB["zbuf"]) < max(2 * K, 32) or (_SUB["n"] % every) != 0:
        return
    with _t.no_grad():
        Zb = _t.cat(_SUB["zbuf"], 0); Db = _t.cat(_SUB["dbuf"], 0)
        Zb = Zb - Zb.mean(0, keepdim=True); Db = Db - Db.mean(0, keepdim=True)
        k = int(min(K, Zb.shape[0] - 1, Db.shape[0] - 1))
        # far fewer samples than dimensions, so the Gram matrix route is cheaper
        def _basis(M):
            G = (M @ M.T).double()
            w, U = _t.linalg.eigh(G)
            o = _t.argsort(w, descending=True)[:k]
            return (M.T.double() @ U[:, o] / _t.sqrt(w[o].clamp_min(1e-9))).float()
        Vz, Vd = _basis(Zb), _basis(Db)
        Fz = (Zb @ Vz).double(); Fd = (Db @ Vd).double()
        A = Fz.T @ Fz + lam * _t.eye(Fz.shape[1], dtype=_t.float64)
        W = _t.linalg.solve(A, Fz.T @ Fd).float()
        dev = z0.device
        _SUB["Vz"], _SUB["Vd"], _SUB["W"] = Vz.to(dev), Vd.to(dev), W.to(dev)
        if not globals().get("_SUB_SHOWN"):
            globals()["_SUB_SHOWN"] = 1
            print("[traj_subspace] basis ready K=%d (buffer %d samples)" % (k, Zb.shape[0]), flush=True)


def _sub_proj(x, which="d"):
    """Project onto the subspace; returns the input unchanged when the basis is not ready, so the caller decides whether to skip."""
    V = _SUB["Vd"] if which == "d" else _SUB["Vz"]
    return x if V is None else (x @ V)


def _std_frame(rec, gt):
    """Standardize both recon and gt with the per-volume (mean,std) of gt, returning (rec_n, gt_n).

    When change losses are computed on raw [0,1] differences, global brightness and contrast
    drift enters delta_gt in full, raising the dt-independent noise level and lowering the cos
    any predictor can reach. recon is standardized with the statistics of gt rather than its
    own, so shifting its own mean is not rewarded."""
    m = gt.mean(dim=(-1, -2, -3), keepdim=True)
    s = gt.std(dim=(-1, -2, -3), keepdim=True).clamp_min(1e-5)
    return (rec - m) / s, (gt - m) / s


def _lsf_on():
    if not int(getattr(args, "loss_std_frame", 0)):
        return False
    if not globals().get("_LSF_SHOWN"):
        globals()["_LSF_SHOWN"] = 1
        print("[loss_std_frame] change losses now use the per-volume standardized frame "
              "(per-volume standardised frame)", flush=True)
    return True


def ae_forward(model, images, mode, share=False):
    """AE forward with encoder-input normalization; reconstruction is mapped back
    to [0,1] so every loss/metric is computed in the same space regardless of mode.
    share=True: the triplet shares the baseline statistics, and _denorm_out reuses the same p,
    so reconstructions and ground truth lie in the same intensity frame."""
    x_in, p = _norm_in(images, mode, share=share)
    recon_n, z_mu, z_sigma = model(x_in)
    return _denorm_out(recon_n, mode, p), z_mu, z_sigma


def _nuis_perturb(x, shift_vox=1.0, bias_amp=0.07, noise_sig=0.01):
    """Acquisition nuisance transform: sub-voxel rigid translation (by interpolation), a smooth bias field and Gaussian noise. Anatomy is unchanged."""
    import torch.nn.functional as _F
    B = x.shape[0]; D, H, W = x.shape[2:]
    if shift_vox > 0:
        t = (torch.rand(B, 3, device=x.device, dtype=torch.float32) * 2 - 1) * shift_vox
        th = torch.zeros(B, 3, 4, device=x.device, dtype=torch.float32)
        th[:, 0, 0] = 1; th[:, 1, 1] = 1; th[:, 2, 2] = 1
        th[:, 0, 3] = 2.0 * t[:, 0] / max(W - 1, 1)
        th[:, 1, 3] = 2.0 * t[:, 1] / max(H - 1, 1)
        th[:, 2, 3] = 2.0 * t[:, 2] / max(D - 1, 1)
        grid = _F.affine_grid(th, list(x.shape), align_corners=False)
        x = _F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)
    if bias_amp > 0:
        g = torch.randn(B, 1, 4, 4, 4, device=x.device, dtype=torch.float32)
        g = _F.interpolate(g, size=(D, H, W), mode="trilinear", align_corners=False)
        g = g / g.abs().amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
        x = x * (1.0 + bias_amp * g)
    if noise_sig > 0:
        x = x + noise_sig * torch.randn_like(x)
    return x.clamp_min(0.0)


@torch.no_grad()
def validate_model(model, dataloader, device, save_root=None, max_batches=6, step_name=""):
    model.eval()
    if save_root is not None:
        os.makedirs(save_root, exist_ok=True)

    avg_psnr, avg_ssim = [], []

    
    for idx, batch in enumerate(dataloader):
        if idx >= max_batches:
            break

        images = batch["starting"].to(device).float()

        with accelerator.autocast():
            reconstruction, z_mu, z_sigma = ae_forward(model, images, getattr(args, "norm_mode", "01"))

        image_np = images.cpu().numpy()
        recon_np = reconstruction.cpu().numpy()

        # Compute PSNR and SSIM
        psnr_val = utils_metric.psnr_3d(image_np, recon_np)
        ssim_val = utils_metric.ssim_3d(image_np, recon_np)

        avg_psnr.append(psnr_val)
        avg_ssim.append(ssim_val)

        # Save middle slice comparison image
        middle_slice = image_np.shape[2] // 2
        image_mid = image_np[:, :, middle_slice, :, :]  # B, C, H, W
        recon_mid = recon_np[:, :, middle_slice, :, :]
        if save_root is not None:
            for b in range(image_mid.shape[0]):
                combined = np.concatenate([image_mid[b], recon_mid[b]], axis=2)[0]  # C=1 → [H, 2W]
                save_path = os.path.join(save_root, f"{step_name}_img_{idx}_{b}.jpg")
                save_image(save_path, combined)

    print(f"{step_name} - AVG_PSNR: {np.mean(avg_psnr):.2f}, AVG_SSIM: {np.mean(avg_ssim):.4f}")
    return float(np.mean(avg_psnr)), float(np.mean(avg_ssim))



class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3, reduction="mean"):
        super().__init__()
        self.eps = eps
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss




if __name__ == '__main__':
    
    # ---------------- Define Dataloader ----------------
    # AD-Progression flat-CSV triplet loader (builds time-ordered triplets on the
    # fly from AD-Progression-All.csv). Falls back to the legacy OASIS pair loader.
    if getattr(args, "dataloader", "ad_progression") == "ad_progression":
        from dataset.ad_progression_3D_triplet import get_brain_dataset
    else:
        from dataset.oasis_dataset_3D_pair_contrastive import get_brain_dataset

    train_loader, ds           = get_brain_dataset(args, mode="train")
    test_loader, test_ds       = get_brain_dataset(args, mode="test")

    # per-visit loader for the latent-trajectory linearity probe (ad_progression only)
    linear_loader = None
    try:
        from dataset.ad_progression_3D_triplet import get_visit_dataset
        from monitor_latent_linearity import evaluate_linearity, format_summary
        linear_loader, _ = get_visit_dataset(args, mode="test")
        print(f"[linearity] probe loader ready: {len(linear_loader.dataset)} visits")
    except Exception as e:
        print(f"[linearity] probe disabled: {e}")

    # ---------------- Define AutoEncoder Model ----------------
    autoencoder_func = import_from_dotted_path(args.autoencoder)
    autoencoder      = autoencoder_func(args).float()

    discriminator = init_patch_discriminator(args.disc_ckpt, spatial_dims=args.dim, 
                                             in_channels=1, num_layers_d=3)

    

    # ---------------- Resume Path ----------------
    if args.aekl_ckpt is not None:
        print("Loading autoencoder from:", args.aekl_ckpt)
        autoencoder.load_state_dict(clean_model_loading(args.aekl_ckpt), strict=True)

    if args.disc_ckpt is not None:
        print("Loading discriminator from:", args.disc_ckpt)
        discriminator.load_state_dict(clean_model_loading(args.disc_ckpt), strict=True)

    autoencoder.to(DEVICE)
    discriminator.to(DEVICE)


    l1_loss_fn  = CharbonnierLoss() #L1Loss()
    ms_ssim_fn  = ms_ssim(data_range=1.0,
                          kernel_size=5,                           # was 11
                          betas=(0.0448, 0.2856, 0.3001)).to(DEVICE)  # keep 5 scales) 

    kl_loss_fn  = KLDivergenceLoss()
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")
    # element-wise variant, for energy weighting (reduction=none)
    adv_loss_fn_none = PatchAdversarialLoss(criterion="least_squares", reduction="none")

    # contrastive loss
    angle_loss_fn      = AngleLoss(temperature=0.07)
    rnc_loss_fn        = RnCLoss(temperature=2, label_diff='l1', feature_sim='l2')
    dist_cross_entropy = nn.CrossEntropyLoss()

    margin_list = (1.0, 0.0, 0.4)
    interclass_filtering_threshold = 0
    margin_loss_fn = CombinedMarginLoss(  # 512
        64,
        margin_list[0],
        margin_list[1],
        margin_list[2],
        interclass_filtering_threshold
    )
    embedding_size = 512
    # num_classes = len(id_to_label.keys())

    sample_rate = 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perc_loss_fn = PerceptualLoss(spatial_dims=args.dim,  
                                      network_type="squeeze",
                                      is_fake_3d=True,
                                      fake_3d_ratio=0.2).to(DEVICE)

    if getattr(args, "freeze_decoder", False):
        _fz = 0
        for n, p in autoencoder.named_parameters():
            if "decod" in n.lower():
                p.requires_grad_(False); _fz += 1
        print(f"[freeze_decoder] froze {_fz} decoder param tensors", flush=True)
    trainable = [p for n, p in autoencoder.named_parameters() if p.requires_grad]
    all_sum = sum(p.numel() for p in autoencoder.parameters())
    print(f"\nAll parameters: {all_sum / 1e6:.2f} M")

    trainable_sum = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)   
    print(f"\nTrainable parameters: {trainable_sum / 1e6:.2f} M")


    optimizer_g = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-6)
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr)

    avgloss = utils_usage.AverageLoss()
    writer = SummaryWriter() if accelerator.is_main_process else type("_NW", (), {"add_scalar": lambda *a, **k: None, "close": lambda *a, **k: None})()
    total_counter = 0


    # ---------------- Prepare Model for training ----------------
    autoencoder, discriminator, optimizer_g, optimizer_d, train_loader, ms_ssim_fn = accelerator.prepare(
        autoencoder, discriminator, optimizer_g, optimizer_d, train_loader, ms_ssim_fn
    )  # test_loader NOT prepared: full/unsharded so every rank evals the same set (DDP sync-safe)

    # Test at starter
    validate_model(accelerator.unwrap_model(autoencoder), test_loader, DEVICE, save_root=None, max_batches=6, step_name="Start")



    for epoch in range(args.n_epochs):
        
        if accelerator.process_index == 1:
            print("EPOCH: ", epoch)
            print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            print(f"Cached GPU memory: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

        autoencoder.train()
        _spe = int(getattr(args, "steps_per_epoch", 0) or 0)
        _total = min(len(train_loader), _spe) if _spe else len(train_loader)
        progress_bar = tqdm(enumerate(train_loader), total=_total)
        progress_bar.set_description(f'Epoch {epoch}')

        for step, batch in progress_bar:
            if _spe and step >= _spe:
                break
            gc.collect()
            torch.cuda.empty_cache()

            if step > 100 and args.DEBUG : break

            images      = batch["starting"].to(DEVICE) #.to(torch.float32)
            followup    = batch["followup"].to(DEVICE) #.to(torch.float32)
            followup2   = batch["followup2"].to(DEVICE) #.to(torch.float32)
            s_id        = batch["subject_id"]
            # print("s_id:", s_id)

            patient_ids = torch.tensor(s_id, device=DEVICE) # it is a list to torch

            patient_age = torch.cat([batch['starting_age'],
                                     batch['followup_age'],
                                     batch['followup2_age']], dim=0).to(DEVICE).unsqueeze(-1)

            # true monotone time axis for the ArcRank ordering: follow_up (months
            # since baseline) is finely increasing, whereas `age` is coarse (year
            # resolution -> ties). Fall back to age if follow_up is not provided.
            if 'starting_follow_up' in batch:
                patient_time = torch.cat([batch['starting_follow_up'],
                                          batch['followup_follow_up'],
                                          batch['followup2_follow_up']], dim=0).to(DEVICE).unsqueeze(-1).float()
            else:
                patient_time = patient_age
            # print("patient_age = ", patient_age )  #

            # PER-VISIT biomarker severity for the disease-anchor loss. Higher =
            # more disease: ventricle enlarges, hippocampus atrophies. Within-patient the
            # head size is constant, so raw volume CHANGES are already comparable.
            patient_biom = None
            if getattr(args, "arc_biomarker", 0.0) > 0 and 'starting_hippocampus' in batch:
                _hip = torch.cat([batch['starting_hippocampus'], batch['followup_hippocampus'],
                                  batch['followup2_hippocampus']], 0).to(DEVICE).float()
                _ven = torch.cat([batch['starting_lateral_ventricle'], batch['followup_lateral_ventricle'],
                                  batch['followup2_lateral_ventricle']], 0).to(DEVICE).float()
                patient_biom = torch.stack([_hip, _ven], -1)   # [3B, 2]

            B = images.shape[0]

            images      = torch.cat([images, followup, followup2], dim=0)  # Concatenate along batch dimension
            patient_ids = torch.cat([patient_ids, patient_ids, patient_ids], dim=0).unsqueeze(-1)

            with accelerator.autocast():

                reconstruction, z_mu, z_sigma = ae_forward(
                    autoencoder, images, getattr(args, "norm_mode", "01"),
                    share=bool(_NORM_SHARED[0]))   # the triplet shares the baseline mu/sigma
                # rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                loss_g = rec_loss
                # ----- (A2) energy-weighted reconstruction: weight the reconstruction loss by the observed cross-scan change magnitude -----
                _erw = float(getattr(args, 'ae_erec_w', 0.0))
                if _erw > 0.0 and reconstruction.shape[0] >= 3 * B:
                    with torch.no_grad():
                        _e = (images[0*B:1*B] - images[2*B:3*B]).abs().float()
                        _fl = _e.reshape(B, -1)
                        _rk = _fl.argsort(dim=1).argsort(dim=1).float() / max(_fl.shape[1] - 1, 1)
                        _E = _rk.reshape(_e.shape)
                        _W = (1.0 + _erw * _E).repeat(3, 1, 1, 1, 1)
                    _err = (reconstruction.float() - images.float()).abs()
                    rec_loss = (_err * _W).sum() / _W.sum().clamp_min(1e-6)
                    loss_g = rec_loss
                    if not globals().get('_AEREC_SHOWN'):
                        globals()['_AEREC_SHOWN'] = 1
                        print('[ae_erec] energy-weighted reconstruction active lam=%s w range [%.2f,%.2f]' % (_erw, float(_W.min()), float(_W.max())), flush=True)

                # ----- (CHG) change preservation: pair the three timepoints and optimize the fidelity of the difference between two scans directly -----
                # The other autoencoder losses act on single images; the difference between two scans is a small
                # fraction of the intensity range and contributes almost nothing to a single-image loss, so the
                # change term is added explicitly. Its magnitude is comparable to rec_loss, so lam=1 is a sane start.
                _acw = float(getattr(args, 'ae_change_w', 0.0))
                if _acw > 0.0 and reconstruction.shape[0] >= 3 * B:
                    _R = reconstruction.float(); _I = images.float(); _chg = 0.0
                    _NP = int(getattr(args, 'ae_change_pairs', 1))
                    _PAIRS = ((0, 2),) if _NP <= 1 else (((0, 2), (0, 1)) if _NP == 2 else ((0, 1), (1, 2), (0, 2)))
                    for _i, _j in _PAIRS:
                        _chg = _chg + l1_loss_fn(_R[_i*B:(_i+1)*B] - _R[_j*B:(_j+1)*B],
                                                 _I[_i*B:(_i+1)*B] - _I[_j*B:(_j+1)*B])
                    _chg = _chg / max(len(_PAIRS), 1)
                    loss_g = loss_g + _acw * _chg
                    if not globals().get('_AECHG_SHOWN'):
                        globals()['_AECHG_SHOWN'] = 1
                        print('[ae_change] active w=%s rec=%.5f chg=%.5f ratio=%.3f' % (_acw, float(rec_loss), float(_chg), float(_chg)/max(float(rec_loss),1e-9)), flush=True)


                # adversarial WARM-UP: hold adv off for the first adv_warmup steps, then
                # ramp adv_weight in over adv_ramp steps (lets reconstruction stabilize first).
                _awu = getattr(args, "adv_warmup", 0); _arp = getattr(args, "adv_ramp", 1000)
                adv_scale = 0.0 if total_counter < _awu else min(1.0, (total_counter - _awu) / max(1, _arp))
                eff_adv = adv_weight * adv_scale
                if use_adv:
                    logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                    # Weight the adversarial patches by energy so the discriminator attends to where change occurs.
                    #   The discriminator output is a per-patch logit map; E is downsampled to that grid and w = 1 + lambda*E
                    #   replaces the plain mean with a weighted mean. E matches the --ae_erec_w definition: rank_normalize(|v1-v3|).
                    _aew = float(getattr(args, "adv_energy_w", 0.0))
                    if _aew > 0.0 and reconstruction.shape[0] >= 3 * B:
                        with torch.no_grad():
                            _ea = (images[0*B:1*B] - images[2*B:3*B]).abs().float()
                            _fa = _ea.reshape(B, -1)
                            _ra = _fa.argsort(dim=1).argsort(dim=1).float() / max(_fa.shape[1] - 1, 1)
                            _Ea = _ra.reshape(_ea.shape).repeat(3, 1, 1, 1, 1)
                            _Wa = F.interpolate(_Ea, size=tuple(logits_fake.shape[-3:]),
                                                mode="trilinear", align_corners=False)
                            _Wa = 1.0 + _aew * _Wa
                        _per = adv_loss_fn_none(logits_fake, target_is_real=True, for_discriminator=False)
                        if isinstance(_per, (list, tuple)):
                            _per = _per[0]
                        while _per.dim() < _Wa.dim():
                            _per = _per.unsqueeze(1)
                        gen_loss = eff_adv * ((_per * _Wa).sum() / _Wa.sum().clamp_min(1e-6))
                        if not globals().get("_ADVE_SHOWN"):
                            globals()["_ADVE_SHOWN"] = 1
                            print("[adv-energy] per-patch weighting active lam=%.2f | logit grid %s | w range [%.2f,%.2f]"
                                  % (_aew, tuple(logits_fake.shape[-3:]),
                                     float(_Wa.min()), float(_Wa.max())), flush=True)
                    else:
                        gen_loss = eff_adv * adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)
                else:
                    gen_loss = torch.tensor(0.0, device=DEVICE)

                if use_kl:
                    # ----- acquisition-nuisance invariance: the latent should be invariant to translation, bias field and noise -----
                    _nw = float(getattr(args, "ae_nuis_w", 0.0))
                    if _nw > 0.0:
                        # the encoder is fully convolutional, so invariance can be applied on random crops, cutting memory roughly 6x
                        _cs = int(getattr(args, "ae_nuis_crop", 64))
                        _D, _H, _W = images.shape[2:]
                        _cs = min(_cs, _D, _H, _W)
                        _d0 = int(torch.randint(0, max(_D - _cs, 1), (1,)).item())
                        _h0 = int(torch.randint(0, max(_H - _cs, 1), (1,)).item())
                        _w0 = int(torch.randint(0, max(_W - _cs, 1), (1,)).item())
                        _xc = images[:, :, _d0:_d0 + _cs, _h0:_h0 + _cs, _w0:_w0 + _cs]
                        with torch.no_grad():
                            _xp = _nuis_perturb(_xc.float(),
                                                float(getattr(args, "ae_nuis_shift", 1.0)),
                                                float(getattr(args, "ae_nuis_bias", 0.07)),
                                                float(getattr(args, "ae_nuis_noise", 0.01))).to(images.dtype)
                        _nm2 = getattr(args, "norm_mode", "01")
                        _enc = accelerator.unwrap_model(autoencoder).encode
                        # stop-gradient: the clean side is a fixed target (no_grad, no stored activations); only the perturbed side backpropagates
                        # both encodes must run under bf16 autocast, matching every other encode/decode in this file; fp32 activations would double memory
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            _xcn, _pc = _norm_in(_xc, _nm2, share=bool(_NORM_SHARED[0]))
                            _zc = _enc(_xcn)
                            _zc = (_zc[0] if isinstance(_zc, (tuple, list)) else _zc).float()
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            _zp = _enc(_norm_in(_xp, _nm2, params=_pc)[0])
                        _zp = _zp[0] if isinstance(_zp, (tuple, list)) else _zp
                        _nl = ((_zp.float() - _zc) ** 2).mean() / (_zc.var() + 1e-6)
                        loss_g = loss_g + _nw * _nl
                        globals()["_NUIS_N"] = globals().get("_NUIS_N", 0) + 1
                        if globals()["_NUIS_N"] == 1 or globals()["_NUIS_N"] % 100 == 0:
                            print("[ae_nuis] step %d active w=%.2f crop=%d | relative latent drift after perturbation %.4f "
                                  "| L_nuis/L_rec=%.2f"
                                  % (globals()["_NUIS_N"], _nw, _cs, float(_nl),
                                     _nw * float(_nl) / max(float(rec_loss.detach()), 1e-8)), flush=True)
                        if not globals().get("_NUIS_SHOWN"):
                            globals()["_NUIS_SHOWN"] = 1
                            print("[ae_nuis] nuisance invariance active w=%s shift=%s bias=%s noise=%s | nuis=%.5f" % (
                                _nw, getattr(args, "ae_nuis_shift", 1.0), getattr(args, "ae_nuis_bias", 0.07),
                                getattr(args, "ae_nuis_noise", 0.01), float(_nl)), flush=True)
                    kld_loss = kl_weight * kl_loss_fn(z_mu, z_sigma)
                else:
                    kld_loss = torch.tensor(0.0, device=DEVICE)
                
                if use_lpips:
                    # LPIPS expects [-1,1] (MONAI calls it with normalize=False); feed the rescaled range.
                    per_loss = perceptual_weight * perc_loss_fn(reconstruction.float()*2-1, images.float()*2-1)
                else:
                    per_loss = torch.tensor(0.0, device=DEVICE)
                
                if use_ssim:
                    ssim_loss = 1 - ms_ssim_fn(reconstruction.float(), images.float()).mean()
                else:
                    ssim_loss = torch.tensor(0.0, device=DEVICE)


                loss_g = rec_loss + kld_loss + gen_loss + per_loss + ssim_loss
                # The assignment above rebuilds loss_g from the base terms, so the change term
                # is re-added here under the same guard that computed it (_chg stays in scope).
                if _acw > 0.0 and reconstruction.shape[0] >= 3 * B:
                    loss_g = loss_g + _acw * _chg
                    if not globals().get("_AECHG_FIX_SHOWN"):
                        globals()["_AECHG_FIX_SHOWN"]=1
                        print("[ae_change_fix] change term reapplied after line 431 w=%s chg=%.5f loss_g=%.5f" % (_acw, float(_chg), float(loss_g)), flush=True)

                # Direction-only change loss: maximize the mean-removed cosine between the reconstructed and true change fields
                # (the directional component of the change PCC, independent of amplitude). Must be added after loss_g is rebuilt above.
                _dcw = float(getattr(args, "ae_dircos_w", 0.0))
                if _dcw > 0.0 and reconstruction.shape[0] >= 3 * B:
                    _Rd = reconstruction.float(); _Id = images.float()
                    # the brain mask uses a 0.05 threshold on the raw [0,1] image and must be computed before standardization
                    _bm = ((_Id[0:B] > 0.05) | (_Id[2*B:3*B] > 0.05)).reshape(B, -1).float()
                    if _lsf_on():
                        _Rd, _Id = _std_frame(_Rd, _Id)
                    _cr = (_Rd[2*B:3*B] - _Rd[0:B]).reshape(B, -1)
                    _cg = (_Id[2*B:3*B] - _Id[0:B]).reshape(B, -1)
                    _nn = _bm.sum(1, keepdim=True).clamp_min(1.0)
                    _cr = (_cr - (_cr*_bm).sum(1, keepdim=True)/_nn) * _bm
                    _cg = (_cg - (_cg*_bm).sum(1, keepdim=True)/_nn) * _bm
                    _cos = (_cr*_cg).sum(1) / (_cr.norm(dim=1)*_cg.norm(dim=1)).clamp_min(1e-6)
                    _dcl = (1.0 - _cos).mean()
                    loss_g = loss_g + _dcw * _dcl
                    if not globals().get("_AEDIRCOS_SHOWN"):
                        globals()["_AEDIRCOS_SHOWN"]=1
                        print("[ae_dircos] directional-cosine change loss active w=%s cos=%.4f loss=%.5f" % (_dcw, float(_cos.mean()), float(_dcl)), flush=True)
                # ===== change-reconstruction loss (optimizes the autoencoder ceiling directly) =====
                # dircos constrains direction only and leaves amplitude unconstrained; the ceiling is set by how accurately dec(z1)-dec(z0) reproduces x1-x0.
                # ===== directional cosine over the precomputed energy region; same loss as dircosR, only the region source differs =====
                _dew = float(getattr(args, "ae_dircosE_w", 0.0))
                if _dew > 0.0 and reconstruction.shape[0] >= 3 * B:
                    import numpy as _np
                    _sh = tuple(images.shape[2:])
                    _ems, _nok = [], 0
                    for _bi in range(B):
                        _e = None
                        for _kk in ("followup_image_path", "followup2_image_path"):
                            _pv = batch.get(_kk)
                            if _pv is None:
                                continue
                            _pp = _pv[_bi] if isinstance(_pv, (list, tuple)) else _pv
                            _x = _load_energy(_pp, _sh)
                            if _x is not None:
                                _e = _x if _e is None else (_e + _x)      # v2+v3 cover the v1-to-v3 interval
                        if _e is None:
                            _ems.append(_np.zeros(_sh, dtype="float32"))
                        else:
                            _ems.append(_e); _nok += 1
                    _Em = torch.from_numpy(_np.stack(_ems)).to(images.device).reshape(B, -1)
                    _epw = float(getattr(args, "ae_dircosE_pow", 0.0))
                    if _epw > 0.0:
                        # soft weighting: weight by the continuous energy magnitude rather than a binary mask
                        _pos = (_Em > 0).float()
                        _mE = (_Em * _pos).sum(1, keepdim=True) / _pos.sum(1, keepdim=True).clamp_min(1.0)
                        _Rm = torch.clamp((_Em / _mE.clamp_min(1e-8)) ** _epw, 0.0, 5.0) * _pos
                    else:
                        _Rm = (_Em > 0).float()                           # pow=0: plain binary mask
                    _ok = ((_Rm > 0).sum(1) > 100).float()                # skip samples whose region is too small
                    _Rd = reconstruction.float(); _Id = images.float()
                    if _lsf_on():
                        _Rd, _Id = _std_frame(_Rd, _Id)
                    _cr = (_Rd[2*B:3*B] - _Rd[0:B]).reshape(B, -1)
                    _cg = (_Id[2*B:3*B] - _Id[0:B]).reshape(B, -1)
                    _nn = _Rm.sum(1, keepdim=True).clamp_min(1.0)
                    _cr = (_cr - (_cr*_Rm).sum(1, keepdim=True)/_nn) * _Rm
                    _cg = (_cg - (_cg*_Rm).sum(1, keepdim=True)/_nn) * _Rm
                    _cosE = (_cr*_cg).sum(1) / (_cr.norm(dim=1)*_cg.norm(dim=1)).clamp_min(1e-6)
                    _del = ((1.0 - _cosE) * _ok).sum() / _ok.sum().clamp_min(1e-6)
                    loss_g = loss_g + _dew * _del
                    globals()["_DE_N"] = globals().get("_DE_N", 0) + B
                    globals()["_DE_OK"] = globals().get("_DE_OK", 0) + _nok
                    if globals()["_DE_N"] % 100 <= B:
                        print("[ae_dircosE] w=%.2f pow=%.2f | region covers %.2f%% of the brain (mean weight %.2f, max %.2f) | cos=%.4f | "
                              "L/L_rec=%.2f | maps available %d/%d"
                              % (_dew, _epw, 100.0*float((_Rm > 0).float().mean()),
                                 float(_Rm[_Rm > 0].mean()) if (_Rm > 0).any() else 0.0,
                                 float(_Rm.max()), float(_cosE.mean()),
                                 _dew*float(_del)/max(float(rec_loss.detach()), 1e-8),
                                 globals()["_DE_OK"], globals()["_DE_N"]), flush=True)

                _crw = float(getattr(args, "ae_chgrec_w", 0.0))
                if _crw > 0.0 and reconstruction.shape[0] >= 3 * B:
                    _Rc = reconstruction.float(); _Ic = images.float()
                    # the brain mask uses a 0.05 threshold on the raw [0,1] image and must be computed before standardization
                    _bmc = ((_Ic[0:B] > 0.05) | (_Ic[2*B:3*B] > 0.05)).reshape(B, -1)
                    if _lsf_on():
                        _Rc, _Ic = _std_frame(_Rc, _Ic)
                    _dr = (_Rc[2*B:3*B] - _Rc[0:B]).reshape(B, -1)
                    _dg = (_Ic[2*B:3*B] - _Ic[0:B]).reshape(B, -1)
                    _ag = _dg.abs()
                    # The change region is defined by a quantile of |true change| rather than a
                    #   fraction of P99, which on small blocks can select nearly the whole brain.
                    #   the quantile is taken within the brain only; including background would pull it down with many zeros.
                    _q = float(getattr(args, "ae_chgrec_q", 0.90))
                    _agb = _ag.detach().masked_fill(~_bmc, float("-inf"))
                    _tau = torch.quantile(_agb.float().clamp_min(-1e30), _q, dim=1, keepdim=True)
                    _Rm = (_ag > _tau) & _bmc
                    _crl = (((_dr - _dg).abs() * _Rm).sum(1) / (_ag * _Rm).sum(1).clamp_min(1e-6)).mean()
                    loss_g = loss_g + _crw * _crl
                    if not globals().get("_AECHGREC_SHOWN"):
                        globals()["_AECHGREC_SHOWN"] = 1
                        print("[ae_chgrec] active w=%.2f q=%.2f | change region covers %.2f%% of the brain | relative residual %.4f | peak memory %.1f GB"
                              % (_crw, _q,
                                 100 * (_Rm.float().sum() / _bmc.float().sum().clamp_min(1)).item(),
                                 float(_crl), torch.cuda.max_memory_allocated() / 1e9), flush=True)

                # Region-restricted directional cosine: computed only inside the change region R={|true change|>0.25*P99}.
                # ===== lower bound on visit separability (guard against collapse) =====
                _spw = float(getattr(args, "ae_sep_w", 0.0))
                if _spw > 0.0 and z_mu.shape[0] >= 3:
                    _s1, _s2, _s3 = z_mu.flatten(1).float().chunk(3, dim=0)
                    _m = float(getattr(args, "ae_sep_margin", 0.02))
                    _r12 = (_s2 - _s1).norm(dim=1) / _s1.norm(dim=1).detach().clamp_min(1e-6)
                    _r23 = (_s3 - _s2).norm(dim=1) / _s2.norm(dim=1).detach().clamp_min(1e-6)
                    _spl = (torch.relu(_m - _r12) + torch.relu(_m - _r23)).mean() * 0.5
                    loss_g = loss_g + _spw * _spl
                    globals()["_SEP_N"] = globals().get("_SEP_N", 0) + 1
                    if globals()["_SEP_N"] % 200 == 1:
                        _rm = float(torch.cat([_r12, _r23]).mean())
                        print("[ae_sep] w=%.3f margin=%.4f | measured separability r=%.5f | "
                              "hinge %s (loss=%.5f, %.2fx rec)"
                              % (_spw, _m, _rm, "active" if _rm < _m else "inactive",
                                 float(_spl), _spw * float(_spl) / max(float(rec_loss.detach()), 1e-8)),
                              flush=True)

                _dcrw = float(getattr(args, "ae_dircosR_w", 0.0))
                if _dcrw > 0.0 and reconstruction.shape[0] >= 3 * B:
                    _Rd = reconstruction.float(); _Id = images.float()
                    if _lsf_on():
                        _Rd, _Id = _std_frame(_Rd, _Id)
                    _cr = (_Rd[2*B:3*B] - _Rd[0:B]).reshape(B, -1)
                    _cg = (_Id[2*B:3*B] - _Id[0:B]).reshape(B, -1)
                    _ag = _cg.abs()
                    _tau = torch.quantile(_ag, 0.99, dim=1, keepdim=True) * 0.25
                    _Rm = (_ag > _tau.clamp_min(1e-6)).float()
                    _nn = _Rm.sum(1, keepdim=True).clamp_min(1.0)
                    _cr = (_cr - (_cr*_Rm).sum(1, keepdim=True)/_nn) * _Rm
                    _cg = (_cg - (_cg*_Rm).sum(1, keepdim=True)/_nn) * _Rm
                    _cos = (_cr*_cg).sum(1) / (_cr.norm(dim=1)*_cg.norm(dim=1)).clamp_min(1e-6)
                    _dcrl = (1.0 - _cos).mean()
                    loss_g = loss_g + _dcrw * _dcrl
                    if not globals().get("_AEDIRCOSR_SHOWN"):
                        globals()["_AEDIRCOSR_SHOWN"]=1
                        print("[ae_dircosR] region-restricted directional cosine active w=%s R share=%.3f cos=%.4f" % (_dcrw, float(_Rm.mean()), float(_cos.mean())), flush=True)
                # placeholders so the progress bar / del are valid in every mode
                angle_loss = rnc_loss = margin_loss = torch.tensor(0.0, device=DEVICE)
                traj_loss = torch.tensor(0.0, device=DEVICE); traj_parts = {}

                # ---- Phase-2 order-based trajectory loss (full latent, per patient) ----
                if getattr(args, "traj", False):
                    _terms = [t.strip() for t in getattr(args, "traj_terms", "T1,C1,C3").split(",") if t.strip()]
                    _wmap = {"T1": args.w_t1, "T3": args.w_t3, "C1": args.w_c1, "C3": args.w_c3,
                             "LINE": args.w_line, "MOVE": getattr(args, "w_move", 0.01)}
                    _wdict = {k: _wmap[k] for k in _terms if k in _wmap}
                    _ramp = min(1.0, (epoch + 1) / max(1, getattr(args, "warmup_epochs", 3)))
                    traj_loss, traj_parts = traj_loss_fn(
                        z_mu, patient_time, n_visits=3, weights=_wdict,
                        margin=getattr(args, "traj_margin", 0.1),
                        move_floor=getattr(args, "move_floor", 0.0))
                    loss_g = loss_g + _ramp * traj_loss

                # Independent switches: each trajectory loss is controlled by its own weight and does not require the --use_contrastive master switch.
                #   (use_contrastive is retained for backward compatibility.)
                _TRAJ_W = ("arc_decode", "arc_decode_extrap_w", "arc_change_w", "arc_change_corr_w",
                           "arc_decode_fixed", "angle_weight", "rnc_weight", "arc_order_w",
                           "arc_curv", "arc_svd", "arc_biomarker", "arc_vsmooth", "arc_isotropy", "arc_extrap_lat_w",
                           "ae_snr_w", "ae_predz_w")
                _traj_any = (any(float(getattr(args, _k, 0.0) or 0.0) > 0 for _k in _TRAJ_W)
                             or int(getattr(args, "ae_axes_k", 0) or 0) > 0
                             or bool(getattr(args, "arc_decode_extrap", False))
                             or (getattr(args, "arc_diversity", False)
                                 and float(getattr(args, "w_div", 0.0) or 0.0) > 0)
                             or (getattr(args, "arc_id_invariance", False)
                                 and float(getattr(args, "w_id", 0.0) or 0.0) > 0))
                if not globals().get("_SWITCH_SHOWN"):
                    globals()["_SWITCH_SHOWN"] = 1
                    _on = [_k for _k in _TRAJ_W if float(getattr(args, _k, 0.0) or 0.0) > 0]
                    print("[switches] active traj terms = %s | use_contrastive=%s"
                          % (_on if _on else "none", bool(use_contrastive)), flush=True)
                if use_contrastive or _traj_any:

                    orig_dtype = z_mu.dtype
                    z_full = z_mu.flatten(1)                    # full latent per visit
                    # progression SUBSPACE: apply trajectory terms to the first
                    # arc_channels latent channels only; the rest stay free to reconstruct
                    # (decouples fidelity vs target). arc_channels=0 -> use full latent.
                    _kc = getattr(args, "arc_channels", 0)
                    z_traj = z_mu[:, :_kc].reshape(z_mu.shape[0], -1) if _kc > 0 else z_full
                    # free "identity" channels — optionally forced CONSTANT across a patient's
                    # visits, so the FULL latent moves only along the progression subspace.
                    z_id = (z_mu[:, _kc:].reshape(z_mu.shape[0], -1)
                            if (_kc > 0 and _kc < z_mu.shape[1]) else None)

                    # TRAIN-TIME DECODE-ALONG-LINE CONSISTENCY — decode the interpolated
                    # (and optionally extrapolated) latent and push it to the REAL middle/
                    # future visit. Optimizes the exact downstream Δ-LFM objective (the
                    # straight line must DECODE to real progression), not the step_cos proxy.
                    _wdec = getattr(args, "arc_decode", 0.0)
                    # EXTRAPOLATION decode gets its OWN (gentler) weight: at the interpolation
                    # weight it over-constrains an out-of-distribution latent and costs image
                    # fidelity. Falls back to _wdec if only the boolean --arc_decode_extrap is set.
                    _wdec_ex = getattr(args, "arc_decode_extrap_w", 0.0)
                    if _wdec_ex == 0.0 and getattr(args, "arc_decode_extrap", False):
                        _wdec_ex = _wdec
                    if _wdec > 0 or _wdec_ex > 0:
                        zs1, zs2, zs3 = z_mu.chunk(3, dim=0)          # spatial [B,C,h,w,d]
                        img1, img2, img3 = images.chunk(3, dim=0)
                        _ae = accelerator.unwrap_model(autoencoder)
                        # decode under an explicit bf16 autocast: MAISI's norm_float16 needs
                        # autocast ON to line up with the float convolutions.
                        # The time coefficients come from patient_time (the same source as arc_decode_fixed) rather than hardcoded 0.5 / 1.0.
                        #   Visit intervals are uneven, so hardcoding them would train an incorrect temporal relation.
                        _pt3 = patient_time.reshape(3, -1).float()          # [3, B] times of the three visits
                        _d01 = (_pt3[1] - _pt3[0]).clamp_min(1e-6)
                        _d02 = (_pt3[2] - _pt3[0]).clamp_min(1e-6)
                        _alpha_v = ((_pt3[1] - _pt3[0]) / _d02).clamp(0.02, 0.98)   # interpolation position
                        _beta_raw = (_pt3[2] - _pt3[1]) / _d01                      # extrapolation factor (unclipped)
                        # Samples with out-of-range beta are skipped rather than clamped: clamping would
                        #   change their target, aligning a much longer extrapolation to the real v3 as if it were 3x.
                        #   Extrapolating from two nearly simultaneous visits to a distant third is ill-conditioned and is not used for training.
                        _beta_ok = ((_beta_raw >= 0.2) & (_beta_raw <= 3.0)).float()
                        _beta_v = _beta_raw.clamp(0.2, 3.0)
                        globals()["_BETA_N"] = globals().get("_BETA_N", 0) + int(_beta_ok.numel())
                        globals()["_BETA_SKIP"] = globals().get("_BETA_SKIP", 0) + int((1 - _beta_ok).sum())
                        if globals()["_BETA_N"] % 200 == 0:
                            print("[beta] %d samples seen, %d extrapolation terms skipped for out-of-range beta (%.1f%%)"
                                  % (globals()["_BETA_N"], globals()["_BETA_SKIP"],
                                     100.0 * globals()["_BETA_SKIP"] / max(globals()["_BETA_N"], 1)), flush=True)
                        _rz = lambda v, ref: v.reshape(-1, *([1] * (ref.dim() - 1))).to(ref.dtype)
                        _ri = lambda v, ref: v.reshape(-1, *([1] * (ref.dim() - 1))).float()
                        _nmx = getattr(args, "norm_mode", "01")

                        def _denorm_like(_y, _mu, _sd):
                            """Map a decoder output in std space back to [0,1]; 01 and pm1 are handled separately."""
                            if _nmx == "std":
                                return _y * _sd + _mu
                            if _nmx == "pm1":
                                return (_y + 1.0) / 2.0
                            return _y

                        # Full resolution with gradient checkpointing (activations recomputed in the backward pass). No cropping; the gradient is mathematically equivalent.
                        from torch.utils.checkpoint import checkpoint as _ckpt
                        _dec_ck = getattr(args, "arc_dec_ckpt", True)

                        def _decode_maybe_ckpt(_z):
                            if _dec_ck and torch.is_grad_enabled():
                                return _ckpt(_ae.decode, _z, use_reentrant=False)
                            return _ae.decode(_z)

                        if not globals().get("_DECCKPT_SHOWN"):
                            globals()["_DECCKPT_SHOWN"] = 1
                            print("[dec-ckpt] decode consistency: gradient checkpointing=%s | norm=%s | full resolution, no cropping | "
                                  "alpha[%.3f,%.3f] beta[%.3f,%.3f]"
                                  % (bool(_dec_ck), _nmx, float(_alpha_v.min()), float(_alpha_v.max()),
                                     float(_beta_v.min()), float(_beta_v.max())), flush=True)

                        _g1 = img1.float(); _g2 = img2.float(); _g3 = img3.float()
                        _m1 = _g1.mean(dim=(-1, -2, -3), keepdim=True)
                        _s1 = _g1.std(dim=(-1, -2, -3), keepdim=True).clamp_min(1e-5)
                        _m2 = _g2.mean(dim=(-1, -2, -3), keepdim=True)
                        _s2 = _g2.std(dim=(-1, -2, -3), keepdim=True).clamp_min(1e-5)
                        _m3 = _g3.mean(dim=(-1, -2, -3), keepdim=True)
                        _s3 = _g3.std(dim=(-1, -2, -3), keepdim=True).clamp_min(1e-5)

                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            dec_loss = images.new_zeros(())
                            if _wdec > 0:      # interpolation -> real v2, restored with a time-weighted mix of the v1/v3 statistics (v2 is not used)
                                _aZ = _rz(_alpha_v, zs1); _aI = _ri(_alpha_v, _g1)
                                pred_mid = _decode_maybe_ckpt((1.0 - _aZ) * zs1 + _aZ * zs3).float()
                                pred_mid = _denorm_like(pred_mid,
                                                        (1.0 - _aI) * _m1 + _aI * _m3,
                                                        (1.0 - _aI) * _s1 + _aI * _s3)
                                dec_loss = dec_loss + _wdec * l1_loss_fn(pred_mid, _g2)
                            if _wdec_ex > 0:   # extrapolation -> real v3, restored by extrapolating the v1/v2 statistics (v3 is not used)
                                _bZ = _rz(_beta_v, zs1); _bI = _ri(_beta_v, _g1)
                                pred_fut = _decode_maybe_ckpt(zs2 + _bZ * (zs2 - zs1)).float()
                                pred_fut = _denorm_like(pred_fut,
                                                        _m2 + _bI * (_m2 - _m1),
                                                        (_s2 + _bI * (_s2 - _s1)).clamp_min(1e-5))
                                # per-sample L1, then weighted by the validity mask so out-of-range beta samples are excluded
                                _ldiff = (pred_fut - _g3).abs().flatten(1).mean(1)      # [B]
                                _wv = _beta_ok.reshape(-1).to(_ldiff.dtype)
                                dec_loss = dec_loss + _wdec_ex * ((_ldiff * _wv).sum()
                                                                  / _wv.sum().clamp_min(1e-6))
                        dec_loss = dec_loss.to(orig_dtype)
                        loss_g = loss_g + dec_loss; traj_parts["dec"] = float(dec_loss.detach())

                    # ---- delta signal-to-noise: push the fitted intercept sigma toward 0 ----
                    # ||d||^2 = sigma^2 + (k*dt)^2, so at sigma=0 the rate r=||d||/dt is independent of dt.
                    # Short intervals are inflated by noise, giving r_short > r_long; this term penalizes the excess one-sidedly.
                    _snw = float(getattr(args, "ae_snr_w", 0.0))
                    if _snw > 0:
                        _s1, _s2, _s3 = z_mu.flatten(1).chunk(3, dim=0)
                        _st = patient_time.reshape(3, -1).float()
                        _dt12 = (_st[1] - _st[0]).abs()
                        _dt23 = (_st[2] - _st[1]).abs()
                        _dt13 = (_st[2] - _st[0]).abs()
                        _mdt = float(getattr(args, "ae_snr_mindt", 1.0))
                        _sok = ((_dt12 >= _mdt) & (_dt23 >= _mdt) & (_dt13 >= _mdt)).float()
                        _eps = 1e-6
                        _lr = lambda _d, _t: (torch.log(_d.norm(dim=1).clamp_min(_eps))
                                              - torch.log(_t.clamp_min(_eps)).to(_d.dtype))
                        _r12 = _lr(_s2 - _s1, _dt12)
                        _r23 = _lr(_s3 - _s2, _dt23)
                        _r13 = _lr(_s3 - _s1, _dt13)
                        _ex = (torch.relu(_r12 - _r13) ** 2 + torch.relu(_r23 - _r13) ** 2) / 2.0
                        _wv = _sok.to(_ex.dtype)
                        snr_loss = _snw * ((_ex * _wv).sum() / _wv.sum().clamp_min(1e-6))
                        loss_g = loss_g + snr_loss.to(orig_dtype)
                        traj_parts["snr"] = float(snr_loss.detach())
                        globals()["_SNR_N"] = globals().get("_SNR_N", 0) + 1
                        if globals()["_SNR_N"] % 50 == 1:
                            print("[ae_snr] w=%.3f | short/long rate excess (log) %.4f/%.4f | "
                                  "skipped %d/%d | L/L_rec=%.2f"
                                  % (_snw, float(torch.relu(_r12 - _r13).mean()),
                                     float(torch.relu(_r23 - _r13).mean()),
                                     int((1 - _wv).sum()), int(_wv.numel()),
                                     float(snr_loss.detach()) / max(float(rec_loss.detach()), 1e-8)),
                                  flush=True)

                    # ---- direction predictability: the direction of delta should be linearly predictable from z0 ----
                    # Constraining the two past segments to be parallel is one step removed from the downstream
                    # objective, and both segments are noise-dominated. This term instead requires delta to follow
                    # from the baseline, which matches the downstream ridge regression.
                    _pzw = float(getattr(args, "ae_predz_w", 0.0))
                    _subk = int(getattr(args, "traj_subspace_k", 0))
                    if _pzw > 0 and _subk > 0:
                        _p1, _p2, _p3 = z_mu.flatten(1).chunk(3, dim=0)
                        _dfull = (_p3 - _p1)
                        _sub_update(_p1, _dfull, _subk)
                        # --traj_center_pop: subtract the population mean drift first, otherwise uniform motion across all subjects scores perfectly
                        if int(getattr(args, "traj_center_pop", 0)) and _SUB.get("dmean") is not None:
                            _dfull = _dfull - _SUB["dmean"].to(_dfull.device).to(_dfull.dtype)
                        if _SUB["W"] is not None:
                            _zp = _p1 @ _SUB["Vz"]
                            _dp = _dfull @ _SUB["Vd"]
                            _hat = _zp @ _SUB["W"]
                            _c = torch.nn.functional.cosine_similarity(
                                _dp.float(), _hat.float(), dim=1)
                            pz_loss = _pzw * (1.0 - _c).mean()
                            loss_g = loss_g + pz_loss.to(orig_dtype)
                            traj_parts["pz"] = float(pz_loss.detach())
                            globals()["_PZ_N"] = globals().get("_PZ_N", 0) + 1
                            if globals()["_PZ_N"] % 100 == 1:
                                print("[ae_predz] w=%.3f K=%d | cos(δ_proj, W·z0_proj)=%.4f | "
                                      "L/L_rec=%.2f"
                                      % (_pzw, _subk, float(_c.mean()),
                                         float(pz_loss.detach()) / max(float(rec_loss.detach()), 1e-8)),
                                      flush=True)

                    # ---- latent-space extrapolation consistency (no decoder) ----
                    _wxl = float(getattr(args, "arc_extrap_lat_w", 0.0))
                    if _wxl > 0:
                        _q1, _q2, _q3 = z_mu.flatten(1).chunk(3, dim=0)
                        _pt = patient_time.reshape(3, -1).float()
                        _b_raw = (_pt[2] - _pt[1]) / (_pt[1] - _pt[0]).clamp_min(1e-6)
                        _ok = ((_b_raw >= 0.2) & (_b_raw <= 3.0)).float()     # skip out-of-range samples rather than clamping
                        _mdt2 = float(getattr(args, "traj_min_dt", 0.0))
                        if _mdt2 > 0.0:                                       # the same long-interval sampling gate
                            _ok = _ok * (((_pt[1] - _pt[0]).abs() >= _mdt2) &
                                         ((_pt[2] - _pt[1]).abs() >= _mdt2)).float()
                        _b = _b_raw.clamp(0.2, 3.0).reshape(-1, 1).to(_q1.dtype)
                        _zhat = _q2 + _b * (_q2 - _q1)
                        # Subspace: over the full latent dimensionality the extrapolation amplifies the noise in z2
                        #   by (1+beta), so rel (in [0,2], where 1.0 means orthogonal) stays near the noise level.
                        #   Projecting onto the top-K principal components first addresses this.
                        if int(getattr(args, "traj_subspace_k", 0)) > 0:
                            _sub_update(_q1, _q3 - _q1, int(args.traj_subspace_k))
                            if _SUB["Vd"] is not None:
                                _Vd = _SUB["Vd"]
                                _q1, _q2, _q3 = _q1 @ _Vd, _q2 @ _Vd, _q3 @ _Vd
                                _zhat = _q2 + _b * (_q2 - _q1)
                        # --traj_center_pop: subtracting the population mean from the displacement means a common drift no longer satisfies the collinearity constraint
                        if int(getattr(args, "traj_center_pop", 0)) and _SUB.get("dmean") is not None:
                            _dm2 = _SUB["dmean"].to(_q1.device).to(_q1.dtype)
                            if _dm2.shape[-1] == _q1.shape[-1]:
                                _q3 = _q3 - _dm2
                                _zhat = _zhat - _dm2
                        _num = ((_q3 - _zhat) ** 2).sum(1)
                        # Bounded form: the denominator |d|^2+|e|^2 gives L in [0,2] and remains strictly scale-invariant.
                        #   Dividing by |d|^2 alone is unbounded, letting large-beta samples dominate the gradient.
                        _dv = (_q3 - _q2)
                        _ev = _b * (_q2 - _q1)
                        _den = ((_dv ** 2).sum(1) + (_ev ** 2).sum(1)).clamp_min(1e-8)
                        _rel = _num / _den
                        _wv = _ok.to(_rel.dtype)
                        # Delta-t weighting: noise sigma is interval-independent while signal scales with delta t, so short-interval directions are almost entirely noise.
                        #   Normalized to mean 1, so it only redistributes weight and does not change the loss scale.
                        _dtp = float(getattr(args, "arc_extrap_dtw", 0.0))
                        if _dtp > 0.0:
                            _dt1 = (_pt[1] - _pt[0]).abs().clamp_min(1e-3).to(_rel.dtype)
                            _sw = _dt1 ** _dtp
                            _sw = _sw / _sw.mean().clamp_min(1e-8)
                            _wv = _wv * _sw
                        xl_loss = _wxl * ((_rel * _wv).sum() / _wv.sum().clamp_min(1e-6))
                        loss_g = loss_g + xl_loss.to(orig_dtype)
                        traj_parts["exl"] = float(xl_loss.detach())
                        globals()["_XL_N"] = globals().get("_XL_N", 0) + int(_wv.numel())
                        globals()["_XL_SK"] = globals().get("_XL_SK", 0) + int((1 - _wv).sum())
                        if globals()["_XL_N"] % 50 == 0:
                            _rr = float(xl_loss.detach()) / max(float(rec_loss.detach()), 1e-8)
                            _dn = (_q3 - _q2).norm(dim=1).mean()
                            _xs = _q2.norm(dim=1).mean()   # at B=1 roll returns the sample itself, so |z| is used as the scale instead
                            print("[extrap-lat] relative residual %.3f | L_extrap/L_rec = %.2f | "
                                  "|dz| %.3f |z| %.3f separability r %.5f | "
                                  "skipped so far %d/%d (%.1f%%)"
                                  % (float((_rel * _wv).sum() / _wv.sum().clamp_min(1e-6)), _rr,
                                     float(_dn), float(_xs), float(_dn / _xs.clamp_min(1e-8)),
                                     globals()["_XL_SK"], globals()["_XL_N"],
                                     100.0 * globals()["_XL_SK"] / max(globals()["_XL_N"], 1)), flush=True)

                    # CHANGE-FIDELITY — bind LATENT movement to REAL image change:
                    # decode(z1),decode(z3), denorm to [0,1], L1 their diff vs real v3-v1.
                    _wchg = getattr(args, "arc_change_w", 0.0)
                    if _wchg > 0:
                        zc1, _zc2, zc3 = z_mu.chunk(3, dim=0)
                        ic1, _ic2, ic3 = images.chunk(3, dim=0)
                        _nm = getattr(args, "norm_mode", "01")
                        _aec = accelerator.unwrap_model(autoencoder)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            r1 = _aec.decode(zc1).float(); r3 = _aec.decode(zc3).float()
                        ic1 = ic1.float(); ic3 = ic3.float()
                        if _nm == "std":
                            m1 = ic1.mean(dim=(-1, -2, -3), keepdim=True); s1 = ic1.std(dim=(-1, -2, -3), keepdim=True).clamp_min(1e-5)
                            m3 = ic3.mean(dim=(-1, -2, -3), keepdim=True); s3 = ic3.std(dim=(-1, -2, -3), keepdim=True).clamp_min(1e-5)
                            r1 = r1 * s1 + m1; r3 = r3 * s3 + m3
                        elif _nm == "pm1":
                            r1 = (r1 + 1) / 2; r3 = (r3 + 1) / 2
                        chg_loss = (_wchg * l1_loss_fn(r3 - r1, ic3 - ic1)).to(orig_dtype)
                        loss_g = loss_g + chg_loss; traj_parts["chg"] = float(chg_loss.detach())

                    # CHANGE-FIDELITY (CORRELATION) — maximize the brain-masked Pearson
                    # correlation between decoded change and real change.
                    _wcc = getattr(args, "arc_change_corr_w", 0.0)
                    if _wcc > 0:
                        zk1, _zk2, zk3 = z_mu.chunk(3, dim=0)
                        jc1, _jc2, jc3 = images.chunk(3, dim=0)
                        _nmc = getattr(args, "norm_mode", "01")
                        _aecc = accelerator.unwrap_model(autoencoder)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            q1 = _aecc.decode(zk1).float(); q3 = _aecc.decode(zk3).float()
                        jc1 = jc1.float(); jc3 = jc3.float()
                        if _nmc == "std":
                            a1 = jc1.mean(dim=(-1, -2, -3), keepdim=True); b1 = jc1.std(dim=(-1, -2, -3), keepdim=True).clamp_min(1e-5)
                            a3 = jc3.mean(dim=(-1, -2, -3), keepdim=True); b3 = jc3.std(dim=(-1, -2, -3), keepdim=True).clamp_min(1e-5)
                            q1 = q1 * b1 + a1; q3 = q3 * b3 + a3
                        elif _nmc == "pm1":
                            q1 = (q1 + 1) / 2; q3 = (q3 + 1) / 2
                        cp = (q3 - q1).flatten(1); cr = (jc3 - jc1).flatten(1)
                        bm = (((jc1 > 0.05) | (jc3 > 0.05)).flatten(1)).float()
                        nn = bm.sum(1, keepdim=True).clamp_min(1.0)
                        cp = (cp - (cp * bm).sum(1, keepdim=True) / nn) * bm
                        cr = (cr - (cr * bm).sum(1, keepdim=True) / nn) * bm
                        corr = (cp * cr).sum(1) / (cp.norm(dim=1) * cr.norm(dim=1)).clamp_min(1e-6)
                        cc_loss = (_wcc * (1.0 - corr).mean()).to(orig_dtype)
                        loss_g = loss_g + cc_loss; traj_parts["cc"] = float(cc_loss.detach())

                    # NORM-CORRECT decode-consistency: decode the interpolated midpoint,
                    # denormalize it (std mode: the time-weighted mix of the v1/v3 statistics)
                    # and L1 it against the real v2, so both sides are compared in [0,1].
                    _wdf = getattr(args, "arc_decode_fixed", 0.0)
                    if _wdf > 0:
                        df1, _df2, df3 = z_mu.chunk(3, dim=0)
                        gi1, gi2, gi3 = images.chunk(3, dim=0)
                        _nmd = getattr(args, "norm_mode", "01")
                        _aed = accelerator.unwrap_model(autoencoder)
                        # TIME-WEIGHTED interp: v2 sits at time-fraction a=(t1-t0)/(t2-t0),
                        # NOT at 0.5, because visit spacing is uneven.
                        if getattr(args, "arc_decode_tw", False):
                            _pt = patient_time.reshape(3, -1).float()
                            _a = ((_pt[1] - _pt[0]) / (_pt[2] - _pt[0]).clamp_min(1e-6)).clamp(0.02, 0.98)
                            _aZ = _a.reshape(-1, *([1] * (df1.dim() - 1))).to(df1.dtype)
                            _aI = _a.reshape(-1, *([1] * (gi1.dim() - 1))).float()
                            if not globals().get("_DECTW_SHOWN"):
                                globals()["_DECTW_SHOWN"] = 1
                                print("[arc_decode_tw] time-weighted interp ON alpha[min %.3f max %.3f mean %.3f]" % (float(_a.min()), float(_a.max()), float(_a.mean())), flush=True)
                        else:
                            _aZ = torch.tensor(0.5, device=df1.device, dtype=df1.dtype)
                            _aI = torch.tensor(0.5, device=gi1.device).float()
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            _zmid = (1.0 - _aZ) * df1 + _aZ * df3
                            if getattr(args, "arc_decode_sg", False): _zmid = _zmid.detach()
                            pmid = _aed.decode(_zmid).float()
                        gi1 = gi1.float(); gi2 = gi2.float(); gi3 = gi3.float()
                        if _nmd == "std":
                            mm = (1.0 - _aI) * gi1.mean(dim=(-1, -2, -3), keepdim=True) + _aI * gi3.mean(dim=(-1, -2, -3), keepdim=True)
                            sm = (1.0 - _aI) * gi1.std(dim=(-1, -2, -3), keepdim=True).clamp_min(1e-5) + _aI * gi3.std(dim=(-1, -2, -3), keepdim=True).clamp_min(1e-5)
                            pmid = pmid * sm + mm
                        elif _nmd == "pm1":
                            pmid = (pmid + 1) / 2
                        df_loss = (_wdf * l1_loss_fn(pmid, gi2)).to(orig_dtype)
                        loss_g = loss_g + df_loss; traj_parts["decf"] = float(df_loss.detach())

                    z_mu = z_mu.mean(1)

                    U, S, Vh = torch.linalg.svd(z_mu.to(torch.float32), full_matrices=False)
                    U = U.to(orig_dtype).contiguous()
                    S = S.to(orig_dtype).contiguous()
                    Vh = Vh.to(orig_dtype).contiguous()

                    U = U.view(U.shape[0], -1)  # Flatten U
                    S = S.view(S.shape[0], -1)  # Flatten S
                    Vh = Vh.view(Vh.shape[0], -1)  # Flatten Vh
                    U = F.normalize(U, dim=1)  # Normalize U
                    Vh = F.normalize(Vh, dim=1)  # Normalize Vh

                    U1, U2, U3 = U.chunk(3, dim=0)  # Split U into two halves
                    S1, S2, S3 = S.chunk(3, dim=0)  # Split S into two halves
                    Vh1, Vh2, Vh3 = Vh.chunk(3, dim=0)  # Split Vh into two halves

                    angle1 = U1  # torch.cat([U1, Vh1], dim=1)
                    angle2 = U2  # torch.cat([U2, Vh2], dim=1).clone().detach()
                    angle3 = U3  # torch.cat([U3, Vh3], dim=1)
                    angle_all = torch.cat([angle1, angle2, angle3], dim=0)

                    # Angular consistency: align each visit's direction to the reference
                    # visit (t2) with a STOP-GRADIENT on the reference, per the paper's ArcRank.
                    U2_ref = U2.detach()
                    if getattr(args, "arc_offline", False):
                        # SIGN-PINNED SVD. L_off is the rank-1 residual of the two step
                        # vectors about the chord u = (z3-z1)/||.|| -- i.e. 1 - line_r2 of
                        # the DELTAS -- written as an energy RATIO, whose gradients scale
                        # far better than a cosine's. SVD alone is sign-blind (zig-zag and
                        # collapse both minimize it), so L_sign pins each step to travel
                        # FORWARD along u.  L_off=0 with both signs positive <=> cos(d1,d2)=+1.
                        z1, z2, z3 = z_traj.float().chunk(3, dim=0)
                        d1, d2 = z2 - z1, z3 - z2
                        u = F.normalize(z3 - z1, dim=1)
                        p1, p2 = (d1 * u).sum(1), (d2 * u).sum(1)
                        e1, e2 = d1.pow(2).sum(1), d2.pow(2).sum(1)
                        off = (e1 - p1.pow(2) + e2 - p2.pow(2)) / (e1 + e2 + 1e-6)
                        sign = (F.relu(-p1 / e1.sqrt().clamp_min(1e-6)) +
                                F.relu(-p2 / e2.sqrt().clamp_min(1e-6)))
                        angle_loss = (off.mean() +
                                      getattr(args, "w_sign", 1.0) * sign.mean()).to(orig_dtype)
                    elif getattr(args, "arc_delta_angle", False):
                        # FIXED arcrank angle: align the trajectory DELTAS (straightness),
                        # not the spatial appearance modes. Pushes cos(d1,d2) -> +1.
                        # No SVD/sign issue and inherently within-patient.
                        z1, z2, z3 = z_traj.chunk(3, dim=0)
                        # -- arc_angle_span: use the full span z3-z1 instead of the adjacent segment z3-z2 --
                        #   Adjacent segments cover a shorter interval and share z2, whose noise enters the two
                        #   directions with opposite sign, biasing the cosine downwards; the full span covers a
                        #   longer interval and shares z1, so the shared noise has the same sign in both.
                        d1 = F.normalize((z2 - z1).float(), dim=1)
                        if int(getattr(args, "arc_angle_span", 0)):
                            d2 = F.normalize((z3 - z1).float(), dim=1)
                            if not globals().get("_SPAN_SHOWN"):
                                globals()["_SPAN_SHOWN"] = 1
                                print("[arc_angle_span] direction loss now uses the full span cos(z2-z1, z3-z1)", flush=True)
                        else:
                            d2 = F.normalize((z3 - z2).float(), dim=1)
                        _cosd = (d1 * d2).sum(1)
                        # Long-interval sampling gate (months): directions from short-interval samples are almost entirely noise
                        _mdt = float(getattr(args, "traj_min_dt", 0.0))
                        if _mdt > 0.0:
                            _ptA = patient_time.reshape(3, -1).float()
                            _g = (((_ptA[1] - _ptA[0]).abs() >= _mdt) &
                                  ((_ptA[2] - _ptA[1]).abs() >= _mdt)).to(_cosd.dtype)
                            angle_loss = (((1.0 - _cosd) * _g).sum() / _g.sum().clamp_min(1e-6)).to(orig_dtype)
                            if not globals().get("_MDT_SHOWN"):
                                globals()["_MDT_SHOWN"] = 1
                                print("[traj_min_dt] long-interval sampling gate active threshold=%.1f months | passed in this batch %d/%d"
                                      % (_mdt, int(_g.sum()), _g.numel()), flush=True)
                        else:
                            angle_loss = (1.0 - _cosd).mean().to(orig_dtype)
                    elif getattr(args, "angle_mode", "infonce") == "l1":
                        # paper's exact form: sum |U_i - U_ref| (L1 on normalized directions)
                        u1 = F.normalize(U1, dim=1); u3 = F.normalize(U3, dim=1)
                        u2 = F.normalize(U2_ref, dim=1)
                        angle_loss = 0.5 * ((u1 - u2).abs().sum(1).mean() +
                                            (u3 - u2).abs().sum(1).mean())
                    else:
                        angle_loss = 0.5 * (angle_loss_fn(U1, U2_ref) +
                                            angle_loss_fn(U3, U2_ref))   # CLIP InfoNCE

                    # ===== bounded progression axes (--ae_axes_k) =====
                    # The straightness terms above leave each patient's direction free and unshared across patients.
                    # This term adds two constraints: (1) the direction must lie in the subspace spanned by k shared
                    #              orthogonal modes, and (2) the coefficients within that subspace must be
                    #              predictable from the baseline z1.
                    # k trades coverage against noise: a rank-1 subspace is too restrictive, while a large k reaches
                    #              full-rank coverage with diminishing returns, so a small k (order 10) is a good start.
                    _ak = int(getattr(args, "ae_axes_k", 0))
                    if _ak > 0:
                        _zt1, _zt2, _zt3 = z_traj.float().chunk(3, dim=0)
                        _dl = torch.cat([_zt2 - _zt1, _zt3 - _zt2], 0).flatten(1)      # (2B, D)
                        _base = torch.cat([_zt1, _zt2], 0).flatten(1)                  # the corresponding baselines
                        _D = _dl.shape[1]
                        if "_AXES_E" not in globals():
                            # shared orthogonal modes E (k, D) and the coefficient head c(z0): both trained with the main model
                            globals()["_AXES_E"] = torch.nn.Parameter(
                                torch.randn(_ak, _D, device=_dl.device) / (_D ** 0.5))
                            globals()["_AXES_HEAD"] = torch.nn.Linear(_D, _ak).to(_dl.device)
                            _almr = float(getattr(args, "ae_axes_lr_mult", 100.0))
                            _allr = float(args.lr) * _almr
                            optimizer_g.add_param_group({"params": [globals()["_AXES_E"]], "lr": _allr})
                            optimizer_g.add_param_group({"params": list(globals()["_AXES_HEAD"].parameters()),
                                                         "lr": _allr})
                            print("[v2-axes] separate lr for E and the coefficient head = %.2e (= args.lr %.2e x %.0f); "
                                  "a critic sharing the generator learning rate may fail to learn"
                                  % (_allr, float(args.lr), _almr), flush=True)
                            print("[v2-axes] k=%d enabled: E %s plus coefficient head, added to optimizer_g"
                                  % (_ak, tuple(globals()["_AXES_E"].shape)), flush=True)
                        _E = torch.linalg.qr(globals()["_AXES_E"].t())[0].t()          # orthogonalized (k, D)
                        _coef = _dl @ _E.t()                                           # (2B, k) true coefficients
                        _proj = _coef @ _E
                        # (1) subspace concentration: a relative residual, scale-free, so it does not compete with reconstruction for amplitude
                        _sub_loss = ((_dl - _proj).pow(2).sum(1) / _dl.pow(2).sum(1).clamp_min(1e-8)).mean()
                        # (2) coefficient predictability: predict these k coefficients from the baseline (direction only, independent of amplitude)
                        _chat = globals()["_AXES_HEAD"](_base)
                        # no detach here: when ae_axes_w=0 this is the only gradient source for E
                        # batch mean removed: only the individual deviation is compared, which prevents E from turning toward a trivially predictable direction shared by everyone
                        # batch_size=1 correction: removing the mean within a batch degenerates (two rows from the same patient give row2 = -row1),
                        #   so a cross-step EMA population mean is used instead, detached from the gradient.
                        _mmt = float(getattr(args, "ae_coef_ema", 0.01))
                        _cmu = globals().get("_COEF_MU")
                        _hmu = globals().get("_CHAT_MU")
                        with torch.no_grad():
                            _cb = _coef.detach().mean(0)
                            _hb = _chat.detach().mean(0)
                            globals()["_COEF_MU"] = _cb.clone() if _cmu is None else (1 - _mmt) * _cmu + _mmt * _cb
                            globals()["_CHAT_MU"] = _hb.clone() if _hmu is None else (1 - _mmt) * _hmu + _mmt * _hb
                        globals()["_COEF_N"] = globals().get("_COEF_N", 0) + 1
                        # defined unconditionally because the print below uses it; the warmup only gates the loss
                        _cc = _coef - globals()["_COEF_MU"].unsqueeze(0)
                        _ch = _chat - globals()["_CHAT_MU"].unsqueeze(0)
                        if globals()["_COEF_N"] <= 100:
                            _coef_loss = _coef.sum() * 0.0        # during EMA warmup this term is inactive (a differentiable zero)
                        else:
                            _coef_loss = (1.0 - F.cosine_similarity(_ch, _cc, dim=1)).mean()
                        if globals()["_COEF_N"] % 100 == 0:
                            print("[v2-axes] step %d | subspace residual %.4f (w=%.3f, %.2fx rec) | "
                                  "both terms together account for %.0f%% of the total loss"
                                  % (globals()["_COEF_N"], float(_sub_loss),
                                     float(getattr(args, "ae_axes_w", 0.3)),
                                     float(getattr(args, "ae_axes_w", 0.3)) * float(_sub_loss)
                                     / max(float(rec_loss.detach()), 1e-8),
                                     100.0 * (float(getattr(args, "ae_axes_w", 0.3)) * float(_sub_loss)
                                              + float(getattr(args, "ae_coef_w", 0.2)) * float(_coef_loss))
                                     / max(float(loss_g.detach()), 1e-8)), flush=True)
                            _rr = float(getattr(args, "ae_coef_w", 0.2)) * float(_coef_loss) / \
                                  max(float(rec_loss.detach()), 1e-8)
                            print("[v2-coef] step %d | 1-cos=%.4f | ‖coef-mu‖=%.4f (‖coef‖=%.4f) | "
                                  "L_coef/L_rec=%.2f"
                                  % (globals()["_COEF_N"], float(_coef_loss),
                                     float((_coef - globals()["_COEF_MU"].unsqueeze(0)).norm(dim=1).mean()),
                                     float(_coef.norm(dim=1).mean()), _rr), flush=True)
                        if not globals().get("_COEF_SHOWN"):
                            globals()["_COEF_SHOWN"] = 1
                            print("[v2-coef] detach removed and mean subtracted | |delta|=%.3f coefficient std=%.4f loss=%.4f"
                                  % (_dl.norm(dim=1).mean().item(), _cc.std().item(),
                                     float(_coef_loss)), flush=True)
                        loss_g = loss_g + (float(getattr(args, "ae_axes_w", 0.3)) * _sub_loss
                                           + float(getattr(args, "ae_coef_w", 0.2)) * _coef_loss).to(orig_dtype)
                        if not globals().get("_AXES_SHOWN"):
                            globals()["_AXES_SHOWN"] = 1
                            print("[v2-axes] active | subspace residual %.4f (lower is more concentrated) | coefficient 1-cos %.4f | w=%.2f/%.2f"
                                  % (_sub_loss.item(), _coef_loss.item(),
                                     float(getattr(args, "ae_axes_w", 0.3)),
                                     float(getattr(args, "ae_coef_w", 0.2))), flush=True)

                    angle_weight = getattr(args, "angle_weight", 0.005)
                    rnc_weight   = getattr(args, "rnc_weight", 0.01)

                    angle_loss = angle_loss * angle_weight
                    margin_loss = torch.tensor(0)  # dist_cross_entropy(logits, patient_ids) * 0.001

                    rnc_loss = monotonicity_triplet_loss(
                        S, patient_time, patient_ids,
                        margin=getattr(args, "rank_margin", 0.1),
                        mode=getattr(args, "rank_mode", "magnitude"),
                    ) * rnc_weight  # RnC Loss (ordered by follow_up)

                    # Each term is independent: a weight of 0 adds nothing, rather than accumulating unconditionally, which would make the master switch implicitly enable angle and rnc.
                    if float(angle_weight) > 0:
                        loss_g = loss_g + angle_loss
                        traj_parts["ang"] = float(angle_loss.detach())
                    if float(rnc_weight) > 0:
                        # --rnc_autoscale R: rescale rnc to R x rec (a detached scalar, so the gradient direction is unchanged),
                        #   which prevents a fixed weight from growing more aggressive as rec falls over training.
                        _ras = float(getattr(args, "rnc_autoscale", 0.0) or 0.0)
                        if _ras > 0:
                            _rv = float(rnc_loss.detach().abs())
                            if _rv > 1e-12:
                                rnc_loss = rnc_loss * (_ras * float(rec_loss.detach()) / _rv)
                                globals()["_RAS_N"] = globals().get("_RAS_N", 0) + 1
                                if globals()["_RAS_N"] % 200 == 1:
                                    print("[rnc_autoscale] R=%.3f | before rescaling %.4f -> after %.4f "
                                          "(rec %.4f)" % (_ras, _rv,
                                                          float(rnc_loss.detach()),
                                                          float(rec_loss.detach())), flush=True)
                        loss_g = loss_g + rnc_loss
                        traj_parts["rnc"] = float(rnc_loss.detach())
                    if torch.is_tensor(margin_loss) and margin_loss.numel() and float(margin_loss) != 0.0:
                        loss_g = loss_g + margin_loss

                    # ---- per-patient trajectory tensor on the (sub)space: [B, 3, d] ----
                    zt1, zt2, zt3 = z_traj.float().chunk(3, dim=0)
                    Bp = zt1.shape[0]

                    # SCALE-FREE TIME-ORDERING — position along the chord is time-monotone
                    # (uses visit-ORDER, normalized by chord length -> no magnitude inflation).
                    _word = getattr(args, "arc_order_w", 0.0)
                    if _word > 0:
                        _chord = zt3 - zt1
                        _L = _chord.norm(dim=1, keepdim=True).clamp_min(1e-6)
                        _u = _chord / _L
                        _p1 = (zt1 * _u).sum(1); _p2 = (zt2 * _u).sum(1); _p3 = (zt3 * _u).sum(1)
                        _Lf = _L.squeeze(1)
                        _m = getattr(args, "order_margin", 0.2)
                        ord_loss = (F.relu(_m - (_p2 - _p1) / _Lf) + F.relu(_m - (_p3 - _p2) / _Lf)).mean()
                        ord_loss = (_word * ord_loss).to(orig_dtype)
                        loss_g = loss_g + ord_loss; traj_parts["ord"] = float(ord_loss.detach())

                    # CURVATURE: straightness = zero acceleration ||z1-2z2+z3||^2
                    # (scale-normalized by the step size so it doesn't fight magnitude/fidelity).
                    _wc = getattr(args, "arc_curv", 0.0)
                    if _wc > 0:
                        accel = (zt1 - 2 * zt2 + zt3)
                        denom = ((zt3 - zt1).pow(2).sum(1) + 1e-6)
                        curv = (accel.pow(2).sum(1) / denom).mean().to(orig_dtype) * _wc
                        loss_g = loss_g + curv; traj_parts["curv"] = float(curv.detach())

                    # SVD collinearity of the per-patient trajectory, applied on the
                    # SUBSPACE for stability: 1 - line_r2 = sigma2^2/(sigma1^2+sigma2^2).
                    _ws = getattr(args, "arc_svd", 0.0)
                    if _ws > 0:
                        Zc = torch.stack([zt1, zt2, zt3], 1)                  # [B,3,d]
                        Zc = Zc - Zc.mean(1, keepdim=True)
                        S = torch.linalg.svdvals(Zc)                         # [B,3]
                        s2 = S.pow(2); line_r2 = s2[:, 0] / s2.sum(1).clamp_min(1e-8)
                        svd_loss = (1.0 - line_r2).mean().to(orig_dtype) * _ws
                        loss_g = loss_g + svd_loss; traj_parts["svd"] = float(svd_loss.detach())

                    # diversity (anti-convergence) on the subspace direction z3-z1
                    if getattr(args, "arc_diversity", False):
                        vp = F.normalize(zt3 - zt1, dim=1)                    # [B, d] per-patient dir
                        if getattr(args, "arc_whiten", False):
                            # whitening: decorrelate directions (Barlow-style), off-diagonal of corr -> 0
                            C = (vp.t() @ vp) / Bp                            # [d,d] feature corr
                            off = (C - torch.diag(torch.diagonal(C))).pow(2).sum() / (vp.shape[1] ** 2)
                            div_loss = off.to(orig_dtype) * getattr(args, "w_div", 0.05)
                        elif getattr(args, "arc_div_hinge", 0.0) > 0:
                            # diversity HINGE — only penalize patient pairs whose directions are
                            # CLOSER than a margin, so well-separated patients aren't dragged and
                            # less straightness is spent on already-diverse pairs.
                            Cmat = (vp @ vp.t()).abs()
                            mrg = getattr(args, "arc_div_hinge", 0.0)
                            off = F.relu(Cmat - torch.eye(Bp, device=Cmat.device) - (1.0 - mrg))
                            div_loss = off.sum() / (Bp * (Bp - 1) + 1e-8)
                            div_loss = div_loss.to(orig_dtype) * getattr(args, "w_div", 0.05)
                        else:
                            Cmat = (vp @ vp.t()).abs()                       # |cos| pairwise
                            div_loss = (Cmat.sum() - Cmat.diagonal().sum()) / (Bp * (Bp - 1) + 1e-8)
                            div_loss = div_loss.to(orig_dtype) * getattr(args, "w_div", 0.05)
                        loss_g = loss_g + div_loss; traj_parts["div"] = float(div_loss.detach())

                    # DISEASE-BIOMARKER ANCHOR — pin the latent trajectory AXIS to real
                    # atrophy: the position along the chord u=(z3-z1) should track disease
                    # severity (ventricle↑ minus hippocampus↓), not merely visit order. This
                    # is the "order because the DISEASE advanced, not age" signal, supervised.
                    _wb = getattr(args, "arc_biomarker", 0.0)
                    if _wb > 0 and patient_biom is not None:
                        u = F.normalize(zt3 - zt1, dim=1)                         # [B,d] chord dir
                        proj = torch.stack([((zt - zt1) * u).sum(1) for zt in (zt1, zt2, zt3)], 1)  # [B,3]
                        hb = patient_biom[:, 0].chunk(3, 0); vb = patient_biom[:, 1].chunk(3, 0)
                        # per-visit severity, z-scored WITHIN each patient (head-size invariant)
                        sev = torch.stack([vb[i] - hb[i] for i in range(3)], 1)   # [B,3] (raw units differ)
                        sev = (sev - sev.mean(1, keepdim=True))
                        prj = (proj - proj.mean(1, keepdim=True))
                        num = (prj * sev).sum(1)
                        den = prj.norm(dim=1).clamp_min(1e-6) * sev.norm(dim=1).clamp_min(1e-6)
                        anchor = (1.0 - (num / den)).mean().to(orig_dtype) * _wb  # maximize corr
                        loss_g = loss_g + anchor; traj_parts["anc"] = float(anchor.detach())

                    # VELOCITY SMOOTHNESS — not constant speed (the objective is order-based), just
                    # penalize abrupt speed CHANGES: |‖Δz2‖−‖Δz1‖| / mean‖Δz‖. A smoother
                    # speed profile makes the Step-3 flow field easier to learn.
                    _wv = getattr(args, "arc_vsmooth", 0.0)
                    if _wv > 0:
                        n1 = (zt2 - zt1).pow(2).sum(1).clamp_min(1e-8).sqrt()
                        n2 = (zt3 - zt2).pow(2).sum(1).clamp_min(1e-8).sqrt()
                        vs = ((n2 - n1).abs() / (0.5 * (n1 + n2) + 1e-6)).mean()
                        vs = vs.to(orig_dtype) * _wv
                        loss_g = loss_g + vs; traj_parts["vsm"] = float(vs.detach())

                    # Δz ISOTROPY PRIOR — push the batch of per-patient steps toward an
                    # isotropic covariance (Barlow-style off-diagonal decorrelation of the
                    # RAW steps, not the normalized directions). A near-white Δz distribution
                    # is what flow-matching marginals prefer. Distinct from `--arc_whiten`,
                    # which whitens direction diversity across patients.
                    _wi = getattr(args, "arc_isotropy", 0.0)
                    if _wi > 0:
                        dz = torch.cat([zt2 - zt1, zt3 - zt2], 0)                 # [2B, d]
                        dz = dz - dz.mean(0, keepdim=True)
                        dz = F.normalize(dz, dim=0)                               # unit per-feature
                        Cc = dz.t() @ dz                                          # [d,d] corr
                        iso = (Cc - torch.eye(Cc.shape[0], device=Cc.device)).pow(2).sum() / (Cc.shape[0] ** 2)
                        iso = iso.to(orig_dtype) * _wi
                        loss_g = loss_g + iso; traj_parts["iso"] = float(iso.detach())

                    # identity invariance. The free (non-progression) channels must be
                    # CONSTANT across a patient's visits, so the FULL latent moves only inside
                    # the progression subspace -> the full-latent trajectory is straight by
                    # construction, while the free channels still carry anatomy for the decoder.
                    # It is the MOTION that must vanish, not the energy: measure the fraction of
                    # the per-patient step that leaves the progression subspace. Bounded in [0,1]
                    # (no blow-up when the progression step is small) and scale-invariant.
                    if getattr(args, "arc_id_invariance", False) and z_id is not None:
                        i1, i2, i3 = z_id.float().chunk(3, dim=0)
                        d_id = (i2 - i1).pow(2).sum(1) + (i3 - i2).pow(2).sum(1)
                        d_pr = (zt2 - zt1).pow(2).sum(1) + (zt3 - zt2).pow(2).sum(1)
                        frac = d_id / (d_id + d_pr + 1e-6)
                        id_loss = frac.mean().to(orig_dtype) * getattr(args, "w_id", 0.05)
                        loss_g = loss_g + id_loss; traj_parts["id"] = float(id_loss.detach())

            accelerator.backward(loss_g)
            optimizer_g.step()
            optimizer_g.zero_grad()

            _tj = " ".join(f"{k}:{v:.3f}" for k, v in traj_parts.items())
            progress_bar.set_description(
                f"Epoch {epoch} | loss: {loss_g.item():.4f} | rec: {rec_loss.item():.4f} | per: {per_loss.item():.4f} | angle: {angle_loss.item():.4f} | rnc: {rnc_loss.item():.4f} | traj[{_tj}]"
            )

            if use_contrastive:
                del angle_loss, margin_loss, rnc_loss, U, S, Vh, U1, U2, U3, S1, S2, S3, Vh1, Vh2, Vh3

            del z_mu, z_sigma

            gc.collect()
            torch.cuda.empty_cache()

            with accelerator.autocast():
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                d_loss_fake = adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
                loss_d = eff_adv *  d_loss_fake * 0.5

            accelerator.backward(loss_d)
            optimizer_d.step()
            optimizer_d.zero_grad()

            with accelerator.autocast():
                logits_real = discriminator(images.contiguous().detach())[-1]
                d_loss_real = adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (d_loss_real) * 0.5
                loss_d = eff_adv * discriminator_loss

            accelerator.backward(loss_d)
            optimizer_d.step()
            optimizer_d.zero_grad()


            avgloss.put('Generator/reconstruction_loss', rec_loss.item())
            avgloss.put('Generator/perceptual_loss', per_loss.item())
            avgloss.put('Generator/adverarial_loss', gen_loss.item())
            avgloss.put('Generator/kl_regularization', kld_loss.item())
            avgloss.put('Discriminator/adverarial_loss', loss_d.item())

            total_counter += 1


            del rec_loss, kld_loss, per_loss, gen_loss, loss_g, logits_fake
            del logits_real, d_loss_fake, d_loss_real, loss_d, discriminator_loss
            del reconstruction, images, followup, followup2, patient_ids, patient_age

            gc.collect()
            torch.cuda.empty_cache()

            # ---- step-based fast eval (streams METRICS_JSON mid-epoch) ----
            if getattr(args, "eval_steps", 0) and total_counter % args.eval_steps == 0:
                _p, _s = validate_model(accelerator.unwrap_model(autoencoder), test_loader, DEVICE, save_root=None,
                                        max_batches=4, step_name=f"step{total_counter}")
                _lin = {}
                if linear_loader is not None:
                    try:
                        _lin = evaluate_linearity(accelerator.unwrap_model(autoencoder), linear_loader, DEVICE,
                                                  max_batches=(getattr(args, "lin_max_batches", 0) or None), norm_mode=getattr(args,"norm_mode","01"))
                        # subspace design: also measure the PROGRESSION SUBSPACE (where the loss acts)
                        _kc = getattr(args, "arc_channels", 0)
                        if _kc > 0:
                            _ls = evaluate_linearity(accelerator.unwrap_model(autoencoder), linear_loader, DEVICE,
                                                     max_batches=(getattr(args, "lin_max_batches", 0) or None),
                                                     channels=_kc, norm_mode=getattr(args,"norm_mode","01"))
                            _lin.update({f"sub_{k}": v for k, v in _ls.items() if k != "n_patients"})
                    except Exception as e:
                        print(f"[linearity] step probe failed: {e}")
                import json as _json
                _r = {"step": total_counter, "epoch": epoch, "psnr": round(_p, 3), "ssim": round(_s, 4)}
                _r.update({k: round(v, 4) for k, v in _lin.items() if k != "n_patients"})
                if "n_patients" in _lin: _r["n_patients"] = _lin["n_patients"]
                if accelerator.is_main_process: print("METRICS_JSON " + _json.dumps(_r), flush=True)
                autoencoder.train()

        image_results = f"{image_root}/epoch_{epoch}"
        os.makedirs(image_results, exist_ok=True)

        val_psnr, val_ssim = validate_model(accelerator.unwrap_model(autoencoder), test_loader, DEVICE,
                       save_root=(image_results if accelerator.is_main_process else None), max_batches=6,
                       step_name="Epoch_{}".format(epoch))

        # ---- latent-trajectory linearity probe + single-line metrics record ----
        lin = {}
        if linear_loader is not None and (epoch % max(1, getattr(args, "eval_every", 2)) == 0):
            try:
                lin = evaluate_linearity(accelerator.unwrap_model(autoencoder), linear_loader, DEVICE,
                                         max_batches=(getattr(args, "lin_max_batches", 0) or None), norm_mode=getattr(args,"norm_mode","01"))
                _kc = getattr(args, "arc_channels", 0)
                if _kc > 0:
                    _ls = evaluate_linearity(accelerator.unwrap_model(autoencoder), linear_loader, DEVICE,
                                             max_batches=(getattr(args, "lin_max_batches", 0) or None),
                                             channels=_kc, norm_mode=getattr(args,"norm_mode","01"))
                    print("[linearity][subspace]", format_summary(_ls))
                    lin.update({f"sub_{k}": v for k, v in _ls.items() if k != "n_patients"})
                print("[linearity]", format_summary(lin))
                for k, v in lin.items():
                    writer.add_scalar(f"Linearity/{k}", v, epoch)
            except Exception as e:
                print(f"[linearity] probe failed this epoch: {e}")
        writer.add_scalar("Val/psnr", val_psnr, epoch)
        writer.add_scalar("Val/ssim", val_ssim, epoch)
        import json as _json
        _rec = {"epoch": epoch, "psnr": round(val_psnr, 3), "ssim": round(val_ssim, 4)}
        _rec.update({k: round(v, 4) for k, v in lin.items() if k != "n_patients"})
        if "n_patients" in lin: _rec["n_patients"] = lin["n_patients"]
        if accelerator.is_main_process: print("METRICS_JSON " + _json.dumps(_rec), flush=True)

        # save the model
        if (epoch + 1) % save_epoch == 0 and accelerator.is_main_process:

            torch.save(accelerator.unwrap_model(discriminator).state_dict(), os.path.join(args.output_dir, f'{args.task}-dis-{epoch + 1}-{args.dim}D.pth'))

            ae_path = os.path.join(args.output_dir, f'{args.task}-ae-{epoch + 1}-{args.dim}D.pth')
            torch.save(accelerator.unwrap_model(autoencoder).state_dict(), ae_path)

            print("Saving models to: ", ae_path)

        gc.collect()
        torch.cuda.empty_cache()

    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(autoencoder).state_dict(),
                   os.path.join(args.output_dir, f'{args.task}-ae-{args.dim}D-final.pth'))

    print("Training finished.")


