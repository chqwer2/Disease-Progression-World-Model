import torch
import torch.nn.functional as F
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import pytorch_msssim

# pip install pytorch-msssim



# Initialize LPIPS model (use 'alex' or 'vgg')
lpips_model = lpips.LPIPS(net='alex')


def psnr_3d(pred, target):
    """
    pred, target: torch.Tensor of shape (1, 1, D, H, W)
    """
    pred   = np.expand_dims(pred, axis=(0,1))   # add new axis at position 0
    target = np.expand_dims(target, axis=(0,1)) # same for target
    # B, C, H, D

  
    # PSNR (scikit-image, over 3D volume)
    psnr_value =  peak_signal_noise_ratio(target, pred, data_range=1.0)
    psnr_value = np.clip(psnr_value, 0, 50)
    
    return psnr_value

def psnr_2d(pred, target):
    """
    pred, target: torch.Tensor of shape (1, 1, H, W)
    """
    # pred = pred.squeeze() #.cpu().numpy()
    # target = target.squeeze()#.cpu().numpy()
    pred   = np.expand_dims(pred, axis=(0,1))   # add new axis at position 0
    target = np.expand_dims(target, axis=(0,1)) # same for target

    # print("pred shape = ", pred.shape, target.shape)
    # PSNR (scikit-image, over 2D image)
    psnr_value = peak_signal_noise_ratio(target, pred, data_range=1.0)

    return psnr_value


def ensure_5d(x):
    t = x
    # Keep adding singleton dimensions at the front until we have 5D
    while t.ndim < 5:
        t = t.unsqueeze(0)
    return t


def ssim_3d(pred, target):
    """
    pred, target: torch.Tensor of shape (1, 1, D, H, W)
    """
    # pred   = pred.squeeze()   #.cpu().numpy()
    # target = target.squeeze() #.cpu().numpy()
    pred   = ensure_5d(torch.from_numpy(pred).float())
    target = ensure_5d(torch.from_numpy(target).float())

    # SSIM (scikit-image, for 3D: need to loop over slices)
    # D, H, W
    ssim_total = 0
    count = 0

    ssim_value   = pytorch_msssim.ssim(pred, target, data_range=1.0)
    # ssim_value = pytorch_msssim.ms_ssim(pred, target, data_range=1.0, win_size=7)


    # print(ssim_val, ssim_value)

    # for i in range(pred.shape[0]):
    #     ssim_slice = structural_similarity(
    #         target[i], pred[i], data_range=1.0
    #     )
    #     ssim_total += ssim_slice
    #     count += 1
    # ssim_value = ssim_total / count


    return ssim_value



def compute_3dmetrics(pred, target):
    """
    pred, target: torch.Tensor of shape (1, 1, D, H, W)
    """
    pred = pred.squeeze() #.cpu().numpy()
    target = target.squeeze()#.cpu().numpy()

    # PSNR (scikit-image, over 3D volume)
    psnr_value = peak_signal_noise_ratio(target, pred, data_range=1.0)

    # SSIM (scikit-image, for 3D: need to loop over slices)
    ssim_total = 0
    count = 0
    for i in range(pred.shape[0]):
        ssim_slice = structural_similarity(
            target[i], pred[i], data_range=1.0
        )
        ssim_total += ssim_slice
        count += 1
    ssim_value = ssim_total / count

    # LPIPS (expects 2D RGB images, so we can take central slices or mean across slices)
    # Take middle slice along z-axis and replicate to 3 channels
    mid_slice_pred   = torch.tensor(pred[pred.shape[0] // 2]).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    mid_slice_target = torch.tensor(target[target.shape[0] // 2]).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    lpips_value = lpips_model(mid_slice_pred, mid_slice_target).item()

    return {
        'PSNR': psnr_value,
        'SSIM': ssim_value,
        'LPIPS': lpips_value
    }


def compute_dice(pred_binary, target_binary, eps=1e-6):
    intersection = (pred_binary & target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


import numpy as np


def _as_np(x):
    """Accept torch tensors or numpy arrays; return a squeezed float32 numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).squeeze()


def _detrend(field, brain, sigma, eps=1e-6):
    """
    Remove low-frequency intensity drift from a change field so the metric reacts
    to the SPATIAL pattern of progression, not to intensity inhomogeneity.
      sigma is None  -> no detrending
      sigma is inf   -> global demean (removes a UNIFORM intensity shift only)
      finite sigma   -> subtract a mask-aware Gaussian local mean (removes a uniform
                        shift AND a smooth NON-UNIFORM bias field, while preserving
                        focal disease change, which is higher-frequency).
    """
    if sigma is None:
        return field
    if not np.isfinite(sigma):
        return field - field[brain].mean()
    from scipy.ndimage import gaussian_filter
    bf = brain.astype(np.float32)
    num = gaussian_filter(field * bf, sigma)     # normalized (Nadaraya-Watson) local
    den = gaussian_filter(bf, sigma)             # mean so mask edges don't bias it
    return field - num / (den + eps)


def compute_change_metrics(pred, target, input, mask=None, detrend_sigma=16.0,
                           tau=None, tau_floor=0.02, tau_rel=0.75, eps=1e-6):
    """
    Robust disease-progression change metrics.

    pred, target, input : volumes in [0, 1] (torch or numpy, any broadcastable shape).
    mask                : brain mask (>0 inside brain). STRONGLY recommended --
                          without it, background / skull / registration edge
                          artifacts leak into the change map. If None, falls back
                          to the foreground of either scan.
    detrend_sigma       : Gaussian sigma (voxels) for high-pass detrending of each
                          change field before scoring. Makes CHANGE_MAE/CHANGE_DICE
                          robust to intensity inhomogeneity -- both a UNIFORM shift
                          (scanner drift / overall brightness) and a smooth NON-UNIFORM
                          bias field -- so they measure the SPATIAL pattern of change.
                          Default 16.0. Use inf for global demean (uniform only), or
                          None to disable detrending.
    tau                 : change threshold. If None it is adaptive and data-relative
                          (tau = max(tau_floor, tau_rel-quantile of |target-input|
                          inside the brain)), so it is not pinned to a magic constant.

    Returns a dict with:
      CHANGE_MAE  : mean |signed_gt_change - signed_pred_change| over the true-change
                    region (bounded ~[0,1], DIRECTIONAL, focused where change happens).
      CHANGE_DICE : direction-aware Dice -- a voxel only agrees if it changes in the
                    SAME direction (atrophy vs growth). A wrong-direction prediction
                    scores ~0 instead of a spurious 1.0.
      CHANGE_PCC  : Pearson correlation of the signed change fields (threshold-free
                    measure of whether the pattern & direction of progression match).
    """
    pred, target, input = _as_np(pred), _as_np(target), _as_np(input)

    if mask is not None:
        brain = _as_np(mask) > 0.5
    else:
        brain = (input > 0.05) | (target > 0.05)
    if brain.sum() == 0:
        brain = np.ones_like(input, dtype=bool)

    gt_full = _detrend(target - input, brain, detrend_sigma)   # signed GT change
    pr_full = _detrend(pred   - input, brain, detrend_sigma)   # signed predicted change
    gt = gt_full[brain]                   # brain-only, intensity-detrended
    pr = pr_full[brain]
    agt = np.abs(gt)

    if tau is None:
        tau = max(tau_floor, float(np.quantile(agt, tau_rel)))

    gt_up, gt_dn = (gt > tau), (gt < -tau)
    pr_up, pr_dn = (pr > tau), (pr < -tau)
    gt_ch = gt_up | gt_dn
    n_change = int(gt_ch.sum())

    # CHANGE_MAE: signed error on the true-change region (guarded against div0).
    if n_change > 0:
        change_mae = float(np.abs(gt[gt_ch] - pr[gt_ch]).mean())
    else:
        change_mae = float(np.abs(gt - pr).mean())

    # CHANGE_DICE: direction-aware, prevalence-weighted across the two directions.
    def _dice(a, b):
        s = a.sum() + b.sum()
        if s == 0:
            return 1.0                    # both agree there is no change here
        return float((2.0 * np.logical_and(a, b).sum()) / (s + eps))
    w_up, w_dn = gt_up.sum(), gt_dn.sum()
    if (w_up + w_dn) > 0:
        change_dice = float((w_up * _dice(pr_up, gt_up) + w_dn * _dice(pr_dn, gt_dn))
                            / (w_up + w_dn))
    else:
        change_dice = 1.0

    # CHANGE_PCC: threshold-free directional pattern agreement.
    active = gt_ch | pr_up | pr_dn
    if active.sum() > 1 and gt[active].std() > eps and pr[active].std() > eps:
        change_pcc = float(np.corrcoef(gt[active], pr[active])[0, 1])
    elif active.sum() == 0:
        change_pcc = 1.0
    else:
        change_pcc = 0.0

    return {'CHANGE_MAE': change_mae, 'CHANGE_DICE': change_dice, 'CHANGE_PCC': change_pcc}


def compute_3dmetrics(pred, target, input, mask=None,
                      change_input=None, change_target=None):
    """
    pred, target, input: torch.Tensor / numpy of shape (1, 1, D, H, W) in [0, 1].
    mask: optional brain mask for the change metrics (see compute_change_metrics).

    change_input / change_target: OPTIONAL references for the CHANGE metrics only.
        For AE / latent generative models, `pred` carries the autoencoder's
        reconstruction error, which -- if compared against the RAW baseline/target --
        dominates the change map and washes out progression differences (every method
        collapses to a similar score). Pass the AE-RECONSTRUCTED baseline and followup
        here (e.g. input_ae = ae(input), target_ae = ae(target)) so the reconstruction
        error is common-mode and cancels, and CHANGE_* measure progression, not AE
        fidelity. PSNR/SSIM/LPIPS still use the RAW target (AE fidelity is legitimately
        part of image quality). Defaults reproduce the raw-domain behavior.
    """
    input = _as_np(input)
    pred = _as_np(pred)
    target = _as_np(target)

    # PSNR (scikit-image, over 3D volume) -- vs the raw target (includes AE fidelity)
    psnr_value = peak_signal_noise_ratio(target, pred, data_range=1.0)

    # SSIM (scikit-image, for 3D: need to loop over slices)
    ssim_total = 0
    count = 0
    for i in range(pred.shape[0]):
        ssim_slice = structural_similarity(
            target[i], pred[i], data_range=1.0
        )
        ssim_total += ssim_slice
        count += 1
    ssim_value = ssim_total / count

    # LPIPS (expects 2D RGB images, so we can take central slices or mean across slices)
    # Take middle slice along z-axis and replicate to 3 channels
    mid_slice_pred   = torch.tensor(pred[pred.shape[0] // 2]).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    mid_slice_target = torch.tensor(target[target.shape[0] // 2]).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    lpips_value = lpips_model(mid_slice_pred, mid_slice_target).item()

    # CHANGE metrics -- optionally in the AE-consistent domain so AE error cancels.
    ci = input  if change_input  is None else _as_np(change_input)
    ct = target if change_target is None else _as_np(change_target)
    change = compute_change_metrics(pred, ct, ci, mask=mask)

    return {
        'PSNR': psnr_value,
        'SSIM': ssim_value,
        'LPIPS': lpips_value,
        'CHANGE_MAE': change['CHANGE_MAE'],
        'CHANGE_DICE': change['CHANGE_DICE'],
        'CHANGE_PCC': change['CHANGE_PCC'],
    }



def compute_2dmetrics(pred, target, lpips_model):
    """
    pred, target: torch.Tensor of shape (1, 1, H, W)
    lpips_model: a preloaded LPIPS model (expects input in shape [N, 3, H, W])
    """
    pred = pred.squeeze()  # shape (H, W)
    target = target.squeeze()  # shape (H, W)
    pred = pred.clip(0, 1)

    # PSNR (scikit-image, over 2D image)
    psnr_value = peak_signal_noise_ratio(target, pred, data_range=1.0)

    # SSIM (scikit-image, over 2D image)
    ssim_value = structural_similarity(target, pred, data_range=1.0)

    # LPIPS (expects 2D RGB images, so we replicate to 3 channels)
    pred_rgb = pred.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)    # shape (1, 3, H, W)
    target_rgb = target.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # shape (1, 3, H, W)
    lpips_value = lpips_model(pred_rgb, target_rgb).item()

    return {
        'PSNR': psnr_value,
        'SSIM': ssim_value,
        'LPIPS': lpips_value
    }