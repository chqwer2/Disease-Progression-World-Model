"""Eval the direct residual regressor on the shared IMAGE-space protocol
(VAL split, fixed-seed subsample, brain mask, copy-input baseline), scored on the AE's
NATIVE grid (const.INPUT_SHAPE_AE = 120x144x120) where decode(latent) actually
lives. Scoring on 128^3 (interpolating the 120^3 decode) misaligns the physical
FOV and depresses PSNR well below the autoencoder's own reconstruction quality,
so score on the native grid.

prediction = decode( start_latent + net([start_latent, Δt]) ).

Usage:
  MC_LATENT_SUFFIX=_new_latent.npz python resreg_eval.py \
    --ckpt .../resreg-ep-N.pth --aekl_ckpt models/autoencoder_maisi_new.pt --num 16 --gpu 1
"""
import os, sys, argparse, numpy as np, pandas as pd, torch, torch.nn.functional as F
ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--aekl_ckpt', default='models/autoencoder_maisi_new.pt')
ap.add_argument('--num', type=int, default=16)
ap.add_argument('--gpu', default='1')
a = ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# External split/metric helpers (final_loader.py, eval_common.py). Point
# MC_EVAL_DATA at the directory holding them.
sys.path.insert(0, os.environ.get("MC_EVAL_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_final_data")))
from monai.networks.nets import DiffusionModelUNet
from monai import transforms
from mambacontrol.autoencoder.MAISI_Unet3D import init_autoencoder
from mambacontrol import const
from final_loader import load_for_method
from eval_common import psnr as ec_psnr, _ssim
DEV = 'cuda'
ROOT = os.environ.get("AD_PROGRESSION_ROOT",
                      "data")
SUF = os.environ.get('MC_LATENT_SUFFIX', '_new_latent.npz')
# Score on the AE's NATIVE 120x144x120 FOV — the region the latent pipeline
# actually covers. The outer 128^3 border is OUTSIDE the AE's FOV, so a latent
# method genuinely can't predict it; scoring there just penalises FOV, not skill.
# Voxel-space methods score the same on either grid, so this is a fairness fix
# rather than a crop that favours one method.
GRID = tuple(const.INPUT_SHAPE_AE)   # (120,144,120)

ck = torch.load(a.ckpt, map_location='cpu'); sf = ck['scale_factor']
def _build_cond():
    return DiffusionModelUNet(spatial_dims=3, in_channels=5, out_channels=4, num_res_blocks=2,
                              channels=(128, 256, 256), attention_levels=(False, False, True),
                              num_head_channels=(0, 0, 256), transformer_num_layers=1,
                              with_conditioning=True, cross_attention_dim=8, upcast_attention=False,
                              norm_num_groups=32, norm_eps=1e-6, resblock_updown=True)
def _build_plain():
    return DiffusionModelUNet(spatial_dims=3, in_channels=5, out_channels=4, num_res_blocks=2,
                              channels=(128, 256, 256), attention_levels=(False, False, False),
                              norm_num_groups=32, norm_eps=1e-6, resblock_updown=True)
net = None; _COND = False
for _cond, _mk in ((True, _build_cond), (False, _build_plain)):
    try:
        _n = _mk(); _n.load_state_dict(ck['net']); net = _n; _COND = _cond; break
    except Exception:
        pass
assert net is not None, "checkpoint matches neither conditioned nor plain architecture"
print(f"model: {'conditioned' if _COND else 'plain'}", flush=True)
net = net.to(DEV).eval()
ae = init_autoencoder(a.aekl_ckpt).to(DEV).float().eval()
for m in ae.modules():
    if hasattr(m, 'norm_float16'): m.norm_float16 = False

_tf = transforms.Compose([transforms.LoadImage(image_only=True), transforms.EnsureChannelFirst(),
    transforms.Spacing(pixdim=const.RESOLUTION), transforms.ResizeWithPadOrCrop(GRID, mode='minimum'),
    transforms.ScaleIntensity(minv=0, maxv=1)])
def raw(p):
    p = p if os.path.isabs(str(p)) else os.path.join(ROOT, str(p)); return _tf(p).float()[0].numpy()
def latent(p):
    p = p if os.path.isabs(str(p)) else os.path.join(ROOT, str(p))
    return torch.from_numpy(np.load(p.replace('.nii.gz', SUF).replace('.nii', SUF))['data']).float()
def pad4(z):
    pads = []
    for d in reversed(z.shape[1:]): pads += [0, (-d) % 4]
    return F.pad(z, pads)
def dec(z): return ae.decode_stage_2_outputs(z.float())[0, 0].clamp(0, 1).cpu().numpy()

df = load_for_method(os.path.join(ROOT, 'AD-Progression-All.csv'), pair=True)
df = df[df.split == 'val'].reset_index(drop=True)
if 'sex' not in df.columns: df['sex'] = 0.5
_COV = ['followup_age', 'sex', 'starting_diagnosis', 'starting_cerebral_cortex',
        'starting_hippocampus', 'starting_amygdala', 'starting_cerebral_white_matter',
        'starting_lateral_ventricle']
for c in _COV:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0).astype('float32')
idx = np.random.RandomState(42).choice(len(df), a.num, replace=False)
rows = []
with torch.no_grad():
    for i in idx:
        r = df.iloc[int(i)]
        z0 = latent(r.starting_image_path)                                    # (4,30,36,30) native
        sL = pad4(z0).unsqueeze(0).to(DEV)
        dt = (float(r.followup_follow_up) - float(r.starting_follow_up)) / 12.0
        sz = sL * sf
        dtp = torch.tensor([dt]).view(1, 1, 1, 1, 1).expand(1, 1, *sz.shape[-3:]).to(DEV)
        _x = torch.cat([sz, dtp], 1).float(); _ts = torch.zeros(1, device=DEV).long()
        if _COND:
            ctx = torch.tensor([[float(r[c]) for c in _COV]], device=DEV).unsqueeze(0).float()  # (1,1,8)
            res = net(x=_x, timesteps=_ts, context=ctx)
        else:
            res = net(x=_x, timesteps=_ts)
        pred_fz = (sz + res)[:, :, :z0.shape[1], :z0.shape[2], :z0.shape[3]]   # UNPAD to native
        x = raw(r.starting_image_path); g = raw(r.followup_image_path)         # raw on native 120^3 grid
        p = dec(pred_fz / sf)                                                  # decode native 120^3 (no pad)
        m = (x > 0.05) | (g > 0.05)
        pt, xt, gt_, mt = torch.from_numpy(p), torch.from_numpy(x), torch.from_numpy(g), torch.from_numpy(m)
        rows.append({
            'pred_psnr': float(ec_psnr(pt, gt_, mt)), 'copy_psnr': float(ec_psnr(xt, gt_, mt)),
            'ssim': float(_ssim(pt, gt_)),
        })
        print(f"  [{len(rows)}/{a.num}] PSNR {rows[-1]['pred_psnr']:.2f} "
              f"(copy {rows[-1]['copy_psnr']:.2f}) SSIM {rows[-1]['ssim']:.4f}", flush=True)
mm = lambda k: float(np.mean([r[k] for r in rows]))
d = np.array([r['pred_psnr'] - r['copy_psnr'] for r in rows])
print("=" * 78)
print(f"MambaControl residual regression — IMAGE space (grid {GRID}, n={len(rows)}, seed-42, brain mask)")
print(f"  PSNR {mm('pred_psnr'):.3f}  copy {mm('copy_psnr'):.3f}  DELTA {d.mean():+.3f}  SSIM {mm('ssim'):.4f}")
