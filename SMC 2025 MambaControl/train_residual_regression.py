"""Direct residual regression baseline for MambaControl.

Instead of diffusion + SDEdit, which generates a plausible follow-up latent rather than
an accurate per-voxel change, train a UNet to DIRECTLY predict the latent residual:
    [start_latent (4ch) + delta_t plane (1ch)]  ->  (followup_latent - start_latent) (4ch)
with an L1 loss. This optimises the CHANGE directly, rather than
generating a plausible latent. Same data pipeline / covariate context as
train_control_mamba.py.

Usage:
  MC_LATENT_SUFFIX=_new_latent.npz python train_residual_regression.py \
    --dataset_csv .../AD-Progression-All.csv --cache_dir ... --output_dir ... \
    --aekl_ckpt models/autoencoder_maisi_new.pt --n_epochs 60 --batch_size 8 --gpus 0
"""
import os, argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_csv', required=True)
parser.add_argument('--cache_dir', required=True)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--aekl_ckpt', required=True)
parser.add_argument('--n_epochs', default=60, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--num_workers', default=3, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--gpus', default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

import sys, torch, numpy as np, pandas as pd
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from monai import transforms
from monai.data.image_reader import NumpyReader
from monai.utils import set_determinism
from monai.networks.nets import DiffusionModelUNet
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# External split/metric helpers (final_loader.py, eval_common.py) -- override with MC_EVAL_DATA.
_ext = os.environ.get("MC_EVAL_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_final_data"))
sys.path.insert(0, os.path.dirname(_ext)); sys.path.insert(0, _ext)
from mambacontrol import get_dataset_from_pd
from final_loader import load_for_method

set_determinism(0)
DEVICE = 'cuda'
_SUF = os.environ.get('MC_LATENT_SUFFIX', '_latent.npz')
_ROOT = os.environ.get('AD_PROGRESSION_ROOT',
                       'data')

# Patient covariates (8-dim cross-attention context): describe the STARTING state
# (diagnosis + baseline regional volumes) so the net can predict how THIS patient
# atrophies, not just the population average. Δt still enters via the input channel.
# ALIGNED (no leak): starting_* only; followup_age is the query time (=Δt), allowed.
_COV = ['followup_age', 'sex', 'starting_diagnosis', 'starting_cerebral_cortex',
        'starting_hippocampus', 'starting_amygdala', 'starting_cerebral_white_matter',
        'starting_lateral_ventricle']
def concat_covariates(_d):
    _d['context'] = torch.tensor([float(_d[c]) for c in _COV]).unsqueeze(0)  # (1,8)
    return _d


if __name__ == '__main__':
    os.makedirs(args.output_dir, exist_ok=True)
    # MC_NOCACHE=1 -> plain non-caching Dataset (the transform is cheap, and this avoids
    # filling up the cache disk). Otherwise PersistentDataset at args.cache_dir.
    _cache = None if os.environ.get('MC_NOCACHE', '0') == '1' else args.cache_dir
    if _cache is not None:
        os.makedirs(_cache, exist_ok=True)
    npz = NumpyReader(npz_keys=['data'])
    tf = transforms.Compose([
        transforms.CopyItemsD(keys=['starting_image_path', 'followup_image_path'], names=['starting_latent', 'followup_latent']),
        transforms.Lambdad(keys=['starting_latent', 'followup_latent'],
                           func=lambda p, _s=_SUF: p.replace('.nii.gz', _s).replace('.nii', _s)),
        transforms.LoadImageD(keys=['starting_latent', 'followup_latent'], reader=npz),
        transforms.EnsureChannelFirstD(keys=['starting_latent', 'followup_latent'], channel_dim=0),
        transforms.DivisiblePadD(keys=['starting_latent', 'followup_latent'], k=4, mode='constant'),
        transforms.Lambda(func=concat_covariates)])

    df = load_for_method(args.dataset_csv, pair=True)
    for c in ('starting_image_path', 'followup_image_path'):
        df[c] = df[c].apply(lambda p: p if os.path.isabs(str(p)) else os.path.join(_ROOT, str(p)))
    if 'sex' not in df.columns: df['sex'] = 0.5
    # Coerce Δt times + all covariate columns to numeric (default collate needs it).
    for c in set(['starting_follow_up', 'followup_follow_up'] + _COV):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0).astype('float32')
    train_df = df[df.split == 'train']; valid_df = df[df.split == 'val']
    print(f"[split] train={len(train_df)} val={len(valid_df)}", flush=True)

    trainset = get_dataset_from_pd(train_df, tf, _cache)
    train_loader = DataLoader(trainset, num_workers=args.num_workers, batch_size=args.batch_size,
                              shuffle=True, persistent_workers=True, pin_memory=True)

    # scale_factor from a followup latent (same as train_control_mamba)
    z0 = trainset[0]['followup_latent']
    scale_factor = float(1.0 / torch.std(z0))
    print(f"scale_factor={scale_factor:.4f}", flush=True)

    # Conditioned residual regressor: Δt via the 5th input channel + 8-dim patient
    # covariates via cross-attention (coarse level only, no fp32 upcast, to keep memory
    # down; MetaTensor is stripped in the loop or attention memory blows up).
    net = DiffusionModelUNet(spatial_dims=3, in_channels=5, out_channels=4, num_res_blocks=2,
                             channels=(128, 256, 256), attention_levels=(False, False, True),
                             num_head_channels=(0, 0, 256), transformer_num_layers=1,
                             with_conditioning=True, cross_attention_dim=8, upcast_attention=False,
                             norm_num_groups=32, norm_eps=1e-6, resblock_updown=True).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    scaler = GradScaler()
    t0 = None

    for epoch in range(args.n_epochs):
        net.train(); pbar = tqdm(enumerate(train_loader), total=len(train_loader)); tot = 0.0; nb = 0
        for step, batch in pbar:
            opt.zero_grad(set_to_none=True)
            _st = lambda t: (t.as_tensor() if hasattr(t, 'as_tensor') else t)
            sz = _st(batch['starting_latent']).to(DEVICE).float() * scale_factor
            fz = _st(batch['followup_latent']).to(DEVICE).float() * scale_factor
            ctx = _st(batch['context']).to(DEVICE).float()   # (n,1,8), MetaTensor stripped
            dt = ((batch['followup_follow_up'].to(DEVICE).float() - batch['starting_follow_up'].to(DEVICE).float()) / 12.0)
            n = sz.shape[0]
            dt_plane = dt.view(n, 1, 1, 1, 1).expand(n, 1, *sz.shape[-3:])
            x = torch.cat([sz, dt_plane], dim=1)  # 5ch
            ts = torch.zeros(n, device=DEVICE).long()
            with autocast(enabled=True):
                pred = net(x=x, timesteps=ts, context=ctx)   # predicted residual (4ch)
                loss = F.l1_loss(pred, fz - sz)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
            pbar.set_description(f"ep{epoch} L1 {tot/nb:.4f}")
        savep = os.path.join(args.output_dir, f'resreg-ep-{epoch}.pth')
        torch.save({'net': net.state_dict(), 'scale_factor': scale_factor}, savep)
        print(f"[ep{epoch}] mean L1 {tot/nb:.5f} -> saved {os.path.basename(savep)}", flush=True)
        old = os.path.join(args.output_dir, f'resreg-ep-{epoch-3}.pth')
        if os.path.exists(old): os.remove(old)
