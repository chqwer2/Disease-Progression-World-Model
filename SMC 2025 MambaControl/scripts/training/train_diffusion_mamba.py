import os
import os, gc, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_csv',  required=True, type=str)
parser.add_argument('--cache_dir',  required=True, type=str)
parser.add_argument('--output_dir', required=True, type=str)
parser.add_argument('--aekl_ckpt',  required=True, type=str)
parser.add_argument('--diff_ckpt',   default=None, type=str)
parser.add_argument('--num_workers', default=8,     type=int)
parser.add_argument('--n_epochs',    default=5,     type=int)
parser.add_argument('--batch_size',  default=16,    type=int)
parser.add_argument('--lr',          default=2.5e-5,  type=float)
parser.add_argument('--DEBUG',       action='store_true', help='If set, will use a smaller dataset for debugging purposes.')

parser.add_argument("--gpus",           default="0", type=str, help="Comma-separated list of GPU ids to use for training.")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from monai import transforms
from monai.utils import set_determinism
from monai.data.image_reader import NumpyReader
from monai.networks.schedulers import DDPMScheduler
from mambacontrol.inferers import DiffusionInferer
from tqdm import tqdm

from mambacontrol import const
from mambacontrol import utils
from mambacontrol import networks
from mambacontrol import (
    get_dataset_from_pd,
    sample_using_diffusion
)


set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# MAISI latents. Two layouts exist for the SAME MAISI autoencoder
# (autoencoder_epoch273.pt), and they are NOT interchangeable:
#
#   sibling  <image>.nii.gz -> <image>_latent.npz      shape (4, 30, 36, 30)
#   ssd      <root>/<subject>/<date>/<stem>_adni_3D_latent-vit.npz   (4, 32, 36, 32)
#
# The shapes differ because the sibling set was encoded at
# const.INPUT_SHAPE_AE = (120,144,120) while the ssd set used 128^3 -- same
# weights, different crop. The sibling layout is the one this pipeline trains
# on and the one extract_latents.py writes, so it is the default;
# MAISI_LATENT_LAYOUT=ssd selects the other path for latents already stored that way.
DATA_ROOT = os.environ.get("AD_PROGRESSION_ROOT",
    "data")
LATENT_LAYOUT = os.environ.get("MAISI_LATENT_LAYOUT", "sibling")
SSD_LATENT_ROOT = os.environ.get("MAISI_LATENT_ROOT",
                                 "data/latents")


def attach_split(df):
    """Attach the persisted SUBJECT-LEVEL split and refuse a positional fallback.

    If the manifest has no `split` column, do NOT fall back to a positional cut of the
    pair table: pairs of the same subject appear many times, so a positional split leaks
    most held-out subjects into training and every metric computed afterwards is measured
    on subjects the model was trained on. Split by subject instead.
    """
    if 'split' not in df.columns:
        import sys as _sys
        _sys.path.insert(0, os.environ.get("MC_EVAL_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_final_data")))
        from final_loader import load_or_make_split
        assignment = load_or_make_split(df)
        df = df.copy()
        df['split'] = df['subject_id'].map(lambda s: assignment.get(s, 'train'))
    return df


def image_to_latent_path(path, root=SSD_LATENT_ROOT, anchor="ADNIv1.0",
                         task="adni", dim=3, diffusion="vit"):
    if LATENT_LAYOUT == "sibling":
        # image_path in the CSV is relative to the dataset root ("Image/<subj>/...").
        # The ssd layout hid this because it rebuilt an absolute path from
        # SSD_LATENT_ROOT; here the root has to be re-attached explicitly.
        lat = path.replace('.nii.gz', '_latent.npz').replace('.nii', '_latent.npz')
        return lat if os.path.isabs(lat) else os.path.join(DATA_ROOT, lat)
    parts = path.split('/')
    sub = parts[parts.index(anchor) + 1:-1] if anchor in parts else parts[-3:-1]
    stem = os.path.basename(path).split('.')[0]
    return os.path.join(root, *sub, f"{stem}_{task}_{dim}D_latent-{diffusion}.npz")


def concat_covariates(_dict):
    """
    Provide context for cross-attention layers and concatenate the
    covariates in the channel dimension.
    """
    _dict['context'] = torch.tensor([ _dict[c] for c in const.CONDITIONING_VARIABLES ]).unsqueeze(0)
    return _dict


def images_to_tensorboard(
    writer,
    epoch, 
    mode, 
    autoencoder, 
    diffusion, 
    scale_factor
):
    """
    Visualize the generation on tensorboard
    """

    for tag_i, size in enumerate([ 'small', 'medium', 'large' ]):

        context = torch.tensor([[
            (torch.randint(60, 99, (1,)) - const.AGE_MIN) / const.AGE_DELTA,  # age 
            (torch.randint(1, 2,   (1,)) - const.SEX_MIN) / const.SEX_DELTA,  # sex
            (torch.randint(1, 3,   (1,)) - const.DIA_MIN) / const.DIA_DELTA,  # diagnosis
            0.567, # (mean) cerebral cortex 
            0.539, # (mean) hippocampus
            0.578, # (mean) amygdala
            0.558, # (mean) cerebral white matter
            0.30 * (tag_i+1), # variable size lateral ventricles
        ]])

        image = sample_using_diffusion(
            autoencoder=autoencoder, 
            diffusion=diffusion, 
            context=context,
            device=DEVICE, 
            scale_factor=scale_factor
        )

        os.makedirs(f"image_results/mamba_diffusion_{mode}", exist_ok=True)

        utils.tb_display_generation(
            writer=writer, 
            step=epoch, 
            tag=f'image_results/mamba_diffusion_{mode}/{size}_ventricles.jpg',
            image=image
        )


if __name__ == '__main__':
    
    

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    npz_reader = NumpyReader(npz_keys=['data'])
    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys=['image_path'], names=['latent']),
        transforms.Lambdad(
            keys="latent",  # holds the image path at this point
            func=image_to_latent_path
        ),
        transforms.LoadImageD(keys=['latent'], reader=npz_reader),
        transforms.EnsureChannelFirstD(keys=['latent'], channel_dim=0), 
        transforms.DivisiblePadD(keys=['latent'], k=4, mode='constant'),
        transforms.Lambda(func=concat_covariates)
    ])

    dataset_df = pd.read_csv(args.dataset_csv)
    # print("keys=", dataset_df.keys())
      # Pair to Single DF
    try:
        cols = dataset_df.columns.tolist()

        # Identify column groups
        starting_cols = [c for c in cols if c.startswith("starting_")]
        followup_cols = [c for c in cols if c.startswith("followup_")]
        other_cols   = [c for c in cols if not (c.startswith("starting_") or c.startswith("followup_"))]

        # Helper to strip a single prefix
        def strip_prefix(c, prefix):
            return c[len(prefix):] if c.startswith(prefix) else c

        # START view: keep non-followup columns, strip "starting_" prefix
        start_keep = other_cols + starting_cols
        start_df = (
            dataset_df[start_keep]
            .rename(columns=lambda c: strip_prefix(c, "starting_"))
        )

        # FOLLOWUP view: keep non-starting columns, strip "followup_" prefix
        follow_keep = other_cols + followup_cols
        follow_df = (
            dataset_df[follow_keep]
            .rename(columns=lambda c: strip_prefix(c, "followup_"))
        )

        # Ensure both have the same column order (union of columns)
        all_cols_after = sorted(set(start_df.columns).union(follow_df.columns))
        start_df = start_df.reindex(columns=all_cols_after)
        follow_df = follow_df.reindex(columns=all_cols_after)

        # Combine, drop duplicates, reset index
        combined_df = (
            pd.concat([start_df, follow_df], ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )

        # If you want this combined result back into dataset_df, assign it:
        dataset_df = combined_df

    except Exception as e:
        print(f"Error while reshaping: {e}")


    # print("keys=", dataset_df.keys())
    if 'sex' in dataset_df.columns:
        dataset_df['sex'] = dataset_df['sex'].fillna(0.5)
    else:
        dataset_df['sex'] = 0.5  # create new column with default

    print("dataset_df key=", dataset_df.keys())

    dataset_df = attach_split(dataset_df)
    train_df = dataset_df[dataset_df.split == 'train']
    valid_df = dataset_df[dataset_df.split == 'val']
    print(f"[split] subject-level: train={len(train_df)} val={len(valid_df)} "
          f"(test held out: {int((dataset_df.split == 'test').sum())})")
    
    trainset = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)
    validset = get_dataset_from_pd(valid_df, transforms_fn, args.cache_dir)

    if args.DEBUG:
        trainset = trainset[:10]
        validset = trainset[:10]

    train_loader = DataLoader(dataset=trainset, 
                              num_workers=args.num_workers, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              persistent_workers=True,
                              pin_memory=True)
    
    valid_loader = DataLoader(dataset=validset, 
                              num_workers=args.num_workers, 
                              batch_size=args.batch_size, 
                              shuffle=False, 
                              persistent_workers=True, 
                              pin_memory=True)

    from mambacontrol.autoencoder.MAISI_Unet3D import init_autoencoder

    autoencoder = init_autoencoder(args).to(DEVICE).float()
    autoencoder.eval()

    diffusion   = networks.init_mamba_diffusion(args.diff_ckpt).to(DEVICE)

    num_params = sum(p.numel() for p in diffusion.parameters())
    print(f"Total parameters in diffusion model: {num_params / 1e6:.2f}M")

    scheduler = DDPMScheduler(
        num_train_timesteps=1000, 
        schedule='scaled_linear_beta', 
        beta_start=0.0015, 
        beta_end=0.0205
    )

    inferer = DiffusionInferer(scheduler=scheduler)

    # ---- Robust latent scaling: average std over many samples (a single outlier
    # ---- latent -> huge scale_factor -> NaN). ----
    with torch.no_grad():
        stds = []
        for i in range(min(len(trainset), 64)):
            z = trainset[i]['latent']
            z = z if torch.is_tensor(z) else torch.as_tensor(z)
            s = torch.std(z.float())
            if torch.isfinite(s) and s > 0:
                stds.append(s)
        if not stds:
            raise RuntimeError("Could not compute a finite scale_factor from the latents.")
        scale_factor = float(1.0 / torch.stack(stds).mean())
    print(f"Scaling factor set to {scale_factor:.5f} (avg std over {len(stds)} samples)")

    # ---- Optional multi-GPU (DataParallel) ----
    use_dp = torch.cuda.device_count() > 1
    if use_dp:
        print(f"Using DataParallel over {torch.cuda.device_count()} GPUs")
        diffusion = torch.nn.DataParallel(diffusion)
    diff_core = diffusion.module if use_dp else diffusion

    optimizer = torch.optim.AdamW(diff_core.parameters(), lr=args.lr)
    scaler = GradScaler()
    nan_events = 0


    writer = SummaryWriter()
    global_counter  = { 'train': 0, 'valid': 0 }
    loaders         = { 'train': train_loader, 'valid': valid_loader }
    datasets        = { 'train': trainset, 'valid': validset }

    for epoch in range(args.n_epochs):
        
        for mode in loaders.keys():
            
            loader = loaders[mode]
            diffusion.train() if mode == 'train' else diffusion.eval()
            epoch_loss = 0
            progress_bar = tqdm(enumerate(loader), total=len(loader))
            progress_bar.set_description(f"Epoch {epoch}")
            
            for step, batch in progress_bar:
                            
                with autocast(enabled=True, device_type='cuda'):
                        
                    if mode == 'train': optimizer.zero_grad(set_to_none=True)
                    image_path = batch['image_path']
                    # print("image_path=", image_path)

                    latents = batch['latent'].to(DEVICE) * scale_factor
                    context = batch['context'].to(DEVICE)
                    n = latents.shape[0]
                    class_labels = None #(batch['followup_age'] - batch['starting_age']).long()         

                    # print("latents=", latents.shape, latents.dtype, latents.min(), latents.max(), latents.mean())

                    with torch.set_grad_enabled(mode == 'train'):
                        
                        noise = torch.randn_like(latents).to(DEVICE)
                        timesteps = torch.randint(0, scheduler.num_train_timesteps, (n,), device=DEVICE).long()

                        noise_pred = inferer(
                            inputs=latents, 
                            diffusion_model=diffusion, 
                            noise=noise, 
                            timesteps=timesteps,
                            condition=context,
                            class_labels=class_labels,
                            mode='crossattn'
                        )

                        loss = F.mse_loss( noise.float(), noise_pred.float() )

                if not torch.isfinite(loss):
                    nan_events += 1
                    print(f"[NaN-GUARD] non-finite loss at epoch {epoch} {mode} "
                          f"step {step} (value={loss.item()}); skipping this batch")
                    if mode == 'train':
                        optimizer.zero_grad(set_to_none=True)
                    global_counter[mode] += 1
                    continue

                if mode == 'train':
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(diff_core.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                writer.add_scalar(f'{mode}/batch-mse', loss.item(), global_counter[mode])
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
                global_counter[mode] += 1
        
            # end of epoch
            epoch_loss = epoch_loss / len(loader)
            writer.add_scalar(f'{mode}/epoch-mse', epoch_loss, epoch)

            # visualize results
            images_to_tensorboard(
                writer=writer, 
                epoch=epoch, 
                mode=mode, 
                autoencoder=autoencoder,
                diffusion=diff_core,
                scale_factor=scale_factor
            )

        # save the model
        savepath = os.path.join(args.output_dir, f'unet-ep-{epoch}.pth')
        torch.save(diff_core.state_dict(), savepath)

        print("Saved:", savepath)

        try:
            os.remove(os.path.join(args.output_dir, f'unet-ep-{epoch-1}.pth'))
        except FileNotFoundError:
            pass

    print(f"TRAINING DONE. non-finite-loss batches skipped: {nan_events}")