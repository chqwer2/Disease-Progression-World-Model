import os
import os, gc, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_csv', type=str, required=True)
parser.add_argument('--aekl_ckpt',   type=str, required=True)
parser.add_argument("--gpus",           default="0", type=str, help="Comma-separated list of GPU ids to use for training.")
parser.add_argument('--DEBUG',       action='store_true', help='If set, will use a smaller dataset for debugging purposes.')
parser.add_argument('--force',       action='store_true', help='If set, will overwrite existing latent files.')

    
args = parser.parse_args()

 

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from monai import transforms
from mambacontrol import init_autoencoder
from mambacontrol import const

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
from mambacontrol import utils_metric



def visualize_reconstruction(images, reconstructions, source_paths, save_root="./image_result/step0_extract_feature"):
    os.makedirs(save_root, exist_ok=True)
    psnr_2D = []
    ssim_2D = []

    for i, (img_tensor, recon) in enumerate(zip(images, reconstructions)):
        base_name = os.path.basename(source_paths[i]).split(".")[0]
        suffix =  "_reconstruction_source.png"
        save_path = os.path.join(save_root, base_name + suffix)

        img = img_tensor.squeeze()  #.cpu().numpy()
        recon = recon.squeeze()

        
        s = recon.shape[-1] // 2
        recon = recon[..., s]
        img = img[..., s]

        # recon_matched = normalize_to_match(img, recon)

        # Compute PSNR and SSIM
        psnr_val = psnr(recon, img, data_range=img.max() - img.min())
        ssim_val = ssim(recon, img,
                        data_range=img.max() - img.min(), channel_axis=0)  # 3, 148, 144

        psnr_2D.append(psnr_val)
        ssim_2D.append(ssim_val)

        recon = recon
        # For multi-channel images, average over channels
        # if args.channel > 1 and args.dim == 2:
        #     recon = np.mean(recon, axis=0)
        #     img = np.mean(img, axis=0)

        # print("recon = ", recon.shape)
        # print("img = ", img.shape)

        # Save side-by-side
        recon_img = np.concatenate([recon, img], axis=1)
        plt.title("Recon | Source")
        plt.imsave(save_path, recon_img, cmap='gray')


if __name__ == '__main__':

    


    # autoencoder = init_autoencoder(args.aekl_ckpt).to(DEVICE).eval()

    # autoencoder_func = import_from_dotted_path(args.autoencoder)
    from mambacontrol.autoencoder.MAISI_Unet3D import init_autoencoder

    autoencoder = init_autoencoder(args).to(DEVICE).float()
    autoencoder.eval()

    image_root = "./image_result/mambacontrol_oasis_step2_visualization"
    os.makedirs(image_root, exist_ok=True)

    avg_psnr, avg_ssim = [], []


    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']), 
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
    ])

    df = pd.read_csv(args.dataset_csv)
    # image_path is relative ('Image/...'); absolutize so LoadImaged finds the files
    _root = os.environ.get('AD_PROGRESSION_ROOT',
        'data')
    df['image_path'] = df['image_path'].apply(lambda p: p if os.path.isabs(str(p)) else os.path.join(_root, str(p)))
    try:
        cols = df.columns.tolist()

        # Identify column groups
        starting_cols = [c for c in cols if c.startswith("starting_")]
        followup_cols = [c for c in cols if c.startswith("followup_")]
        other_cols = [c for c in cols if not (c.startswith("starting_") or c.startswith("followup_"))]


        # Helper to strip a single prefix
        def strip_prefix(c, prefix):
            return c[len(prefix):] if c.startswith(prefix) else c


        # START view: keep non-followup columns, strip "starting_" prefix
        start_keep = other_cols + starting_cols
        start_df = (
            df[start_keep]
            .rename(columns=lambda c: strip_prefix(c, "starting_"))
        )

        # FOLLOWUP view: keep non-starting columns, strip "followup_" prefix
        follow_keep = other_cols + followup_cols
        follow_df = (
            df[follow_keep]
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
        df = combined_df

    except Exception as e:
        print(f"Error while reshaping: {e}")

    if args.DEBUG:
        df = df[:10]


    with torch.no_grad():
        for idx, image_path in tqdm(enumerate(df.image_path), total=len(df)):

            destpath = image_path.replace('.nii.gz', '_new_latent.npz').replace('.nii', '_new_latent.npz')
            if os.path.exists(destpath) and not args.force: continue

            mri_tensor = transforms_fn({'image_path': image_path})['image'].to(DEVICE)
            source_paths = image_path

            from torch.cuda.amp import autocast

            with autocast(dtype=torch.float16):
                mri_latent, _ = autoencoder.encode(mri_tensor.unsqueeze(0))
                mri_latent = mri_latent.cpu().squeeze(0).numpy()
            print("mri_latent shape:", mri_latent.shape, destpath)  # (4, 30, 36, 30)

            np.savez_compressed(destpath, data=mri_latent)

            with autocast(dtype=torch.float16):
                recon, _, _ = autoencoder(mri_tensor.unsqueeze(0))
                image_np = mri_tensor.cpu().numpy()
                recon_np = recon.cpu().squeeze(0).numpy()


            psnr_val = utils_metric.psnr_3d(image_np, recon_np)
            ssim_val = utils_metric.ssim_3d(image_np, recon_np)

            if idx % 100 == 0:
                print("images shape:", image_np.shape)
                print(f"Batch {idx} - PSNR/SSIM for batch:", psnr_val, ssim_val)
                visualize_reconstruction(image_np, recon_np, source_paths, save_root=image_root)

            avg_psnr.append(psnr_val)
            avg_ssim.append(ssim_val)


            # Cleanup
            del mri_latent
            gc.collect()
            torch.cuda.empty_cache()

        print(f" Final AVG_PSNR: {np.mean(avg_psnr):.2f}, AVG_SSIM: {np.mean(avg_ssim):.4f}")


