import os
import os, gc, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_csv', required=True, type=str)
parser.add_argument('--cache_dir', required=True, type=str)
parser.add_argument('--output_dir', required=True, type=str)
parser.add_argument('--aekl_ckpt', required=True, type=str)
parser.add_argument('--diff_ckpt', required=True, type=str)
parser.add_argument('--cnet_ckpt', required=True, type=str)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--n_epochs', default=5, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--DEBUG', action='store_true', help='If set, will use a smaller dataset for debugging purposes.')

parser.add_argument('--lr', default=2.5e-5, type=float)

parser.add_argument("--gpus", default="0", type=str, help="Comma-separated list of GPU ids to use for training.")

parser.add_argument('--save_dir', type=str, default="./diffusemorph_test_results",
                    help='Directory to save test results')

# Other parameters
parser.add_argument('--save', action='store_true', help='Save visualization images')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.makedirs(args.save_dir, exist_ok=True)

import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from monai import transforms
from monai.data.image_reader import NumpyReader
from monai.networks.schedulers import DDPMScheduler
from mambacontrol.utils_metric import psnr_3d, ssim_3d

from tqdm import tqdm

from mambacontrol import const
from mambacontrol import utils
from mambacontrol import networks
from mambacontrol import (
    get_dataset_from_pd,
    sample_using_controlnet_and_z_batch_test
)
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import std, tqdm
from math import *

warnings.filterwarnings("ignore")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# def modify_context(context, starting_a, followup_a):


def get_middle_slices(volume):
    h, w, d = volume.shape
    axial = volume[:, :, d // 2]  # Z-axis
    sagittal = volume[:, w // 2, :]  # Y-axis
    coronal = volume[h // 2, :, :]  # X-axis
    return axial, sagittal, coronal


# Get slices for real and reconstructed

def pad_to_shape(img, target_shape):
    """Pad a 2D image (slice) to the target shape with zeros (centered)."""
    pad_height = target_shape[0] - img.shape[0]
    pad_width = target_shape[1] - img.shape[1]

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')


def concat_covariates(_dict):
    """
    Provide context for cross-attention layers and concatenate the
    covariates in the channel dimension.
    """
    conditions = [
        _dict['followup_age'],
        _dict['sex'],
        _dict['followup_diagnosis'],
        _dict['followup_cerebral_cortex'],
        _dict['followup_hippocampus'],
        _dict['followup_amygdala'],
        _dict['followup_cerebral_white_matter'],
        _dict['followup_lateral_ventricle']
    ]
    _dict['context'] = torch.tensor(conditions).unsqueeze(0)
    return _dict


def images_to_tensorboard(
        writer,
        epoch,
        mode,
        autoencoder,
        diffusion,
        controlnet,
        dataset,
        scale_factor
):
    """
    Visualize the generation on tensorboard
    """
    resample_fn = transforms.Spacing(pixdim=1.5)
    random_indices = np.random.choice(range(len(dataset)), 3)

    for tag_i, i in enumerate(random_indices):
        starting_z = dataset[i]['starting_latent'] * scale_factor
        context = dataset[i]['context']  # .squeeze(0)
        starting_a = dataset[i]['starting_age']

        starting_image = torch.from_numpy(nib.load(dataset[i]['starting_image_path']).get_fdata()).unsqueeze(0)
        followup_image = torch.from_numpy(nib.load(dataset[i]['followup_image_path']).get_fdata()).unsqueeze(0)
        starting_image = resample_fn(starting_image).squeeze(0)
        followup_image = resample_fn(followup_image).squeeze(0)

        predicted_image = sample_using_controlnet_and_z(
            autoencoder=autoencoder,
            diffusion=diffusion,
            controlnet=controlnet,
            starting_z=starting_z,
            starting_a=starting_a,
            context=context,
            device=DEVICE,
            scale_factor=scale_factor
        )

        utils.tb_display_cond_generation(
            writer=writer,
            step=epoch,
            tag=f'{mode}/comparison_{tag_i}',
            starting_image=starting_image,
            followup_image=followup_image,
            predicted_image=predicted_image
        )


if __name__ == '__main__':

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    image_path_keys = ['starting_image_path', 'followup_image_path']
    image_keys      = ['starting_image', 'followup_image']

    npz_reader = NumpyReader(npz_keys=['data'])

    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys="starting_image_path", names='starting_image'),
        transforms.CopyItemsD(keys="followup_image_path", names='followup_image'),

        transforms.LoadImageD(image_only=True, keys=image_keys),
        transforms.EnsureChannelFirstD(keys=image_keys),
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=image_keys),
        transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=image_keys),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=image_keys),
        transforms.Lambda(func=concat_covariates),
    ])

    dataset_df = pd.read_csv(args.dataset_csv)
    if "sex" not in dataset_df.columns:
        dataset_df['sex'] = 0.5

    df_length = len(dataset_df)
    # train_df = dataset_df[:int(0.8 * df_length)]  # 80% for training
    valid_df = dataset_df[int(0.8 * df_length):]  # 20% for validation

    # trainset = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)
    validset = get_dataset_from_pd(valid_df, transforms_fn, args.cache_dir)
    if args.DEBUG:
        # trainset = trainset[:10]
        validset = validset[:10]

    valid_loader = DataLoader(dataset=validset,
                              num_workers=args.num_workers,
                              batch_size=args.batch_size,
                              shuffle=True,
                              persistent_workers=True,
                              pin_memory=True)

    from mambacontrol.autoencoder.MAISI_Unet3D import init_autoencoder

    autoencoder = init_autoencoder(args).to(DEVICE).float()
    autoencoder.eval()

    # autoencoder = networks.init_autoencoder(args.aekl_ckpt)
    diffusion   = networks.init_mamba_diffusion(args.diff_ckpt)
    controlnet  = networks.init_mamba_controlnet(args.cnet_ckpt)
    controlnet.load_state_dict(torch.load(args.cnet_ckpt))

    print("ControlNet initialized with weights from:", args.cnet_ckpt)

    # freeze the unet weights
    for p in diffusion.parameters():
        p.requires_grad = False

    for p in autoencoder.parameters():
        p.requires_grad = False

    for p in controlnet.parameters():
        p.requires_grad = False

    autoencoder.eval()
    diffusion.eval()
    controlnet.eval()  # Set ControlNet to evaluation mode

    # Move everything to DEVICE
    autoencoder.to(DEVICE)
    diffusion.to(DEVICE)
    controlnet.to(DEVICE)

    scaler = GradScaler()
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.lr)

    scale_factor = 0.96111625
    print(f"Scaling factor set to {scale_factor}")

    scheduler = DDPMScheduler(num_train_timesteps=1000,
                              schedule='scaled_linear_beta',
                              beta_start=0.0015,
                              beta_end=0.0205)

    writer = SummaryWriter()

    progress_bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    progress_bar.set_description(f"Test ")


    # pad_transform = DivisiblePadD(keys=['starting_latent'], kpredicted_z=4, mode='constant')

    def pad_to_divisible(x: torch.Tensor, k: int):
        # x: (B, C, D, H, W)
        D, H, W = x.shape[-3:]
        pad_d = (k - D % k) % k
        pad_h = (k - H % k) % k
        pad_w = (k - W % k) % k

        # F.pad uses (W_left, W_right, H_left, H_right, D_left, D_right)
        padding = (0, pad_w, 0, pad_h, 0, pad_d)
        x_padded = F.pad(x, padding, mode='constant', value=0)
        return x_padded, (D, H, W)


    def unpad_to_shape(x: torch.Tensor, orig_shape):
        D, H, W = orig_shape
        return x[..., :D, :H, :W]


    psnr_before_list = []
    psnr_after_list = []
    ssim_before_list = []
    ssim_after_list = []
    res_mae_list = []

    # Dataset-wise metrics
    dataset_metrics = defaultdict(lambda: {
        "psnr_before": [], "psnr_after": [],
        "ssim_before": [], "ssim_after": [],
        "res_mae": [], "year_diff": []
    })


    for idx, batch in progress_bar:
        test_data = batch
        dataset_name = test_data['dataset'][0]
        starting_diagnosis = test_data['starting_diagnosis'][0].item() if 'starting_diagnosis' in test_data else 0
        followup_diagnosis = test_data['followup_diagnosis'][0].item() if 'followup_diagnosis' in test_data else 0
        starting_age = test_data['starting_age'][0].item() if 'starting_age' in test_data else 0
        followup_age = test_data['followup_age'][0].item() if 'followup_age' in test_data else 0
        starting_path = test_data['starting_image_path'][0] if 'starting_image_path' in test_data else ''
        followup_path = test_data['followup_image_path'][0] if 'followup_image_path' in test_data else ''

        starting_age = test_data['starting_age'][0].item() if 'starting_age' in test_data else 0
        followup_age = test_data['followup_age'][0].item() if 'followup_age' in test_data else 0

        starting_path = test_data['starting_image_path'][0] if 'starting_image_path' in test_data else ''
        followup_path = test_data['followup_image_path'][0] if 'followup_image_path' in test_data else ''

        year_diff = ceil((followup_age - starting_age) * 100)

        starting_image = batch['starting_image'].to(DEVICE)  # * scale_factor
        followup_image = batch['followup_image'].to(DEVICE)  # * scale_factor
        context = batch['context'].to(DEVICE)
        starting_a = batch['starting_age'].to(DEVICE)
        followup_a = batch['followup_age'].to(DEVICE)

        n = starting_a.shape[0]

        with autocast(enabled=True, device_type='cuda'):

            starting_z, _ = autoencoder.encode(starting_image)

            starting_z = starting_z * scale_factor  # Encode starting image
            starting_z, orig_shape = pad_to_divisible(starting_z, k=4)

            print("starting_z stat:", starting_z.shape, starting_z.min(), starting_z.max(), starting_z.mean())

            # context = modify_context(context, starting_a, followup_a)

            predicted_z = sample_using_controlnet_and_z_batch_test(
                autoencoder=autoencoder,
                diffusion=diffusion,
                controlnet=controlnet,
                starting_z=starting_z,
                starting_a=starting_a,
                context=context,
                device=DEVICE,
                num_inference_steps=50,  # It is important
                beta_start = 0.0015,
                beta_end = 0.0205,
                scale_factor=scale_factor,
                ratio=0.8  #
            )

            # print("output:", predicted_z.shape, predicted_z.min(), predicted_z.max(), predicted_z.mean())

            predicted_z = unpad_to_shape(predicted_z, orig_shape)

            predicted_z = torch.clip(predicted_z, starting_z.min(), starting_z.max())

            predicted_image = autoencoder.decode(predicted_z.to(DEVICE))  # _stage_2_outputs

            # x = autoencoder.decode_stage_2_outputs( predicted_z.to(DEVICE) )
            # predicted_image = trans.to_mni_space_1p5mm_trick( x.cpu() )#.squeeze(0)

        followup_image = followup_image.cpu().numpy()
        predicted_image = predicted_image.cpu().numpy()
        starting_image = starting_image.cpu().numpy()

        starting_diagnosis_list = batch['starting_diagnosis'] if 'starting_diagnosis' in batch else 0
        followup_diagnosis_list = batch['followup_diagnosis'] if 'followup_diagnosis' in batch else 0

        starting_age_list = batch['starting_age'] if 'starting_age' in batch else 0
        followup_age_list = batch['followup_age'] if 'followup_age' in batch else 0

        starting_path_list = batch['starting_image_path'] if 'starting_image_path' in batch else ''
        followup_path_list = batch['followup_image_path'] if 'followup_image_path' in batch else ''

        # print("pred image:",     predicted_image.shape, predicted_image.min(), predicted_image.max(), predicted_image.mean(), "std=", np.std(predicted_image.astype(np.float64)))
        # print("followup image:", followup_image.shape, followup_image.min(), followup_image.max(), followup_image.mean())

        whole_comparison = []
        for (moving_np, fixed_np, deformed_np, starting_diagnosis, followup_diagnosis,
             starting_age, followup_age, starting_path, followup_path) in zip(
            starting_image, followup_image, predicted_image, starting_diagnosis_list, followup_diagnosis_list,
            starting_age_list, followup_age_list, starting_path_list, followup_path_list):

            moving_np = moving_np.squeeze()  # Remove channel dimension if present
            fixed_np = fixed_np.squeeze()  # Remove channel dimension if present
            deformed_np = deformed_np.squeeze()  # Remove channel dimension if present


            # deformed_np = np.clip(deformed_np, 0, 1)  # Ensure values are in [0, 1] range

            # fixed_np <- deformed_np match the distribution
            def match_mean_std(source, target):
                src_mean, src_std = source.mean(), source.astype(np.float64).std()
                tgt_mean, tgt_std = target.mean(), np.std(target)
                return ((source - src_mean) / (src_std + 1e-8)) * tgt_std + tgt_mean






            deformed_np = match_mean_std(deformed_np, fixed_np)
            deformed_np = np.clip(deformed_np, 0, 1)

            # ------------------ Calculate metrics -------------------

            # match the mean
            mean_ = fixed_np.mean()
            std_ = fixed_np.std()
            deformed_np = deformed_np - deformed_np.mean() + mean_
            deformed_np = deformed_np / (deformed_np.std() + 1e-8) * std_  # normalize to original std

            # ------------------ Metrics -------------------
            psnr_before = psnr_3d(moving_np, fixed_np)
            psnr_after = psnr_3d(deformed_np, fixed_np)
            ssim_before = ssim_3d(moving_np, fixed_np)
            ssim_after = ssim_3d(deformed_np, fixed_np)

            res_image_moving = moving_np - fixed_np
            res_image_deformed = deformed_np - fixed_np

            denominator = np.maximum(
                (np.abs(res_image_moving) + np.abs(res_image_deformed)) / 2,
                np.mean(np.abs(res_image_moving) + np.abs(res_image_deformed)) / 2
            )

            # denominator = (np.abs(res_image_moving) + np.abs(res_image_deformed)) / 2

            res_mae = np.clip(np.abs(res_image_deformed - res_image_moving) / denominator, 0, 2.0)  # , like sMAPE style

            # Collect global lists
            psnr_before_list.append(psnr_before)
            psnr_after_list.append(psnr_after)
            ssim_before_list.append(ssim_before)
            ssim_after_list.append(ssim_after)
            res_mae_list.append(np.mean(res_mae))

            # Collect dataset-specific
            dataset_metrics[dataset_name]["psnr_before"].append(psnr_before)
            dataset_metrics[dataset_name]["psnr_after"].append(psnr_after)
            dataset_metrics[dataset_name]["ssim_before"].append(ssim_before)
            dataset_metrics[dataset_name]["ssim_after"].append(ssim_after)
            dataset_metrics[dataset_name]["res_mae"].append(np.mean(res_mae))
            dataset_metrics[dataset_name]["year_diff"].append(year_diff)

            # Crop some border
            border = 6
            fixed_np = fixed_np[border:-border, border:-border, border:-border]
            deformed_np = deformed_np[border:-border, border:-border, border:-border]

            gamma = 1.4
            gamma_transform = lambda img, gamma=2.0: np.power(img, gamma)

            # apply to image (values in [0,1])
            fixed_np = gamma_transform(fixed_np, gamma=gamma)
            deformed_np = gamma_transform(deformed_np, gamma=gamma)

            r = 0.03
            fixed_np[fixed_np < r] = 0
            deformed_np[deformed_np < r] = 0

            real_axial, real_sagittal, real_coronal = get_middle_slices(fixed_np)
            recon_axial, recon_sagittal, recon_coronal = get_middle_slices(deformed_np)

            # ------------------- Save image ---------------------
            if idx % 25 == 0 or idx == len(valid_loader) - 1:
                save_root = "test_image_results"

                os.makedirs(save_root, exist_ok=True)

                real_slices = [real_axial, real_sagittal, real_coronal]
                recon_slices = [recon_axial, recon_sagittal, recon_coronal]

                max_height = max(slice.shape[0] for slice in real_slices + recon_slices)
                max_width = max(slice.shape[1] for slice in real_slices + recon_slices)
                target_shape = (max_height, max_width)

                real_slices_padded = [pad_to_shape(s, target_shape) for s in real_slices]
                recon_slices_padded = [pad_to_shape(s, target_shape) for s in recon_slices]

                real_row = np.concatenate(real_slices_padded, axis=1)
                recon_row = np.concatenate(recon_slices_padded, axis=1)
                comparison_img = np.concatenate([real_row, recon_row], axis=0)

                save_path = os.path.join(save_root, f"{idx}_compare.png")
                plt.imsave(save_path, comparison_img, cmap='gray')

                print(f"""
                        [Validation Summary up to IDX {idx}]
                        -------------------------------------------------------------------------
                        | Pair              |   PSNR (↑)     |   SSIM (↑)      |    Res MAE
                        -------------------------------------------------------------------------
                        | Fake → Target     | {np.mean(psnr_after_list):.3f}      | {np.mean(ssim_after_list):.3f}      |  {np.mean(res_mae_list):.3f}  
                        | Input → Target    | {np.mean(psnr_before_list):.3f}      | {np.mean(ssim_before_list):.3f}      | 
                        -------------------------------------------------------------------------
                        """)

            # ------------------- Save npz if requested ---------------------
            if args.save:
                starting_diagnosis = int(starting_diagnosis * 3)
                followup_diagnosis = int(followup_diagnosis * 3)
                starting_age = int(starting_age * 100)
                followup_age = int(followup_age * 100)
                patient_id = starting_path.split('/')[-3]

                save_target = os.path.join(args.save_dir, patient_id)
                os.makedirs(save_target, exist_ok=True)

                start_image_id = starting_path.split('/')[-2]
                follow_image_id = followup_path.split('/')[-2]

                image_name = (f"{start_image_id}-{follow_image_id}_Age_{starting_age}-{followup_age}"
                              f"_Diag_{starting_diagnosis}-{followup_diagnosis}"
                              f"_P-{psnr_after:.4f}_S-{ssim_after:.4f}_M-{np.mean(res_mae):.4f}")

                residual_axial = np.abs(recon_axial - real_axial)
                residual_sagittal = np.abs(recon_sagittal - real_sagittal)
                residual_coronal = np.abs(recon_coronal - real_coronal)


                vmin = max(residual_axial.min(), residual_sagittal.min(), residual_coronal.min(), 0)
                vmax = min(residual_axial.max(), residual_sagittal.max(), residual_coronal.max(), 1)


                def stack_image(axial, sagittal, coronal):
                    """
                    Stack axial, sagittal, and coronal slices into one combined image.
                    Assumes inputs are PyTorch tensors (2D or 3D slices).
                    """

                    # axial    = np.ascontiguousarray(np.rot90(axial,    k=1, axes=(0, 1)))
                    sagittal = np.ascontiguousarray(np.rot90(sagittal, k=1, axes=(0, 1)))
                    coronal = np.ascontiguousarray(np.rot90(coronal, k=1, axes=(0, 1)))

                    # Ensure torch tensors with batch + channel dims

                    # print("stack shapes:", axial.shape, sagittal.shape, coronal.shape)

                    def to_nchw(x):
                        if isinstance(x, np.ndarray):
                            x = torch.from_numpy(x)
                        if x.ndim == 2:  # H, W
                            x = x.unsqueeze(0).unsqueeze(0)
                        elif x.ndim == 3:  # C, H, W
                            x = x.unsqueeze(0)
                        return x.float()

                    axial = to_nchw(axial)
                    sagittal = to_nchw(sagittal)
                    coronal = to_nchw(coronal)

                    # Resize sagittal & coronal to half of axial
                    target_size = (axial.shape[-2] // 2 - 1, axial.shape[-1] // 2)
                    sagittal = F.interpolate(sagittal, size=target_size, mode='bilinear', align_corners=False)
                    coronal = F.interpolate(coronal, size=target_size, mode='bilinear', align_corners=False)

                    # Squeeze to numpy
                    axial = axial.squeeze().cpu().numpy()
                    sagittal = sagittal.squeeze().cpu().numpy()
                    coronal = coronal.squeeze().cpu().numpy()

                    # Stack sagittal + coronal vertically
                    bottom = np.concatenate([sagittal, coronal], axis=1)

                    # bottom make it 0.9 original bottom height /

                    axial = axial[:-3]
                    bottom = bottom[3:]

                    # Stack axial on top
                    stacked = np.concatenate([axial, bottom], axis=0)

                    return stacked


                # --- make stacked images ---
                stacked_real = stack_image(real_axial, real_sagittal, real_coronal)
                stacked_recon = stack_image(recon_axial, recon_sagittal, recon_coronal)
                stacked_residual = stack_image(residual_axial, residual_sagittal, residual_coronal)

                stacked_real = (stacked_real - stacked_real.min()) / (stacked_real.max() - stacked_real.min() + 1e-8)
                stacked_recon = (stacked_recon - stacked_recon.min()) / (
                            stacked_recon.max() - stacked_recon.min() + 1e-8)


                # stacked = np.clip(stacked, 0, 1)

                # --- save helper ---
                def save_img(img, out_path, cmap, vmin, vmax):
                    dpi = 100
                    fig = plt.figure(figsize=(4, 6), dpi=dpi)
                    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                    plt.axis("off")
                    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
                    plt.close()


                # --- prepare folder ---
                save_dir = save_target
                os.makedirs(save_dir, exist_ok=True)

                # --- save files ---
                save_img(stacked_real, os.path.join(save_dir, f"{image_name}_real.jpg"), cmap="gray", vmin=0, vmax=1)
                save_img(stacked_recon, os.path.join(save_dir, f"{image_name}_recon.jpg"), cmap="gray", vmin=0, vmax=1)
                save_img(stacked_residual, os.path.join(save_dir, f"{image_name}_residual.jpg"), cmap="magma",
                         vmin=vmin, vmax=vmax)
                save_img(stacked_residual, os.path.join(save_dir, f"{image_name}_residual_01.jpg"), cmap="magma",
                         vmin=0, vmax=1)

                print(f"Saved images to {save_dir}")

            # ================= FINAL SUMMARY =================
            print(f"""
                        [Validation Summary Final - Global]
                        -------------------------------------------------------------------------
                        | Pair              |   PSNR (↑)     |   SSIM (↑)      |    Res MAE
                        -------------------------------------------------------------------------
                        | Fake → Target     | {np.mean(psnr_after_list):.3f}      | {np.mean(ssim_after_list):.3f}      |  {np.mean(res_mae_list):.3f}  
                        | Input → Target    | {np.mean(psnr_before_list):.3f}      | {np.mean(ssim_before_list):.3f}      | 
                        -------------------------------------------------------------------------
                        """)

            # ================= DATASET SUMMARY =================
            print("\n[Validation Summary Final by Dataset]")
            print("-------------------------------------------------------------------------")
            print("| Dataset        | PSNR_before | PSNR_after | SSIM_before | SSIM_after | Res_MAE |")
            print("-------------------------------------------------------------------------")

            summary_rows = []
            for dset, vals in dataset_metrics.items():
                row = {
                    "Dataset": dset,
                    "PSNR_before": np.mean(vals["psnr_before"]),
                    "PSNR_after": np.mean(vals["psnr_after"]),
                    "SSIM_before": np.mean(vals["ssim_before"]),
                    "SSIM_after": np.mean(vals["ssim_after"]),
                    "Res_MAE": np.mean(vals["res_mae"])
                }
                summary_rows.append(row)
                print(f"| {dset:<13} | {row['PSNR_before']:.3f}     | {row['PSNR_after']:.3f}    "
                      f"| {row['SSIM_before']:.3f}     | {row['SSIM_after']:.3f}    "
                      f"| {row['Res_MAE']:.3f} |")

            print("-------------------------------------------------------------------------")

            # ================= YEAR-WISE SUMMARY =================
            print("\n[Validation Summary Final by Year Difference]")
            year_bins = defaultdict(list)
            for dset, vals in dataset_metrics.items():
                for i, y in enumerate(vals["year_diff"]):
                    year_group = int(round(y))  # bucket into nearest year
                    year_bins[year_group].append({
                        "psnr_before": vals["psnr_before"][i],
                        "psnr_after": vals["psnr_after"][i],
                        "ssim_before": vals["ssim_before"][i],
                        "ssim_after": vals["ssim_after"][i],
                        "res_mae": vals["res_mae"][i]
                    })

            print("-------------------------------------------------------------------------")
            print("| Years Δ       | PSNR_before | PSNR_after | SSIM_before | SSIM_after | Res_MAE |")
            print("-------------------------------------------------------------------------")

            for year, metrics in sorted(year_bins.items()):
                psnr_b = np.mean([m["psnr_before"] for m in metrics])
                psnr_a = np.mean([m["psnr_after"] for m in metrics])
                ssim_b = np.mean([m["ssim_before"] for m in metrics])
                ssim_a = np.mean([m["ssim_after"] for m in metrics])
                res_m = np.mean([m["res_mae"] for m in metrics])

                print(f"| {year:<11} | {psnr_b:.3f}     | {psnr_a:.3f}    "
                      f"| {ssim_b:.3f}     | {ssim_a:.3f}    "
                      f"| {res_m:.3f} |")
            print("-------------------------------------------------------------------------")

            # ================= SAVE TO CSV =================
            df = pd.DataFrame(summary_rows)
            df.to_csv(os.path.join(args.save_dir, "dataset_metrics_summary.csv"), index=False)

            print(f"Testing complete! Results saved in {args.save_dir} and dataset_metrics_summary.csv")
