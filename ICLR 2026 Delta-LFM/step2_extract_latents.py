import os
import sys
import gc
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from diffusers.models import AutoencoderKL
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Add parent folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.options import args
from utils.utils_image import standardize_images, min_max_normalize
try:
    from utils.utils_wandb import WandBLogger
except Exception:
    WandBLogger = None  # wandb not installed; not needed for extraction
try:
    from src import get_autoencoder
except Exception:
    get_autoencoder = None
from utils import args, import_from_dotted_path, utils_metric
import accelerate
from accelerate import Accelerator

# ------------------------ Configuration ------------------------

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
accelerator = Accelerator()
DEVICE = accelerator.device
extract_latent = args.extract_latent



# ------------------------ Helper Functions ------------------------
from collections import OrderedDict
def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = v
    return new_state_dict

    
def save_latent_to_file(latent: np.ndarray, source_path: str):

    filename = os.path.basename(source_path).split(".")[0]
    source_path_split = source_path.split('/')

    save_path = os.path.join(args.output_dir, f"{source_path_split[-3]}/{source_path_split[-2]}")
    os.makedirs(save_path, exist_ok=True)

    save_path = os.path.join(save_path, f"{filename}.npz")
    
    # print("latent shape = ", latent.shape)
    
    np.savez_compressed(save_path, data=latent)


def normalize_to_match(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    mean_r, std_r = np.mean(reference), np.std(reference)
    mean_t, std_t = np.mean(target), np.std(target)
    target = (target - mean_t) / (std_t + 1e-8)
    return target * std_r + mean_r


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

        if args.dim == 3:
            s = recon.shape[-1] // 2
            recon = recon[..., s]
            img = img[..., s]

        # Match img distribution to recon
        recon_matched = normalize_to_match(img, recon)

        # Compute PSNR and SSIM
        psnr_val = psnr(recon_matched, img, data_range=img.max() - img.min())
        ssim_val = ssim(recon_matched, img,
                        data_range=img.max() - img.min(), channel_axis=0)  # 3, 148, 144

        psnr_2D.append(psnr_val)
        ssim_2D.append(ssim_val)

        # print(f"[{base_name}] PSNR: {psnr_val:.4f} | SSIM: {ssim_val:.4f}")
        # print("recon/img shape:", recon.shape, img.shape)

        recon = recon_matched
        # For multi-channel images, average over channels
        if args.channel > 1 and args.dim == 2:
            recon = np.mean(recon, axis=0)
            img = np.mean(img, axis=0)

        # print("recon = ", recon.shape)
        # print("img = ", img.shape)

        # Save side-by-side
        recon_img = np.concatenate([recon, img], axis=1)
        plt.title("Recon | Source")
        plt.imsave(save_path, recon_img, cmap='gray')


    print(f"2D midslice PSNR: {np.mean(psnr_2D):.4f} | SSIM: {np.mean(ssim_2D):.4f}")
# ------------------------ Main Execution ------------------------

if __name__ == '__main__':
    
    # saved_latent_name

    print("cache dir=", args.cache_dir)
    print("latent_path=", args.latent_path)
    print("data_dir=", args.data_dir)

    # ---------------- Define Dataloader ----------------
    from dataset.ad_progression_3D_triplet import get_visit_dataset

    data_loader, test_ds = get_visit_dataset(args, mode="test_all", min_visits=1)

    # ---------------- Define AutoEncoder Model ----------------
    autoencoder_func = import_from_dotted_path(args.autoencoder)
    autoencoder      = autoencoder_func(args).to(DEVICE).float()
    autoencoder.eval()

    try:
        weight = torch.load(args.aekl_ckpt)
        weight = remove_module_prefix(weight)

        autoencoder.load_state_dict(weight)
        print("Successful load: ", args.aekl_ckpt)
    except FileNotFoundError:
        print(f"File {args.aekl_ckpt} not found, using random initialization for autoencoder.")
        

    image_root = "./image_result/all_step2_visualization"
    os.makedirs(image_root, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    avg_psnr, avg_ssim = [], []
    autoencoder = accelerator.prepare(autoencoder)

    for idx, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
        _img_mt = batch['image']
        source_paths = list(_img_mt.meta['filename_or_obj']) if hasattr(_img_mt, 'meta') else batch.get('image_path')
        images = _img_mt.to(DEVICE).float()

        mri_latent = None
        if extract_latent:
            with accelerator.autocast(), torch.no_grad():
                _nmx = getattr(args, "norm_mode", "01")
                if _nmx == "std":
                    _mx = images.mean(dim=(-1, -2, -3), keepdim=True); _sx = images.std(dim=(-1, -2, -3), keepdim=True).clamp_min(1e-5)
                    _enc_in = (images - _mx) / _sx
                elif _nmx == "pm1":
                    _enc_in = images * 2.0 - 1.0
                else:
                    _enc_in = images
                mri_latent, _ = accelerator.unwrap_model(autoencoder).encode(_enc_in)
            mri_latent = mri_latent.float().cpu().numpy()
            

            for i in range(mri_latent.shape[0]):  # Loop over batch
                latent_i = mri_latent[i].copy()    # 4, 30, 36, 30
                image_path = source_paths[i]  # absolute path of the source volume; the latent is saved alongside it
                save_latent_to_file(latent_i, image_path)


        # Visualize every 10th sample
        with accelerator.autocast(), torch.no_grad():
            # recon, _, _ = autoencoder(images)
            _nmv = getattr(args, "norm_mode", "01")
            if _nmv == "std":
                _mv = images.mean(dim=(-1, -2, -3), keepdim=True); _sv = images.std(dim=(-1, -2, -3), keepdim=True).clamp_min(1e-5)
                _vin = (images - _mv) / _sv
            elif _nmv == "pm1":
                _vin = images * 2.0 - 1.0
            else:
                _vin = images
            mri_latent, mri_sigma = accelerator.unwrap_model(autoencoder).encode(_vin)
            mri_latent = accelerator.unwrap_model(autoencoder).sampling(mri_latent, mri_sigma)
            recon = accelerator.unwrap_model(autoencoder).decode(mri_latent)
            if _nmv == "std":
                recon = recon * _sv + _mv
            elif _nmv == "pm1":
                recon = (recon + 1.0) / 2.0

            # recon = accelerator.unwrap_model(autoencoder).reconstruct(images)

        image_np = images.cpu().numpy()     
        recon_np = recon.cpu().numpy()  

        # print(" image_np shape:", image_np.shape)
        # print(" recon_np shape:", recon_np.shape)

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


    print(f" AVG_PSNR: {np.mean(avg_psnr):.2f}, AVG_SSIM: {np.mean(avg_ssim):.4f}")

    # Save latent representations
    print("Latent representations saved successfully.")
