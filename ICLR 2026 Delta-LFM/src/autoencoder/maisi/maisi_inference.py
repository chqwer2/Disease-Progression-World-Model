# pip install -q "monai-weekly[nibabel, tqdm]"
import argparse
import json
import os
import tempfile

import monai
import torch
from monai.apps import download_url
from monai.config import print_config
from monai.transforms import LoadImage, Orientation
from monai.utils import set_determinism
from scripts.sample import LDMSampler, check_input
from scripts.utils import define_instance
from scripts.utils_plot import find_label_center_loc, get_xyz_plot, show_image
from scripts.diff_model_setting import setup_logging



maisi_version = "maisi3d-rflow"
if maisi_version == "maisi3d-ddpm":
    model_def_path = "./configs/config_maisi3d-ddpm.json"
elif maisi_version == "maisi3d-rflow":
    model_def_path = "./configs/config_maisi3d-rflow.json"

with open(model_def_path, "r") as f:
    model_def = json.load(f)




os.environ["MONAI_DATA_DIRECTORY"] = "temp_work_dir_inference_demo"
directory = os.environ.get("MONAI_DATA_DIRECTORY")
if directory is not None:
    os.makedirs(directory, exist_ok=True)
root_dir = tempfile.mkdtemp() if directory is None else directory

# TODO: remove the `files` after the files are uploaded to the NGC
files = [
    {
        "path": "models/autoencoder_epoch273.pt",
        "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials"
        "/model_zoo/model_maisi_autoencoder_epoch273_alternative.pt",
    },
    {
        "path": "models/mask_generation_autoencoder.pt",
        "url": "https://developer.download.nvidia.com/assets/Clara/monai" "/tutorials/mask_generation_autoencoder.pt",
    },
    # {
    #     "path": "models/mask_generation_diffusion_unet.pt",
    #     "url": "https://developer.download.nvidia.com/assets/Clara/monai"
    #     "/tutorials/model_zoo/model_maisi_mask_generation_diffusion_unet_v2.pt",
    # },
    {
        "path": "configs/all_anatomy_size_condtions.json",
        "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/all_anatomy_size_condtions.json",
    },
    # {
    #     "path": "datasets/all_masks_flexible_size_and_spacing_4000.zip",
    #     "url": "https://developer.download.nvidia.com/assets/Clara/monai"
    #     "/tutorials/all_masks_flexible_size_and_spacing_4000.zip",
    # },
]



for file in files:
    file["path"] = file["path"] if "datasets/" not in file["path"] else os.path.join(root_dir, file["path"])
    download_url(url=file["url"], filepath=file["path"])

args = argparse.Namespace()

if maisi_version == "maisi3d-ddpm":
    environment_file = "./configs/environment_maisi3d-ddpm.json"
elif maisi_version == "maisi3d-rflow":
    environment_file = "./configs/environment_maisi3d-rflow.json"
else:
    raise ValueError(f"maisi_version has to be chosen from ['maisi3d-ddpm', 'maisi3d-rflow'], yet got {maisi_version}.")

with open(environment_file, "r") as f:
    env_dict = json.load(f)
for k, v in env_dict.items():
    # Update the path to the downloaded dataset in MONAI_DATA_DIRECTORY
    val = v if "datasets/" not in v else os.path.join(root_dir, v)
    setattr(args, k, val)


with open(model_def_path, "r") as f:
    model_def = json.load(f)
for k, v in model_def.items():
    setattr(args, k, v)


config_infer_file = "./configs/config_infer.json"
with open(config_infer_file, "r") as f:
    config_infer_dict = json.load(f)
for k, v in config_infer_dict.items():
    setattr(args, k, v)


print("args.trained_autoencoder_path = ", args.trained_autoencoder_path)
print(args.autoencoder_def)
args.output_size = (128, 128, 128)



import pickle
with open("inference_args.pkl", "wb") as f:
    pickle.dump(args, f)

with open("inference_args.pkl", "rb") as f:
    args = pickle.load(f)


autoencoder = define_instance(args, "autoencoder_def")# .to(device)
checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=True, map_location="cpu")
autoencoder.load_state_dict(checkpoint_autoencoder)
autoencoder = autoencoder.to("cuda").half()

# model = model
print("\nargs.output_size = ", args.output_size)


img = torch.randn(1, 1, 128, 144, 128).half().to("cuda")

autoencoder.eval()
with torch.no_grad():
    latent = autoencoder.encode(img)[0]  # z_mu, z_sigma
    print("latent = ", latent.shape)
    recon = autoencoder.decode(latent[0])


print("args.autoencoder_sliding_window_infer_size, ", args.autoencoder_sliding_window_infer_size)
print("args.autoencoder_sliding_window_infer_overlap, ", args.autoencoder_sliding_window_infer_overlap)

# print("args.autoencoder_sliding_window_infer_mode, ", args.autoencoder_sliding_window_infer_mode)
# print(autoencoder



# inferer = SlidingWindowInferer(
#     roi_size=autoencoder_sliding_window_infer_size,
#     sw_batch_size=1,
#     progress=True,
#     mode="gaussian",
#     overlap=autoencoder_sliding_window_infer_overlap,
#     sw_device=device,
#     device=torch.device("cpu"),
# )
