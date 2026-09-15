import torch.nn as nn
from monai.networks.nets import (
    AutoencoderKL,
)

from .maisi.scripts.utils import define_instance
import os
from typing import Optional
import torch
from monai.apps import download_url

def load_if(checkpoints_path: Optional[str], network: nn.Module) -> nn.Module:
    """
    Load pretrained weights if available.

    Args:
        checkpoints_path (Optional[str]): path of the checkpoints
        network (nn.Module): the neural network to initialize

    Returns:
        nn.Module: the initialized neural network
    """



    if checkpoints_path is not None:
        assert os.path.exists(checkpoints_path), 'Invalid path'
        # device = next(network.parameters()).device # Use the same device as the model
        checkpoint = torch.load(checkpoints_path, map_location='cpu', weights_only=True)

        new_state_dict = {}
        if "autoencoder" in checkpoints_path:
            for key in checkpoint:
                new_key = key
                if (
                        "decoder.blocks.3.conv.conv" in key or
                        "decoder.blocks.6.conv.conv" in key or
                        "decoder.blocks.9.conv.conv" in key
                ):
                    new_key = key.replace("conv.conv", "postconv.conv")
                new_state_dict[new_key] = checkpoint[key]

        elif "controlnet" in checkpoints_path or "cnet" in checkpoints_path:
            for key, val in checkpoint.items():
                if "module." in key:
                    key = key.replace("module.", "")

                if "to_out.0" in key:
                    new_key = key.replace("to_out.0", "out_proj")
                    new_state_dict[new_key] = val
                else:
                    new_state_dict[key] = val

        elif "diffusion" in checkpoints_path:
            for k, v in checkpoint.items():
                new_k = k

                # Common renames
                new_k = new_k.replace("to_out.0", "out_proj")
                new_k = new_k.replace("proj_out.0", "proj_out.conv")
                new_k = new_k.replace("proj_in.0", "proj_in.conv")
                new_k = new_k.replace("conv_in.0", "conv_in.conv")
                new_k = new_k.replace("conv_out.0", "out.2.conv")  # if out is ConvBlock
                new_k = new_k.replace("time_embedding.linear_1", "time_embed.0")
                new_k = new_k.replace("time_embedding.linear_2", "time_embed.2")

                new_state_dict[new_k] = v


        print("Loaded pretrained weights from", checkpoints_path)
        network.load_state_dict(new_state_dict)


    return network.float()



def init_autoencoder(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the KL autoencoder (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the KL autoencoder
    """
    files = [
        {
            "path": "models/autoencoder_epoch273.pt",
            "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials"
                   "/model_zoo/model_maisi_autoencoder_epoch273_alternative.pt",
        },
    ]
    root_dir = "./"
    for file in files:
        file["path"] = file["path"] if "datasets/" not in file["path"] else os.path.join(root_dir, file["path"])
        download_url(url=file["url"], filepath=file["path"])

    import pickle

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "maisi", "inference_args.pkl"), "rb") as f:
        args = pickle.load(f)

    autoencoder  = define_instance(args, "autoencoder_def")# .to(device)
    checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=True, map_location="cpu")
    autoencoder.load_state_dict(checkpoint_autoencoder)

    return autoencoder.float()


if __name__ == "__main__":
    import time
    autoencoder = Autoencoder3D()
    autoencoder.eval()
    img = torch.randn(1, 1, 128, 144, 128)  # (128, 144, 128)


    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
        img = img.cuda()

    start = time.time()
    with torch.no_grad():
        latent, z_sigma = autoencoder.encode(img)
        print("latent shape = ", latent.shape)  # torch.Size([1, 3, 16, 18, 16])
        reconstruction = autoencoder.decode(latent)
        print("recon shape = ", reconstruction.shape)


    print("time elapsed = ", time.time() - start)