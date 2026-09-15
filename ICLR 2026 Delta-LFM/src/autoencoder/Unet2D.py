import torch.nn as nn
from diffusers.models import AutoencoderKL

import os
from typing import Optional
import torch




def init_autoencoder(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the KL autoencoder (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the KL autoencoder
    """
    vae_model = "stabilityai/sd-vae-ft-ema"
    autoencoder = AutoencoderKL.from_pretrained(vae_model, force_upcast=True)

    return autoencoder




if __name__ == "__main__":
    # Test 2D

    autoencoder = init_autoencoder()
    autoencoder.eval()

    # img = torch.randn(1, 3, 256, 256)
    img = torch.randn(1, 3, 144, 128)  # img = torch.randn(1, 1, 128, 144, 128)
    with torch.no_grad():
        latent = autoencoder.encode(img).latent_dist.mean
        print("latent shape = ", latent.shape)  # torch.Size([1, 4, 32, 32])
        reconstruction = autoencoder.decode(latent).sample
        print("recon shape = ", reconstruction.shape)






