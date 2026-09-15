import numpy as np
import torch
import imageio.v2 as imageio
import torch.nn.functional as F

def standardize_images(images):
    """
    Standardize images to have mean 0 and std 1.
    """
    mean_ = images.mean(dim=(-1, -2, -3), keepdim=True)  # Compute mean over all dimensions except the first (batch)
    std_ = images.std(dim=(-1, -2, -3), keepdim=True)  # Compute std over all dimensions except the first (batch)

    too_small = std_ < 1e-5
    if torch.any(too_small):
        std_ = std_.clone()
        std_[too_small] = 1  #e-5

    images = (images - mean_) / (std_ + 1e-6)
    return images


def min_max_normalize(images):
    """
    Normalize images to the range [0, 1].
    """
    division = images.max() - images.min()
    too_small = division < 1e-5
    if too_small:
        division = 1

    min      = images.min()
    images   = (images - min) / (division + 1e-6)
    return images


def pad_to_shape(tensor, target_shape):
    """
    Pads a 5D tensor [B, C, D, H, W] to the given target shape.
    Pads only D, H, W dimensions.

    Args:
        tensor (torch.Tensor): input tensor of shape [B, C, D, H, W]
        target_shape (tuple): desired shape, e.g., (B, C, 32, 32, 32)

    Returns:
        torch.Tensor: padded tensor
    """
    _, _, D, H, W = tensor.shape
    _, _, TD, TH, TW = target_shape

    pad_D = TD - D
    pad_H = TH - H
    pad_W = TW - W

    # Compute symmetric padding (left, right) for each dimension
    pad = [
        pad_W // 2, pad_W - pad_W // 2,
        pad_H // 2, pad_H - pad_H // 2,
        pad_D // 2, pad_D - pad_D // 2,
    ]

    return F.pad(tensor, pad, mode='constant', value=0)



def save_image(path, x):
    """
    Save a single image to the specified path.
    """
    # Convert to uint8
    # (x - x.min()) / (x.max() - x.min() + 1e-8)

    x = np.clip(x, 0, 1)
    x = (x * 255).astype(np.uint8)

    # Save the image
    imageio.imwrite(path, x)

