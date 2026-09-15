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


    return network



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

    with open("src/autoencoder/maisi/inference_args.pkl", "rb") as f:
        args = pickle.load(f)

    print("args = ", args.autoencoder_def)

    autoencoder  = define_instance(args, "autoencoder_def")# .to(device)
    checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=True, map_location="cpu")
    autoencoder.load_state_dict(checkpoint_autoencoder)
    # autoencoder = autoencoder.to("cuda").half()



    lora_targets = [
        "quant_conv_mu.conv",
        "quant_conv_log_sigma.conv",
        "post_quant_conv.conv",  # Important: needs ".conv" at end
        "encoder.blocks.0.conv.conv",
        "encoder.blocks.1.conv1.conv.conv",
        "encoder.blocks.1.conv2.conv.conv",
        "encoder.blocks.8.conv1.conv.conv",
        "decoder.blocks.0.conv.conv",
        "decoder.blocks.1.conv1.conv.conv"
        # add more as needed
    ]

    for param in autoencoder.parameters():
        param.requires_grad = False

    # inject_lora_by_name(autoencoder, lora_targets, rank=4, alpha=8.0)
    inject_lora_all_conv3d(autoencoder, rank=32, alpha=8.0)   # 32
    # autoencoder = autoencoder.to("cuda").half()

    return autoencoder


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def inject_lora_all_conv3d(model: nn.Module, rank=4, alpha=1.0):
    for name, child in model.named_children():
        if isinstance(child, nn.Conv3d):
            # Replace with LoRA3DLayer
            lora_module = LoRA3DLayer(child, rank=rank, alpha=alpha)
            model.add_module(name, lora_module)

        else:
            inject_lora_all_conv3d(child, rank=rank, alpha=alpha)



# === LoRA Wrapper for Conv3D ===
class LoRA3DLayer(nn.Module):
    def __init__(self, conv: nn.Conv3d, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        assert isinstance(conv, nn.Conv3d), "LoRA3DLayer supports only Conv3d"

        self.conv = conv
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_channels = conv.in_channels
        out_channels = conv.out_channels
        kx, ky, kz = conv.kernel_size

        # print("in_channels * kx * ky * kz:", in_channels * kx * ky * kz)

        full_rank = in_channels * kx * ky * kz
        middle_rank = max(full_rank // rank, 4)

        self.lora_A = nn.Parameter(
            torch.randn(middle_rank, in_channels * kx * ky * kz, device=conv.weight.device,
                        dtype=conv.weight.dtype, requires_grad=True) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_channels, middle_rank, device=conv.weight.device,
                        dtype=conv.weight.dtype, requires_grad=True)
        )

        # ------------ Debugging Info ------------
        conv_params = sum(p.numel() for p in conv.parameters())
        A_params = self.lora_A.numel()
        B_params = self.lora_B.numel()
        total_lora_params = A_params + B_params

        # print(f"\n[LoRA Injection]")
        # print(f"Conv3D layer: {conv.__class__.__name__}")
        # print(f"→ Conv params: {conv_params:,}")
        # print(f"→ LoRA A params: {A_params:,}")
        # print(f"→ LoRA B params: {B_params:,}")
        # print(f"→ Total LoRA params: {total_lora_params:,} ({total_lora_params / 1e6:.2f} M)")

    def forward(self, x):
        # out = self.conv(x)
        #
        # delta_weight = torch.matmul(self.lora_B, self.lora_A)
        # delta_weight = delta_weight.view(
        #     self.conv.out_channels, self.conv.in_channels, *self.conv.kernel_size
        # )
        #
        # lora_out = F.conv3d(
        #     x, delta_weight, bias=None,
        #     stride=self.conv.stride,
        #     padding=self.conv.padding,
        #     dilation=self.conv.dilation,
        #     groups=self.conv.groups
        # )
        #
        # return out + self.scaling * lora_out

        delta_weight = torch.matmul(self.lora_B, self.lora_A)
        delta_weight = delta_weight.view(
            self.conv.out_channels,
            self.conv.in_channels,
            *self.conv.kernel_size
        )

        # Merge weights: base weight + scaled LoRA delta
        merged_weight = self.conv.weight + self.scaling * delta_weight
        out = F.conv3d(
            x, merged_weight, bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups
        )

        return out


# === Utility: Get module by dot-path ===
def get_module_by_path(model, path):
    module = model
    for attr in path.split('.'):
        if attr.isdigit():
            module = module[int(attr)]
        else:
            module = getattr(module, attr)
    return module


# === Utility: Set module by dot-path ===
def set_module_by_path(model, path, new_module):
    parts = path.split('.')
    parent = model
    for attr in parts[:-1]:
        if attr.isdigit():
            parent = parent[int(attr)]
        else:
            parent = getattr(parent, attr)
    final_attr = parts[-1]
    if final_attr.isdigit():
        parent[int(final_attr)] = new_module
    else:
        setattr(parent, final_attr, new_module)


# === Main Injection Function ===
def inject_lora_by_name(model: nn.Module, layer_names: list[str], rank=4, alpha=1.0):
    for name in layer_names:
        try:
            target_module = get_module_by_path(model, name)
            if not isinstance(target_module, nn.Conv3d):
                raise TypeError(f"{name} is not a Conv3d layer")

            lora_module = LoRA3DLayer(target_module, rank=rank, alpha=alpha)
            set_module_by_path(model, name, lora_module)
            # print(f"Injected LoRA into: {name}")
        except Exception as e:
            print(f"❌ Failed to inject LoRA at '{name}': {e}")



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