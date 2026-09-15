import os
from typing import Optional

import torch
import torch.nn as nn
from monai.networks.nets import (
    AutoencoderKL, 
    PatchDiscriminator,
    DiffusionModelUNet, 
    ControlNet
)
from collections import OrderedDict


def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = v
    return new_state_dict


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
        assert os.path.exists(checkpoints_path), f'Invalid path: {checkpoints_path}'
        weight = torch.load(checkpoints_path, weights_only=True)
        weight = remove_module_prefix(weight)
        network.load_state_dict(weight)


    return network



def init_autoencoder(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the KL autoencoder (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the KL autoencoder
    """
    autoencoder = AutoencoderKL(spatial_dims=3, 
                                in_channels=1, 
                                out_channels=1, 
                                latent_channels=3,
                                channels=(64, 128, 128, 128),
                                num_res_blocks=2, 
                                norm_num_groups=32,
                                norm_eps=1e-06,
                                attention_levels=(False, False, False, False), 
                                with_decoder_nonlocal_attn=False, 
                                with_encoder_nonlocal_attn=False)
    return load_if(checkpoints_path, autoencoder)


def init_patch_discriminator(checkpoints_path: Optional[str] = None, spatial_dims=3, in_channels=1, num_layers_d=3) -> nn.Module:
    """
    Load the patch discriminator (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the parch discriminator
    """
    patch_discriminator = PatchDiscriminator(spatial_dims=spatial_dims,
                                             num_layers_d=num_layers_d,
                                             channels=32,
                                             in_channels=in_channels,
                                             out_channels=1)

    return load_if(checkpoints_path, patch_discriminator)

    

def init_latent_diffusion(args: Optional[str] = None, in_channels=4, use_image=True, out_channels=None, num_class_embeds=15) -> nn.Module:
    """
    Load the UNet from the diffusion model (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the UNet
    """
    if use_image:
        from .unet_image_cond import DiffusionModelUNet     # Use image as condition
    else:
        from .unet import DiffusionModelUNet                # Use vector as condition

    latent_diffusion = DiffusionModelUNet(spatial_dims=3, 
                                          in_channels=in_channels, 
                                          out_channels=(out_channels if out_channels is not None else in_channels),
                                          num_res_blocks=2,
                                          channels=(256, 512, 768),
                                          attention_levels=(False, True, True), 
                                          norm_num_groups=32, 
                                          norm_eps=1e-6, 
                                          resblock_updown=True, 
                                          num_head_channels=(0, 512, 768), 
                                          transformer_num_layers=1,
                                          with_conditioning=True,
                                          cross_attention_dim=128,  # 128
                                          num_class_embeds=num_class_embeds,   # None
                                          upcast_attention=True, 
                                          use_flash_attention=False)
    return latent_diffusion



def init_large_latent_diffusion(args: Optional[str] = None, in_channels=4, use_image=True, out_channels=None, num_class_embeds=15, cond_dim=0, spade_dim=0) -> nn.Module:
    """
    Load the UNet from the diffusion model (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the UNet
    """
    if use_image:
        from .unet_image_cond import DiffusionModelUNet     # Use image as condition
    else:
        from .unet import DiffusionModelUNet                # Use vector as condition

    latent_diffusion = DiffusionModelUNet(spatial_dims=3, 
                                          in_channels=in_channels, 
                                          out_channels=(out_channels if out_channels is not None else in_channels),
                                          num_res_blocks=2,
                                          channels=(256, 512, 768),
                                          attention_levels=(False, True, True), 
                                          norm_num_groups=32, 
                                          norm_eps=1e-6, 
                                          resblock_updown=True, 
                                          num_head_channels=(0, 96, 96),   # 64?
                                          transformer_num_layers=1,
                                          with_conditioning=True,
                                          cross_attention_dim=128,  # 128
                                          num_class_embeds=num_class_embeds,
                                          cond_dim=cond_dim,
                                          spade_dim=spade_dim, 
                                          upcast_attention=True, 
                                          use_flash_attention=False)
    return latent_diffusion


def init_controlnet(checkpoints_path: Optional[str] = None, 
                    in_channels=3, 
                    conditioning_embedding_in_channels=2,
                    cross_attention_dim=9) -> nn.Module:
    """
    Load the ControlNet (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the ControlNet
    """
    from .controlnet_seg import ControlNet

    controlnet = ControlNet(spatial_dims=3, 
                            in_channels=in_channels,
                            num_res_blocks=2, 
                            channels=(256, 512, 768),
                            attention_levels=(False, True, True), 
                            norm_num_groups=32, 
                            norm_eps=1e-6, 
                            resblock_updown=True, 
                            num_head_channels=(256, 512, 768), 
                            transformer_num_layers=1, 
                            with_conditioning=True,
                            cross_attention_dim=cross_attention_dim, 
                            num_class_embeds=15, 
                            upcast_attention=True, 
                            use_flash_attention=False, 
                            conditioning_embedding_in_channels=conditioning_embedding_in_channels,  
                            conditioning_embedding_num_channels=(256,))

                            
    return load_if(checkpoints_path, controlnet)




def init_large_controlnet(checkpoints_path: Optional[str] = None, 
                    in_channels=3, 
                    conditioning_embedding_in_channels=2,
                    cross_attention_dim=9) -> nn.Module:
    """
    Load the ControlNet (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the ControlNet
    """
    from .controlnet import ControlNet

    controlnet = ControlNet(spatial_dims=3, 
                            in_channels=in_channels,
                            num_res_blocks=2, 
                            channels=(256, 512, 768),
                            attention_levels=(False, True, True), 
                            norm_num_groups=32, 
                            norm_eps=1e-6, 
                            resblock_updown=True, 
                            num_head_channels=(0, 96, 96), 
                            transformer_num_layers=1, 
                            with_conditioning=True,
                            cross_attention_dim=cross_attention_dim, 
                            num_class_embeds=15, 
                            upcast_attention=True, 
                            use_flash_attention=False, 
                            conditioning_embedding_in_channels=conditioning_embedding_in_channels,  
                            conditioning_embedding_num_channels=(256,))

                            
    return load_if(checkpoints_path, controlnet)
