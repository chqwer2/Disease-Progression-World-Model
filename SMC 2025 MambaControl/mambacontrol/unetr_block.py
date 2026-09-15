# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, CrossAttentionBlock, MLPBlock, SABlock, SpatialAttentionBlock, Upsample
from monai.networks.layers.factories import Pool
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module



class DiffusionUNetTransformerBlock(nn.Module):
    """
    A Transformer block that allows for the input dimension to differ from the hidden dimension.

    Args:
        num_channels: number of channels in the input and output.
        num_attention_heads: number of heads to use for multi-head attention.
        num_head_channels: number of channels in each attention head.
        dropout: dropout probability to use.
        cross_attention_dim: size of the context vector for cross attention.
        upcast_attention: if True, upcast attention operations to full precision.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.

    """

    def __init__(
        self,
        num_channels: int,
        num_attention_heads: int,
        num_head_channels: int,
        dropout: float = 0.0,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
        include_fc: bool = True,
        use_combined_linear: bool = False,
    ) -> None:
        super().__init__()
        self.attn1 = SABlock(
            hidden_size=num_attention_heads * num_head_channels,
            hidden_input_size=num_channels,
            num_heads=num_attention_heads,
            dim_head=num_head_channels,
            dropout_rate=dropout,
            attention_dtype=torch.float if upcast_attention else None,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
        self.ff = MLPBlock(hidden_size=num_channels, mlp_dim=num_channels * 4, act="GEGLU", dropout_rate=dropout)
        self.attn2 = CrossAttentionBlock(
            hidden_size=num_attention_heads * num_head_channels,
            num_heads=num_attention_heads,
            hidden_input_size=num_channels,
            context_input_size=cross_attention_dim,
            dim_head=num_head_channels,
            dropout_rate=dropout,
            attention_dtype=torch.float if upcast_attention else None,
            use_flash_attention=use_flash_attention,
        )
        self.norm1 = nn.LayerNorm(num_channels)
        self.norm2 = nn.LayerNorm(num_channels)
        self.norm3 = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        # 1. Self-Attention
        x = self.attn1(self.norm1(x)) + x

        # 2. Cross-Attention
        x = self.attn2(self.norm2(x), context=context) + x

        # 3. Feed-forward
        x = self.ff(self.norm3(x)) + x
        return x

class DiffusionUNetResnetBlock(nn.Module):
    """
    Residual block with timestep conditioning.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        temb_channels: number of timestep embedding  channels.
        out_channels: number of output channels.
        up: if True, performs upsampling.
        down: if True, performs downsampling.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        temb_channels: int,
        out_channels: int | None = None,
        up: bool = False,
        down: bool = False,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.channels = in_channels
        self.emb_channels = temb_channels
        self.out_channels = out_channels or in_channels
        self.up = up
        self.down = down

        self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.nonlinearity = nn.SiLU()
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        self.upsample = self.downsample = None
        if self.up:
            self.upsample = WrappedUpsample(
                spatial_dims=spatial_dims,
                mode="nontrainable",
                in_channels=in_channels,
                out_channels=in_channels,
                interp_mode="nearest",
                scale_factor=2.0,
                align_corners=None,
            )
        elif down:
            self.downsample = DiffusionUnetDownsample(spatial_dims, in_channels, use_conv=False)

        self.time_emb_proj = nn.Linear(temb_channels, self.out_channels)

        self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=self.out_channels, eps=norm_eps, affine=True)
        self.conv2 = zero_module(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )
        self.skip_connection: nn.Module
        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)

        if self.upsample is not None:
            x = self.upsample(x)
            h = self.upsample(h)
        elif self.downsample is not None:
            x = self.downsample(x)
            h = self.downsample(h)

        h = self.conv1(h)

        if self.spatial_dims == 2:
            temb = self.time_emb_proj(self.nonlinearity(emb))[:, :, None, None]
        else:
            temb = self.time_emb_proj(self.nonlinearity(emb))[:, :, None, None, None]

        h = h + temb

        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.conv2(h)
        output: torch.Tensor = self.skip_connection(x) + h
        return output



class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data. First, project the input (aka embedding) and reshape to b, t, d. Then apply
    standard transformer action. Finally, reshape to image.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of channels in the input and output.
        num_attention_heads: number of heads to use for multi-head attention.
        num_head_channels: number of channels in each attention head.
        num_layers: number of layers of Transformer blocks to use.
        dropout: dropout probability to use.
        norm_num_groups: number of groups for the normalization.
        norm_eps: epsilon for the normalization.
        cross_attention_dim: number of context dimensions to use.
        upcast_attention: if True, upcast attention operations to full precision.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_attention_heads: int,
        num_head_channels: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        inner_dim = num_attention_heads * num_head_channels

        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)

        self.proj_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=inner_dim,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                DiffusionUNetTransformerBlock(
                    num_channels=inner_dim,
                    num_attention_heads=num_attention_heads,
                    num_head_channels=num_head_channels,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
                for _ in range(num_layers)
            ]
        )

        self.proj_out = zero_module(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=inner_dim,
                out_channels=in_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        # note: if no context is given, cross-attention defaults to self-attention
        batch = channel = height = width = depth = -1
        if self.spatial_dims == 2:
            batch, channel, height, width = x.shape
        if self.spatial_dims == 3:
            batch, channel, height, width, depth = x.shape

        residual = x
        x = self.norm(x)
        x = self.proj_in(x)

        inner_dim = x.shape[1]

        if self.spatial_dims == 2:
            x = x.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        if self.spatial_dims == 3:
            x = x.permute(0, 2, 3, 4, 1).reshape(batch, height * width * depth, inner_dim)

        for block in self.transformer_blocks:
            x = block(x, context=context)

        if self.spatial_dims == 2:
            x = x.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        if self.spatial_dims == 3:
            x = x.reshape(batch, height, width, depth, inner_dim).permute(0, 4, 1, 2, 3).contiguous()

        x = self.proj_out(x)
        return x + residual



class UnetrUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        self.time_emb_proj = nn.Linear(temb_channels, out_channels)

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )   

        self.nonlinearity = nn.SiLU()

    def forward(self, inp, skip, temb=None, context=None):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)

        temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None, None]
        out = out + temb


        return out





class UnetrPrUpBlock(nn.Module):
    """
    A projection upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_layer: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        conv_block: bool = False,
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            num_layer: number of upsampling blocks.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()

        upsample_stride = upsample_kernel_size
        self.transp_conv_init = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
        if conv_block:
            if res_block:
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            get_conv_layer(
                                spatial_dims,
                                out_channels,
                                out_channels,
                                kernel_size=upsample_kernel_size,
                                stride=upsample_stride,
                                conv_only=True,
                                is_transposed=True,
                            ),
                            UnetResBlock(
                                spatial_dims=spatial_dims,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                norm_name=norm_name,
                            ),
                        )
                        for i in range(num_layer)
                    ]
                )
            else:
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            get_conv_layer(
                                spatial_dims,
                                out_channels,
                                out_channels,
                                kernel_size=upsample_kernel_size,
                                stride=upsample_stride,
                                conv_only=True,
                                is_transposed=True,
                            ),
                            UnetBasicBlock(
                                spatial_dims=spatial_dims,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                norm_name=norm_name,
                            ),
                        )
                        for i in range(num_layer)
                    ]
                )
        else:
            self.blocks = nn.ModuleList(
                [
                    get_conv_layer(
                        spatial_dims,
                        out_channels,
                        out_channels,
                        kernel_size=upsample_kernel_size,
                        stride=upsample_stride,
                        conv_only=True,
                        is_transposed=True,
                    )
                    for i in range(num_layer)
                ]
            )

    def forward(self, x):
        x = self.transp_conv_init(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class UnetrBasicBlock(nn.Module):
    """
    A CNN module that can be used for UNETR, based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int, 
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()

        self.time_emb_proj = nn.Linear(temb_channels, out_channels)

        if res_block:
            self.layer = UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )
        else:
            self.layer = UnetBasicBlock(  # type: ignore
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )

        self.nonlinearity = nn.SiLU()

        num_res_blocks    = 1
        num_head_channels = 1
        norm_num_groups = 48
        norm_eps = 1e-6
        transformer_num_layers = 0
        cross_attention_dim = 128

        dropout_cattn = 0.0
        include_fc = True
        use_combined_linear = False
        use_flash_attention = False
        upcast_attention = False
        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels

            resnets.append(
                DiffusionUNetResnetBlock(
                    spatial_dims=3,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

            attentions.append(
                SpatialTransformer(
                    spatial_dims=3,
                    in_channels=out_channels,
                    num_attention_heads=out_channels // num_head_channels,
                    num_head_channels=num_head_channels,
                    num_layers=transformer_num_layers,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    dropout=dropout_cattn,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
            )   
        
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)


    def forward(self, inp, temb=None, context=None):
        input_temb = temb

        h = self.layer(inp)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None, None]
            h = h + temb

        if context is not None:
            for resnet, attn in zip(self.resnets, self.attentions):
                h = resnet(h, emb=input_temb)  # , temb  
                h = attn(h, context=context).contiguous()

        return h
