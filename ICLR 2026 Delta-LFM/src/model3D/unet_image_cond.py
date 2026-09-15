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
#
# =========================================================================
# Adapted from https://github.com/huggingface/diffusers
# which has the following license:
# https://github.com/huggingface/diffusers/blob/main/LICENSE
#
# Copyright 2022 UC Berkeley Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from monai.networks.blocks import Convolution, MLPBlock, SABlock, SpatialAttentionBlock, Upsample
from monai.networks.layers.factories import Pool
from monai.utils import ensure_tuple_rep, optional_import

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")

__all__ = ["DiffusionModelUNet"]


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def use_fused_attn():
    # Dummy placeholder — replace with actual fused attention check if needed
    return False


from typing import Optional, Tuple

import torch
import torch.nn as nn

from monai.networks.layers.utils import get_rel_pos_embedding_layer
from monai.utils import optional_import

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")

class CrossAttentionBlock(nn.Module):
    """
    A cross-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    One can setup relative positional embedding as described in <https://arxiv.org/abs/2112.01526>
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        hidden_input_size: int | None = None,
        context_input_size: int | None = None,
        dim_head: int | None = None,
        qkv_bias: bool = False,
        save_attn: bool = False,
        causal: bool = False,
        sequence_length: int | None = None,
        rel_pos_embedding: Optional[str] = None,
        input_size: Optional[Tuple] = None,
        attention_dtype: Optional[torch.dtype] = None,
        use_flash_attention: bool = False,
    ) -> None:
        """
        Args:
            hidden_size (int): dimension of hidden layer.
            num_heads (int): number of attention heads.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            hidden_input_size (int, optional): dimension of the input tensor. Defaults to hidden_size.
            context_input_size (int, optional): dimension of the context tensor. Defaults to hidden_size.
            dim_head (int, optional): dimension of each head. Defaults to hidden_size // num_heads.
            qkv_bias (bool, optional): bias term for the qkv linear layer. Defaults to False.
            save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.
            causal (bool, optional): whether to use causal attention.
            sequence_length (int, optional): if causal is True, it is necessary to specify the sequence length.
            rel_pos_embedding (str, optional): Add relative positional embeddings to the attention map. For now only
                "decomposed" is supported (see https://arxiv.org/abs/2112.01526). 2D and 3D are supported.
            input_size (tuple(spatial_dim), optional): Input resolution for calculating the relative positional
                parameter size.
            attention_dtype: cast attention operations to this dtype.
            use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
                (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if dim_head:
            inner_size = num_heads * dim_head
            self.head_dim = dim_head
        else:
            if hidden_size % num_heads != 0:
                raise ValueError("hidden size should be divisible by num_heads.")
            inner_size = hidden_size
            self.head_dim = hidden_size // num_heads

        if causal and sequence_length is None:
            raise ValueError("sequence_length is necessary for causal attention.")

        if use_flash_attention and save_attn:
            raise ValueError(
                "save_attn has been set to True, but use_flash_attention is also set"
                "to True. save_attn can only be used if use_flash_attention is False"
            )

        if use_flash_attention and rel_pos_embedding is not None:
            raise ValueError("rel_pos_embedding must be None if you are using flash_attention.")

        self.num_heads = num_heads
        self.hidden_input_size = hidden_input_size if hidden_input_size else hidden_size
        self.context_input_size = context_input_size if context_input_size else hidden_size
        self.out_proj = nn.Linear(inner_size, self.hidden_input_size)


        # self.context_input_size = 128  # TODO: Fixed

        # self.hidden_input_size =  512 self.context_input_size =  8
        # print("self.hidden_input_size = ", self.hidden_input_size, "self.context_input_size = ", self.context_input_size)
        
        
        self.attention_type = "q"  # TODO


        # key, query, value projections
        if self.attention_type == "q": # "kv"
            self.to_q = nn.Linear(self.hidden_input_size, inner_size, bias=qkv_bias)
            self.to_k = nn.Linear(self.context_input_size, inner_size, bias=qkv_bias)
            self.to_v = nn.Linear(self.context_input_size, inner_size, bias=qkv_bias)
            self.to_kx = nn.Linear(self.hidden_input_size, inner_size, bias=qkv_bias)
            self.to_vx = nn.Linear(self.hidden_input_size, inner_size, bias=qkv_bias)

            # print("using self.attention_type = q")

        elif self.attention_type == "kv": # "q"
            self.to_q = nn.Linear(self.context_input_size, inner_size, bias=qkv_bias)
            self.to_k = nn.Linear(self.hidden_input_size, inner_size, bias=qkv_bias)
            self.to_v = nn.Linear(self.hidden_input_size, inner_size, bias=qkv_bias)
            self.to_qx = nn.Linear(self.hidden_input_size, inner_size, bias=qkv_bias)

        else:
            raise ValueError("attention_type must be either 'q' or 'kv'.")


        
        self.input_rearrange = Rearrange("b h (l d) -> b l h d", l=num_heads)

        self.out_rearrange = Rearrange("b l h d -> b h (l d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate

        self.scale = self.head_dim**-0.5
        self.save_attn = save_attn
        self.attention_dtype = attention_dtype

        self.causal = causal
        self.sequence_length = sequence_length
        self.use_flash_attention = use_flash_attention

        if causal and sequence_length is not None:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(sequence_length, sequence_length)).view(1, 1, sequence_length, sequence_length),
            )
            self.causal_mask: torch.Tensor
        else:
            self.causal_mask = torch.Tensor()

        self.att_mat = torch.Tensor()
        self.rel_positional_embedding = (
            get_rel_pos_embedding_layer(rel_pos_embedding, input_size, self.head_dim, self.num_heads)
            if rel_pos_embedding is not None
            else None
        )
        self.input_size = input_size

        

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        """
        Args:
            x (torch.Tensor): input tensor. B x (s_dim_1 * ... * s_dim_n) x C
            context (torch.Tensor, optional): context tensor. B x (s_dim_1 * ... * s_dim_n) x C

        Return:
            torch.Tensor: B x (s_dim_1 * ... * s_dim_n) x C
        """
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        b, t, c = x.size()  # batch size, sequence length, embedding dimensionality (hidden_size)
        
        if self.attention_type == "q":
            q = self.input_rearrange(self.to_q(x))
            if context is not None:
                kv = context if context is not None else x
                _, kv_t, _ = kv.size()
                k = self.input_rearrange(self.to_k(kv))
                v = self.input_rearrange(self.to_v(kv))
            else:
                kv = x
                _, kv_t, _ = kv.size()
                k = self.input_rearrange(self.to_kx(kv))
                v = self.input_rearrange(self.to_vx(kv))

        elif self.attention_type == "kv":
            if context is not None:
                q = self.input_rearrange(self.to_q(context if context is not None else x))
            else:
                q = self.input_rearrange(self.to_qx(x))

            kv = x # if context is None else x
            _, kv_t, _ = kv.size()
            k = self.input_rearrange(self.to_k(kv))
            v = self.input_rearrange(self.to_v(kv))

        else:
            raise ValueError("attention_type must be either 'q' or 'kv'.")


        if self.attention_dtype is not None:
            q = q.to(self.attention_dtype)
            k = k.to(self.attention_dtype)

        if self.use_flash_attention:
            x = torch.nn.functional.scaled_dot_product_attention(
                query=q, key=k, value=v, scale=self.scale, dropout_p=self.dropout_rate, is_causal=self.causal
            )
        else:
            att_mat = torch.einsum("blxd,blyd->blxy", q, k) * self.scale
            # apply relative positional embedding if defined
            if self.rel_positional_embedding is not None:
                att_mat = self.rel_positional_embedding(x, att_mat, q)

            if self.causal:
                att_mat = att_mat.masked_fill(self.causal_mask[:, :, :t, :kv_t] == 0, float("-inf"))

            att_mat = att_mat.softmax(dim=-1)

            if self.save_attn:
                # no gradients and new tensor;
                # https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
                self.att_mat = att_mat.detach()

            att_mat = self.drop_weights(att_mat)
            x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)

        x = self.out_rearrange(x)
        x = self.out_proj(x)
        x = self.drop_output(x)
        return x


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        # Used only when context is None (self-attention)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Used when context is not None (cross-attention)
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.context_proj = nn.Linear(64, dim, bias=proj_bias)

        self.atten_type = "both"  # q | kv | both
        
        if self.atten_type == "both":
            self.to_q2 = nn.Linear(dim, dim, bias=qkv_bias)
            self.to_k2 = nn.Linear(dim, dim, bias=qkv_bias)
            self.to_v2 = nn.Linear(dim, dim, bias=qkv_bias)

            self.q_norm2 = norm_layer(self.head_dim) if qk_norm else nn.Identity()
            self.k_norm2 = norm_layer(self.head_dim) if qk_norm else nn.Identity()

            self.attn_drop2 = nn.Dropout(attn_drop)
            self.proj2      = nn.Linear(dim, dim, bias=proj_bias)
            self.proj_drop2 = nn.Dropout(proj_drop)


    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:

        # context = context.flatten(2).transpose(1, 2)  # [B, C, H, W] -> [B, S, C]
        input_x = x

        B, N, C = x.shape

        if context is not None:
            B, C, H, W = context.shape  # (12, C, 36, 36)
            context = context.view(B, C, H * W).permute(0, 2, 1)
            context = self.context_proj(context)
            B, N, C = x.shape

            # Cross-attention: q from x, k/v from context
            if self.atten_type == "q":
                q = self.to_q(context).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]

                S = context.shape[1]  # sequence length of conv features
                k = self.to_k(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D]
                v = self.to_v(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D]

                q = self.q_norm(q)
                k = self.k_norm(k)
            
            elif self.atten_type == "kv":
                q = self.to_q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]

                S = context.shape[1]  # sequence length of conv features
                k = self.to_k(context).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D]
                v = self.to_v(context).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D]

                q = self.q_norm(q)
                k = self.k_norm(k)
            else:
                S = context.shape[1]  # sequence length of conv features

                q = self.to_q(context).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
                k = self.to_k(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D]
                v = self.to_v(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D]

                q = self.q_norm(q)
                k = self.k_norm(k)

                q2 = self.to_q2(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
                k2 = self.to_k2(context).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D]
                v2 = self.to_v2(context).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D]

                q2 = self.q_norm2(q2)
                k2 = self.k_norm2(k2)
        else:
            # Self-attention
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
            q, k, v = qkv.unbind(0)

            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.atten_type == "kv" or self.atten_type == "q":
            attn = (q * self.scale) @ k.transpose(-2, -1)  # [B, H, N, S]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v  # [B, H, N, D]

            x = x.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
            x = self.proj(x)
            x = self.proj_drop(x)
            return x + input_x
        
        elif self.atten_type == "both":
            attn = (q * self.scale) @ k.transpose(-2, -1)  # [B, H, N, S]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x1 = attn @ v  # [B, H, N, D]
            x1 = x1.transpose(1, 2).reshape(B, N, C)
            x1 = self.proj(x1)
            x1 = self.proj_drop(x1)

            attn2 = (q2 * self.scale) @ k2.transpose(-2, -1)  # [B, H, N, S]
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop2(attn2)
            x2 = attn2 @ v2  # [B, H, N, D]
            x2 = x2.transpose(1, 2).reshape(B, N, C)
            x2 = self.proj2(x2)
            x2 = self.proj_drop2(x2)

        return x1 + x2   + input_x


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


# Attention
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

        # print("input x shape:", x.shape, context.shape)  # input x shape: torch.Size([1, 512, 12, 12, 12])
        
        if self.spatial_dims == 2:
            x = x.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        if self.spatial_dims == 3:
            x = x.permute(0, 2, 3, 4, 1).reshape(batch, height * width * depth, inner_dim)
        
        if isinstance(context, torch.Tensor):
            batch, channel, c_height, c_width, c_depth = context.shape
            context = context.permute(0, 2, 3, 4, 1).reshape(batch, c_height * c_width * c_depth, context.shape[1]) if context is not None else None
            
        # x shape: torch.Size([1, 1728, 512]) torch.Size([1, 1728, 64])
        # print("x shape:", x.shape, context.shape)  # Debugging line to check the shape of x
        # TODO: context is at fixed 1, 1728, 128

        for block in self.transformer_blocks:
            x = block(x, context=context)   # Here is the problem

        if self.spatial_dims == 2:
            x = x.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        if self.spatial_dims == 3:
            x = x.reshape(batch, height, width, depth, inner_dim).permute(0, 4, 1, 2, 3).contiguous()

        x = self.proj_out(x)
        return x + residual


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings following the implementation in Ho et al. "Denoising Diffusion Probabilistic
    Models" https://arxiv.org/abs/2006.11239.

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
        embedding_dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
    """
    if timesteps.ndim != 1:
        raise ValueError("Timesteps should be a 1d-array")

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    freqs = torch.exp(exponent / half_dim)

    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        embedding = torch.nn.functional.pad(embedding, (0, 1, 0, 0))

    return embedding


class DiffusionUnetDownsample(nn.Module):
    """
    Downsampling layer.

    Args:
        spatial_dims: number of spatial dimensions.
        num_channels: number of input channels.
        use_conv: if True uses Convolution instead of Pool average to perform downsampling. In case that use_conv is
            False, the number of output channels must be the same as the number of input channels.
        out_channels: number of output channels.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension.
    """

    def __init__(
        self, spatial_dims: int, num_channels: int, use_conv: bool, out_channels: int | None = None, padding: int = 1
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.out_channels = out_channels or num_channels
        self.use_conv = use_conv
        if use_conv:
            self.op = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.num_channels,
                out_channels=self.out_channels,
                strides=2,
                kernel_size=3,
                padding=padding,
                conv_only=True,
            )
        else:
            if self.num_channels != self.out_channels:
                raise ValueError("num_channels and out_channels must be equal when use_conv=False")
            self.op = Pool[Pool.AVG, spatial_dims](kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        del emb
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Input number of channels ({x.shape[1]}) is not equal to expected number of channels "
                f"({self.num_channels})"
            )
        output: torch.Tensor = self.op(x)
        return output


class WrappedUpsample(Upsample):
    """
    Wraps MONAI upsample block to allow for calling with timestep embeddings.
    """

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        del emb
        upsampled: torch.Tensor = super().forward(x)
        return upsampled


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


class DownBlock(nn.Module):
    """
    Unet's down block containing resnet and downsamplers blocks.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        temb_channels: number of timestep embedding channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_downsample: if True add downsample block.
        resblock_updown: if True use residual blocks for downsampling.
        downsample_padding: padding used in the downsampling block.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []

        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsampler: nn.Module | None
            if resblock_updown:
                self.downsampler = DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    down=True,
                )
            else:
                self.downsampler = DiffusionUnetDownsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            self.downsampler = None

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del context
        output_states = []

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)

        return hidden_states, output_states


class AttnDownBlock(nn.Module):
    """
    Unet's down block containing resnet, downsamplers and self-attention blocks.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        temb_channels: number of timestep embedding  channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_downsample: if True add downsample block.
        resblock_updown: if True use residual blocks for downsampling.
        downsample_padding: padding used in the downsampling block.
        num_head_channels: number of channels in each attention head.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
        num_head_channels: int = 1,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            attentions.append(
                SpatialAttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    num_head_channels=num_head_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.downsampler: nn.Module | None
        if add_downsample:
            if resblock_updown:
                self.downsampler = DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    down=True,
                )
            else:
                self.downsampler = DiffusionUnetDownsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            self.downsampler = None

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del context
        output_states = []

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states).contiguous()
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)

        return hidden_states, output_states


class CrossAttnDownBlock(nn.Module):
    """
    Unet's down block containing resnet, downsamplers and cross-attention blocks.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        temb_channels: number of timestep embedding channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_downsample: if True add downsample block.
        resblock_updown: if True use residual blocks for downsampling.
        downsample_padding: padding used in the downsampling block.
        num_head_channels: number of channels in each attention head.
        transformer_num_layers: number of layers of Transformer blocks to use.
        cross_attention_dim: number of context dimensions to use.
        upcast_attention: if True, upcast attention operations to full precision.
        dropout_cattn: if different from zero, this will be the dropout value for the cross-attention layers.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
        num_head_channels: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        dropout_cattn: float = 0.0,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

            attentions.append(
                SpatialTransformer(
                    spatial_dims=spatial_dims,
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

        self.downsampler: nn.Module | None
        if add_downsample:
            if resblock_updown:
                self.downsampler = DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    down=True,
                )
            else:
                self.downsampler = DiffusionUnetDownsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            self.downsampler = None

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        output_states = []

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, context=context).contiguous()
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)

        return hidden_states, output_states


class AttnMidBlock(nn.Module):
    """
    Unet's mid block containing resnet and self-attention blocks.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        temb_channels: number of timestep embedding channels.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        num_head_channels: number of channels in each attention head.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        temb_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        num_head_channels: int = 1,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()

        self.resnet_1 = DiffusionUNetResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )
        self.attention = SpatialAttentionBlock(
            spatial_dims=spatial_dims,
            num_channels=in_channels,
            num_head_channels=num_head_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )

        self.resnet_2 = DiffusionUNetResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        del context
        hidden_states = self.resnet_1(hidden_states, temb)
        hidden_states = self.attention(hidden_states).contiguous()
        hidden_states = self.resnet_2(hidden_states, temb)

        return hidden_states


class CrossAttnMidBlock(nn.Module):
    """
    Unet's mid block containing resnet and cross-attention blocks.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        temb_channels: number of timestep embedding channels
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        num_head_channels: number of channels in each attention head.
        transformer_num_layers: number of layers of Transformer blocks to use.
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
        temb_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        num_head_channels: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        dropout_cattn: float = 0.0,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()

        self.resnet_1 = DiffusionUNetResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )
        self.attention = SpatialTransformer(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_attention_heads=in_channels // num_head_channels,
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
        self.resnet_2 = DiffusionUNetResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden_states = self.resnet_1(hidden_states, temb)
        hidden_states = self.attention(hidden_states, context=context)
        hidden_states = self.resnet_2(hidden_states, temb)

        return hidden_states


class UpBlock(nn.Module):
    """
    Unet's up block containing resnet and upsamplers blocks.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        prev_output_channel: number of channels from residual connection.
        out_channels: number of output channels.
        temb_channels: number of timestep embedding channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_upsample: if True add downsample block.
        resblock_updown: if True use residual blocks for upsampling.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown
        resnets = []

        for i in range(num_res_blocks):
            res_skip_channels = in_channels if (i == num_res_blocks - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        self.upsampler: nn.Module | None
        if add_upsample:
            if resblock_updown:
                self.upsampler = DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,
                )
            else:
                post_conv = Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
                self.upsampler = WrappedUpsample(
                    spatial_dims=spatial_dims,
                    mode="nontrainable",
                    in_channels=out_channels,
                    out_channels=out_channels,
                    interp_mode="nearest",
                    scale_factor=2.0,
                    post_conv=post_conv,
                    align_corners=None,
                )

        else:
            self.upsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_list: list[torch.Tensor],
        temb: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_list[-1]
            res_hidden_states_list = res_hidden_states_list[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, temb)

        return hidden_states


class AttnUpBlock(nn.Module):
    """
    Unet's up block containing resnet, upsamplers, and self-attention blocks.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        prev_output_channel: number of channels from residual connection.
        out_channels: number of output channels.
        temb_channels: number of timestep embedding channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_upsample: if True add downsample block.
        resblock_updown: if True use residual blocks for upsampling.
        num_head_channels: number of channels in each attention head.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
        num_head_channels: int = 1,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            res_skip_channels = in_channels if (i == num_res_blocks - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            attentions.append(
                SpatialAttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    num_head_channels=num_head_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        self.upsampler: nn.Module | None
        if add_upsample:
            if resblock_updown:
                self.upsampler = DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,
                )
            else:

                post_conv = Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
                self.upsampler = WrappedUpsample(
                    spatial_dims=spatial_dims,
                    mode="nontrainable",
                    in_channels=out_channels,
                    out_channels=out_channels,
                    interp_mode="nearest",
                    scale_factor=2.0,
                    post_conv=post_conv,
                    align_corners=None,
                )
        else:
            self.upsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_list: list[torch.Tensor],
        temb: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_list[-1]
            res_hidden_states_list = res_hidden_states_list[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states).contiguous()

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, temb)

        return hidden_states


class CrossAttnUpBlock(nn.Module):
    """
    Unet's up block containing resnet, upsamplers, and self-attention blocks.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        prev_output_channel: number of channels from residual connection.
        out_channels: number of output channels.
        temb_channels: number of timestep embedding channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_upsample: if True add downsample block.
        resblock_updown: if True use residual blocks for upsampling.
        num_head_channels: number of channels in each attention head.
        transformer_num_layers: number of layers of Transformer blocks to use.
        cross_attention_dim: number of context dimensions to use.
        upcast_attention: if True, upcast attention operations to full precision.
        dropout_cattn: if different from zero, this will be the dropout value for the cross-attention layers.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
        num_head_channels: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        dropout_cattn: float = 0.0,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            res_skip_channels = in_channels if (i == num_res_blocks - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            attentions.append(
                SpatialTransformer(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    num_attention_heads=out_channels // num_head_channels,
                    num_head_channels=num_head_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    num_layers=transformer_num_layers,
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

        self.upsampler: nn.Module | None
        if add_upsample:
            if resblock_updown:
                self.upsampler = DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,
                )
            else:

                post_conv = Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
                self.upsampler = WrappedUpsample(
                    spatial_dims=spatial_dims,
                    mode="nontrainable",
                    in_channels=out_channels,
                    out_channels=out_channels,
                    interp_mode="nearest",
                    scale_factor=2.0,
                    post_conv=post_conv,
                    align_corners=None,
                )
        else:
            self.upsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_list: list[torch.Tensor],
        temb: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:


        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_list[-1]
            res_hidden_states_list = res_hidden_states_list[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, context=context)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, temb)

        return hidden_states


def get_down_block(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    num_res_blocks: int,
    norm_num_groups: int,
    norm_eps: float,
    add_downsample: bool,
    resblock_updown: bool,
    with_attn: bool,
    with_cross_attn: bool,
    num_head_channels: int,
    transformer_num_layers: int,
    cross_attention_dim: int | None,
    upcast_attention: bool = False,
    dropout_cattn: float = 0.0,
    include_fc: bool = True,
    use_combined_linear: bool = False,
    use_flash_attention: bool = False,
) -> nn.Module:
    if with_attn:
        return AttnDownBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_downsample=add_downsample,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
    elif with_cross_attn:
        return CrossAttnDownBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_downsample=add_downsample,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            dropout_cattn=dropout_cattn,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
    else:
        return DownBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_downsample=add_downsample,
            resblock_updown=resblock_updown,
        )


def get_mid_block(
    spatial_dims: int,
    in_channels: int,
    temb_channels: int,
    norm_num_groups: int,
    norm_eps: float,
    with_conditioning: bool,
    num_head_channels: int,
    transformer_num_layers: int,
    cross_attention_dim: int | None,
    upcast_attention: bool = False,
    dropout_cattn: float = 0.0,
    include_fc: bool = True,
    use_combined_linear: bool = False,
    use_flash_attention: bool = False,
) -> nn.Module:
    if with_conditioning:
        return CrossAttnMidBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            num_head_channels=num_head_channels,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            dropout_cattn=dropout_cattn,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
    else:
        return AttnMidBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            num_head_channels=num_head_channels,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )


def get_up_block(
    spatial_dims: int,
    in_channels: int,
    prev_output_channel: int,
    out_channels: int,
    temb_channels: int,
    num_res_blocks: int,
    norm_num_groups: int,
    norm_eps: float,
    add_upsample: bool,
    resblock_updown: bool,
    with_attn: bool,
    with_cross_attn: bool,
    num_head_channels: int,
    transformer_num_layers: int,
    cross_attention_dim: int | None,
    upcast_attention: bool = False,
    dropout_cattn: float = 0.0,
    include_fc: bool = True,
    use_combined_linear: bool = False,
    use_flash_attention: bool = False,
) -> nn.Module:
    if with_attn:
        return AttnUpBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_upsample=add_upsample,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
    elif with_cross_attn:
        return CrossAttnUpBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_upsample=add_upsample,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            dropout_cattn=dropout_cattn,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
    else:
        return UpBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_upsample=add_upsample,
            resblock_updown=resblock_updown,
        )


class DiffusionModelUNet(nn.Module):
    """
    Unet network with timestep embedding and attention mechanisms for conditioning based on
    Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" https://arxiv.org/abs/2112.10752
    and Pinaya et al. "Brain Imaging Generation with Latent Diffusion Models" https://arxiv.org/abs/2209.07162

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see _ResnetBlock) per level.
        channels: tuple of block output channels.
        attention_levels: list of levels to add attention.
        norm_num_groups: number of groups for the normalization.
        norm_eps: epsilon for the normalization.
        resblock_updown: if True use residual blocks for up/downsampling.
        num_head_channels: number of channels in each attention head.
        with_conditioning: if True add spatial transformers to perform conditioning.
        transformer_num_layers: number of layers of Transformer blocks to use.
        cross_attention_dim: number of context dimensions to use.
        num_class_embeds: if specified (as an int), then this model will be class-conditional with `num_class_embeds`
            classes.
        upcast_attention: if True, upcast attention operations to full precision.
        dropout_cattn: if different from zero, this will be the dropout value for the cross-attention layers.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to True.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        resblock_updown: bool = False,
        num_head_channels: int | Sequence[int] = 8,
        with_conditioning: bool = False,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        num_class_embeds: int | None = None,
        upcast_attention: bool = False,
        dropout_cattn: float = 0.0,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        if with_conditioning is True and cross_attention_dim is None:
            raise ValueError(
                "DiffusionModelUNet expects dimension of the cross-attention conditioning (cross_attention_dim) "
                "when using with_conditioning."
            )
        if cross_attention_dim is not None and with_conditioning is False:
            raise ValueError(
                "DiffusionModelUNet expects with_conditioning=True when specifying the cross_attention_dim."
            )
        if dropout_cattn > 1.0 or dropout_cattn < 0.0:
            raise ValueError("Dropout cannot be negative or >1.0!")

        # All number of channels should be multiple of num_groups
        if any((out_channel % norm_num_groups) != 0 for out_channel in channels):
            raise ValueError("DiffusionModelUNet expects all num_channels being multiple of norm_num_groups")

        if len(channels) != len(attention_levels):
            raise ValueError("DiffusionModelUNet expects num_channels being same size of attention_levels")

        if isinstance(num_head_channels, int):
            num_head_channels = ensure_tuple_rep(num_head_channels, len(attention_levels))

        if len(num_head_channels) != len(attention_levels):
            raise ValueError(
                "num_head_channels should have the same length as attention_levels. For the i levels without attention,"
                " i.e. `attention_level[i]=False`, the num_head_channels[i] will be ignored."
            )

        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(channels))

        if len(num_res_blocks) != len(channels):
            raise ValueError(
                "`num_res_blocks` should be a single integer or a tuple of integers with the same length as "
                "`num_channels`."
            )

        self.in_channels = in_channels
        self.block_out_channels = channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.num_head_channels = num_head_channels
        self.with_conditioning = with_conditioning

        # input
        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        # time
        time_embed_dim = channels[0] * 4
        self.time_embed = nn.Sequential(
            nn.Linear(channels[0], time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
        )

        # class embedding
        self.num_class_embeds = num_class_embeds
        if num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)


        y_cond_dim = [64, 64, 128]
        self.y_embedder = nn.Sequential(  # For context, shrink 8 size use convolution
            nn.Conv3d(
                in_channels=1,  # e.g., 4 if using latent space from VAE
                out_channels=y_cond_dim[0],
                kernel_size=5,
                stride=2, padding=2
            ), nn.SiLU(),

            nn.Conv3d(
                in_channels=y_cond_dim[0],  # e.g., 4 if using latent space from VAE
                out_channels=y_cond_dim[1],
                kernel_size=5,
                stride=2, padding=2
            ), nn.SiLU(),  # 24

            nn.Conv3d(
                in_channels=y_cond_dim[1],  # e.g., 4 if using latent space from VAE
                out_channels=y_cond_dim[2],
                kernel_size=5,
                stride=2, padding=2
            ), nn.SiLU(),

        )


        # down
        self.down_blocks = nn.ModuleList([])
        output_channel = channels[0]
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_final_block = i == len(channels) - 1

            down_block = get_down_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                num_res_blocks=num_res_blocks[i],
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_downsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(attention_levels[i] and not with_conditioning),
                with_cross_attn=(attention_levels[i] and with_conditioning),
                num_head_channels=num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                dropout_cattn=dropout_cattn,
                include_fc=include_fc,
                use_combined_linear=use_combined_linear,
                use_flash_attention=use_flash_attention,
            )

            self.down_blocks.append(down_block)

        # mid
        self.middle_block = get_mid_block(
            spatial_dims=spatial_dims,
            in_channels=channels[-1],
            temb_channels=time_embed_dim,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            with_conditioning=with_conditioning,
            num_head_channels=num_head_channels[-1],
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            dropout_cattn=dropout_cattn,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )

        # up
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(channels))
        reversed_num_res_blocks = list(reversed(num_res_blocks))
        reversed_attention_levels = list(reversed(attention_levels))
        reversed_num_head_channels = list(reversed(num_head_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(channels) - 1)]

            is_final_block = i == len(channels) - 1

            up_block = get_up_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                prev_output_channel=prev_output_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                num_res_blocks=reversed_num_res_blocks[i] + 1,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_upsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(reversed_attention_levels[i] and not with_conditioning),
                with_cross_attn=(reversed_attention_levels[i] and with_conditioning),
                num_head_channels=reversed_num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                dropout_cattn=dropout_cattn,
                include_fc=include_fc,
                use_combined_linear=use_combined_linear,
                use_flash_attention=use_flash_attention,
            )

            self.up_blocks.append(up_block)

        # out
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=channels[0], eps=norm_eps, affine=True),
            nn.SiLU(),
            zero_module(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=channels[0],
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        down_block_additional_residuals: tuple[torch.Tensor] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor (N, C, SpatialDims).
            timesteps: timestep tensor (N,).
            context: context tensor (N, 1, ContextDim).
            class_labels: context tensor (N, ).
            down_block_additional_residuals: additional residual tensors for down blocks (N, C, FeatureMapsDims).
            mid_block_additional_residual: additional residual tensor for mid block (N, C, FeatureMapsDims).
        """
        # 1. time
        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0])

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embed(t_emb)

        # 2. class
        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            class_emb = self.class_embedding(class_labels)
            class_emb = class_emb.to(dtype=x.dtype)
            emb = emb + class_emb

        # 3. initial convolution
        h = self.conv_in(x)
        context = self.y_embedder(context) if context is not None else None
        # print("first h shape:", h.shape)  # h shape: torch.Size([1, 32, 21, 21, 21])
        # context: context tensor (N, 1, ContextDim).
        # print("context shape:", context.shape if context is not None else "None")  # context shape: torch.Size([1, 16, 21, 21, 21])


        # 4. down
        if context is not None and self.with_conditioning is False:
            raise ValueError("model should have with_conditioning = True if context is provided")
        
        down_block_res_samples: list[torch.Tensor] = [h]
        for downsample_block in self.down_blocks:
            h, res_samples = downsample_block(hidden_states=h, temb=emb, context=context)
            for residual in res_samples:
                down_block_res_samples.append(residual)

        # Additional residual conections for Controlnets
        if down_block_additional_residuals is not None:
            new_down_block_res_samples: list[torch.Tensor] = []
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += [down_block_res_sample]

            down_block_res_samples = new_down_block_res_samples

        # 5. mid
        h = self.middle_block(hidden_states=h, temb=emb, context=context)

        # Additional residual conections for Controlnets
        if mid_block_additional_residual is not None:
            h = h + mid_block_additional_residual

        # 6. up
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            h = upsample_block(hidden_states=h, res_hidden_states_list=res_samples, temb=emb, context=context)

        # 7. output block
        output: torch.Tensor = self.out(h)

        return output

    def load_old_state_dict(self, old_state_dict: dict, verbose=False) -> None:
        """
        Load a state dict from a DiffusionModelUNet trained with
        [MONAI Generative](https://github.com/Project-MONAI/GenerativeModels).

        Args:
            old_state_dict: state dict from the old DecoderOnlyTransformer  model.
        """

        new_state_dict = self.state_dict()
        # if all keys match, just load the state dict
        if all(k in new_state_dict for k in old_state_dict):
            print("All keys match, loading state dict.")
            self.load_state_dict(old_state_dict)
            return

        if verbose:
            # print all new_state_dict keys that are not in old_state_dict
            for k in new_state_dict:
                if k not in old_state_dict:
                    print(f"key {k} not found in old state dict")
            # and vice versa
            print("----------------------------------------------")
            for k in old_state_dict:
                if k not in new_state_dict:
                    print(f"key {k} not found in new state dict")

        # copy over all matching keys
        for k in new_state_dict:
            if k in old_state_dict:
                new_state_dict[k] = old_state_dict.pop(k)

        # fix the attention blocks
        attention_blocks = [k.replace(".attn.to_k.weight", "") for k in new_state_dict if "attn.to_k.weight" in k]
        for block in attention_blocks:
            new_state_dict[f"{block}.attn.to_q.weight"] = old_state_dict.pop(f"{block}.to_q.weight")
            new_state_dict[f"{block}.attn.to_k.weight"] = old_state_dict.pop(f"{block}.to_k.weight")
            new_state_dict[f"{block}.attn.to_v.weight"] = old_state_dict.pop(f"{block}.to_v.weight")
            new_state_dict[f"{block}.attn.to_q.bias"] = old_state_dict.pop(f"{block}.to_q.bias")
            new_state_dict[f"{block}.attn.to_k.bias"] = old_state_dict.pop(f"{block}.to_k.bias")
            new_state_dict[f"{block}.attn.to_v.bias"] = old_state_dict.pop(f"{block}.to_v.bias")

            # projection
            if f"{block}.attn.out_proj.weight" in new_state_dict and f"{block}.attn.out_proj.bias" in new_state_dict:
                new_state_dict[f"{block}.attn.out_proj.weight"] = old_state_dict.pop(f"{block}.proj_attn.weight")
                new_state_dict[f"{block}.attn.out_proj.bias"] = old_state_dict.pop(f"{block}.proj_attn.bias")
        # fix the cross attention blocks
        cross_attention_blocks = [
            k.replace(".out_proj.weight", "")
            for k in new_state_dict
            if "out_proj.weight" in k and "transformer_blocks" in k
        ]
        for block in cross_attention_blocks:
            new_state_dict[f"{block}.out_proj.weight"] = old_state_dict.pop(f"{block}.to_out.0.weight")
            new_state_dict[f"{block}.out_proj.bias"] = old_state_dict.pop(f"{block}.to_out.0.bias")

        # fix the upsample conv blocks which were renamed postconv
        for k in new_state_dict:
            if "postconv" in k:
                old_name = k.replace("postconv", "conv")
                new_state_dict[k] = old_state_dict.pop(old_name)
        if verbose:
            # print all remaining keys in old_state_dict
            print("remaining keys in old_state_dict:", old_state_dict.keys())
        self.load_state_dict(new_state_dict)


class DiffusionModelEncoder(nn.Module):
    """
    Classification Network based on the Encoder of the Diffusion Model, followed by fully connected layers. This network is based on
    Wolleb et al. "Diffusion Models for Medical Anomaly Detection" (https://arxiv.org/abs/2203.04306).

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see _ResnetBlock) per level.
        channels: tuple of block output channels.
        attention_levels: list of levels to add attention.
        norm_num_groups: number of groups for the normalization.
        norm_eps: epsilon for the normalization.
        resblock_updown: if True use residual blocks for downsampling.
        num_head_channels: number of channels in each attention head.
        with_conditioning: if True add spatial transformers to perform conditioning.
        transformer_num_layers: number of layers of Transformer blocks to use.
        cross_attention_dim: number of context dimensions to use.
        num_class_embeds: if specified (as an int), then this model will be class-conditional with `num_class_embeds` classes.
        upcast_attention: if True, upcast attention operations to full precision.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        resblock_updown: bool = False,
        num_head_channels: int | Sequence[int] = 8,
        with_conditioning: bool = False,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        num_class_embeds: int | None = None,
        upcast_attention: bool = False,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        if with_conditioning is True and cross_attention_dim is None:
            raise ValueError(
                "DiffusionModelEncoder expects dimension of the cross-attention conditioning (cross_attention_dim) "
                "when using with_conditioning."
            )
        if cross_attention_dim is not None and with_conditioning is False:
            raise ValueError(
                "DiffusionModelEncoder expects with_conditioning=True when specifying the cross_attention_dim."
            )

        # All number of channels should be multiple of num_groups
        if any((out_channel % norm_num_groups) != 0 for out_channel in channels):
            raise ValueError("DiffusionModelEncoder expects all num_channels being multiple of norm_num_groups")
        if len(channels) != len(attention_levels):
            raise ValueError("DiffusionModelEncoder expects num_channels being same size of attention_levels")

        if isinstance(num_head_channels, int):
            num_head_channels = ensure_tuple_rep(num_head_channels, len(attention_levels))

        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(channels))

        if len(num_head_channels) != len(attention_levels):
            raise ValueError(
                "num_head_channels should have the same length as attention_levels. For the i levels without attention,"
                " i.e. `attention_level[i]=False`, the num_head_channels[i] will be ignored."
            )

        self.in_channels = in_channels
        self.block_out_channels = channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.num_head_channels = num_head_channels
        self.with_conditioning = with_conditioning

        # input
        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        # time
        time_embed_dim = channels[0] * 4
        self.time_embed = nn.Sequential(
            nn.Linear(channels[0], time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
        )

        # class embedding
        self.num_class_embeds = num_class_embeds
        if num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)

        # down
        self.down_blocks = nn.ModuleList([])
        output_channel = channels[0]
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_final_block = i == len(channels)  # - 1

            down_block = get_down_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                num_res_blocks=num_res_blocks[i],
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_downsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(attention_levels[i] and not with_conditioning),
                with_cross_attn=(attention_levels[i] and with_conditioning),
                num_head_channels=num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                include_fc=include_fc,
                use_combined_linear=use_combined_linear,
                use_flash_attention=use_flash_attention,
            )

            self.down_blocks.append(down_block)

        self.out = nn.Sequential(nn.Linear(4096, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, self.out_channels))

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor (N, C, SpatialDims).
            timesteps: timestep tensor (N,).
            context: context tensor (N, 1, ContextDim).
            class_labels: context tensor (N, ).
        """
        # 1. time
        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0])

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embed(t_emb)

        # 2. class
        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            class_emb = self.class_embedding(class_labels)
            class_emb = class_emb.to(dtype=x.dtype)
            emb = emb + class_emb

        # 3. initial convolution
        h = self.conv_in(x)

        # 4. down
        if context is not None and self.with_conditioning is False:
            raise ValueError("model should have with_conditioning = True if context is provided")
        for downsample_block in self.down_blocks:
            h, _ = downsample_block(hidden_states=h, temb=emb, context=context)

        h = h.reshape(h.shape[0], -1)
        output: torch.Tensor = self.out(h)

        return output
