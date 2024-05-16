import math
from torch import nn
import torch
from .functional_layers import Swish


class Block(nn.Module):
    """
    Basic convolutional block with GroupNorm, optional dropout, and Swish activation.

    Args:
        dim (int): Number of input channels to the block.
        dim_out (int): Number of output channels from the block.
        groups (int): Number of groups for the GroupNorm.
        dropout (float): Dropout rate; if 0, dropout is not applied.
    """

    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    """
    Residual block that integrates feature-wise affine transformations with an optional noise embedding.

    Args:
        dim (int): Number of input channels to the block.
        dim_out (int): Number of output channels from the block.
        noise_level_emb_dim (int, optional): Dimension of the noise level embedding.
        dropout (float): Dropout rate.
        use_affine_level (bool): Whether to use affine transformations conditioned on noise level.
        norm_groups (int): Number of groups for the GroupNorm.
    """

    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    """
    Self-attention module for convolutional networks.

    Args:
        in_channel (int): Number of channels in the input tensor.
        n_head (int): Number of attention heads.
        norm_groups (int): Number of groups for the GroupNorm.
    """

    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    """
    Residual block with optional self-attention.

    Args:
        dim (int): Number of input channels to the block.
        dim_out (int): Number of output channels from the block.
        noise_level_emb_dim (int, optional): Dimension of the noise level embedding for feature-wise affine transformations.
        norm_groups (int): Number of groups for the GroupNorm.
        dropout (float): Dropout rate.
        with_attn (bool): Flag to enable self-attention within the block.
    """

    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if self.with_attn:
            x = self.attn(x)
        return x


class FeatureWiseAffine(nn.Module):
    """
    Feature-wise affine transformation for conditional instance normalization. This module applies a learned affine
    transformation to the input features, which can be conditioned on external data like noise levels.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after the transformation.
        use_affine_level (bool): Flag to use separate parameters for scaling and shifting.
    """

    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x
