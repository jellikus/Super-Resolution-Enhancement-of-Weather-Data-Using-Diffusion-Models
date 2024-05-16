from torch import nn
import torch
import math


class HF_guided_CA(nn.Module):
    """
    Implementation of HF_guided_CA block in ResDiff model

    Args:
        in_channel (int): Number of input channels.
        norm_groups (int): Number of groups for group normalization.
        image_channels (int): Number of channels in the input images.
        wavelet_components (int): Number of wavelet components.
    """
    def __init__(self, in_channel, norm_groups=32, image_channels=3, wavelet_components=1):
        super().__init__()

        self.norm = nn.GroupNorm(norm_groups, in_channel).to('cuda')
        self.q = nn.Conv2d(image_channels * wavelet_components, in_channel, 1, bias=False)
        self.kv = nn.Conv2d(in_channel, in_channel * 2, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input, quary):
        batch, channel, height, width = input.shape
        head_dim = channel

        norm = self.norm(input)

        kv = self.kv(norm).view(batch, 1, head_dim * 2, height, width)
        key, value = kv.chunk(2, dim=2)  # bhdyx
        quary = self.q(quary).unsqueeze(1)

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", quary, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, 1, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, 1, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input

