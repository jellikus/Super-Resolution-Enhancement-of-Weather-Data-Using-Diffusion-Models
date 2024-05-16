import math
import torch
from torch import nn
from inspect import isfunction
import torch.nn.functional as F

def exists(x):
    """Check if a variable is not None."""
    return x is not None


def default(val, d):
    """
    Return the value 'val' if it exists, otherwise return 'd'.
    If 'd' is a function, it calls it to get the default value.
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d

class PositionalEncoding(nn.Module):
    """
    Positional Encoding module that encodes a noise level into a sinusoidal positional encoding.
    Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py

    Args:
        dim (int): The dimensionality of the encoding to generate.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class Swish(nn.Module):
    """Swish activation function."""
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(nn.Module):
    """Mish activation function."""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Upsample(nn.Module):
    """
    Upsampling module that doubles the spatial dimensions using nearest neighbor interpolation followed by a convolution.

    Args:
        dim (int): The number of channels in the input tensor to be upsampled.
    """
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    """
    Downsampling module that reduces the spatial dimensions by half using a strided convolution.

    Args:
        dim (int): The number of channels in the input tensor to be downsampled.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)
