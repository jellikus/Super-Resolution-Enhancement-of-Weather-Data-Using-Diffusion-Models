"""
source: https://github.com/LeiaLi/SRDiff/blob/56285083a2d64ed321c249e6ef5d6e745a5d817e/models/diffsr_modules.py
"""

import functools
import torch
from torch import nn
import torch.nn.functional as F


class RRDBNet(nn.Module):
    """
    RRDBNet: Residual in Residual Dense Block Network.

    Args:
        in_nc (int): Number of input channels.
        out_nc (int): Number of output channels.
        nf (int): Number of features.
        nb (int): Number of RRDB blocks.
        gc (int): Growth channels (intermediate channels).
    """
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, get_fea=False):
        feas = []
        x = (x + 1) / 2
        fea_first = fea = self.conv_first(x)
        for l in self.RRDB_trunk:
            fea = l(fea)
            feas.append(fea)
        trunk = self.trunk_conv(fea)
        fea = fea_first + trunk
        feas.append(fea)

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))

        fea_hr = self.HRconv(fea)
        out = self.conv_last(self.lrelu(fea_hr))
        out = out.clamp(0, 1)
        out = out * 2 - 1
        if get_fea:
            return out, feas
        else:
            return out


def make_layer(block, n_layers, seq=False):
    """
    Create a sequence of blocks.

    Args:
        block (callable): A callable that creates a block (e.g., nn.Module).
        n_layers (int): Number of layers (blocks) to create.
        seq (bool): If True, return a nn.Sequential container for the blocks.

    Returns:
        nn.ModuleList or nn.Sequential: List or sequence of created blocks.
    """
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    if seq:
        return nn.Sequential(*layers)
    else:
        return nn.ModuleList(layers)


class ResidualDenseBlock_5C(nn.Module):
    """
    Residual Dense Block with 5 Convolutional Layers.

    Args:
        nf (int): Number of input features.
        gc (int): Growth channels (intermediate channels).
        bias (bool): Whether to use bias in convolutional layers.
    """
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block (RRDB).

    Args:
        nf (int): Number of input features.
        gc (int): Growth channels (intermediate channels).
    """

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


if __name__ == '__main__':
    image_channels_o = 1
    batch = 17
    height = 32
    width = 64
    img = torch.randn(batch, image_channels_o, height, width).to('cuda')

    hidden_size = 64
    num_block = 17
    net = RRDBNet(image_channels_o, image_channels_o, hidden_size, num_block, hidden_size // 2).to('cuda')
    rrdb_sr, rrdb_encoded = net(img, True)

    print("rrdb_sr shape: ", rrdb_sr.shape)
    print("rrdb_encoded shape: ", rrdb_encoded.shape)
