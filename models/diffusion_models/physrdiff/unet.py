
import torch
from torch import nn
import pytorch_wavelets as pw
from ..nn_modules.functional_layers import default, exists, PositionalEncoding, Mish, Upsample, Downsample
from ..nn_modules.resnet import Block, ResnetBlocWithAttn
from ..resdiff.guided_cross_attention import HF_guided_CA


class UNet(nn.Module):
    """
    U-Net architecture for SRDiff+Physics model.

    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        inner_channel (int): Number of channels in the inner layers.
        norm_groups (int): Number of groups for group normalization.
        channel_mults (tuple): Multipliers for the number of channels throughout the U-Net levels.
        attn_res (tuple): Resolutions at which to apply self-attention.
        res_blocks (int): Number of residual blocks per level.
        dropout (float): Dropout rate.
        with_noise_level_emb (bool): Flag to include positional encoding for noise level embedding.
        image_width (int): Width of the input images.
        image_height (int): Height of the input images.
        image_channels (int): Number of channels in the input images.
    """
    def __init__(
            self,
            in_channel=9,
            out_channel=3,
            inner_channel=32,
            norm_groups=32,
            channel_mults=(1, 2, 4, 8, 8),
            attn_res=(8,),
            res_blocks=3,
            dropout=0,
            with_noise_level_emb=True,
            image_width=128,
            image_height=128,
            image_channels=1,
    ):
        super().__init__()

        device = 'cuda'
        self.hidden_size = 64
        self.num_block = 17
        self.cond_proj = nn.ConvTranspose2d(self.hidden_size * ((self.num_block + 1) // 3),
                                            self.hidden_size, 4 * 2, 4,
                                            4 // 2)

        kernel_x = torch.tensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=torch.float, device=device).view(1, 1, 3, 3)
        self.kernel_x = torch.cat([kernel_x] * image_channels, dim=1)

        kernel_y = torch.tensor([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=torch.float, device=device).view(1, 1, 3, 3)
        self.kernel_y = torch.cat([kernel_y] * image_channels, dim=1)

        kernel_xy = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float, device=device).view(1, 1, 3, 3)
        self.kernel_xy = torch.cat([kernel_xy] * image_channels, dim=1)

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Mish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None
        self.image_channels = image_channels

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        self.image_height = image_height
        self.image_width = image_width
        now_res = image_height
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]

        self.hf_ca_list = []

        self.J = 4
        for i in range(self.J):
            self.hf_ca_list.append(HF_guided_CA(inner_channel * (2 ** i), image_channels=self.image_channels, wavelet_components = 3))
        self.hf_ca_list = nn.ModuleList(self.hf_ca_list)

        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel + feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time):

        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        interpolated_img, rrdb_encoded, x = x
        cond = self.cond_proj(torch.cat(rrdb_encoded[2::3], 1))

        dwt_x = interpolated_img

        J = self.J
        dwt_img_list = []
        dwt_f = pw.DWTForward(J=J, wave='haar', mode='symmetric')
        dwt_f.cuda()
        x_dwt = dwt_f(dwt_x)[1]
        for i in range(J):
            dwt_img_list.append(x_dwt[i][:, :, 0, :, :] + x_dwt[i][:, :, 1, :, :] + x_dwt[i][:, :, 2, :, :])

        x = self.fd_spliter(torch.cat([interpolated_img, x], dim=1), t)
        feats = []
        idx = 0
        for i, layer in enumerate(self.downs):
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            if i == 2:
                x = x + cond
            if len(feats) != 0 and feats[-1].shape[2:] != x.shape[2:]:
                hf_ca = self.hf_ca_list[idx]
                idx += 1
                query = dwt_img_list.pop(0)
                hdd = hf_ca(x, query)
                feats.append(hdd)
            else:
                feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)



if __name__ == '__main__':
    image_channels_o = 1
    batch = 8
    height = 128
    width = 256
    img = [torch.randn(batch, image_channels_o, height, width).to('cuda'), torch.randn(batch, image_channels_o, height, width).to('cuda')]
    t = torch.tensor([0.645] * batch).to('cuda')
    net = UNet(in_channel=image_channels_o * 1, out_channel=image_channels_o, image_channels=image_channels_o,
               norm_groups=32, inner_channel=64, channel_mults=[1, 2, 4, 8, 8], attn_res=[16, ], res_blocks=2,
               dropout=0.2, image_height=height, image_width=width).to('cuda')
    y = net(img, t)
    print(y.shape)
