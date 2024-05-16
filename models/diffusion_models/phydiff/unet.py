import torch
from torch import nn
import pytorch_wavelets as pw
import torch.nn.functional as F
from models.diffusion_models.phydiff.constrain_moments import K2M
from ..resdiff.guided_cross_attention import HF_guided_CA
from ..nn_modules.functional_layers import PositionalEncoding, Swish, Upsample, Downsample, default, exists
from ..nn_modules.resnet import Block, ResnetBlocWithAttn


class PhyConv(nn.Module):
    """
    Physically Interpretable Convolutional Layer.

    This module implements a convolutional layer with physically interpretable filters.
    The kernels can be randomly initialized or initialized with specified values.

    Args:
        n_filters (int): Number of convolutional filters (kernels).
        kernel_size (int): Size of the square convolutional kernels (e.g., 3 for 3x3 kernels).
        device (str): Device to use for computation ('cpu' or 'cuda').
        init_values (list of list of float, optional): Initial values for kernels. Defaults to None.
        in_channels (int, optional): Number of input channels. Defaults to 1.
        J (int, optional): Number of down-sampling stages. Defaults to 4.
    """
    def __init__(self, n_filters, kernel_size, device, init_values=None, in_channels=1, J=4):
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.device = device
        self.in_channels = in_channels
        self.conv_1x1 = nn.Conv2d(n_filters, 1, 1)

        if init_values is None:
            self.kernels = self._init_kernels_random()
        else:
            self.kernels = self._init_kernels(init_values)

        self.constraints = self.get_constraints(self.device)
        self.J = J

    def forward(self, x):
        # get img wihout noise
        x, _ = torch.split(x, self.in_channels, dim=1)
        convolved_tensor = self.convolve_tensor(x, self.kernels, self.in_channels)

        convolved_layers = []
        for i in range(self.J):
            x_downsampled = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

            # Apply convolutions using learned kernels
            convolved_tensor = self.convolve_tensor(x_downsampled, self.kernels, self.in_channels)
            convolved_layers.append(convolved_tensor)

            x = x_downsampled

        convolved_tensor = self.conv_1x1(convolved_tensor)

        # Compute moments
        m = self.get_moments(self.kernels)

        # print("m shape: ", m.shape)
        # print("convolved_tensor shape: ", convolved_tensor.shape)

        return convolved_tensor, m

    def convolve_tensor(self, input_tensor: torch.Tensor, kernels, in_channels: int) -> torch.Tensor:
        """
        Convolve input tensor with learned kernels.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            kernels (torch.Tensor): Learned kernels of shape (1, 1, kernel_size, kernel_size).
            in_channels (int): Number of input channels.
        Returns:
            torch.Tensor: Output tensor after convolution.
        """
        kernel_size = kernels.size(2)
        padding = (kernel_size - 1) // 2

        padded_tensor = F.pad(input_tensor, (padding, padding, padding, padding), mode='reflect')

        convolved_tensors = []

        for learned_kernel in kernels:
            kernel = learned_kernel.unsqueeze(dim=0).detach()
            kernel = torch.cat([kernel] * in_channels, dim=1)
            convolved_tensors.append(F.conv2d(padded_tensor, kernel))

        return torch.cat(convolved_tensors, dim=1)

    def _init_kernels_random(self) -> nn.Parameter:
        """
        Initialize kernels randomly.

        Returns:
            nn.Parameter: Parameter containing randomly initialized kernels.
        """

        # Concatenate randomly initialized kernels into a single tensor
        tensors = [torch.randn(1, 1, self.kernel_size, self.kernel_size, device=self.device) for _ in
                   range(self.n_filters)]
        combined_tensor = torch.cat(tensors, dim=0)
        return nn.Parameter(combined_tensor, requires_grad=True)

    def get_moments(self, kernels):
        """
        Compute moments of the learned kernel matrix.

        Args:
            kernels (torch.Tensor): Learned kernels.

        Returns:
            torch.Tensor: Moments of the kernels.
        """
        filter_size = self.kernel_size
        k2m = K2M((filter_size, filter_size)).to(kernels.device)
        moments = k2m(kernels.double())
        moments = moments.float()
        return moments

    def get_constraints(self, device: str) -> torch.Tensor:
        """
        Create constraints for the kernels.

        Returns:
            nn.Parameter: Parameter containing constraints for the kernels.
        """
        constraints = torch.zeros((self.n_filters, self.kernel_size, self.kernel_size), device=device)
        ind = 0
        for i in range(0, self.kernel_size):
            for j in range(0, self.kernel_size):
                if ind < self.n_filters:
                    constraints[ind, i, j] = 1
                    ind += 1
        constraints = constraints.unsqueeze(dim=1)
        return constraints


class UNet(nn.Module):
    """
    U-Net architecture for Resdiff+Physics model.

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
            device='cuda'
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None
        self.image_channels = image_channels

        kernel_x = torch.tensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=torch.float, device=device).view(1, 1, 3, 3)
        self.kernel_x = torch.cat([kernel_x] * image_channels, dim=1)

        kernel_y = torch.tensor([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=torch.float, device=device).view(1, 1, 3, 3)
        self.kernel_y = torch.cat([kernel_y] * image_channels, dim=1)

        kernel_xy = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float, device=device).view(1, 1, 3, 3)
        self.kernel_xy = torch.cat([kernel_xy] * image_channels, dim=1)

        self.wavelet_components = 3
        self.J = 4

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        self.image_height = image_height
        self.image_width = image_width
        # not sure why height and not width
        now_res = image_height
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]

        self.hf_ca_list = []

        for i in range(self.J):
            self.hf_ca_list.append(HF_guided_CA(inner_channel * (2 ** i), image_channels=self.image_channels,
                                                wavelet_components=self.wavelet_components))
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

    def forward(self, x, time_emb):
        residual_cond, _ = torch.split(x, self.image_channels, dim=1)

        J = self.J
        dwt_img_list = []
        dwt_f = pw.DWTForward(J=J, wave='haar', mode='symmetric')
        dwt_f.cuda()
        x_dwt = dwt_f(residual_cond)[1]

        # ------------------------------------------------------------------------------------------------------------------------------------
        # for i in range(J):
        #     dwt_img_list.append(x_dwt[i][:, :, 0, :, :] + x_dwt[i][:, :, 1, :, :] + x_dwt[i][:, :, 2, :, :])
        # ------------------------------------------------------------------------------------------------------------------------------------
        #
        for i in range(J):
            dwt_img_list.append(
                torch.cat([x_dwt[i][:, :, 0, :, :], x_dwt[i][:, :, 1, :, :], x_dwt[i][:, :, 2, :, :]], dim=1))

        # ------------------------------------------------------------------------------------------------------------------------------------
        # device = x.device
        # kernel = torch.tensor([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]])
        # kernel_x = torch.tensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=torch.float).to(device)
        # kernel_x = torch.cat([kernel_x] * self.image_channels, dim=0).view(1, self.image_channels, 3, 3)
        # kernel_y = torch.tensor([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=torch.float).to(device)
        # kernel_y = torch.cat([kernel_y] * self.image_channels, dim=0).view(1, self.image_channels, 3, 3)
        # kernel_xy = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float).to(device)
        # kernel_xy = torch.cat([kernel_xy] * self.image_channels, dim=0).view(1, self.image_channels, 3, 3)
        #
        # dwt_x_copy = dwt_x.clone()
        #
        # for i in range(J):
        #     dwt_x_downsampled = F.interpolate(dwt_x_copy, scale_factor=0.5, mode='bilinear', align_corners=False)
        #     padded_tensor = F.pad(dwt_x_downsampled, (1, 1, 1, 1), mode='reflect')
        #     # Save convolved images
        #     convolved_tensors = [F.conv2d(padded_tensor, kernel_x),
        #                          F.conv2d(padded_tensor, kernel_y),
        #                          F.conv2d(padded_tensor, kernel_xy)]
        #     #print("convolved tensors:", convolved_tensors[0].shape)
        #     dwt_x_copy = dwt_x_downsampled
        #     dwt_img_list.append(torch.cat([x_dwt[i][:, :, 0, :, :], x_dwt[i][:, :, 1, :, :], x_dwt[i][:, :, 2, :, :],
        #                                    F.conv2d(padded_tensor, kernel_x), F.conv2d(padded_tensor, kernel_y),
        #                                    F.conv2d(padded_tensor, kernel_xy)], dim=1))
        #     #print(dwt_img_list[i].shape)
        # layers_images, moments = self.phy_conv(x)
        # moments = None
        # dwt_img_list = layers_images
        # for i in range(J):
        #     dwt_img_list[i] = torch.cat([dwt_img_list[i], layers_images[i]], dim=1)
        # -------------------------------------------------------------------------------------------------------------------------------------
        # convolved_tensor, moments = self.phy_conv(x)

        padded_tensor = F.pad(residual_cond, (1, 1, 1, 1), mode='reflect')

        x = torch.cat([x, F.conv2d(padded_tensor, self.kernel_x), F.conv2d(padded_tensor, self.kernel_y),
                       F.conv2d(padded_tensor, self.kernel_xy)], dim=1)
        t = self.noise_level_mlp(time_emb) if exists(
            self.noise_level_mlp) else None

        skips = []
        idx = 0
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            if len(skips) != 0 and skips[-1].shape[2:] != x.shape[2:]:
                hf_ca = self.hf_ca_list[idx]
                idx += 1
                query = dwt_img_list.pop(0)
                hdd = hf_ca(x, query)
                skips.append(hdd)
            else:
                skips.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, skips.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)


if __name__ == '__main__':
    image_channels_o = 1
    batch = 7
    height = 128
    width = 256
    img = torch.randn(batch, image_channels_o * 2, height, width).to('cuda')
    t = torch.tensor([0.645] * batch).to('cuda')
    net = UNet(in_channel=image_channels_o * 5, out_channel=image_channels_o, image_channels=image_channels_o,
               norm_groups=32, inner_channel=64, channel_mults=[1, 2, 4, 8, 8], attn_res=[16, ], res_blocks=2,
               dropout=0.2, image_height=height, image_width=width).to('cuda')
    y = net(img, t)
    print(y.shape)
