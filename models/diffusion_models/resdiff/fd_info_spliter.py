from torch import nn
import torch


class FD_Info_Spliter(nn.Module):
    """
    Implementation of FD_Info_Spliter block in ResDiff model

    Args:
        dim (int): Number of input channels.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        image_height (int): Height of the input images.
        image_width (int): Width of the input images.
    """

    def __init__(self, dim, in_channels, out_channels, image_height=128, image_width=128):
        super().__init__()

        self.in_channels = in_channels
        self.dim = dim
        self.image_height = image_height
        self.image_width = image_width
        self.noise_func = nn.Linear(dim, self.image_width)
        reduction = 2
        if in_channels == 1:
            reduction = 1

        self.noise_resSE = ResSE(in_channels, reduction=reduction)
        self.sigma_resSE = ResSE(in_channels * 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.HF_guided_resSE = ResSE(in_channels * 2)

        # changed (for any number of channels)
        self.channel_transform = nn.Conv2d(in_channels * 2, out_channels, 1)

    def forward(self, x, noise_embed):
        cnn_x, x = torch.split(x, self.in_channels, dim=1)

        assert x.shape == cnn_x.shape
        # Noise image suppression

        b, c, h, w = x.shape
        noise_embed = self.noise_func(noise_embed.view(b, -1))
        noise_embed = noise_embed.unsqueeze(1).unsqueeze(2).repeat(1, self.in_channels, self.image_height, 1)
        noise_atten = self.noise_resSE(noise_embed)
        denoise_x = x * noise_atten

        # High and low frequency information separation
        n, m = x.shape[-2:]
        device = x.device
        # create frequency grid

        xx = torch.arange(n, dtype=torch.float, device=device)
        yy = torch.arange(m, dtype=torch.float, device=device)
        u, v = torch.meshgrid(xx, yy)
        u = u - n / 2
        v = v - m / 2

        # convert tensor to complex tensor and apply FFT
        tensor_complex = torch.stack([cnn_x, torch.zeros_like(cnn_x)], dim=-1)
        tensor_complex = torch.view_as_complex(tensor_complex)
        tensor_fft = torch.fft.fftn(tensor_complex)

        # Concat the real and imaginary parts
        x_real, x_imag = torch.real(tensor_fft), torch.imag(tensor_fft)
        x_fd = torch.cat([x_real, x_imag], dim=1)

        # get sigma, numerical stabilization was performed
        l = min(self.image_height, self.image_width)
        sigma_pre = torch.abs(torch.mean(self.avg_pool(self.sigma_resSE(x_fd)), dim=1)) + l / 2
        sigma_min = torch.tensor(l - 10, device=device).view(1, 1, 1).expand_as(sigma_pre)
        sigma = torch.minimum(sigma_pre, sigma_min)

        # calculate Gaussian high-pass filter
        D = torch.sqrt(u ** 2 + v ** 2).to(device)
        H = 1 - torch.exp(-D ** 2 / (2 * sigma ** 2))
        H = H.to(device).unsqueeze(1)

        # changed (for any number of channels)
        if self.in_channels > 1:
            H = torch.cat([H] * self.in_channels, dim=1)

        # apply Gaussian high-pass filter to FFT
        tensor_filtered_fft = tensor_fft * H

        # get Frequency-domain guided attention weight,thus obtain Low-frequency feature map
        x_real_filterd, x_imag_filterd = torch.real(tensor_filtered_fft), torch.imag(tensor_filtered_fft)
        x_fd_filterd = torch.cat([x_real_filterd, x_imag_filterd], dim=1)
        x_hf_guided_atten = self.HF_guided_resSE(x_fd_filterd)

        # 1d conv
        x_lf_feature = cnn_x * self.channel_transform(x_hf_guided_atten)
        # IFFT，get High-frequency feature map
        tensor_filtered = torch.fft.ifftn(tensor_filtered_fft)
        x_hf_feature = torch.abs(tensor_filtered)

        # # plot convolved tensors
        # convolved_tensors = []
        # padded_tensor = F.pad(cnn_x, (1, 1, 1, 1), mode='reflect')
        #
        # for kernel in [kernel_x, kernel_y, kernel_xy]:
        #     convolved_tensors.append(F.conv2d(padded_tensor, kernel))

        import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        #
        # # Original Image
        # axes[0].imshow(cnn_x[0, 0].cpu().numpy(), cmap='plasma')
        # axes[0].set_title('Original Image')
        #
        # # Convolved Image
        # axes[1].imshow(convolved_tensors[0][0, 0].detach().cpu().numpy(), cmap='plasma')
        # axes[1].set_title('Convolved Image')
        # plt.show()
        # return torch.cat([x, cnn_x, denoise_x, x_lf_feature, x_hf_feature, *convolved_tensors], dim=1)
        return torch.cat([x, cnn_x, denoise_x, x_lf_feature, x_hf_feature], dim=1)


class ResSE(nn.Module):
    """
    implementation of ResSE block in ResDiff model

    Args:
        ch_in (int): Number of input channels.
        reduction (int): Reduction factor

    Returns:
        torch.Tensor: Output tensor
    """

    def __init__(self, ch_in, reduction=2):
        super(ResSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        tmp = x
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        tmp = x * y.expand_as(x) + tmp
        return tmp

