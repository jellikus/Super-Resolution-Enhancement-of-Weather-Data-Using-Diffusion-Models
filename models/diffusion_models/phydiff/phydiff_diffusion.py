import torch
import numpy as np
from ..diffusion import GaussianDiffusion
from ..nn_modules.functional_layers import default


class PhyDiffDiffusion(GaussianDiffusion):
    """
    A class for ResDiff+Physics diffusion model.
    Args:
        denoise_fn (Callable): denoising neural network.
        image_height (int): The height of images that will be processed.
        image_width (int): The width of images that will be processed.
        channels (int): The number of channels in input images.
        loss_type (str): The type of loss function to use ('l1' or 'l2').
        conditional (bool): If True, the model expects conditional inputs.
        schedule_opt (dict, optional): Configuration for the noise schedule, containing
        keys like 'schedule', 'n_timestep', 'linear_start', and 'linear_end'.
        pretrained_model_path (str, optional): Path to a pretrained model to load weights from.
        lock_weights (bool): If True, weights are frozen and not updated during training.

    """

    def __init__(
            self,
            denoise_fn,
            image_height,
            image_width,
            channels=3,
            loss_type='l1',
            conditional=True,
            schedule_opt=None,
            pretrained_model_path=None,
            lock_weights=True,
    ):
        super().__init__(
            denoise_fn=denoise_fn,
            channels=channels,
            loss_type=loss_type,
            conditional=conditional,
            schedule_opt=schedule_opt,
            image_height=image_height,
            image_width=image_width,
            pretrained_model_path=pretrained_model_path,
            lock_weights=lock_weights,
        )


    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        """
        Implementation of ResDiff sampling algorithm.
        Args:
            x_in: Shape of the tensor to be sampled or initial noisy image for conditional models.
            continous: Not used for effciency.

        Returns:
            Generated samples after the complete reverse diffusion process.
        """
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps // 10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in reversed(range(0, self.num_timesteps)):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x
            for i in reversed(range(0, self.num_timesteps)):
                img = self.p_sample(img, i, condition_x=x)
                if i % sample_inter == 0:
                    # ret_img = torch.cat([ret_img, img], dim=0)
                    ret_img = img

        return ret_img + x_in

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        """
        Perform super-resolution on the given low-resolution input using the trained diffusion.

        Args:
            x_in: Input dictionary containing low-resolution (and potentially high-resolution) images for reference.
            continous (bool): Boolean indicating whether to output samples continuously during the process.

        Returns:
            Super-resolved images.
        """
        return self.p_sample_loop(x_in["SR"], continous)

    def p_losses(self, x_in, noise=None):
        """
        Calculate the training loss for a single step of the reverse diffusion process for ResDiff+Physics model
        including denoise fucntion utilization withoud FD info Spliter.

        Args:
            x_in: Dictionary containing input images and potentially other conditioning information.
            noise: Optional external noise to add for the diffusion process.

        Returns:
            The calculated loss for the diffusion to optimize.
        """

        cnn_prediction = x_in['SR']
        x_start = x_in['HR'] - cnn_prediction
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start,
            continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            # x_recon, moments = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            # x_recon, moments = self.denoise_fn(torch.cat([cnn_prediction, x_noisy], dim=1), continuous_sqrt_alpha_cumprod)
            x_recon = self.denoise_fn(torch.cat([cnn_prediction, x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        noise_loss = self.loss_func(noise, x_recon)
        # moment_loss = self.moment_loss_func(moments, self.denoise_fn.phy_conv.constraints)

        return noise_loss
