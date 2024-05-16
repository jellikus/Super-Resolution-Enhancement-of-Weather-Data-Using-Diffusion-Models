import torch
import numpy as np
from tqdm import tqdm
from ..nn_modules.functional_layers import default
from ..diffusion import GaussianDiffusion


class SR3Diffusion(GaussianDiffusion):
    """
    A class for Gaussian diffusion model.
    Args:
        denoise_fn (Callable): denoising neural network.
        channels (int): The number of channels in input images.
        loss_type (str): The type of loss function to use ('l1' or 'l2').
        conditional (bool): If True, the model expects conditional inputs.
        schedule_opt (dict, optional): Configuration for the noise schedule, containing keys like
                                       'schedule', 'n_timestep', 'linear_start', and 'linear_end'.
        image_height (int): The height of images that will be processed.
        image_width (int): The width of images that will be processed.
        pretrained_model_path (str, optional): Path to a pretrained model to load weights from.
        lock_weights (bool): If True, weights are frozen and not updated during training.
    """

    def __init__(
            self,
            denoise_fn,
            channels=1,
            image_height=128,
            image_width=256,
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
        Perform a loop of T sampling steps, sampling from p(x_{t-1} | x_t) starting with gaussian noise.

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
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                          total=self.num_timesteps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                          total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x)
                if i % sample_inter == 0:
                    ret_img = img

        return ret_img

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
        Calculate the training loss for a single step of the reverse diffusion process.

        Args:
            x_in: Dictionary containing input images and potentially other conditioning information.
            noise: Optional external noise to add for the diffusion process.

        Returns:
            The calculated loss for the diffusion to optimize.
        """
        x_start = x_in['HR']
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
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        loss = self.loss_func(noise, x_recon)
        print("loss:", loss)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
