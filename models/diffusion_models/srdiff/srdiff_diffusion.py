import torch
import numpy as np
from models.rrdb_encoder.RRDBNet import RRDBNet
import torch.nn.functional as F
from ..nn_modules.functional_layers import default
from ..diffusion import GaussianDiffusion


class SRDiffDiffusion(GaussianDiffusion):
    """
    A class for SRDiff diffusion model.
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
            channels=1,
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
        self.lock_weights = lock_weights
        self.rrdb_encoder = None

        if pretrained_model_path is not None:
            self.init_rrdb_encoder(pretrained_model_path, lock_weights)

        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)


    def init_rrdb_encoder(self, pretrained_model_path, lock_weights=True):
        """
        Initialize the RRDB encoder with the pretrained model weights.
        Args:
            pretrained_model_path (str): Path to the pretrained model weights.
            lock_weights (bool): If True, the weights are frozen and not updated during training.
        """
        hidden_size = 64
        num_block = 17
        self.rrdb_encoder = RRDBNet(self.channels, self.channels, hidden_size, num_block, hidden_size // 2)
        self.rrdb_encoder.load_state_dict(torch.load(pretrained_model_path))
        if lock_weights:
            self.rrdb_encoder.eval()
            for name, para in self.rrdb_encoder.named_parameters():
                # RRDB weights are all frozen
                para.requires_grad_(False)

    def p_sample_loop(self, x_in, continous=False):
        """
        Implementation of SRDiffusion sampling algorithm.
        Args:
            x_in: Shape of the tensor to be sampled or initial noisy image for conditional models.
            continous: Not used for effciency.

        Returns:
            Generated samples after the complete reverse diffusion process.
        """
        x_interpolated = x_in['SR']
        x_lr = x_in['LR']

        batch_size = x_interpolated.size(0)
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps // 10))
        if not self.conditional:
            shape = x_interpolated
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in reversed(range(0, self.num_timesteps)):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            shape = x_interpolated.shape
            img = torch.randn(shape, device=device)
            ret_img = x_interpolated

            # encode the LR image
            if self.rrdb_encoder:
                _, rrdb_encoded = self.rrdb_encoder(x_lr, True)
            else:
                rrdb_encoded = x_lr

            for i in reversed(range(0, self.num_timesteps)):
                img = self.p_sample(img, i, condition_x=rrdb_encoded)
                if i % sample_inter == 0:
                    ret_img = img

        return ret_img + x_interpolated

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
        return self.p_sample_loop(x_in, continous)

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        """
        Compute: mu_theta(x_t, t, condition) = 1/sqrt(alpha_t) * (x_t - (1 - alpha_t) / sqrt(1 - alpha_t_hat)) * model_predicted_noise).
        and conditions srdiff in right way

        Args:
            x: Current noisy image x_t.
            t (int): Current timestep index.
            clip_denoised (bool): Boolean flag to clip the denoised image values.
            condition_x: Optional conditional information (e.g., class labels).

        Returns:
            A tuple containing the diffusion mean and posterior log variance.
        """
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)

        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.denoise_fn((condition_x, x), noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    def p_losses(self, x_in, noise=None):
        """
        Calculate the training loss for a single step of the reverse diffusion process for SRDiff model
        including denoise fucntion utilization and rrdb encoder.

        Args:
            x_in: Dictionary containing input images and potentially other conditioning information.
            noise: Optional external noise to add for the diffusion process.

        Returns:
            The calculated loss for the diffusion to optimize.
        """

        img_lr_up = x_in['SR']
        img_lr = x_in['LR']
        img_hr = x_in['HR']

        if self.rrdb_encoder:
            rrdb_sr, rrdb_encoded = self.rrdb_encoder(img_lr, True)
        else:
            rrdb_sr = img_lr_up
            rrdb_encoded = img_lr

        # compute residual img
        x_start = img_hr - img_lr_up

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
            x_recon = self.denoise_fn((rrdb_encoded, x_noisy), continuous_sqrt_alpha_cumprod)

        loss = self.loss_func(noise, x_recon)

        if self.rrdb_encoder and not self.lock_weights:
            rrdb_loss = F.l1_loss(rrdb_sr, img_hr)
            return loss + rrdb_loss

        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
