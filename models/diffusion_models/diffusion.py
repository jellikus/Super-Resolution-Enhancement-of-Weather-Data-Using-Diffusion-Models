import torch
from torch import nn
from functools import partial
import numpy as np
from .nn_modules.functional_layers import default
from .sheduler import make_beta_schedule
from abc import abstractmethod


class GaussianDiffusion(nn.Module):
    """
    A class for Gaussian diffusion model with abstract methods for each type of architecture (SRDiff, ResDiff, SR3, ...)
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
            loss_type='l1',
            conditional=True,
            schedule_opt=None,
            image_height=128,
            image_width=256,
            pretrained_model_path=None,
            lock_weights=True,

    ):
        super().__init__()
        self.channels = channels
        self.image_height = image_height
        self.image_width = image_width
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass

    def set_new_noise_schedule(self, schedule_opt, device):
        """
        Configure the noise schedule for the diffusion process using specified options.

        Args:
            schedule_opt (dict): Options for the noise schedule, containing schedule type and parameters.
            device (str or torch.device): The device to use for creating tensors.
        """
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
                             (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def set_loss(self, device):
        """
        Set the loss function based on the specified type.

        Args:
            device (str or torch.device): The device tensors should be moved to.
        """
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict the denoised image (start image) from the noisy image and the noise using the diffusion model's
        parameters using the formula: mu_t(x_t, x_0) = 1/sqrt(alpha_t) * (x_t - (1 - alpha_t) / sqrt(1 -
        alpha_t_hat)) * noise).
        Args:
            x_start: The original image (start image).
            x_t: The noisy image at timestep t.
            t (int): Current timestep index.
        Returns:
            The predicted start image (denoised image).
        """
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        """
        Calculate the posterior distribution parameters for the reverse diffusion process.

        Args:
            x_start: The original image (start image).
            x_t: The noisy image at timestep t.
            t (int): Current timestep index.

        Returns:
            A tuple containing the mean and log variance of the posterior distribution.
        """
        posterior_mean = self.posterior_mean_coef1[t] * \
                         x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        """
        Compute the mean and variance of the diffusion distribution p(x_{t-1} | x_t) during the reverse process using formula:
        mu_theta(x_t, t, condition) = 1/sqrt(alpha_t) * (x_t - (1 - alpha_t) / sqrt(1 - alpha_t_hat)) * model_predicted_noise).

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
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        """
        Sample image x_{t-1} from the diffusion distribution p(x_{t-1} | x_t) during the reverse process.

        Args:
            x: Current noisy image x_t.
            t (int): Current timestep index.
            clip_denoised (bool): Boolean flag to clip the denoised image values.
            condition_x: Optional conditional information for conditional sampling.

        Returns:
            A sampled image from the diffusion distribution.
        """
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        """
        Generate samples from the diffusion. This method initializes the sampling process and specifies the batch size and dimensions.

        Args:
            batch_size (int): The number of samples to generate.
            continous (bool): Boolean indicating whether to output samples continuously during the process.

        Returns:
            A batch of generated samples.
        """
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, self.image_height, self.image_height), continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        """
        Sample from the noise diffusion q using the reparameterization trick to add specified noise levels. This function
        does forward diffusion process.

        Args:
            x_start: The initial clean image.
            continuous_sqrt_alpha_cumprod: Precomputed square roots of alpha cumulatives for noise scaling.
            noise: Optional external noise to add to the image. If None, random noise is generated.

        Returns:
            Noisy image generated from the clean image by applying the noise diffusion.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
                continuous_sqrt_alpha_cumprod * x_start +
                (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise
        )

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)

    @abstractmethod
    def p_sample_loop(self, x_in, continous=False) -> torch.Tensor:
        """
        Perform a loop of T sampling steps, sampling from p(x_{t-1} | x_t) starting with gaussian noise.

        Args:
            x_in: Shape of the tensor to be sampled or initial noisy image for conditional models.
            continous: Not used for effciency.

        Returns:
            Generated samples after the complete reverse diffusion process.
        """
        pass

    @abstractmethod
    def super_resolution(self, x_in, continous=False) -> torch.Tensor:
        """
        Perform super-resolution on the given low-resolution input using the trained diffusion.

        Args:
            x_in: Input dictionary containing low-resolution (and potentially high-resolution) images for reference.
            continous (bool): Boolean indicating whether to output samples continuously during the process.

        Returns:
            Super-resolved images.
        """
        pass

    @abstractmethod
    def p_losses(self, x_in, noise=None) -> torch.Tensor:
        """
        Calculate the training loss for a single step of the reverse diffusion process.

        Args:
            x_in: Dictionary containing input images and potentially other conditioning information.
            noise: Optional external noise to add for the diffusion process.

        Returns:
            The calculated loss for the diffusion to optimize.
        """
        pass
