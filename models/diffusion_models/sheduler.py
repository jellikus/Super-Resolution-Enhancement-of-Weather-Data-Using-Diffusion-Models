import math
import numpy as np
import torch

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    """
    Generate beta values for the diffusion process with a linear warmup phase.

    Args:
        linear_start (float): Starting value for the linear warmup.
        linear_end (float): Ending value for the linear warmup.
        n_timestep (int): Total number of timesteps in the diffusion process.
        warmup_frac (float): Fraction of the total timesteps where warmup is applied.

    Returns:
        np.ndarray: Array of beta values with warmup applied to the initial segment.
    """
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    """
    Create a schedule for the beta parameter in the diffusion process.

    Args:
        schedule (str): Type of schedule to use ('quad', 'linear', 'warmup10', 'warmup50', 'const', 'jsd', 'cosine').
        n_timestep (int): Total number of timesteps in the diffusion process.
        linear_start (float): Starting value for linear and quadratic schedules.
        linear_end (float): Ending value for linear and quadratic schedules.
        cosine_s (float): Smoothing parameter for the cosine schedule.

    Returns:
        np.ndarray or torch.Tensor: Array of beta values according to the specified schedule.
    """
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) /
                n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas