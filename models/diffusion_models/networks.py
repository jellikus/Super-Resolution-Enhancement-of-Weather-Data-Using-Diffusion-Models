import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init

logger = logging.getLogger('base')


def weights_init_normal(m, std=0.02):
    """
    Initialize weights with a normal distribution for specific module types.

    Args:
        m: Module to initialize.
        std: Standard deviation for the normal distribution.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    """
    Initialize weights with the Kaiming (He) initialization suited for ReLUs.

    Args:
        m: Module to initialize.
        scale: Scaling factor applied after initialization.
    """
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    """
    Initialize weights with an orthogonal matrix which helps in reducing
    the dependency amongst model parameters.

    Args:
        m: Module to initialize.
    """
    classname = m.__class__.__name__
    if classname.find('PhyConv') != -1:
        return
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    """
    Apply specified initialization to a neural network.

    Args:
        net: Network to initialize.
        init_type: Type of initialization ('normal', 'kaiming', 'orthogonal').
        scale: Scaling factor for 'kaiming'.
        std: Standard deviation for 'normal'.
    """
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))


# Generator
def define_diffusion(opt):
    """
    Define a diffusion model based on the architecture specified in the opt dictionary.

    Args:
        opt: Dictionary containing model specifications.

    Returns:
        diffusion_model: Configured network diffusion.
    """
    model_opt = opt['model']
    if model_opt['architecture'] == 'sr3':
        from .sr3 import unet
        from .sr3.sr3_diffusion import SR3Diffusion as GaussianDiffusion
    elif model_opt['architecture'] == 'resdiff':
        from .resdiff import unet
        from .resdiff.resdiff_diffusion import ResDiffDiffusion as GaussianDiffusion
    elif model_opt['architecture'] == 'phydiff':
        from .phydiff import unet
        from .phydiff.phydiff_diffusion import PhyDiffDiffusion as GaussianDiffusion
    elif model_opt['architecture'] == 'srdiff':
        from .srdiff import unet
        from .srdiff.srdiff_diffusion import SRDiffDiffusion as GaussianDiffusion
    elif model_opt['architecture'] == 'physrdiff':
        from .physrdiff import unet
        from .physrdiff.physrdiff_diffusion import PhySRDiffDiffusion as GaussianDiffusion
    else:
        raise NotImplementedError(
            'Architecture [{:s}] is not implemented.'.format(model_opt['architecture']))

    if ('norm_groups' not in model_opt['unet']) or model_opt['unet']['norm_groups'] is None:
        model_opt['unet']['norm_groups'] = 32

    model = unet.UNet(
        in_channel=model_opt['unet']['in_channel'],
        out_channel=model_opt['unet']['out_channel'],
        norm_groups=model_opt['unet']['norm_groups'],
        inner_channel=model_opt['unet']['inner_channel'],
        channel_mults=model_opt['unet']['channel_multiplier'],
        attn_res=model_opt['unet']['attn_res'],
        res_blocks=model_opt['unet']['res_blocks'],
        dropout=model_opt['unet']['dropout'],
        image_height=model_opt['diffusion']['image_height'],
        image_width=model_opt['diffusion']['image_width'],
        image_channels=model_opt['diffusion']['image_channels']
    )

    diffusion_model = GaussianDiffusion(
        model,
        image_height=model_opt['diffusion']['image_height'],
        image_width=model_opt['diffusion']['image_width'],
        channels=model_opt['diffusion']['image_channels'],
        loss_type='l1',  # L1 or L2
        conditional=model_opt['diffusion']['conditional'],
        schedule_opt=model_opt['beta_schedule']['train'],
        pretrained_model_path=model_opt['pretrained_model']['model_path'],
        lock_weights=model_opt['pretrained_model']['lock_weights']
    )
    if opt['phase'] == 'train':
        init_weights(diffusion_model, init_type='orthogonal')
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        diffusion_model = nn.DataParallel(diffusion_model)
    return diffusion_model
