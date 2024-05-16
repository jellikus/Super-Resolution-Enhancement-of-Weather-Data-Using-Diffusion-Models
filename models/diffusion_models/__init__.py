import logging
logger = logging.getLogger('base')


def create_model(opt):
    """
    Create an instance of the DDPM model defined in the local diffusion module.

    Args:
        opt: A dictionary containing configuration options for the model.

    Returns:
        An instance of the DDPM model with the provided configuration.
    """
    from .model import DDPM
    model = DDPM(opt)
    logger.info('Model [{:s}] is created.'.format(model.__class__.__name__))
    return model
