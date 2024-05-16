from models.simple_cnn.Simple_CNN import SimpleCNN
import torch


def prepare_cnn(path, scale_factor=4, channels=1):
    """Creates a CNN model.

    Args:
        path: A path to the model.
        scale_factor: How many times are images upscaled in each dim.
        channels: The number input channels of the images.

    Returns:
        A CNN model.
    """
    cnn = SimpleCNN(scale_factor=scale_factor, channels=channels)
    cnn.load_state_dict(torch.load(path))
    cnn.eval()
    for name, para in cnn.named_parameters():
        # CNN weights are all frozen
        para.requires_grad_(False)
    return cnn
