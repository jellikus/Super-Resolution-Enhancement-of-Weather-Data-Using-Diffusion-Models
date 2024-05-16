import typing
import torch
import torch.nn as nn
from abc import abstractmethod
# diffusion models part of the code is based on https://github.com/LYL1015/ResDiff#resdiff-combining-cnn-and-diffusion-model-for-image-super-resolution

class BaseModel:
    """Base class for all main models."""

    def __init__(self, opt):
        """Initializes BaseModel.

        Args:
            opt: Configuration parameters.
        """
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() and opt['gpu_ids'] else "cpu")
        self.begin_step, self.begin_epoch = 0, 0

    @abstractmethod
    def feed_data(self, data) -> None:
        """Provides model with data.

        Args:
            data: A tuple containing dictionary with the following keys: (
                {HR: a batch of high-resolution images [B, C, H, W]},
                {LR: a batch of low-resolution images [B, C, H, W]},
                {INTERPOLATED: a batch of upsampled (via interpolation) images [B, C, H, W]},
                [list of corresponding months of samples in a batch]
        """
        pass

    @abstractmethod
    def optimize_parameters(self) -> None:
        """Computes loss and performs GD step on learnable parameters.
        """
        pass

    @abstractmethod
    def get_images(self) -> dict:
        """Returns reconstructed data points.
        """
        pass

    @abstractmethod
    def print_network(self) -> None:
        """Prints the network architecture.
        """
        pass

    def set_device(self, x):
        """Sets values of x onto device specified by an attribute of the same name.

        Args:
            x: Value storage.

        Returns:
            x set on self.device.
        """
        if isinstance(x, dict):
            x = {key: (item.to(self.device) if item.numel() else item) for key, item in x.items()}
        elif isinstance(x, list):
            x = [item.to(self.device) if item else item for item in x]
        else:
            x = x.to(self.device)
        return x

    @staticmethod
    def get_network_description(network: nn.Module) -> typing.Tuple[str, int]:
        """Get the network name and parameters.

        Args:
            network: The neural netowrk.

        Returns:
            Name of the network and the number of parameters.
        """
        if isinstance(network, nn.DataParallel):
            network = network.module
        n_params = sum(map(lambda x: x.numel(), network.parameters()))
        return str(network), n_params

    def get_loaded_epoch(self) -> int | None:
        """Returns the current epoch of the training process.
        """
        return self.begin_epoch

    def get_loaded_iter(self) -> int | None:
        """Returns the current iteration of the training process.
        """
        return self.begin_step

    @abstractmethod
    def prepare_to_train(self) -> None:
        """Sets model to train mode and activates implementation-specific training procedures.
        """
        pass

    @abstractmethod
    def prepare_to_eval(self) -> None:
        """Sets model to eval mode and activates implementation-specific evaluation procedures.
        """
        pass

    @abstractmethod
    def generate_sr(self, continuous: bool = False) -> None:
        """Constructs the super-resolution image and assiggns to SR attribute.

        Args:
            continuous: If True save ouput for each iteration T while sampling.
        """
        pass

    @abstractmethod
    def save_network(self):
        """Saves the network weights.
        """
        pass

    @abstractmethod
    def get_current_log(self) -> dict:
        """Returns current log.
        """
        pass


def create_model(opt, optimizer):
    """Creates model which is child of BaseModel.

    Args:
        opt: Configuration parameters.
        optimizer: Optimization algorithm.

    Returns:
        Model instance.
    """
    if opt["model"]["model_name"] == "diffusion":
        import models.diffusion_models as model
        return model.create_model(opt)
    else:
        raise NotImplementedError("Model {} not implemented.".format(opt["model"]["model_name"]))
