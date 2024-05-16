import torch
import os
import random
import numpy as np

def get_optimizer(name: str) -> torch.optim:
    """Return optimization algorithm class corresponding to name.

    Args:
        name: The name of optimizer.

    Returns:
        A torch optimizer.
    """
    if name == "adam":
        from torch.optim import Adam as Optimizer
    elif name == "adamw":
        from torch.optim import AdamW as Optimizer
    elif name == "sgd":
        from torch.optim import RMSprop as Optimizer
    elif name == "adadelta":
        from torch.optim import Adadelta as Optimizer
    elif name == "adagrad":
        from torch.optim import Adagrad as Optimizer
    elif name == "adamax":
        from torch.optim import Adamax as Optimizer
    elif name == "asgd":
        from torch.optim import Rprop as Optimizer
    elif name == "sparseadam":
        from torch.optim import SparseAdam as Optimizer
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented.")
    return Optimizer


def get_valid_optimizer():
    return ["adam", "adamw", "sgd", "adadelta", "adagrad", "adamax", "asgd", "sparseadam"]

def set_seeds(seed: int = 0):
    """Sets random seeds of Python, NumPy and PyTorch.

    Args:
        seed: Seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

