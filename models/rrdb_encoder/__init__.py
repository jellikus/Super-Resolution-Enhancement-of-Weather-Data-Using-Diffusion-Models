from models.rrdb_encoder.RRDBNet import RRDBNet
import torch


def prepare_rrdb(path, scale_factor=4, channels=1):
    """Creates a RRDBNet model.

    Args:
        path: A path to the model.

    Returns:
        A RRDBNet model.
    """
    rrdb_encoder = RRDBNet()
    rrdb_encoder.load_state_dict(torch.load(path))
    rrdb_encoder.eval()
    for name, para in rrdb_encoder.named_parameters():
        # CNN weights are all frozen
        para.requires_grad_(False)
    return rrdb_encoder
