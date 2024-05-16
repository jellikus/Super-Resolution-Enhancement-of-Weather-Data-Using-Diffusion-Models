import logging
import os


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    """Sets up the logger.

    Args:
        logger_name: The logger name.
        root: The directory of logger.
        phase: Either train or val.
        level: The level of logging.
        screen: If True then write logging records to a stream.
    """
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter("%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s", datefmt="%y-%m-%d %H:%M:%S")
    log_file = os.path.join(root, "{}.log".format(phase))
    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)