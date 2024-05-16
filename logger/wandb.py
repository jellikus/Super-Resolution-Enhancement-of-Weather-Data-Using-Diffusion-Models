import wandb
class WandbLogger:
    """
    Wrapper for logging metrics and images to Weights & Biases (W&B).

    Args:
        config_dict (dict): Configuration dictionary for W&B and experiment settings.
    """
    def __init__(self, config_dict):
        self._config = config_dict
        self._wandb = wandb
        self._init_wandb_logger()

    def _init_wandb_logger(self):
        """
        Initialize Weights & Biases (W&B) logger. Create a new run
        if it does not exist and Table for evaluation metrics.
        """
        if self._wandb.run is None:
            self._wandb.init(
                project=self._config['wandb']['project'],
                config=self._config,
                dir='./experiments/wandb/',
                name=self._config['name']
            )

        self._config = self._wandb.config

        self.eval_table = self._wandb.Table(columns=['fake_image',
                                                     'sr_image',
                                                     'hr_image',
                                                     'psnr',
                                                     'ssim'])

    def log_metrics(self, metrics, commit=True, step=None):
        """
        Log metrics onto W&B.

        Args:
            metrics (dict): Dictionary of metrics to be logged.
            commit (bool, optional): Whether to commit the logged metrics.
            step (int, optional): Step number for the logged metrics.
        """
        self._wandb.log(metrics, commit=commit, step=step)

    def log_val_metrics(self, metrics, commit=True, step=None):
        """
        Log validation metrics onto W&B.

        Args:
            metrics (dict): Dictionary of validation metrics to be logged.
            commit (bool, optional): Whether to commit the logged metrics.
            step (int, optional): Step number for the logged metrics.
        """
        self._wandb.log({"val": metrics}, commit=commit, step=step)

    def log_train_metrics(self, metrics, commit=True, step=None):
        """
        Log training metrics onto W&B.

        Args:
            metrics (dict): Dictionary of training metrics to be logged.
            commit (bool, optional): Whether to commit the logged metrics.
            step (int, optional): Step number for the logged metrics.
        """
        self._wandb.log({"train": metrics}, commit=commit, step=step)

    def log_train_mean_metrics(self, metrics, commit=True, step=None):
        """
        Log mean training metrics onto W&B.

        Args:
            metrics (dict): Dictionary of mean training metrics to be logged.
            commit (bool, optional): Whether to commit the logged metrics.
            step (int, optional): Step number for the logged metrics.
        """
        self._wandb.log({"train_mean": metrics}, commit=commit, step=step)

    def log_sr_hr_it_image(self, figure, commit=True, step=None):
        """
        Log image which contains super-resolved, high-resolution, and interpolated images.

        Args:
            figure (PIL.Image.Image or numpy.ndarray): Image to be logged.
            commit (bool, optional): Whether to commit the logged image.
            step (int, optional): Step number for the logged image.
        """
        self._wandb.log({'sr_hr_it': wandb.Image(figure)}, commit=commit, step=step)

    def log_val_time(self, time, commit=True, step=None):
        """
        Log how much time it took to validate over the val dataset.

        Args:
            time (float): Validation time value to be logged.
            commit (bool, optional): Whether to commit the logged time value.
            step (int, optional): Step number for the logged time value.
        """
        self._wandb.log({'val_time': time}, commit=commit, step=step)