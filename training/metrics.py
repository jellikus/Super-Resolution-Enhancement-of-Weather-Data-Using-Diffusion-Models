from abc import ABC, abstractmethod
from torcheval.metrics import PeakSignalNoiseRatio, MeanSquaredError, StructuralSimilarity
import numpy as np
import torch
from skimage.metrics import structural_similarity
import warnings


class MetricsContainer(ABC):
    """
    Abstract base class for containers that manage various metrics.
    """

    @abstractmethod
    def metrics2dict(self):
        """
        Convert metrics to a dictionary format. Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def metrics2str(self):
        """
        Convert metrics to a string format for display. Must be implemented by subclasses.
        """
        pass


class Metric(ABC):
    """
    Abstract base class for defining metrics.
    """

    def __init__(self, device=None):
        """
        Initialize the metric with default values on the specified device.

        Args:
            device (str, optional): The device to allocate tensors (e.g., 'cuda', 'cpu').
        """
        self.count = torch.tensor(0, device=device, dtype=torch.float)
        self.sum = torch.tensor(0, device=device, dtype=torch.float)
        self.device = device
        self.reset()

    def reset(self):
        """
        Reset the metric counters.
        """
        self.sum = torch.tensor(0, device=self.device, dtype=torch.float)
        self.count = torch.tensor(0, device=self.device, dtype=torch.float)

    @abstractmethod
    def update(self, predicted, target):
        """
        Update the metric based on the predictions and targets.

        Args:
            predicted (torch.Tensor): Predicted outputs.
            target (torch.Tensor): Ground truth targets.
        """
        pass

    @abstractmethod
    def compute(self):
        """
        Compute the final metric.

        Returns:
            float: The computed metric value.
        """
        pass


class MAE(Metric):
    """
    Mean Absolute Error (MAE) metric class.
    """

    def __init__(self, device=None):
        super().__init__(device)

    @torch.inference_mode()
    def update(self, predicted, target):
        """
        Update MAE sum and count with the absolute differences between predicted and target tensors.

        Args:
            predicted (torch.Tensor): Predicted outputs.
            target (torch.Tensor): Ground truth targets.
        """
        self.sum += (predicted - target).abs().sum()
        self.count += predicted.numel()

    @torch.inference_mode()
    def compute(self):
        """
        Compute the MAE.

        Returns:
            float: The MAE value, or 0 if count is zero.
        """
        if self.count == 0:
            return 0.0
        return self.sum / self.count


class MSE(Metric):
    """
    Mean Squared Error (MSE) metric class.
    """
    def __init__(self, device=None):
        super().__init__(device)

    @torch.inference_mode()
    def update(self, predicted, target):
        """
        Update MSE sum and count with the squared differences between predicted and target tensors.

        Args:
            predicted (torch.Tensor): Predicted outputs.
            target (torch.Tensor): Ground truth targets.
        """
        self.sum += (predicted - target).pow(2).sum()
        self.count += predicted.numel()

    @torch.inference_mode()
    def compute(self):
        """
        Compute the MSE.

        Returns:
            float: The MSE value, or 0 if count is zero.
        """
        if self.count == 0:
            return 0.0
        return self.sum / self.count


class RMSE(Metric):
    """
    Root Mean Squared Error (RMSE) metric class.
    """
    def __init__(self, device=None):
        super().__init__(device)

    @torch.inference_mode()
    def update(self, predicted, target):
        """
        Update RMSE sum and count with the squared differences between predicted and target tensors.

        Args:
            predicted (torch.Tensor): Predicted outputs.
            target (torch.Tensor): Ground truth targets.
        """
        self.sum += (predicted - target).pow(2).sum()
        self.count += predicted.numel()

    @torch.inference_mode()
    def compute(self):
        """
        Compute the RMSE.

        Returns:
            float: The RMSE value, or 0 if count is zero.
        """
        if self.count == 0:
            return 0.0
        return torch.sqrt(self.sum / self.count)


class MR(Metric):
    """
    Mean Residual (MR) metric class.
    """
    def __init__(self, device=None):
        super().__init__(device)

    @torch.inference_mode()
    def update(self, predicted, target):
        """
        Update the sum and count with the differences between predicted and target tensors.

        Args:
            predicted (torch.Tensor): Predicted outputs.
            target (torch.Tensor): Ground truth targets.
        """
        self.sum += (predicted - target).sum()
        self.count += predicted.numel()

    @torch.inference_mode()
    def compute(self):
        """
        Compute the mean residual.

        Returns:
            float: The mean residual value, or 0 if count is zero.
        """
        if self.count == 0:
            return 0.0
        return self.sum / self.count


class PSNR(Metric):
    """
    Peak Signal-to-Noise Ratio (PSNR) metric class.
    """
    def __init__(self, device=None):
        self.device = device
        self.psnr = PeakSignalNoiseRatio(data_range=None, device=self.device)

    def reset(self):
        """
        Reset the PSNR metric.
        """
        self.psnr = PeakSignalNoiseRatio(data_range=None, device=self.device)

    @torch.inference_mode()
    def update(self, predicted, target):
        """
        Update PSNR computation based on predicted and target tensors.

        Args:
            predicted (torch.Tensor): Predicted outputs.
            target (torch.Tensor): Ground truth targets.
        """
        self.psnr.update(predicted, target)

    @torch.inference_mode()
    def compute(self):
        """
        Compute the final PSNR value.

        Returns:
            float: Computed PSNR value.
        """
        return self.psnr.compute()


class SSIM(Metric):
    """
    Structural Similarity Index (SSIM) metric class.
    """
    def __init__(self, device=None):
        super().__init__(device)

    @torch.inference_mode()
    def update(self, predicted: torch.Tensor, target: torch.Tensor, ):
        """
        Update SSIM computation for each pair of predicted and target images in the batch.

        Args:
            predicted (torch.Tensor): Predicted outputs.
            target (torch.Tensor): Ground truth targets.

        Raises:
            RuntimeError: If the shapes of the predicted and target tensors do not match.
        """

        if predicted.shape != target.shape:
            raise RuntimeError("The two sets of images must have the same shape.")

        batch_size = predicted.shape[0]

        for idx in range(batch_size):
            img_pred = predicted[idx][0].detach().cpu().numpy()
            img_target = target[idx][0].detach().cpu().numpy()
            mssim = structural_similarity(
                img_target,
                img_pred,
                data_range=img_pred.max() - img_pred.min(),
            )
            self.sum += mssim

        self.count += batch_size

    @torch.inference_mode()
    def compute(self):
        """
        Compute the average SSIM for the batch.

        Returns:
            float: The average SSIM value, or raise a warning if count is zero.
        """
        if self.count == 0:
            warnings.warn(
                "The number of images must be greater than 0.",
                RuntimeWarning,
                stacklevel=2,
            )
        return self.sum / self.count

    @torch.inference_mode()
    def merge_state(self, metrics):
        """
        Merge the metric state with its counterparts from other metric instances.
        Args:
            metrics (Iterable[Metric]): metric instances whose states are to be merged.
        """
        for metric in metrics:
            self.sum += metric.mssim_sum.to(self.device)
            self.count += metric.num_images.to(self.device)

        return self


class ValidationMetrics(MetricsContainer):
    """
    Container class for managing and aggregating validation metrics.
    """
    def __init__(self, metrics_dict):
        """
        Initialize the ValidationMetrics with a dictionary of metric objects.

        Args:
            metrics_dict (dict): A dictionary of metric objects to manage.
        """
        self.metrics_objects = metrics_dict
        self.metrics = {}
        for metric in self.metrics_objects.values():
            metric.reset()

    def reset(self):
        """
        Reset all metrics in the container.
        """
        for metric in self.metrics_objects.values():
            metric.reset()

    def update(self, predicted, target):
        """
        Update all metrics with the new predicted and target data.

        Args:
            predicted (torch.Tensor): The predicted outputs.
            target (torch.Tensor): The ground truth targets.
        """
        for metric in self.metrics_objects.values():
            metric.update(predicted, target)

    def metrics2str(self):
        """
        Convert all metrics to a formatted string for display.

        Returns:
            str: A formatted string representing all metrics.
        """
        message = ""
        for metric, value in self.metrics.items():
            message = f"{message}  |  {metric:s}: {value:.5f}"
        return message

    def metrics2dict(self):
        """
        Convert all metrics to a dictionary.

        Returns:
            dict: A dictionary with metric names as keys and their values.
        """
        return self.metrics

    def compute_metrics(self):
        """
        Compute all metrics and store the results in the internal dictionary.

        Returns:
            dict: A dictionary with the computed metric values.
        """
        metric_values = {}
        for metric_name, metric in self.metrics_objects.items():
            value = metric.compute()
            metric_values[metric_name] = value

        self.metrics = metric_values
        return metric_values


class TrainMetrics(MetricsContainer):
    """
    Container class for managing and reporting training metrics over epochs.
    """
    def __init__(self):
        """
        Initialize the TrainMetrics container with empty dictionaries for metrics and log data.
        """
        self.metrics = {}
        self.last_log = {}

    def reset(self):
        """
        Reset all stored metrics and logs. This method is typically called at the start of a new epoch.
        """
        self.metrics = {}
        self.last_log = {}

    def update(self, new_dict):
        """
        Update the metrics with a new dictionary of values typically received after a training batch.

        Args:
            new_dict (dict): A dictionary containing metric names as keys and their new values.
        """
        self.last_log = new_dict
        for key, value in new_dict.items():
            if key in self.metrics:
                self.metrics[key].append(value)
            else:
                self.metrics[key] = [value]

    def metrics2str(self):
        """
        Convert the current metrics to a formatted string to facilitate easy reading during training.

        Returns:
            str: A formatted string that provides an overview of current and mean metric values.
        """
        message = ""
        for metric, values in self.metrics.items():
            mean_value = np.mean(values)
            message = f"{message}  |  {metric:s}: mean = {mean_value:.5f}, curr = {values[-1]:.5f}"

        return message

    def metrics2dict(self):
        """
        Convert the last logged metrics to a dictionary. Useful for logging the last batch's metrics.

        Returns:
            dict: A dictionary of the last logged metrics.
        """
        return self.last_log

    def mean_metrics2dict(self):
        """
        Retrieve the mean of each metric over the current epoch or logging period.

        Returns:
            dict: A dictionary containing the mean values of all metrics.
        """
        return self.get_last_mean_metrics()

    def get_last_mean_metrics(self):
        """
        Compute the mean values of each metric stored.

        Returns:
            dict: A dictionary with the mean values of each metric.
        """
        mean_metric_values = {}
        for metric_name, metric in self.metrics.items():
            value = np.mean(metric)
            mean_metric_values[metric_name] = value
        return mean_metric_values

    def get_last_metrics(self):
        """
        Retrieve the latest values of each metric.

        Returns:
            dict: A dictionary with the last recorded values for each metric.
        """
        last_metric_values = {}
        for metric_name, metric in self.metrics.items():
            value = metric[-1]
            last_metric_values[metric_name] = value
        return last_metric_values

    def get_metrics(self):
        """
        Accessor method for retrieving all metrics.

        Returns:
            dict: A dictionary containing all metrics and their respective lists of values.
        """
        return self.metrics


def create_metric_dict(torch_device=None):
    """
    Creates a dictionary of metrics used for validation.
    Args:
        data_range: Range of data values. Usually 1.0 for normalized data and 255.0 for non-normalized data.
        torch_device: Device to use for computation. Defaults to None.
    """
    metrics = {}
    metrics["MSE"] = MSE(device=torch_device)
    metrics["RMSE"] = RMSE(device=torch_device)
    metrics["MAE"] = MAE(device=torch_device)
    metrics["MR"] = MR(device=torch_device)
    metrics["PSNR"] = PSNR(device=torch_device)
    metrics["SSIM"] = SSIM(device=torch_device)
    return metrics
