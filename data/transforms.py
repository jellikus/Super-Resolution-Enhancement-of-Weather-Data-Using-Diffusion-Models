import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from .utils import validate_group_months_subset, date_to_str, unpack_datasets, flatten, str_to_date, get_month_datetime, \
    find_group_idx
from .datasets import TimeVariateData
from collections import OrderedDict
from .npy_reader import WNPYReader
import os


class DataTransformer:
    """
    The DataTransformer class handles the preprocessing and transformation of meteorological data.
    It supports fitting transformation models to specific subsets of data defined by groups of months.
    """

    def __init__(self, variables: list, dataroot: str, months_subset, groups=None):
        """
        Initializes the DataTransformer with transformations and variables.

        Args:
            transformations (dict): A dictionary of monthly fitted LR/HR transformations for each variable used for training.
            variables (list): A list of WeatherBench variables.
            dataroot (str): Path to the dataset.
            transformation (Transform): A transformation model to fit.
            groups (optional): Groups of months that should be transformed together.
            months_subset (list): A list of month indices.
        """
        self.transformation_dict = {}
        self.variables = variables
        self.dataroot = dataroot
        self.groups = groups
        self.months_subset = months_subset

    def transform(self, min_date: str, max_date: str, data_type: str, variable: str, transformation) -> dict:
        """fit and stores transofrmation for each group.

        Args:
            min_date: Minimum date starting from which to read the data.
            max_date: Maximum date until which to read the date.
            data_type: Either lr or hr string.
            variable: Weather variable name.
            transformation: A transformation object.

        Returns:
            dict: A dictionary of transformations.
        """
        validate_group_months_subset(self.months_subset, self.groups)
        group_data = self._create_group_datasets(min_date, max_date, data_type, variable)

        fitted_transforms = {group: self._fit_month(datasets, transformation) for group, datasets in
                             group_data.items()}
        mapped_transforms = {}
        for idx, group in enumerate(self.groups):
            for month in group:
                mapped_transforms[month] = fitted_transforms[idx + 1]

        # if variable not in transformation_dict create a new entry else
        # add new transformation to the existing entry
        if variable not in self.transformation_dict:
            self.transformation_dict[variable] = {data_type: mapped_transforms}
        else:
            self.transformation_dict[variable][data_type] = mapped_transforms

        return mapped_transforms

    def get_transform(self, variable: str, data_type: str) -> dict:
        """Returns a dictionary of transformations for a given variable and data type.

        Args:
            variable: Variable name from WeatherBench dataset.
            data_type: Either lr or hr.

        Returns:
            dict: A dictionary of transformations.
        """
        return self.transformation_dict[variable][data_type]

    def inverse_transform(self, data: dict, batch_months: list) -> dict:
        """
        Inverse transforms batch of data.

        Args:
            data (dict): Dictionary of data tensors.
            batch_months (list): A list of month indices in given batch.

        Returns:
            dict: Inversed data.
        """
        reverse_transformed_batch = OrderedDict()
        for key, tensor in data.items():
            data_type = "lr" if key == "LR" else "hr"
            reverse_transformed_batch[key] = self._inverse_tensor(tensor, data_type, batch_months)
        return reverse_transformed_batch

    def _fit_month(self, datasets: list, transformation):
        """Fits a transformation to a list of datasets.

        Args:
            datasets: A list of datasets corresponding to the same month.
            transformation: A transformation to fit.

        Returns:
            Transform: A fitted transformation.

        """
        transform = transformation()
        for data in flatten(unpack_datasets(datasets)):
            transform.fit(data)
            transform.clear_data_source()
        return transform

    def _inverse_tensor(self, tensor: torch.tensor, data_type: str, months_subset: list) -> torch.tensor:
        """
        Inverse transforms a tensor.

        Args:
            tensor (torch.tensor): Tensor data of shape (batch_size, number of variables, H, W).
            data_type (str): Either 'lr' or 'hr'.
            months_subset (list): A list of month indices.

        Returns:
            torch.tensor: Inverse transformed single data point.
        """
        reverse_transformed_tensors = []
        for index, variable in enumerate(self.variables):
            tensor_of_variable = tensor[:, index].unsqueeze(1)

            batch_size = tensor.shape[0]
            # reverse transformation for the variable
            reversed_variable = torch.cat(
                [self.transformation_dict[variable][data_type][months_subset[idx]].revert(tensor_of_variable[idx]) for
                 idx in
                 range(batch_size)])
            reverse_transformed_tensors.append(reversed_variable)
        return torch.cat(reverse_transformed_tensors, dim=1)

    def _create_group_datasets(self, min_date: str, max_date: str, data_type: str, variable: str) -> dict:
        """It processes data on a monthly basis, combining datasets that correspond to the same goup index into
        a single entity. It then creates a dictionary where each group index is paired with its respective dataset.

        Args:
            min_date: Minimum date starting from which to read the data.
            max_date: Maximum date until which to read the date.
            data_type: Either lr or hr.
            variable: Variable name from WeatherBench dataset.

        Returns:
            dict: Month to data mapping.
        """
        month2data = {}
        max_date_datetime = str_to_date(max_date)
        start = str_to_date(min_date)
        start_of_next_month = start + get_month_datetime()
        npy_reader = WNPYReader(os.path.join(self.dataroot, data_type, variable))

        while start_of_next_month < max_date_datetime:
            group_idx = find_group_idx(start.month, self.groups)

            if group_idx is not None:
                data = TimeVariateData(npy_reader,
                                       name=f"{variable}_{data_type}{date_to_str(start)}",
                                       lead_time=0, min_date=date_to_str(start),
                                       max_date=date_to_str(start_of_next_month))
                month2data[group_idx] = ConcatDataset(
                    [month2data[group_idx], data]) if group_idx in month2data else data

            start = start_of_next_month
            start_of_next_month = (start_of_next_month + get_month_datetime()).replace(day=1)

        group_idx = find_group_idx(start.month, self.groups)

        if group_idx is not None:
            data = TimeVariateData(npy_reader,
                                   name=f"{variable}_{data_type}{date_to_str(start)}",
                                   lead_time=0, min_date=date_to_str(start), max_date=date_to_str(max_date_datetime))

            month2data[group_idx] = ConcatDataset([month2data[group_idx], data]) if group_idx in month2data else data
        return month2data


class Transform(nn.Module):
    """Base class for transformations, extending PyTorch's nn.Module."""

    def __init__(self, requires_fit, exclude_at_evaluation=False):
        super(Transform, self).__init__()
        self.requires_fit = requires_fit
        self.exclude_at_evaluation = exclude_at_evaluation

    def transform(self, data):
        """Transform the data; this method must be implemented by subclasses."""
        raise NotImplementedError()

    def out_channels(self, in_channels):
        """
        Define the number of output channels given the input channels.
        """
        return in_channels

    def forward(self, data):
        return self.transform(data)

    def is_data_adaptive(self):
        """
        Check if the transform requires data adaptation (fitting).

        Returns:
            bool: True if fitting is required, False otherwise.
        """
        return self.requires_fit

    def summarize(self):
        """
        Summarize the transform; useful for introspection or debugging.

        Returns:
            dict: Dictionary containing the transform type.
        """
        return {"transform_type": self.__class__.__name__}


class IdentityTransform(Transform):
    """A transformation class that returns input data unchanged."""

    def __init__(self):
        super(IdentityTransform, self).__init__(requires_fit=False, exclude_at_evaluation=False)

    def transform(self, data):
        """Apply the transformation on the input data.

        Args:
            data (Any): Input data to be transformed.

        Returns:
            Any: The input data, unchanged.
        """
        return data

    def fit(self, dataset, batch_size=None, previous_transforms=None, disable_fitting_mode=False):
        """Fit method for the IdentityTransform (no actual fitting needed).

        Args:
            dataset (Dataset): Dataset for fitting.
            batch_size (int, optional): Batch size for fitting. Defaults to None.
            previous_transforms (list, optional): List of previous transforms. Defaults to None.
            disable_fitting_mode (bool, optional): Whether to disable fitting mode. Defaults to False.

        Returns:
            IdentityTransform: The fitted IdentityTransform object.
        """
        return self

    def revert(self, data):
        """Revert the transformation (returns data unchanged).

        Args:
            data (Any): Transformed data to revert.

        Returns:
            Any: The input data, unchanged.
        """
        return data

    def _update_parameters(self, data):
        """Update internal parameters based on data (no implementation needed)."""
        return None

    def summarize(self):
        """Summarizes information about transformation"""

        summary = super(IdentityTransform, self).summarize()
        summary.update({"identity_transform": True})
        return summary

    def clear_data_source(self):
        self._data_source = None


class StandardScaling(Transform):
    def __init__(self, unbiased=True, exclude_at_evaluation=False):
        """
        Initialize the StandardScaling transform with the option to apply bias correction.
        Args:
            unbiased (bool): If True, uses unbiased estimator in the computation of variance.
            exclude_at_evaluation (bool): If True, excludes this transformation during evaluation phase.
        """
        super(StandardScaling, self).__init__(requires_fit=True, exclude_at_evaluation=exclude_at_evaluation)
        self._count = 0  # Counter for the number of data points processed
        self._bias_correction = int(unbiased)  # Convert boolean to integer for correction calculation
        self.register_buffer("_mean", None)  # Buffer for storing the running mean
        self.register_buffer("_squared_differences", None)  # Buffer for storing the running squared differences
        self._data_source = None  # Placeholder for data source info

    def fit(self, dataset, batch_size=None, previous_transforms=None, disable_fitting_mode=False):
        """
        Fit the scaling parameters to the dataset.
        Args:
            dataset (Dataset): The dataset to fit the transformation.
            batch_size (int, optional): Size of batches to use for fitting. If None, fits to the entire dataset at once.
            previous_transforms (list of Transform objects, optional): Previous transformations to apply before fitting.
            disable_fitting_mode (bool): If True, temporarily disables dataset fitting mode.
        """
        if self._data_source is not None:
            raise Exception("[ERROR] Fit should only be called once on adaptive transform objects.")
        if previous_transforms is not None:
            assert isinstance(previous_transforms, list)
            for t in previous_transforms:
                assert isinstance(t, Transform)
        if not dataset.is_time_variate():
            self._fit_to_batch(dataset, [0], previous_transforms)
        else:
            in_fitting_mode = dataset.get_fitting_mode()
            if in_fitting_mode != disable_fitting_mode:
                dataset.set_fitting_mode(disable_fitting_mode)
            if batch_size is None:
                self._fit_to_batch(dataset, np.arange(len(dataset)), previous_transforms)
            else:
                assert isinstance(batch_size, int)
                idx = np.arange(len(dataset))
                batches = np.array_split(idx, np.ceil(len(idx) / batch_size))
                for idx_batch in batches:
                    self._fit_to_batch(dataset, idx_batch, previous_transforms)
            dataset.set_fitting_mode(in_fitting_mode)
        self._fill_data_source(dataset, previous_transforms)
        return self

    def _fill_data_source(self, dataset, previous_transforms):
        """
        Records summary of the dataset and any applied transformations.
        Args:
            dataset: The dataset being summarized.
            previous_transforms: List of transformations applied before this one.
        """
        self._data_source = dataset.summarize()
        if previous_transforms is not None:
            self._data_source.update({
                "previous_transforms": [
                    t.summarize() for t in reversed(previous_transforms)
                ]
            })

    def _update_stats(self, data_count, data_mean, data_squared_differences):
        """
        Updates the scaling parameters with new data.

        Args:
            data_count: Number of data points in the new data.
            data_mean: Mean of the new data.
            data_squared_differences: Sum of squared differences of the new data.

        Returns:
            StandardScaling: The updated StandardScaling object.
        """
        new_count = self._count + data_count
        self._squared_differences += data_squared_differences
        self._squared_differences += (data_mean - self._mean) ** 2 * ((data_count * self._count) / new_count)
        self._mean = ((self._count * self._mean) + (data_count * data_mean)) / new_count
        self._count = new_count
        return self

    def clear_data_source(self):
        """
        Clears the stored data source information.
        """
        self._data_source = None

    def _fit_to_batch(self, dataset, batch, previous_transforms):
        """
        Fits the scaling parameters to a specific batch of data.
        Args:
            dataset: The dataset containing the batch.
            batch: Indices representing the batch within the dataset.
            previous_transforms: List of transformations to apply before fitting.
        """
        for data in dataset.get_batch(batch):
            if previous_transforms is not None:
                for t in previous_transforms:
                    data = t.transform(data)
            self._update_parameters(data)

    def _std(self):
        """
        Computes the standard deviation using the accumulated squared differences and count.
        Returns:
            torch.tensor: Computed standard deviation.
        """
        return torch.sqrt(self._squared_differences / (self._count - self._bias_correction))

    def transform(self, data):
        """
        Applies the standard scaling transformation to the data.
        Args:
            data (tensor): Input data to transform.
        Returns:
            tensor: Transformed data.
        """
        return (data - self._mean) / self._std()

    def revert(self, data):
        """
        Reverts the standard scaling transformation.
        Args:
            data (tensor): Scaled data to revert.
        Returns:
            tensor: Original data before scaling.
        """
        return (self._std() * data) + self._mean

    def _update_parameters(self, data):
        """
        Updates the scaling parameters with new data.
        Args:
            data (tensor): New data to update parameters with.
        """
        data_stats = self._compute_stats(data)
        if self._mean is None:
            self._count, self._mean, self._squared_differences = data_stats
            return self
        return self._update_stats(*data_stats)

    def summarize(self):
        """
        Provides a summary of the transform's state including mean and standard deviation.
        Returns:
            dict: Summary of the transformation.
        """
        summary = super(StandardScaling, self).summarize()
        if self._mean and self._bias_correction and self._count and self._squared_differences:
            summary.update({"mean": self._mean})
            summary.update({"std": self._std()})
        else:
            summary.update({"mean": None})
            summary.update({"std": None})
        return summary


class LocalStandardScaling(StandardScaling):
    """
    Standard scaling transformation that computes mean and standard deviation over the batch dimension.
    , mchanges way how to compute mean and std, data.shape = (batch_size, channels, lat, lon)
    , mean and std is computed over dim 0

    Return:
        StandardScaling: The updated StandardScaling object.
    """

    def _compute_stats(self, data):
        data_count = data.shape[0]
        # computed batch size, mean over batch and SD over batch
        data_mean = torch.mean(data, dim=0, keepdim=True)
        return data_count, data_mean, torch.sum(torch.square(data - data_mean), dim=0, keepdim=True)


class GlobalStandardScaling(StandardScaling):
    """
    Standard scaling transformation that computes mean and standard deviation over the batch, lat, and lon dimensions,
    mean and std computed over dim 0 (batches), 2 (lat) and 3 (lon),
    data.shape = (batch_size, channels, lat, lon)

    Return:
        StandardScaling: The updated StandardScaling object.
    """

    def _compute_stats(self, data):
        shape = data.shape
        data_count = shape[0] * shape[2] * shape[3]
        data_mean = torch.mean(data, dim=(0, 2, 3), keepdim=True)
        return data_count, data_mean, torch.sum(torch.square(data - data_mean), dim=(0, 2, 3), keepdim=True)


def get_transformation_by_name(name):
    """
    Returns a transformation by its name.

    Args:
        name: Name of the transformation.

    Rraises:
        Exception: If the transformation is unknown.

    Returns:
        Transform: A transformation object.
    """
    if name == "GlobalStandardScaling":
        return GlobalStandardScaling
    elif name == "LocalStandardScaling":
        return LocalStandardScaling
    elif name == "IdentityTransform":
        return IdentityTransform
    else:
        raise Exception("[ERROR] Unknown transformation <{}>.".format(name))
