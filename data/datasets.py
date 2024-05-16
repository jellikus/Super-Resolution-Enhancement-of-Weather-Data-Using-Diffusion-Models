from collections import OrderedDict
from datetime import datetime
from torch.utils.data import Dataset
from intervaltree import IntervalTree
from .utils import str_to_date

import numpy as np
import torch

from configs.config import DataConfig
from .npy_reader import WNPYReader

config = DataConfig()
TEMPORAL_RESOLUTION = np.timedelta64(config.temporal_resolution_value, config.temporal_resolution_unit)
DATETIME_FORMAT = config.datetime_format

"""
source: https://github.com/davitpapikyan/Probabilistic-Downscaling-of-Climate-Variables/blob/main/weatherbench_data/datasets.py
!! We modified this code with intervaltree data handling for multi month data and added the ability to get data by date. !!
"""


# Parse a date input into a NumPy datetime64 object.
def _parse_date_input(date_input, datetime_format=None):
    """
    Parse a date input into a NumPy datetime64 object.

    Args:
        date_input: The input date, which can be of type string, datetime, or np.datetime64.
        datetime_format: The format of the date string if the date_input is a string (default is None).

    Returns:
        np.datetime64: Parsed date in NumPy datetime64 format.
    """
    if date_input is None:
        return None
    input_type = type(date_input)
    if input_type == np.datetime64:
        return date_input
    elif input_type == datetime:
        return np.datetime64(date_input)
    elif input_type == str:
        if datetime_format is None:
            datetime_format = DATETIME_FORMAT
        try:
            date = datetime.strptime(date_input, datetime_format)
        except Exception:
            raise Exception(
                "[ERROR] Encountered invalid date string input (input: {}, datetime format: {}).".format(
                    date_input, datetime_format
                )
            )
        return np.datetime64(date)
    else:
        raise Exception("[ERROR] Encountered invalid date input.")


def _verify_date_bounds(min_date, max_date):
    """
    Verify that the date bounds are valid numpy.datetime64 objects and within acceptable bounds.

    Args:
        min_date: Minimum date of the range (np.datetime64 or None).
        max_date: Maximum date of the range (np.datetime64 or None).

    Raises:
        AssertionError: If the date bounds are invalid.
    """
    assert (isinstance(min_date, np.datetime64) or min_date is None) and (
            isinstance(max_date, np.datetime64) or max_date is None), \
        "[ERROR] Date bounds must be given as numpy.datetime64 objects."
    if min_date is not None:
        # check if date is consistent with temporal resolution of data set, in other words modulo of date and temporal resolution must be zero
        assert (min_date - np.datetime64("2020-01-01T00")) % TEMPORAL_RESOLUTION == np.timedelta64(0, "ms"), \
            "[ERROR] Date bounds must be consistent with the temporal resolution of the data set ({}).".format(
                TEMPORAL_RESOLUTION
            )
    if max_date is not None:
        assert (max_date - np.datetime64("2020-01-01T00")) % TEMPORAL_RESOLUTION == np.timedelta64(0, "ms"), \
            "[ERROR] Date bounds must be consistent with the temporal resolution of the data set ({}).".format(
                TEMPORAL_RESOLUTION
            )
    if min_date is not None and max_date is not None:
        assert max_date > min_date, "[ERROR] Lower date bound ({}) must be earlier than upper ({}).".format(min_date,
                                                                                                            max_date)


# method returns a lambda function. This lambda function takes an argument x and simply returns x
class DefaultIdentityMapping(dict):
    """
    A dictionary that returns a lambda function returning its input for missing keys.
    """
    def __missing__(self, key):
        return lambda x: x


class TimeVariateData(Dataset):
    """
    A class representing a dataset with time-varying data.
    """
    def __init__(self, source: WNPYReader, name=None, lead_time=None, delays=None, min_date=None,
                 max_date=None, transform: dict = None):
        """
        Initialize the TimeVariateData object.

        Args:
            source: The data source, expected to be a WNPYReader object.
            name: Name of the dataset (default is None, which uses the source name).
            lead_time: Lead time for data preparation (default is None).
            delays: List of delays to apply (default is None).
            min_date: Minimum date for the dataset (default is None).
            max_date: Maximum date for the dataset (default is None).
            transform: Transformation dictionary to apply (default is None).
        """
        assert isinstance(source, WNPYReader)
        assert source.is_time_variate()
        self.name = name if name is not None else source.name
        if name is not None:
            assert isinstance(name, str)
        self.wnpy_reader = source
        self._lead_time = TEMPORAL_RESOLUTION * lead_time if lead_time is not None else None
        if delays is not None:
            assert isinstance(delays, list), "[ERROR] Delay parameter must be given as list."
            for d in delays:
                assert isinstance(d, int), "[ERROR] Delay parameter must be given as list of ints."
            if 0 not in delays:
                delays = [0] + delays
            delays = np.array(delays)
            assert len(delays) == len(np.unique(delays)), "[ERROR] Delays must be unique."
            self._delays = TEMPORAL_RESOLUTION * delays
        else:
            self._delays = None
        self.min_date = None
        self.max_date = None
        self._sample_index = None
        self.set_date_range(min_date, max_date)
        self._fitting_mode = False
        self._transform = transform if transform else DefaultIdentityMapping()
        self.date_ranges_tree = IntervalTree()
        self.date_ranges_tree[self.min_date:self.max_date] = (self.min_date, self.max_date)

    def set_date_range(self, min_date=None, max_date=None, datetime_format=None):
        """
        Set the date range for the dataset.

        Args:
            min_date: Minimum date for the dataset (default is None).
            max_date: Maximum date for the dataset (default is None).
            datetime_format: Format of the date string if the date inputs are strings (default is None).

        Returns:
            self: Updated TimeVariateData object.
        """
        min_date = _parse_date_input(min_date, datetime_format)
        max_date = _parse_date_input(max_date, datetime_format)

        # Verify that the specified date range is within acceptable bounds
        _verify_date_bounds(min_date, max_date)

        # Get the valid time stamps from the source data np.arrange(min_date, max_date, TEMPORAL_RESOLUTION)
        source_time_stamps = self.wnpy_reader.get_valid_time_stamps()

        # Calculate the minimum and maximum dates from the source data
        source_min_date = np.min(source_time_stamps)
        source_max_date = np.max(source_time_stamps) + TEMPORAL_RESOLUTION

        # Initialize admissible minimum and maximum dates
        admissible_min_date = source_min_date
        admissible_max_date = source_max_date

        # Adjust admissible dates based on the lead time if it is specified
        if self._lead_time is not None:
            admissible_min_date = admissible_min_date - self._lead_time
            admissible_max_date = admissible_max_date - self._lead_time
        if self._delays is not None:
            admissible_min_date = admissible_min_date - np.min(self._delays)
            admissible_max_date = admissible_max_date - np.max(self._delays)
        if min_date is None:
            self.min_date = admissible_min_date
        else:
            assert min_date >= admissible_min_date, \
                "[ERROR] Requested minimum date ({}) is beyond the range of admissible dates ([{}] – [{}]).".format(
                    min_date, admissible_min_date, admissible_max_date
                )
            self.min_date = min_date
        if max_date is None:
            self.max_date = admissible_max_date
        else:
            assert max_date <= admissible_max_date, \
                "[ERROR] Requested maximum date ({}) is beyond the range of admissible dates ([{}] – [{}]).".format(
                    max_date, admissible_min_date, admissible_max_date
                )
            self.max_date = max_date
        self._build_sample_index()
        return self

    def _update_sample_index(self, new_min_date, new_max_date):
        """
        Update the sample index with new date range.

        Args:
            new_min_date: The new minimum date for the dataset.
            new_max_date: The new maximum date for the dataset.
        """
        max_idx = max(self._sample_index)
        valid_samples = np.arange(new_min_date, new_max_date, TEMPORAL_RESOLUTION)
        index = {(i + max_idx + 1): time_stamp for i, time_stamp in enumerate(valid_samples)}
        self._sample_index.update(index)

    def add_data_by_date(self, min_date, max_date, datetime_format=None):
        """
        Add data to the dataset by specifying a date range.

        Args:
            min_date: The minimum date of the range to add.
            max_date: The maximum date of the range to add.
            datetime_format: The format of the date string if the date inputs are strings (default is None).

        Raises:
            AssertionError: If the date range overlaps with existing ranges or is invalid.
        """
        min_date = _parse_date_input(min_date, datetime_format)
        max_date = _parse_date_input(max_date, datetime_format)

        _verify_date_bounds(min_date, max_date)

        assert bool(self.date_ranges_tree[
                    min_date:max_date]) is False, f"[ERROR] Requested date range ({min_date}, {max_date}) overlaps with existing date ranges."

        source_time_stamps = self.wnpy_reader.get_valid_time_stamps()
        admissible_min_date = np.min(source_time_stamps)
        admissible_max_date = np.max(source_time_stamps) + TEMPORAL_RESOLUTION

        assert min_date != None, "[ERROR] Requested minimum date is None."

        assert min_date >= admissible_min_date, \
            "[ERROR] Requested minimum date ({}) is beyond the range of admissible dates ([{}] – [{}]).".format(
                min_date, admissible_min_date, admissible_max_date
            )

        assert max_date != None, "[ERROR] Requested maximum date is None."

        assert max_date <= admissible_max_date, \
            "[ERROR] Requested maximum date ({}) is beyond the range of admissible dates ([{}] – [{}]).".format(
                max_date, admissible_min_date, admissible_max_date
            )

        if min_date < self.min_date:
            self.min_date = min_date

        if max_date > self.max_date:
            self.max_date = max_date

        self.date_ranges_tree[min_date:max_date] = (min_date, max_date)
        self._update_sample_index(min_date, max_date)

    def _build_sample_index(self):
        """
         Build the sample index for the dataset.
         """
        valid_samples = np.arange(self.min_date, self.max_date, TEMPORAL_RESOLUTION)
        self._sample_index = {i: time_stamp for i, time_stamp in enumerate(valid_samples)}

    def set_transform(self, transform: dict):
        """
        Set the transformation to be applied to the data.

        Args:
            transform: A dictionary of transformations to apply.
        """
        self._transform = transform

    def get_transform(self):
        """
        Get the current transformation applied to the data.

        Returns:
            dict: The transformation dictionary.
        """
        return self._transform

    def __getitem__(self, item):
        """
        Get a data item by index or timestamp.

        Args:
            item: The index or timestamp of the data to retrieve.

        Returns:
            tuple: The transformed data, dataset name, and month index.
        """

        # if time si datetime object return the item
        if isinstance(item, np.datetime64):
            time_stamp = item
        else:
            time_stamp = self._sample_index[item]

        month = int(time_stamp.astype("datetime64[M]").astype(int) % 12 + 1)

        if month not in self._transform:
            month = 0

        if self._lead_time is not None:
            time_stamp = time_stamp + self._lead_time
        if self._fitting_mode or self._delays is None:
            return self._transform[month](self.wnpy_reader[time_stamp]), self.name, month
        else:
            return tuple((self._transform[month](self.wnpy_reader[delayed_time]), self.name, month)
                         for delayed_time in (time_stamp + self._delays))

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self._sample_index)

    def get_channel_count(self):
        """
        Get the number of channels in the dataset.

        Returns:
            int: The number of channels.
        """
        source_channels = self.wnpy_reader.get_channel_count()
        if self._delays is not None:
            return len(self._delays) * source_channels
        else:
            return source_channels

    @staticmethod
    def _generate_batches(data, data_length: int, chunk_size: int = 50000):
        """
        Generate batches of data.

        Args:
            data: The data to be batched.
            data_length: The length of the data.
            chunk_size: The size of each batch (default is 50000).

        Yields:
            list: A batch of data.
        """
        for start in range(0, data_length, chunk_size):
            yield [next(data) for _ in range(start, min(data_length, start + chunk_size))]

    def get_batch(self, indices):
        """
        Get a batch of data by indices.

        Args:
            indices: The indices of the data to retrieve.

        Yields:
            torch.Tensor: A batch of concatenated data.
        """
        data = (self.__getitem__(i) for i in indices)
        for chunk_of_data in self._generate_batches(data, len(indices)):
            if self._delays is not None:
                yield tuple(torch.cat(d[0], dim=0) for d in chunk_of_data)
            else:
                yield torch.cat([tup[0] for tup in chunk_of_data], dim=0)

    def enable_fitting_mode(self):
        """
        Enable fitting mode for the dataset.

        Returns:
            self: Updated TimeVariateData object.
        """
        return self.set_fitting_mode(True)

    def disable_fitting_mode(self):
        """
        Disable fitting mode for the dataset.

        Returns:
            self: Updated TimeVariateData object.
        """
        return self.set_fitting_mode(False)

    def set_fitting_mode(self, mode):
        """
        Set the fitting mode for the dataset.

        Args:
            mode: Boolean indicating whether to enable or disable fitting mode.

        Returns:
            self: Updated TimeVariateData object.

        Raises:
            AssertionError: If the mode is not a boolean.
        """
        assert isinstance(mode, bool)
        self._fitting_mode = mode
        return self

    def get_fitting_mode(self):
        """
        Get the current fitting mode status.

        Returns:
            bool: The current fitting mode status.
        """
        return self._fitting_mode

    @staticmethod
    def is_time_variate():
        """
        Check if the data is time-varying.

        Returns:
            bool: True, indicating the data is time-varying.
        """
        return True

    def summarize(self):
        return {
            "data_type": "TimeVariateData",
            "path": self.wnpy_reader.path,
            "date_range": [
                self._numpy_date_to_datetime(min(self.date_ranges_tree)[0]).strftime(DATETIME_FORMAT),
                self._numpy_date_to_datetime(max(self.date_ranges_tree)[1]).strftime(DATETIME_FORMAT)
            ],
            "lead_time": self._lead_time,
            "delays": self._delays,
            "name": self.name,
            "number_of_intervals": len(self.date_ranges_tree),
        }

    def get_time_intervals(self):
        """
        Get the time intervals of the dataset.

        Returns:
            generator: Time intervals of the dataset.
        """
        return ((iv.begin, iv.end) for iv in self.date_ranges_tree.items())

    @staticmethod
    def _numpy_date_to_datetime(time_stamp):
        """
        Convert a numpy datetime64 to a datetime object.

        Args:
            time_stamp: The numpy datetime64 object.

        Returns:
            datetime: Converted datetime object.
        """
        total_seconds = (time_stamp - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s")
        return datetime.utcfromtimestamp(total_seconds)

    def get_valid_time_stamps(self):
        """
        Get valid time stamps from the dataset.

        Returns:
            list: List of valid time stamps.
        """
        return sorted(self._sample_index.values())


class ConstantData(Dataset):
    """
    A class representing a dataset with constant data.
    """
    def __init__(self, source, name=None, min_date=None, max_date=None, datetime_format=None):
        """
        Initialize the ConstantData object.

        Args:
            source: The data source, expected to be a WNPYReader object.
            name: Name of the dataset (default is None, which uses the source name).
            min_date: Minimum date for the dataset (default is None).
            max_date: Maximum date for the dataset (default is None).
            datetime_format: Format of the date string if the date inputs are strings (default is None).
        """
        assert isinstance(source, WNPYReader)
        assert not source.is_time_variate()
        if name is not None:
            assert isinstance(name, str)
        self.name = name if name is not None else source.name
        self.wnpy_reader = source
        min_date = _parse_date_input(min_date, datetime_format)
        max_date = _parse_date_input(max_date, datetime_format)
        _verify_date_bounds(min_date, max_date)
        self.min_date = min_date
        self.max_date = max_date
        self._num_samples = None
        self.set_date_range(min_date, max_date)
        self._fitting_mode = False

    def __getitem__(self, item):
        """
        Get a data item by index.

        Args:
            item: The index of the data to retrieve.

        Returns:
            data: The data at the specified index.

        Raises:
            Exception: If the requested item is out of range.
        """
        if item < self._num_samples:
            return self.wnpy_reader[item]
        else:
            raise Exception(
                "[ERROR] Requested item ({}) is beyond the range of valid item numbers ([0, {}]).".format(item,
                                                                                                          self._num_samples))

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return self._num_samples

    def set_date_range(self, min_date=None, max_date=None, datetime_format=None):
        """
        Set the date range for the dataset.

        Args:
            min_date: Minimum date for the dataset (default is None).
            max_date: Maximum date for the dataset (default is None).
            datetime_format: Format of the date string if the date inputs are strings (default is None).

        Returns:
            self: Updated ConstantData object.
        """
        min_date = _parse_date_input(min_date, datetime_format)
        max_date = _parse_date_input(max_date, datetime_format)
        _verify_date_bounds(min_date, max_date)
        self.min_date = min_date
        self.max_date = max_date
        if min_date is None or max_date is None:
            self._num_samples = 1
        else:
            self._num_samples = int((max_date - min_date) / TEMPORAL_RESOLUTION)
        return self

    def get_channel_count(self):
        """
        Get the number of channels in the dataset.

        Returns:
            int: The number of channels.
        """
        return self.wnpy_reader.get_channel_count()

    def enable_fitting_mode(self):
        """
        Enable fitting mode for the dataset.

        Returns:
            self: Updated ConstantData object.
        """
        return self.set_fitting_mode(True)

    def disable_fitting_mode(self):
        """
        Disable fitting mode for the dataset.

        Returns:
            self: Updated ConstantData object.
        """
        return self.set_fitting_mode(False)

    def set_fitting_mode(self, mode):
        """
        Set the fitting mode for the dataset.

        Args:
            mode: Boolean indicating whether to enable or disable fitting mode.

        Returns:
            self: Updated ConstantData object.

        Raises:
            AssertionError: If the mode is not a boolean.
        """
        assert isinstance(mode, bool)
        self._fitting_mode = mode
        return self

    def get_fitting_mode(self):
        """
        Get the current fitting mode status.

        Returns:
            bool: The current fitting mode status.
        """
        return self._fitting_mode

    @staticmethod
    def is_time_variate():
        """
        Check if the data is time-varying.

        Returns:
            bool: False, indicating the data is constant.
        """
        return False

    def summarize(self):
        """
        Summarize the dataset information.

        Returns:
            dict: Summary of the dataset.
        """
        return {
            "data_type": "ConstantData",
            "path": self.wnpy_reader.path
        }


class WeatherBenchData(Dataset):
    """
    A class representing a collection of time-varying and constant datasets.
    """
    def __init__(self, min_date=None, max_date=None, datetime_format=None):
        """
        Initialize the WeatherBenchData object.

        Args:
            min_date: Minimum date for the dataset (default is None).
            max_date: Maximum date for the dataset (default is None).
            datetime_format: Format of the date string if the date inputs are strings (default is None).
        """
        min_date = _parse_date_input(min_date, datetime_format)
        max_date = _parse_date_input(max_date, datetime_format)
        _verify_date_bounds(min_date, max_date)
        self.min_date = min_date
        self.max_date = max_date
        self.data_groups = OrderedDict({})

    def add_data_group(self, group_key, datasets, _except_on_changing_date_bounds=False):
        """
        Add a group of datasets to the collection.

        Args:
            group_key: A string key for the group.
            datasets: A list of datasets to add.
            _except_on_changing_date_bounds: Whether to raise an exception if date bounds change (default is False).

        Returns:
            self: Updated WeatherBenchData object.

        Raises:
            AssertionError: If the input is invalid or the date bounds change when _except_on_changing_date_bounds is True.
        """

        self._verify_data_group_inputs(group_key, datasets)
        min_dates = [dataset.min_date for dataset in datasets if dataset.min_date is not None]

        if len(min_dates) > 0:
            common_min_date = np.max(min_dates)
        else:
            common_min_date = None
        if _except_on_changing_date_bounds:
            assert common_min_date == self.min_date, "[ERROR] Encountered missing time stamps."
        else:
            if (common_min_date is not None) and (self.min_date is None or common_min_date > self.min_date):
                self.min_date = common_min_date
        max_dates = [dataset.max_date for dataset in datasets if dataset.max_date is not None]
        if len(max_dates) > 0:
            common_max_date = np.min(max_dates)
        else:
            common_max_date = None
        if _except_on_changing_date_bounds:
            assert common_max_date == self.max_date, "[ERROR] Encountered missing time stamps."
        else:
            if (common_max_date is not None) and (self.max_date is None or common_max_date < self.max_date):
                self.max_date = common_max_date

        self.data_groups.update({group_key: {dataset.name: dataset for dataset in datasets}})

        self._check_groups_date_bounds()
        return self

    def _check_groups_date_bounds(self):
        """
        Check that all datasets in all groups have the same date bounds.

        Raises:
            AssertionError: If any dataset has different date bounds.
        """
        assert self.min_date is not None and self.max_date is not None, "[ERROR] Date bounds must be set."

        for group in self.data_groups.values():
            for dataset in group.values():
                assert dataset.min_date == self.min_date, "[ERROR] Date bounds are not same for all groups."
                assert dataset.max_date == self.max_date, "[ERROR] Date bounds are not same for all groups."

    def _verify_data_group_inputs(self, group_key, datasets):
        """
        Verify the inputs for adding a data group.

        Args:
            group_key: A string key for the group.
            datasets: A list of datasets to add.

        Raises:
            AssertionError: If the inputs are invalid.
        """
        assert isinstance(group_key, str), "[ERROR] Group keys must be of type string."
        assert group_key not in self.data_groups, "[ERROR] Group keys must be unique. Key <{}> is already existing.".format(
            group_key)
        if not isinstance(datasets, list):
            datasets = [datasets]
        for dataset in datasets:
            assert isinstance(dataset, (ConstantData, TimeVariateData))
            "[ERROR] Datasets must be given as TimeVariateData or ConstantData objects or a list thereof."
        data_names = [dataset.name for dataset in datasets]
        assert len(data_names) == len(
            np.unique(data_names)), "[ERROR] Dataset names must be unique within a common parameter group."

    def remove_data_group(self, group_key):
        """
        Remove a data group from the collection.

        Args:
            group_key: The key of the group to remove.

        Returns:
            self: Updated WeatherBenchData object.
        """
        if group_key in self.data_groups:
            del self.data_groups[group_key]
        return self

    def _update_date_bounds(self):
        """
        Update the date bounds for all datasets in all groups.
        """
        for group in self.data_groups.values():
            for dataset in group.values():
                dataset.set_date_range(self.min_date, self.max_date)

    def set_date_range(self, min_date=None, max_date=None, datetime_format=None):
        """
        Set the date range for the dataset.

        Args:
            min_date: Minimum date for the dataset (default is None).
            max_date: Maximum date for the dataset (default is None).
            datetime_format: Format of the date string if the date inputs are strings (default is None).

        Returns:
            self: Updated WeatherBenchData object.
        """
        min_date = _parse_date_input(min_date, datetime_format)
        max_date = _parse_date_input(max_date, datetime_format)
        _verify_date_bounds(min_date, max_date)
        self.min_date = min_date
        self.max_date = max_date
        self._update_date_bounds()
        return self

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        if len(self.data_groups) == 0:
            return 0

        return len(list(list(self.data_groups.values())[0].values())[0])
        # return max((len(group) for group in self.data_groups.values()))

    def __getitem__(self, item):
        # dataset[item][1] is the integer indicating month index (from 1 for Januray to 12 for December).
        # dataset[item][2] name of the variable.
        # dataset[item][0] tensor data.
        return tuple(tuple(dataset[item] for dataset in group.values()) for group in self.data_groups.values())

    def get_data_names(self):
        """
        Get the names of all data items.

        Returns:
            dict: A dictionary of data names.
        """
        return {
            group_key: tuple(dataset.name for dataset in group.values())
            for group_key, group in self.data_groups.items()
        }

    def get_named_item(self, item):
        return {
            group_key: {dataset.name: dataset[item] for dataset in group.values()}
            for group_key, group in self.data_groups.items()
        }

    def get_channel_count(self, group_key=None):
        if group_key is None:
            return {gk: self.get_channel_count(group_key=gk) for gk in self.data_groups}
        elif group_key in self.data_groups:
            return np.sum([dataset.get_channel_count() for dataset in self.data_groups[group_key].values()])
        else:
            raise Exception("[ERROR] Dataset does not contain a data group named <{}>.".format(group_key))

    def get_valid_time_stamps(self):
        """
        Get valid time stamps from the dataset.

        Returns:
            np.ndarray: An array of valid time stamps.
        """
        return np.arange(self.min_date, self.max_date, TEMPORAL_RESOLUTION)

    def get_data_by_date(self, date):
        date = str_to_date(date)

        # check if date time is within the date range
        assert self.min_date <= date <= self.max_date, "[ERROR] Requested date is beyond the range of valid dates. Use a date between {} and {} for this validation dataset configuration.".format(
            self.min_date, self.max_date)

        date = np.datetime64(date)
        return tuple(tuple(dataset[date] for dataset in group.values()) for group in self.data_groups.values())

    def summarize(self):
        return {
            "data_type": "WeatherBenchData",
            "date_range": [
                self._numpy_date_to_datetime(self.min_date).strftime(DATETIME_FORMAT),
                self._numpy_date_to_datetime(self.max_date).strftime(DATETIME_FORMAT)
            ],
            "data_groups": {
                group_key: {
                    dataset.name: dataset.summarize()
                    for dataset in group.values()
                }
                for group_key, group in self.data_groups.items()
            }
        }

    @staticmethod
    def _numpy_date_to_datetime(time_stamp):
        """
        Convert a numpy datetime64 to a datetime object.

        Args:
            time_stamp: The numpy datetime64 object.

        Returns:
            datetime: Converted datetime object.
        """
        total_seconds = (time_stamp - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s")
        return datetime.utcfromtimestamp(total_seconds)
