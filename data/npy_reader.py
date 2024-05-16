import json
import os
from datetime import datetime
from itertools import chain

import numpy as np
import torch
from configs.config import DataConfig

config = DataConfig()
DATETIME_FORMAT = config.datetime_format
TEMPORAL_RESOLUTION = np.timedelta64(config.temporal_resolution_value, config.temporal_resolution_unit)
DIRECTORY_NAME_META_DATA = config.directory_name_meta_data
FILE_NAME_META_DATA = config.file_name_meta_data
DIRECTORY_NAME_SAMPLE_DATA = config.directory_name_sample_data

"""
source: https://github.com/davitpapikyan/Probabilistic-Downscaling-of-Climate-Variables/blob/main/weatherbench_data/datastorage.py
There are done small changes in this code.
"""

class WNPYReader(object):
    """
    creates object that can be used to read .npy data and reshape it to right format [batch_size, channels, height, width]
    """

    def __init__(self, path, domain_dimension=2, sample_index=None):
        """
        Initialize the WNPYReader.

        Args:
            path (str): The path to the data file.
            domain_dimension (int): The dimension of the data domain. Default is 2.
            sample_index (np.array, np.array): indexed files in directories if exists
        """

        self._verify_path(path)
        self.path = os.path.abspath(path)
        self.domain_dimension=domain_dimension
        self.meta_data = None
        self._load_meta_data()
        assert len(self.meta_data["dims"]) >= domain_dimension
        self.name = self.meta_data["name"]
        self._is_time_variate = self.meta_data["time_variate"]
        self._samples = None
        if sample_index is None:
            self._read_sample_directory()
        else:
            self._samples = sample_index

    @staticmethod
    def _verify_path(path):
        """
        Verify the structure of the specified directory path.

        Args:
            path (str): The path to verify.

        Asserts:
            The path must be a directory containing exactly two subdirectories, one for metadata and one for sample data,
            matching the expected folder structure for a WeatherBench parameter directory.
        """
        assert os.path.isdir(path), "[ERROR] <{}> is not a valid directory path.".format(path)
        contents = os.listdir(path)
        assert len(contents) == 2 and os.path.isdir(os.path.join(path, DIRECTORY_NAME_META_DATA)) and os.path.isdir(os.path.join(path, DIRECTORY_NAME_SAMPLE_DATA)), \
            "[ERROR] <{}> does not follow the expected folder structure of a WeatherBench parameter directory.".format(path)

    def _load_meta_data(self):
        """
        Load metadata from a JSON file located in the metadata directory.
        Converts coordinate lists into numpy arrays for easier manipulation later.
        """
        with open(os.path.join(self.path, DIRECTORY_NAME_META_DATA, FILE_NAME_META_DATA + ".json"), "r") as f:
            self.meta_data = json.load(f)
        coordinates = self.meta_data["coords"]
        # convert coordinate value lists to numpy arrays
        for c in coordinates:
            c.update({"values": np.array(c["values"])})

    def _read_sample_directory(self):
        """
        Read the sample directory and depending on the data variability (time-variate or not),
        load the appropriate data format.
        """
        sample_directory = os.path.join(self.path, DIRECTORY_NAME_SAMPLE_DATA)
        if self._is_time_variate:
            sample_time_stamps = self._build_sample_index(sample_directory)
            self._verify_data_completeness(sample_time_stamps)
        else:
            self._load_constant_data(sample_directory)

    def _build_sample_index(self, sample_directory):
        """
        Build an index of sample files and their corresponding time stamps.

        Args:
            sample_directory (str): The path to the directory containing the sample files.

        Returns:
            numpy.array: Sorted array of time stamps corresponding to the samples.
        """
        sub_directories = [
            d for d in sorted(os.listdir(sample_directory))
            if os.path.isdir(os.path.join(sample_directory, d))
        ]
        samples = []
        time_stamps = []

        for sub_directory in sub_directories:
            contents_s = []
            contents_t = []
            for f in sorted(os.listdir(os.path.join(sample_directory, sub_directory))):
                if self._matches_sample_file_convention(f):
                    contents_s.append(os.path.join(sample_directory, sub_directory, f))
                    contents_t.append(self._file_name_to_datetime(f))
            samples.append(contents_s)
            time_stamps.append(contents_t)

        # flatten lists
        samples = np.array(list(chain.from_iterable(samples)))
        time_stamps = np.array(list(chain.from_iterable(time_stamps)))

        # sort lists indexes
        sorting_index = np.argsort(time_stamps)

        # create tuple where first element is the first time stamp and the second element are the sorted sample paths by time stamp
        self._samples = (time_stamps[0], samples[sorting_index])
        # return sorted time_stamps
        return time_stamps[sorting_index]

    @staticmethod
    def _verify_data_completeness(sample_time_stamps):
        """
        Verify the completeness of the time-series data against expected intervals.

        Args:
            sample_time_stamps (numpy.array): Array of time stamps of the data samples.

        Asserts:
            The data covers a comprehensive and continuous range based on the defined temporal resolution.
        """
        min_date = sample_time_stamps[0]
        max_date = sample_time_stamps[-1]
        assert len(sample_time_stamps) == int((max_date - min_date) / TEMPORAL_RESOLUTION) + 1, \
            "[ERROR] encountered missing data values."
        assert np.all(np.diff(sample_time_stamps) == TEMPORAL_RESOLUTION)

    def _matches_sample_file_convention(self, f):
        """
        Check if the filename matches the expected convention for sample files.

        Args:
            filename (str): The name of the file to check.

        Returns:
            bool: True if the file matches the expected convention, False otherwise.
        """
        if not f.endswith(".npy"):
            return False
        f_split = f.split(".")
        if len(f_split) > 2:
            return False
        try:
            date = self._file_name_to_datetime(f)
        except:
            return False
        return True

    @staticmethod
    def _file_name_to_datetime(f):
        """
        Convert a filename string to a numpy datetime64 object.

        Args:
            f (str): The filename to convert, expected to be in the format "datetime.extension".

        Returns:
            numpy.datetime64: The datetime extracted from the filename, converted to datetime64.
        """
        return np.datetime64(datetime.strptime(f.split(".")[0], DATETIME_FORMAT))

    def _load_constant_data(self, sample_directory):
        """
        Load constant data from a numpy file stored in the sample directory.

        Args:
            sample_directory (str): Directory containing the constant data file.
        """
        data = torch.tensor(np.load(os.path.join(sample_directory, "constant.npy")))
        self._samples = self._to_pytorch_standard_shape(data)

    def _to_pytorch_standard_shape(self, data):
        """
        Adjust data dimensions to fit PyTorch's expected format for models.

        Args:
            data (torch.Tensor): Data to adjust.

        Returns:
            torch.Tensor: Data reshaped to [batch_size, channels, height, width].
        """
        dim = len(data.shape)
        domain_dim = self.domain_dimension
        # care for channel dimensions
        if dim == domain_dim:
            data = data.unsqueeze(dim=0)
        elif dim > domain_dim + 1:
            data = torch.flatten(data, start_dim=0, end_dim=-(domain_dim + 1))

        # add batch (time) dimension
        # return [batch_size, channels, height, width]
        return data.unsqueeze(dim=0)

    def __len__(self):
        """
        Define the length of the dataset, which varies depending on whether the data is time-variate.

        Returns:
            int: Number of data entries in the dataset.
        """
        if self._is_time_variate:
            return len(self._samples[1])
        else:
            return 1

    def __getitem__(self, item):
        """
        Retrieve an item from the dataset by index or timestamp.

        Args:
            item (int or datetime): Index or explicit datetime of the sample to retrieve.

        Returns:
            torch.Tensor: The requested data sample reshaped to PyTorch's standard format.
        """
        if self._is_time_variate:
            if isinstance(item, int):
                item = self.get_valid_time_stamps()[item]

            idx = int((item - self._samples[0]) / TEMPORAL_RESOLUTION)
            data = torch.tensor(np.load(self._samples[1][idx]))
            return self._to_pytorch_standard_shape(data)
        else:
            return self._samples

    def get_valid_time_stamps(self):
        """
        Generate a list of valid timestamps for time-variate data.

        Returns:
            numpy.array: Array of timestamps from start to end at specified temporal resolution.
        """
        if self._is_time_variate:
            min_date = self._samples[0]
            return np.arange(min_date, min_date + len(self._samples[1]) * TEMPORAL_RESOLUTION, TEMPORAL_RESOLUTION)
        else:
            return None

    def is_time_variate(self):
        """
        Check if the dataset contains time-variate data.

        Returns:
            bool: True if the dataset contains time-variate data, False otherwise.
        """
        return self._is_time_variate

    def get_channel_count(self):
        """
        Calculate the number of channels in the dataset based on the metadata dimensions excluding the last dimensions.

        Returns:
            int: Number of channels derived from the product of dimensions excluding the spatial (last) dimensions.
        """
        count = 1
        for axis_length in self.meta_data["shape"][0:-self.domain_dimension]:
            count = count * axis_length
        return int(count)