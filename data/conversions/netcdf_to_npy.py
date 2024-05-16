import json
import os
import shutil
from datetime import datetime

import numpy as np
import xarray as xr
import argparse
from pathlib import Path
from tqdm import tqdm
from configs.config import DataConfig

config = DataConfig()
DATETIME_FORMAT = config.datetime_format
TEMPORAL_RESOLUTION = np.timedelta64(config.temporal_resolution_value, config.temporal_resolution_unit)
DIRECTORY_NAME_META_DATA = config.directory_name_meta_data
FILE_NAME_META_DATA = config.file_name_meta_data
FILE_NAME_CONSTANT_DATA = config.file_name_constant_data
DIRECTORY_NAME_SAMPLE_DATA = config.directory_name_sample_data
NETCDF_EXTENSION = config.netcdf_extension
NUMPY_EXTENSION = config.numpy_extension

"""
source: https://github.com/davitpapikyan/Probabilistic-Downscaling-of-Climate-Variables/blob/main/weatherbench_data/datasets.py
"""


class NetcdfToNpy:
    """Class to convert NetCDF files to numpy binary format, handling both metadata and sample data."""

    def __init__(self, source_dir, target_dir, netcdf_extension=NETCDF_EXTENSION, numpy_extension=NUMPY_EXTENSION,
                 datetime_format=DATETIME_FORMAT):
        """
        Initialize the converter with directory and file settings.

        Args:
            source_dir (str): Path to the directory containing the NetCDF files.
            target_dir (str): Path where the numpy files will be saved.
            netcdf_extension (str): File extension for NetCDF files (default is set by configuration).
            numpy_extension (str): File extension for numpy files (default is set by configuration).
            datetime_format (str): Date and time format to use in filenames.
        """
        self.NETCDF_EXTENSION = netcdf_extension
        self.NUMPY_EXTENSION = numpy_extension
        self.DATETIME_FORMAT = datetime_format
        self.source_dir = source_dir
        self.target_dir = target_dir

    def set_source_dir(self, source_directory):
        self.source_dir = os.path.abspath(source_directory)
        return self

    def set_target_dir(self, target_directory):
        self.target_dir = os.path.abspath(target_directory)
        return self

    def open_files(self, chunks=None, parallel=True, verbose=False):
        """
        Open NetCDF files using xarray with optional chunking and parallel processing.

        Args:
            chunks (dict, optional): Mapping from dimension names to chunk sizes.
            parallel (bool): If True, enables parallel I/O.
            verbose (bool): If True, print detailed information about the dataset.

        Returns:
            xarray.Dataset: Dataset loaded from the NetCDF files.
        """

        if self.source_dir is None:
            raise Exception("[ERROR] Source directory must be set or given")

        # check if source directory exists
        if not os.path.exists(self.source_dir):
            raise Exception("[ERROR] Folder does not exist: {}".format(self.source_dir))

        # check if folder is a directory
        if not os.path.isdir(self.source_dir):
            raise Exception("[ERROR] Path is not a directory: {}".format(self.source_dir))

        # check if folder is empty
        if not os.listdir(self.source_dir):
            raise Exception("[ERROR] Directory is empty: {}".format(self.source_dir))
        # open files using xarray
        xarray_data = xr.open_mfdataset(os.path.join(self.source_dir, "*" + self.NETCDF_EXTENSION), parallel=parallel,
                                        chunks=chunks)

        if chunks is None and "time" in xarray_data.dims:
            xarray_data = xr.open_mfdataset(
                os.path.join(self.source_dir, "*" + self.NETCDF_EXTENSION), parallel=parallel,
                chunks={"time": 12}
            )

        # print information about the dataset
        if verbose:
            print("[INFO] Dataset information:")
            print(xarray_data)

        return xarray_data

    def _create_new_dir_structure(self, target_directory, directory_name, overwrite_previous_data):
        """
        Create a new directory structure for storing converted data, with subdirectories for metadata and sample data.

        Args:
            target_directory (str): The base directory where new directories will be created.
            directory_name (str): The name of the main directory to create under the target directory.
            overwrite_previous_data (bool): If True, any existing data in the target path will be overwritten.

        Returns:
            tuple: A tuple containing paths to the created subdirectories for metadata and sample data.

        Raises:
            Exception: If the directory exists and is not empty, and overwriting is not allowed.
        """
        var_directory = os.path.join(target_directory, directory_name)
        if os.path.isdir(var_directory):
            if len(os.listdir(var_directory)) > 0 and not overwrite_previous_data:
                raise Exception(
                    "[ERROR] Tried to create variable directory at <{}> but directory existed and was found to be not empty.")
            else:
                print("[INFO] Removing previously existing variable directory.")
                shutil.rmtree(var_directory, ignore_errors=True)
        os.makedirs(var_directory)
        print("[INFO] Created new variable directory at <{}>.".format(var_directory))
        sub_directories = []

        for folder_name in (DIRECTORY_NAME_META_DATA, DIRECTORY_NAME_SAMPLE_DATA):
            sub_dir = os.path.join(var_directory, folder_name)
            os.makedirs(sub_dir)
            sub_directories.append(sub_dir)
        return tuple(sub_directories)

    def convert_to_npy(self, chunks=None, parallel=True, verbose=False, overwrite_previous_data=False, batch_size=16):
        """
        Convert NetCDF files into numpy arrays and save them to the target directory as bin.

        Args:
            chunks (dict, optional): Specifies chunk sizes for dimensions, e.g., {'time': 12}.
            parallel (bool): Whether to use parallel I/O.
            verbose (bool): If True, print detailed processing information.
            overwrite_previous_data (bool): Whether to overwrite existing files.
            batch_size (int): Number of samples to process in each batch.
        """
        xarray_data = self.open_files(parallel=parallel, chunks=chunks, verbose=verbose)

        # check if target directory is set
        if self.target_dir is None:
            raise Exception("[ERROR] Target directory must be set or given")

        # check if target directory exists
        if not os.path.isdir(self.target_dir):
            os.makedirs(self.target_dir)
            print("[INFO] Created target directory at <{}>.".format(self.target_dir))

        for var_name in xarray_data.data_vars:
            print("[INFO] Processing data variable <{}>.".format(var_name))
            var_data = xarray_data[var_name]

            meta_folder, samples_folder = self._create_new_dir_structure(target_directory=self.target_dir,
                                                                         directory_name=var_name,
                                                                         overwrite_previous_data=overwrite_previous_data)
            self._convert_meta_data(var_data, meta_folder, xarray_data)
            print("[INFO] Converted meta data.")
            self._convert_sample_data(var_data, samples_folder, batch_size)

    def _convert_meta_data(self, data_var, meta_folder, xarray_data):
        """
        Extract and save metadata for a given data variable to a JSON file.

        Args:
            data_var (xarray.DataArray): Data variable from which metadata is extracted.
            meta_folder (str): Folder path where the metadata JSON file will be saved.
            xarray_data (xarray.Dataset): Dataset containing coordinates and attributes for the data variable.

        Stores metadata including the variable's name, its non-time dimensions, shape without time, and
        coordinate information.
        """
        print("[INFO] Reading meta data.")
        meta_data = {}
        meta_data.update({"name": data_var.name})
        meta_data.update({"time_variate": "time" in list(data_var.dims)})
        meta_data.update({"dims": [dim_name for dim_name in data_var.dims if dim_name != "time"]})
        meta_data.update({"shape": [dim_length for dim_name, dim_length in zip(data_var.dims, data_var.data.shape) if
                                    dim_name != "time"]})
        meta_data.update({"coords": []})
        data_coords = xarray_data.coords
        for coord_key in data_coords:
            if coord_key != "time":
                axis = data_coords[coord_key]
                meta_data["coords"].append({
                    "name": axis.name,
                    "values": axis.values.tolist(),
                    "dims": list(axis.dims)
                })
        meta_data.update({"attrs": {**xarray_data.attrs, **data_var.attrs}})
        meta_data_file = os.path.join(meta_folder, FILE_NAME_META_DATA + ".json")
        with open(meta_data_file, "w") as f:
            json.dump(meta_data, f)
        print("[INFO] Stored meta data in <{}>.".format(meta_data_file))

    def _convert_sample_data(self, data_var, samples_folder, batch_size):
        """
        Convert and save sample data into numpy binary files, organized by year if the data has a time dimension.

        Args:
            data_var (xarray.DataArray): The data variable to convert.
            samples_folder (str): The directory where the numpy files will be stored.
            batch_size (int): Number of time steps to process in each batch for efficiency.
        """

        if "time" in data_var.dims:
            print("[INFO] Converting temporal samples.")
            time_stamps = data_var["time"].values
            time_axis = tuple(data_var.dims).index("time")
            assert len(time_stamps) == len(
                np.unique(time_stamps)), "[ERROR] Encountered data variable with non-unique time stamps."
            batches = np.array_split(time_stamps, np.ceil(len(time_stamps) / batch_size))
            current_year = None
            storage_folder = None

            for sample_batch in tqdm(batches[0:1]):
                # creates array of arrays with shape (batch_size, 32, 64) for LR
                batch_data = np.array_split(data_var.sel(time=sample_batch).values, len(sample_batch), axis=time_axis)

                for time_stamp, data in zip(sample_batch, batch_data):
                    # data are shape (32, 64) for LR
                    # 32 is the latitude dimension and 64 is the longitude dimension where are temperature values
                    time_stamp = self._numpy_date_to_datetime(time_stamp)
                    if time_stamp.year != current_year:
                        current_year = time_stamp.year
                        storage_folder = os.path.join(samples_folder, "{}".format(current_year))
                        if not os.path.isdir(storage_folder):
                            os.makedirs(storage_folder)

                    # data is shape (32, 64) for LR
                    np.save(
                        os.path.join(storage_folder, self._file_name_from_time_stamp(time_stamp)),
                        np.squeeze(data, axis=time_axis)
                    )
        else:
            data = data_var.values
            np.save(
                os.path.join(samples_folder, FILE_NAME_CONSTANT_DATA + self.NUMPY_EXTENSION),
                data
            )

    def _numpy_date_to_datetime(self, time_stamp):
        """
        Convert numpy datetime64 object to a Python datetime object.

        Args:
            time_stamp (numpy.datetime64): The timestamp to convert.

        Returns:
            datetime.datetime: Converted Python datetime object.
        """
        total_seconds = (time_stamp - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s")
        return datetime.utcfromtimestamp(total_seconds)

    def _file_name_from_time_stamp(self, time_stamp):
        """
        Generate a file name based on a datetime object's format.

        Args:
            time_stamp (datetime.datetime): Timestamp to be formatted into a file name.

        Returns:
            str: Formatted file name including the proper extension.
        """
        return time_stamp.strftime(self.DATETIME_FORMAT) + self.NUMPY_EXTENSION


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, default='/home/jellik/Datasets/2m_temperature_5.625deg')
    parser.add_argument('--out', '-o', type=str, default='/home/jellik/Datasets/2m_temperature_5.625deg_numpy_test')
    parser.add_argument('--batch_size', '-b', type=int, default=300)
    parser.add_argument('--overwrite_previous_data', '-ov', type=bool, default=True)
    parser.add_argument('--verbose', '-v', type=bool, default=False)

    # parse all arguments
    args = parser.parse_args()
    converter = NetcdfToNpy(source_dir=args.path, target_dir=args.out)
    converter.convert_to_npy(overwrite_previous_data=args.overwrite_previous_data, batch_size=args.batch_size,
                             verbose=args.verbose)
