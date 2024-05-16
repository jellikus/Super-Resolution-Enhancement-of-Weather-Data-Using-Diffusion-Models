import os
from datetime import datetime

import numpy as np
import xarray as xr
import lmdb
import argparse
from tqdm import tqdm
from configs.config import DataConfig

config = DataConfig()
DATETIME_FORMAT = config.datetime_format
NETCDF_EXTENSION = config.netcdf_extension

class NetcdfToLMDB:
    """
    This class handles the conversion of NetCDF files to LMDB format
    """
    def __init__(self, source_dir, target_dir, netcdf_extension=NETCDF_EXTENSION, datetime_format=DATETIME_FORMAT):
        """
        Initialize the converter with directory paths and configuration settings.

        Args:
            source_dir (str): Path to the directory containing the NetCDF files.
            target_dir (str): Path where the LMDB database will be created.
            netcdf_extension (str): File extension for NetCDF files, defaulted from a global setting.
            datetime_format (str): Format string for datetime objects, used for key generation in LMDB.
        """
        self.NETCDF_EXTENSION = netcdf_extension
        self.DATETIME_FORMAT = datetime_format
        self.source_dir = source_dir
        self.target_dir = target_dir

    def open_files(self, chunks=None, parallel=True):
        """
        Open NetCDF files using xarray with optional chunking and parallel processing.

        Args:
            chunks (dict, optional): How to chunk the data, e.g., {'time': 10} for efficient loading.
            parallel (bool): Whether to enable parallel I/O operations.

        Returns:
            xarray.Dataset: The dataset loaded from the NetCDF files.

        Raises:
            Exception: If the source directory does not exist, is not a directory, or is empty.
        """
        if not os.path.exists(self.source_dir):
            raise Exception(f"[ERROR] Source directory does not exist: {self.source_dir}")
        if not os.path.isdir(self.source_dir):
            raise Exception(f"[ERROR] Path is not a directory: {self.source_dir}")
        if not os.listdir(self.source_dir):
            raise Exception(f"[ERROR] Directory is empty: {self.source_dir}")

        return xr.open_mfdataset(os.path.join(self.source_dir, "*" + self.NETCDF_EXTENSION),
                                 parallel=parallel, chunks=chunks or {"time": 12})

    def convert_to_lmdb(self, overwrite_previous_data=False, batch_size=16, map_size=1e12):
        """
        Convert the data from NetCDF format to LMDB format.

        Args:
            overwrite_previous_data (bool): If True, overwrite any existing data in the target directory.
            batch_size (int): Number of time steps to process in each batch for efficiency.
            map_size (int, float): The maximum size of the database in bytes.

        Uses time stamps as keys in the database, and stores serialized numpy arrays as values.
        """
        xarray_data = self.open_files()
        env = lmdb.open(self.target_dir, map_size=int(map_size), writemap=True)

        with env.begin(write=True) as txn:
            for var_name in xarray_data.data_vars:
                var_data = xarray_data[var_name]
                print(f"[INFO] Processing variable {var_name}")

                if "time" in var_data.dims:
                    time_stamps = var_data["time"].values
                    for time_stamp in tqdm(time_stamps):
                        data = var_data.sel(time=time_stamp).values
                        key = datetime.utcfromtimestamp((time_stamp - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')).strftime(DATETIME_FORMAT).encode('utf-8')
                        txn.put(key, data.tobytes())

        env.close()
        print("[INFO] Conversion to LMDB completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, required=True)
    parser.add_argument('--out', '-o', type=str, required=True)
    parser.add_argument('--batch_size', '-b', type=int, default=300)
    parser.add_argument('--overwrite_previous_data', '-ov', type=bool, default=True)

    args = parser.parse_args()
    converter = NetcdfToLMDB(source_dir=args.path, target_dir=args.out)
    converter.convert_to_lmdb(overwrite_previous_data=args.overwrite_previous_data, batch_size=args.batch_size)
