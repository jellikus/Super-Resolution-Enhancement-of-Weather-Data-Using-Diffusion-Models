from .utils import date_to_str, get_month_datetime, str_to_date, validate_month_subset, is_full_year, save_object
from .datasets import TimeVariateData, WeatherBenchData
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
from .transforms import DataTransformer
from .npy_reader import WNPYReader
from types import SimpleNamespace
from .utils import log_dataset_info
import logging
import torch
import os


class DataHandler:
    """
    Handles loading, processing, and transformation of weather data for training and validation purposes.
    Manages the creation of data loaders and datasets for specified date ranges and configurations.
    """

    def __init__(self, dataroot, variables, storage_root, months_subset, groups, transformation, train_min_date=None,
                 train_max_date=None, val_min_date=None, val_max_date=None, val_batch_size=None, train_batch_size=None,
                 shuffle_data=True, num_workers=None):
        """
        Initialize the data handler with configuration for data processing.

        Args:
            dataroot (str): Root directory for dataset storage.
            variables (list): List of variables to process.
            storage_root (str): Directory for storing processed data and metadata.
            months_subset (list): List of months to include in the dataset.
            groups (list): Group names for organizing the dataset.
            transformation (dict): Transformations to apply to the data.
            train_min_date (str, optional): Minimum date for training dataset.
            train_max_date (str, optional): Maximum date for training dataset.
            val_min_date (str, optional): Minimum date for validation dataset.
            val_max_date (str, optional): Maximum date for validation dataset.
            val_batch_size (int, optional): Batch size for validation data loader.
            train_batch_size (int, optional): Batch size for training data loader.
            shuffle_data (bool): Whether to shuffle the data.
            num_workers (int, optional): Number of worker threads for loading data.
        """
        self.metadata = {}
        self.dataroot = dataroot
        self.variables = variables
        self.storage_root = storage_root
        self.months_subset = months_subset
        self.groups = groups
        self.transformation = transformation
        self.train_min_date = train_min_date
        self.train_max_date = train_max_date
        self.val_min_date = val_min_date
        self.val_max_date = val_max_date
        self.val_batch_size = val_batch_size
        self.train_batch_size = train_batch_size
        self.shuffle_data = shuffle_data
        self.num_workers = num_workers
        self.data_transformer = DataTransformer(variables, dataroot, months_subset, groups)
        validate_month_subset(months_subset)

        self.train_loader = None
        self.val_loader = None
        self.train_dataset = None
        self.val_dataset = None

        self.data_transformer = DataTransformer(variables, dataroot, months_subset, groups)
        validate_month_subset(months_subset)

    def get_datasets(self):
        """
        Retrieve the train and validation datasets.

        Returns:
            tuple: A tuple containing the training dataset and validation dataset.
        """
        return self.train_dataset, self.val_dataset

    def get_data_loaders(self):
        """
        Retrieve the data loaders for training and validation.

        Returns:
            tuple: A tuple containing the DataLoader for training and validation.
        """
        return self.train_loader, self.val_loader

    def get_metadata(self):
        """
        Retrieve the metadata associated with the datasets.

        Returns:
            SimpleNamespace: Metadata organized in an easy-to-access namespace format.
        """
        return SimpleNamespace(**self.metadata)

    def get_data_transformer(self):
        """
        Get the data transformer used for processing data.

        Returns:
            DataTransformer: The transformer object used for applying data transformations.
        """
        return self.data_transformer

    def get_tensor_by_date(self, date):
        # return self.val_dataset[date]
        pass

    def create_val_set(self, val_min_date=None, val_max_date=None, transform=None):
        """
        Create and set up the validation dataset using the specified date range.

        Args:
            val_min_date (str, optional): Minimum date for the validation dataset.
            val_max_date (str, optional): Maximum date for the validation dataset.

        Returns:
            Dataset: The created validation dataset.
        """
        if val_min_date:
            self.val_min_date = val_min_date
        if val_max_date:
            self.val_max_date = val_max_date

        self.val_dataset = self._create_set(self.val_min_date, self.val_max_date, train=False)
        return self.val_dataset

    def create_train_set(self, train_min_date=None, train_max_date=None):
        """
        Create and set up the training dataset using the specified date range.

        Args:
            train_min_date (str, optional): Minimum date for the training dataset.
            train_max_date (str, optional): Maximum date for the training dataset.
        """
        if train_min_date:
            self.train_min_date = train_min_date
        if train_max_date:
            self.train_max_date = train_max_date

        self.train_dataset = self._create_set(self.train_min_date, self.train_max_date, train=True)

    def create_train_loader(self, batch_size, use_shuffle, num_workers):
        """
        Create a DataLoader for the training dataset.

        Args:
            batch_size (int): The number of items in each batch.
            use_shuffle (bool): Whether to shuffle the data.
            num_workers (int): The number of worker threads to use.

        Returns:
            DataLoader: The DataLoader for the training dataset.
        """
        if self.train_dataset is None:
            raise ValueError("Training dataset is not created. Call create_train_set() first.")

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=batch_size,
                                       collate_fn=self._form_batch,
                                       shuffle=use_shuffle,
                                       pin_memory=True,
                                       drop_last=True,
                                       num_workers=num_workers)
        return self.train_loader

    def create_val_loader(self, batch_size, use_shuffle, num_workers):
        """
        Create a DataLoader for the validation dataset.

        Args:
            batch_size (int): The number of items in each batch.
            use_shuffle (bool): Whether to shuffle the data.
            num_workers (int): The number of worker threads to use.
        """
        if self.val_dataset is None:
            raise ValueError("Validation dataset is not created. Call create_val_set() first.")

        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=batch_size,
                                     collate_fn=self._form_batch,
                                     pin_memory=True,
                                     drop_last=True,
                                     num_workers=num_workers)

    def log_info(self):
        """
        Log detailed information about the training and validation datasets.

        Uses the logging module to output the dataset sizes and other relevant information.
        """
        logger = logging.getLogger("base")
        log_dataset_info(self.train_dataset, f"Train WeatherDataset", logger)
        log_dataset_info(self.val_dataset, f"Validation WeatherDataset", logger)
        logger.info("Finished.\n")

    def get_all(self):
        return self.train_dataset, self.val_dataset, self.get_metadata(), self.data_transformer

    def process_data(self):
        """
        Main method to process and set up all necessary datasets and data loaders.

        This method orchestrates the creation of datasets, data loaders, and logs the process,
        encapsulating the entire setup in one method call.

        Returns:
            tuple: Contains the train and validation DataLoaders, metadata, and data transformer.
        """
        self.create_train_set()
        self.create_val_set()
        self._save_metadata_and_transformations()
        self.create_train_loader(self.train_batch_size, self.shuffle_data, self.num_workers)
        self.create_val_loader(self.val_batch_size, self.shuffle_data, self.num_workers)
        self.log_info()

        return self.train_loader, self.val_loader, self.get_metadata(), self.get_data_transformer()

    def get_data_by_date(self, date):
        """
        Retrieve the data for a specific date from the validation dataset.

        Args:
            date (str): The date to retrieve data for.

        Returns:
            tuple: A tuple containing the low-resolution , high-resolution and super-resolution images.
        """
        # check if date is in valid format
        return self._form_batch([self.val_dataset.get_data_by_date(date)])

    def _create_set(self, min_date=None, max_date=None, train=True):
        """
        Internal method to create a dataset for either training or validation based on a date range.

        Args:
            min_date (str): The start date for the dataset.
            max_date (str): The end date for the dataset.
            train (bool): Flag to indicate whether this is for training or validation.

        Returns:
            WeatherBenchData: The assembled dataset for the specified date range.
        """
        datasets = {"lr": [], "hr": []}

        for variable in self.variables:
            for data_type in ("lr", "hr"):
                reader = WNPYReader(os.path.join(self.dataroot, data_type, variable))

                if train:
                    transform = self.data_transformer.transform(min_date, max_date, data_type, variable,
                                                                self.transformation)
                    self._update_metadata(data_type, reader)
                else:
                    transform = self.data_transformer.get_transform(variable, data_type)

                if is_full_year(self.months_subset):
                    data = TimeVariateData(reader, name=f"{data_type}_{variable}", lead_time=0,
                                           min_date=min_date, max_date=max_date,
                                           transform=transform)
                else:
                    data = self._create_dataset_by_month_subset(reader, f"{data_type}_{variable}", 0,
                                                                min_date, max_date, transform)

                datasets[data_type].append(data)

        dataset = WeatherBenchData(min_date=min_date, max_date=max_date)
        dataset.add_data_group("lr", datasets["lr"])
        dataset.add_data_group("hr", datasets["hr"])
        return dataset

    def _update_metadata(self, data_type, wbd_reader):
        """
        Update the metadata from a data reader.

        Args:
            data_type (str): The type of data (e.g., 'lr' for low resolution).
            reader (WNPYReader): The reader object containing metadata to extract.

        Updates the class's metadata dictionary with information from the reader.
        """
        for dimension in wbd_reader.meta_data["coords"]:
            key = f"{data_type}_{dimension['name']}"
            value = dimension["values"]
            self.metadata[key] = value

    def _save_metadata_and_transformations(self):
        """
        Save the current metadata and transformations to disk.

        Stores the metadata and transformation settings in the specified storage root directory.
        """
        save_object(self.metadata, self.storage_root, "metadata")
        save_object(self.data_transformer.transformation_dict, self.storage_root, "transformations")

    def _create_dataset_by_month_subset(self, source, name, lead_time, min_date, max_date, transform):
        """
        Create a dataset limited to specific months within the provided date range.

        Args:
            source (WNPYReader): Source object for reading data.
            name (str): The name for the dataset.
            lead_time (int): Lead time in hours for forecasting purposes.
            min_date (str): Start date for the dataset range.
            max_date (str): End date for the dataset range.
            transform (callable): Transformation function to apply to the data.

        Returns:
            TimeVariateData: The dataset created with data only from specified months.
        """
        max_date_datetime = str_to_date(max_date)
        start = str_to_date(min_date)
        start_of_next_month = start + get_month_datetime()
        npy_reader = source
        dataset = None

        while start_of_next_month < max_date_datetime:
            current_month = start.month
            if current_month not in self.months_subset:
                start = start_of_next_month
                start_of_next_month = (start_of_next_month + get_month_datetime()).replace(day=1)
                continue

            if dataset is None:
                dataset = TimeVariateData(npy_reader,
                                          name=name,
                                          lead_time=lead_time, min_date=date_to_str(start),
                                          max_date=date_to_str(start_of_next_month), transform=transform)
            else:
                dataset.add_data_by_date(date_to_str(start), date_to_str(start_of_next_month))
            start = start_of_next_month
            start_of_next_month = (start_of_next_month + get_month_datetime()).replace(day=1)

        if start.month in self.months_subset:
            if dataset is None:
                dataset = TimeVariateData(npy_reader,
                                          name=name,
                                          lead_time=lead_time, min_date=date_to_str(start),
                                          max_date=date_to_str(max_date_datetime), transform=transform)
            else:
                dataset.add_data_by_date(date_to_str(start), date_to_str(max_date_datetime))

        return dataset

    def _form_batch(self, samples: list):
        """Creates batch from sample list and interpolates lr images

        Args:
            samples: A list of data points LR and HR datapoints.

        Returns:
            A dictionary containing the following items:
                LR – a low-resolution tensor,
                HR – a high-resolution tensor,
                SR – an upsampled low-resolution tensor with bicubic interpolation
            and a list of month indices corresponding to each sample.
        """
        lr_tensors = []
        hr_tensors = []
        months = []
        output_batch = {}

        # Iterate through each sample to extract and process data
        for low_res, high_res in samples:
            # Concatenate low-resolution tensors along the channel dimension
            lr_concatenated = torch.cat([variable[0] for variable in low_res], dim=1)
            hr_concatenated = torch.cat([variable[0] for variable in high_res], dim=1)

            lr_tensors.append(lr_concatenated)
            hr_tensors.append(hr_concatenated)
            months.append(low_res[0][2])  # Assuming month information is in the first element

        output_batch["HR"] = torch.cat(hr_tensors)
        output_batch["LR"] = torch.cat(lr_tensors)
        # Upsample low-resolution tensors using bicubic interpolation
        interpolated_tensors = []
        for lr_tensor in lr_tensors:
            interpolated_tensor = interpolate(lr_tensor, scale_factor=4, mode="bicubic")
            interpolated_tensors.append(interpolated_tensor)

        output_batch["SR"] = torch.cat(interpolated_tensors)

        return output_batch, months
