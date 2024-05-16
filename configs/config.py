import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime


class DataConfig:
    """Handles configuration loading from a JSON file."""

    def __init__(self, json_path: str = None, json_name: str = None):
        """Initializes the DataConfig object.

        Args:
            json_path (str, optional): Path to the JSON configuration file.
            json_name (str, optional): Name of the JSON configuration file without extension.

        Raises:
            FileNotFoundError: Raised if the JSON file does not exist.
        """
        if json_name and not json_path:
            if json_name.endswith(".json"):
                json_name = json_name[:-5]
            json_path = os.path.join(os.path.dirname(__file__), "data_config", json_name + ".json")
            if not os.path.exists(json_path):
                raise FileNotFoundError("Json file with given name does not exist: {}".format(json_path))

        if not json_path:
            json_files = [pos_json for pos_json in os.listdir(os.path.join(os.path.dirname(__file__), "data_config")) if
                          pos_json.endswith('.json')]
            if len(json_files) == 0:
                raise FileNotFoundError("No json file found in data_config directory")
            json_path = os.path.join(os.path.dirname(__file__), "data_config", json_files[0])

        with open(json_path, "r") as f:
            self.config = json.load(f)
            self.name = self.config["name"]
            self.datetime_format = self.config["datetime_format"]
            self.temporal_resolution_unit = self.config["temporal_resolution"]["unit"]
            self.temporal_resolution_value = self.config["temporal_resolution"]["value"]
            self.directory_name_meta_data = self.config["directory_name_meta_data"]
            self.file_name_meta_data = self.config["file_name_meta_data"]
            self.file_name_constant_data = self.config["file_name_constant_data"]
            self.directory_name_sample_data = self.config["directory_name_sample_data"]
            self.netcdf_extension = self.config["netcdf_extension"]
            self.numpy_extension = self.config["numpy_extension"]


class Config:
    """Configuration class for managing experiment parameters."""

    def __init__(self, args: argparse.Namespace, experiment=True):
        """Initializes the Config object.

        Args:
            args (argparse.Namespace): Command line arguments.
            experiment (bool): Whether to log experiment create folders etc.
        """
        self.args = args
        self.root = self.args.config  # json file path
        self.gpu_ids = self.args.gpu_ids  # gpu ids
        self.params = {}
        self.experiments_root = None
        self.__parse_configs(experiment)
        self.params["gpu_ids"] = self.gpu_ids
        self.params["data"]["transform_groups"] = list(self.params["data"]["transform_groups"].values())

    def get_opt(self):
        """Retrieve parsed configurations.

        Returns:
            dict: A dictionary containing configurations.
        """
        return self.params

    def __parse_configs(self, experiment=True):
        """Reads and parses the configuration JSON file, creates experiment directories, and sets GPU ids.

        Args:
            experiment (bool): Whether to log experiment create folders etc.
        """
        json_str = ""
        with open(self.root, "r") as f:
            for line in f:
                json_str = f"{json_str}{line.split('//')[0]}\n"

        self.params = json.loads(json_str, object_pairs_hook=OrderedDict)

        if experiment:
            self.handle_experiment_configs()

        if self.gpu_ids:
            self.params["gpu_ids"] = [int(gpu_id) for gpu_id in self.gpu_ids.split(",")]
            gpu_list = self.gpu_ids
        else:
            gpu_list = ",".join(str(x) for x in self.params["gpu_ids"])

        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        self.params["distributed"] = True if len(gpu_list) > 1 else False

    def handle_experiment_configs(self):
        """Handle the creation of directories for experiments."""
        if not self.params["path"]["resume_state"]:
            if self.params["path"]["experiments_folder_path"]:
                self.experiments_root = os.path.join(self.params["path"]["experiments_folder_path"], "experiments",
                                                     f"{self.params['name']}_{get_current_datetime()}")
            else:
                self.experiments_root = os.path.join("experiments",
                                                     f"{self.params['name']}_{get_current_datetime()}")
        else:
            self.experiments_root = "/".join(self.params["path"]["resume_state"].split("/")[:-2])

        for key, path in self.params["path"].items():
            if not key.startswith("resume") and not key.startswith("experiments"):
                self.params["path"][key] = os.path.join(self.experiments_root, path)
                mkdirs(self.params["path"][key])
        self.params["path"]['experiments_root'] = self.experiments_root

    def __getattr__(self, item):
        """Returns None when attribute doesn't exist.

        Args:
            item: Attribute to retrieve.

        Returns:
            None
        """
        return None

    def get_hyperparameters_as_dict(self):
        """Returns dictionary containg parsed configuration json file.
        """
        return self.params


def dict2str(opt, indent_l=1):
    """dict to string for logger"""
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


def get_current_datetime() -> str:
    """Converts the current datetime to a string in the format '%y%m%d_%H%M%S'.

    Returns:
        str: String version of the current datetime.
    """
    return datetime.now().strftime("%y%m%d_%H%M%S")


def mkdirs(paths) -> None:
    """Creates directories represented by paths argument.

    Args:
        paths (list or str): Either a list of paths or a single path.
    """
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)
