# Super-Resolution Enhancement of Weather Data Using Diffusion Models

---
This repository contains implementaiton of batchelor thesis focused on super-resolution enhancement of weather data using diffusion models. This work is based on [[1](https://github.com/LYL1015/ResDiff#resdiff-combining-cnn-and-diffusion-model-for-image-super-resolution), [2](https://github.com/davitpapikyan/Probabilistic-Downscaling-of-Climate-Variables/tree/main?tab=readme-ov-file)]
## 📖 Table of Contents

---

- [💻 Installation Guide](#-installation-guide)
- [📡 Dataset](#-dataset)
- [💡 Training](#-training)
- [🧪 Experiments](#-experiments)
- [🌅 Sampling](#-sampling)
- [📁 Project file structure](#-project-file-structure)


 <!-- headings -->
<a id="item-one"></a>
## 💻 Installation Guide

---
### Conda Environment

1. **Install Anaconda or Miniconda** from [here](https://www.anaconda.com/products/distribution)
2. **Create and activate a new Conda environment**:
   ```shell
   conda create -n myenv python=3.11 pip
   conda activate myenv
   ```

### Installing Required Packages

1. **Install PyTorch**: PyTorch has specific installation commands depending on your system's CUDA version (if applicable). Visit the [PyTorch Get Started Page](https://pytorch.org/get-started/locally/)

2. **Install Weights & Biases (wandb)**: Weights & Biases is used for experiment tracking and visualization. It can be installed via `pip`:

   ```shell
   pip install wandb
   ```

   After installation, you'll need to log in or sign up for a wandb account, which you can do by following the instructions provided after running the command above or by visiting [wandb login](https://wandb.ai/login).

3. **Install Additional Requirements**: Install the remaining requirements from the `requirements.txt` file, which includes other necessary packages.

   ```shell
   pip install -r requirements.txt
   ```
## 📡 Dataset

---

The original WeatherBench dataset can be donwloaded from this [link](https://github.com/pangeo-data/WeatherBench).
If you prefer not to download the full 50 GB of data, you can download a smaller example subset from [google drive](https://drive.google.com/drive/folders/1H4-dN-UOwTJz__D_vWs7UfYFSx83o2GJ?usp=sharing). This subset contains data from January 2017 to February 2017 and is already converted to .npy format. To use this dataset for training or sampling, specify the location of the downloaded `temperature_data_numpy` folder in the JSON configuration file.
### Data conversion
If you downloaded the original WeatherBench dataset in NetCDF format, you will need to convert it to .npy format using the provided script.

```shell
# this porcess can take a while depending on the size of the dataset and requires a lot of memory
python netcdf_to_npy.py -p [path to netcdf file] -o [output folder] -b 256
```
### Data configuration
To adjust data parameters, you can modify the JSON configuration file. Below is an example of a section from a JSON configuration file that specifies data details:
```json5
{
   "data": {
      "name": "WeatherBench", // Name of the dataset
      "dataroot": "/home/jellik/Datasets/temperature_data_numpy_all", // Path to the dataset
      "batch_size": 4, // Batch size for training
      "val_batch_size": 8, // Batch size for validation
      "num_workers": 24, // Number of workers for data loading
      "use_shuffle": true, // Enable shuffling of the data
      "train_min_date": "1979-01-01-00", // Start date for training data
      "train_max_date": "2016-01-01-00", // End date for training data
      "train_subset_min_date": "2014-01-01-00", // Start date for a subset of training data
      "train_subset_max_date": "2016-01-01-00", // End date for a subset of training data
      "transformation": "GlobalStandardScaling", // Transformation method applied to data
      "months_subset": [1], // Months included in the dataset
      "transform_groups": {"january": [1]}, // Month groups which will be transformed separately
      "val_min_date": "2016-01-01-00", // Start date for validation data
      "val_max_date": "2017-01-01-00", // End date for validation data
      "variables": ["t2m"], // Variables included in the dataset
      "height": 128 // Height of the images in the dataset
   }
}
```
## 💡 Training
Due to the computational demands of diffusion models, it is recommended to use a GPU for training. All training details are configured in a JSON config file.
If you prefer not to train your own model, you can download the pre-trained model weights from  [google drive](https://drive.google.com/drive/folders/1J5LD_m9QQcCQ6DAXgvJ8bvzsI7sHL79W?usp=sharing) and link them in the configuration file.

### Scripts
1. To **pre-train** **RRDB encoder** or **Simple CNN**  run following script:
```shell
# example config files are in configs/experiment_configs folder
python pretrain.py -p train -gpu 0 -c [config file]
```
2. To train **Diffusion Model** run following script:
```shell
# example config files are in configs/experiment_configs folder
python train.py -p train -gpu 0 -c [config file]
```
### Usage
Description of the command line arguments for the training scripts:


| Option                | Type | Description                                                                       | Default Value |
|-----------------------| ---- |-----------------------------------------------------------------------------------|---------------|
| `--phase`, `-p`       | String | Determines whether the model will be trained or only validated                    | `Train`       |
| `--gpu_ids`, `-gpu`   | String | Gpu id used for training and validation.                                          | `None`        |
| `--config`, `-c`      | String | Path to json configuration file.                                                  | `None`        |
| `--help`              | | Displays help information about the command and its options.                      |               |

### Json training configuration
To adjust training parameters, you can modify the JSON configuration file. Below is an example of a section from a JSON configuration file that specifies training details:
```json5
// Part of json configuration file specifying training details
// Config file for pre-trained models is simpler and can be foud in configs/experiment_configs/rrdb folder 
{
   "train": {
      "save_visualizations": true, // Enable saving visualizations
      "n_iter": 1, // Number of iterations
      "val_freq": 1, // Frequency of validation on validation substet for efficient validation
      "full_val_freq": 1, // Frequency of validation on full validation set
      "save_checkpoint_freq": 1, // Frequency to save checkpoints
      "print_freq": 1, // Frequency of printing logs
      "val_vis_freq": 1, // Frequency of visualization during validation
      "optimizer": {
         "type": "adam", // Type of optimizer
         "lr": 1e-4 // Learning rate for the optimizer
      },
      "ema_scheduler": { // EMA scheduler configuration (currently not used)
         "step_start_ema": 5000, // Step to start EMA
         "update_ema_every": 1, // Frequency to update EMA
         "ema_decay": 0.9999 // Decay rate of EMA
      }
   }
}
```



## 🧪 Experiments

---
### Local experiment files

During each run of `train.py` or `pretrain.py`, a structure for storing experiment files is created in the `experiments/` folder. Below is an example of the folder structure with description what each folder contains:
```shell
# This file structure can be modified in json configuration file.

experiments/
├── wandb/ # Folder for storing local wandb experiment files
├── experiment_1/
│   ├── checkpoint/  # Folder for storing model checkpoints (use lot of disk space)
│   ├── logs/        # Folder for storing logs from python logging
│   ├── results/
│   │   ├── 1/  # Results images from validation set in epoch 1
│   │   └── 2/  # Results images from validation set in epoch 2
│   └── tb_logger/   # Folder for storing TensorBoard logs
└── experiment_2/
```
### Weights and Biases
All experiment results are logged using Weights and Biases. To view the training and validation results, log in to your Weights and Biases account [here](https://wandb.ai/site) and navigate to the project page specified in the JSON configuration file.
```json5
{
  "wandb": {
    "project": "Climate-Variables-SR", // Name of the project
    "entity": "jellik_dgx" // Name of the entity
  }
}
```
All run parameters, losses, visualizations, and metrics are logged.

![Wandb logs](example/images/wandb.gif)
## 🌅 Sampling

---
To sample images, use the same configuration file as for training. You can provide the path to the trained model either as a script argument or in the JSON configuration file.
```shell
# samples can be also generated during training using train.py script
# keep in mind that date has to be in format YYYY-MM-DD-HH and has to be within dataset range specified in config file
python sample.py -gpu 0 -p [model folder] -c [config file] -o [output folder] -n 1 -t SR HR AE DELTA INTERPOLATED -d 2017-01-01-00
```

### Usage
Description of the command line arguments for the sampling script:

| Option                      | Type    | Description                                                                                                | Default Value |
|-----------------------------|---------|------------------------------------------------------------------------------------------------------------|---------------|
| `--model_path`, `-p`        | String  | Path to trained model. Can be specified in config file.                                                    | `None`        |
| `--gpu_ids`, `-gpu`         | String  | Gpu id used for training and validation.                                                                   | `None`        |
| `--config`, `-c`            | String  | Path to json configuration file.                                                                           | `None`        |
| `--output_path`, `-o`       | String  | Path where to save ouput images.                                                                           | `None`        |
| `--number_of_samples`, `-n` | Integer | Number of samples to visualize                                                                             | `1`           |
| `--image_types`, `-t`       | List    | list of image types to vizualize: (HR, SR, LR, INTERPOLATED, DELTA, AE).                                   | `SR`          |
| `--color_map`, `-m`         | String  | Color map used for temperature visualistaion.                                                              | `coolwarm`    |
| `--date`, `-d`              | String  | Specification of exact date from which to sample images. If not provided first batch from dataset is used. | `None`        |
| `--help`                    |         | Displays help information about the command and its options.                                               |               |

## 📁 Project file structure



```shell
Climate-Variables-Downscaling/
├── configs/                         # Directory containing model and data configurations
│   ├── data_config/                 # Data-specific configurations
│   └── experiment_configs/          # Configurations for different experiments
│       └── config.py                # Script that parse json configuration files
├── data/                            # Data handling modules
│   ├── conversions/                 # Conversion scripts or utilities
│   ├── dataset_builder.py           # Script to create datasets and dataloaders for training and validation
│   ├── datasets.py                  # Module to handle weatherbench datasets
│   ├── npy_reader.py                # Utility to read numpy weather files
│   ├── transforms.py                # Data transformation functions
│   └── utils.py                     # General utilities for data manipulation
├── example/                         # Example files for demonstration how to use this project
│   └── images/                      # Examples of generated images
├── experiments/                     # Folder where experiment outputs are defaultly stored
├── logger/                          # Logging utilities
├── models/                          # Models implementations
│   ├── diffusion_models/            # Diffusion models implementations
│   │   ├── nn_modules/              # common neural network building blocks
│   │   ├── phydiff/                 # Phydiff model implementation
│   │   ├── physrdiff/               # Physrdiff model implementation
│   │   ├── resdiff/                 # Resdiff model implementation
│   │   ├── sr3/                     # SR3 model implementation
│   │   ├── srdiff/                  # Srdiff model implementation
│   │   ├── __init__.py              # Initialization file for Python module
│   │   ├── diffusion.py             # Common parts of diffusion process implementations for all models
│   │   ├── model.py                 # Main model definitions
│   │   ├── networks.py              # Creating and defining which model is used
│   │   └── scheduler.py             # Noise scheduler
│   ├── rrdb_encoder/                # RRDB encoder model
│   │   ├── __init__.py              # Initialization file for Python module
│   │   └── RRDBNet.py               # RRDB Network definition
│   ├── simple_cnn/                  # Simple CNN implementation
│   │   ├── __init__.py              # Initialization file for Simple CNN
│   │   ├── loss.py                  # Loss functions for training
│   │   └── Simple_CNN.py            # Simple CNN model definition
│   └──  base_model.py               # Base model abstract class
├── thesis/                          # Thesis documents
├── training/                        # Training utilities
│   ├── metrics.py                   # Metrics for evaluating model performance
│   ├── utils.py                     # Utilities used in training scripts
│   └── visualization.py             # Visualization tools for sampling 
├── pretrain.py                      # Script for pre-training models
├── requirements.txt                 # List of dependencies
├── sample.py                        # Sample script for quick testing
└── train.py                         # Main training script
└── example.ipynb                    # Jupyter notebook with simple run example
```

