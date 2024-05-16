import argparse
import os
import time
import torch
import wandb
import logging
from tqdm import tqdm
from tensorboardX import SummaryWriter

# imports for data handling, configurations, logging, model creation, and training
from data.transforms import get_transformation_by_name
from data.dataset_builder import DataHandler
from configs.config import Config
from models.base_model import create_model
from logger.python_logging import setup_logger
from logger.wandb import WandbLogger
from training.utils import get_optimizer, set_seeds
from training.metrics import ValidationMetrics, create_metric_dict, TrainMetrics
from training.visualization import ImageContainer


def torch_cudnn():
    """
    Enable CuDNN acceleration and benchmark mode in PyTorch to enhance GPU utilization.
    """
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def train(train_loader, model, logger, data_transformer, train_metrics, val_metrics, wandb_logger, curr_iter,
          curr_epoch, opt, val_loader, metadata):
    """
    Training process of a model.

    Args:
        train_loader (DataLoader): Loader for training data.
        model: The neural network model to be trained.
        logger (Logger): Logger for training process.
        data_transformer: Transformations applied to the data.
        train_metrics (TrainMetrics): Container for tracking training metrics.
        val_metrics (ValidationMetrics): Container for tracking validation metrics.
        wandb_logger (WandbLogger): Logger for Weights & Biases platform.
        curr_iter (int): Iteration from which to start training.
        curr_epoch (int): Epoch from which to start training.
        opt (dict): Configuration options.
        val_loader (DataLoader): Loader for validation data.
    """

    n_iter, print_freq = opt["train"]["n_iter"], opt["train"]["print_freq"]
    val_freq, val_vis_freq, full_val_freq = opt["train"]["val_freq"], opt["train"]["val_vis_freq"], opt["train"][
        "full_val_freq"]
    save_checkpoint_freq, variables, results = opt["train"]["save_checkpoint_freq"], opt["data"]["variables"], \
        opt["path"]["results"]

    while curr_iter <= n_iter:
        curr_epoch += 1
        wandb_logger.log_metrics({'epoch': curr_epoch}, commit=False, step=curr_iter + 1)

        for train_data in train_loader:
            curr_iter += 1

            if curr_iter > n_iter:
                break

            model.feed_data(train_data)
            model.optimize_parameters()

            # log model loss and other important metrics using get_current_log
            train_metrics.update(model.get_current_log())
            # get train info
            if curr_iter % print_freq == 0:
                logger.info(f"Epoch: {curr_epoch:5}  |  Iteration: {curr_iter:8} | {train_metrics.metrics2str()}")
                wandb_logger.log_train_metrics(train_metrics.metrics2dict(), commit=False, step=curr_iter)
                wandb_logger.log_train_mean_metrics(train_metrics.mean_metrics2dict(), commit=False, step=curr_iter)
                train_metrics.reset()

            # start validation
            if curr_iter % val_freq == 0:
                model.prepare_to_eval()
                logger.info("Starting validation.")
                result_path = f"{results}/{curr_epoch}"
                os.makedirs(result_path, exist_ok=True)

                val_iter = 0
                # start timer for validation
                start_val_time = time.time()

                for val_data in tqdm(val_loader):
                    val_iter += 1

                    if val_iter > 1 and curr_iter % full_val_freq != 0:
                        break

                    model.feed_data(val_data)
                    model.generate_sr(False)

                    model_images = model.get_images(need_LR=True)
                    inversed_tensors = data_transformer.inverse_transform(model_images, model.get_months())

                    val_metrics.update(inversed_tensors["HR"], inversed_tensors["SR"])

                    if val_iter == 1 and opt["train"]["save_visualizations"]:
                        path = f"{result_path}/{curr_epoch}_{curr_iter}_{val_iter}"
                        logger.info(f"[{val_iter // val_vis_freq}] Visualizing and storing some examples.")

                        n_val_vis = 1
                        image_container = ImageContainer(inversed_tensors, metadata, n_val_vis)
                        fig = image_container.make_wandb_plot()
                        wandb_logger.log_sr_hr_it_image(fig, commit=False, step=curr_iter)
                        image_container.save_all_images(path)

                end_val_time = time.time()

                val_time = end_val_time - start_val_time
                wandb_logger.log_val_time(val_time, commit=False, step=curr_iter)
                val_metrics.compute_metrics()
                logger.info(f"Epoch: {curr_epoch:5}  |  Iteration: {curr_iter:8} | {val_metrics.metrics2str()}")
                wandb_logger.log_val_metrics(val_metrics.metrics2dict(), commit=False, step=curr_iter)
                val_metrics.reset()

                model.prepare_to_train()

            if curr_iter % save_checkpoint_freq == 0:
                logger.info("Saving models and training states.")
                model.save_network(curr_epoch, curr_iter)

            wandb_logger.log_metrics({}, commit=True, step=curr_iter)

    logger.info("End of training.")


def validate(val_loader, model, logger, data_transformer, val_metrics, wandb_logger, opt,  curr_iter, curr_epoch, metadata):
    """
    Validation process of a model.

    Args:
        model: The neural network model to be validated.
        logger (Logger): Logger for the validation process.
        data_transformer: Transformations applied to the data for validation.
        val_metrics (ValidationMetrics): Container for validation metrics.
        wandb_logger (WandbLogger): Logger for Weights & Biases platform.
        opt (dict): Configuration options.
        val_loader (DataLoader): DataLoader for the validation dataset.
        curr_iter (int): Iteration from which to start validation.
        curr_epoch (int): Epoch from which to start validation.
    """

    # load configuration options
    val_freq, val_vis_freq, full_val_freq = opt["train"]["val_freq"], opt["train"]["val_vis_freq"], opt["train"][
        "full_val_freq"]
    save_checkpoint_freq, variables, results = opt["train"]["save_checkpoint_freq"], opt["data"]["variables"], \
        opt["path"]["results"]
    model.prepare_to_eval()
    logger.info("Starting validation.")
    result_path = f"{results}/{curr_epoch}"
    os.makedirs(result_path, exist_ok=True)


    val_iter = 0
    start_val_time = time.time()

    # start validation
    for val_data in tqdm(val_loader):
        val_iter += 1
        model.feed_data(val_data)
        model.generate_sr(False)

        model_images = model.get_images(need_LR=True)
        inversed_tensors = data_transformer.inverse_transform(model_images, model.get_months())
        val_metrics.update(inversed_tensors["HR"], inversed_tensors["SR"])

        # save visualisations
        if val_iter == 1 and opt["train"]["save_visualizations"]:
            path = f"{result_path}/{curr_epoch}_{curr_iter}_{val_iter}"
            logger.info(f"[{val_iter // val_vis_freq}] Visualizing and storing some examples.")

            image_container = ImageContainer(inversed_tensors, metadata, 1)
            image_container.set_min_max(220, 315)

            # log image to wandb
            fig = image_container.make_wandb_plot()
            wandb_logger.log_sr_hr_it_image(fig, commit=False, step=curr_iter)

            # make and save visualisations
            image_container.save_all_images(path)

    end_val_time = time.time()
    val_time = end_val_time - start_val_time

    wandb_logger.log_val_time(val_time, commit=False, step=curr_iter)
    val_metrics.compute_metrics()
    logger.info(f"Epoch: {curr_epoch:5}  |  Iteration: {curr_iter:8} | {val_metrics.metrics2str()}")

    wandb_logger.log_val_metrics(val_metrics.metrics2dict(), commit=False, step=curr_iter)
    wandb_logger.log_metrics({}, commit=True, step=curr_iter)
    val_metrics.reset()

    logger.info("End of validation.")


def main():
    """
    Main function to handle the workflow of training, evaluating, logginng, and saving results of the model.
    """

    # load configuration
    set_seeds()
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="JSON file for configuration")
    parser.add_argument("-p", "--phase", type=str, choices=["train", "val"],
                        help="Run either training or validation(inference).", default="train")
    parser.add_argument("-gpu", "--gpu_ids", type=str, default=None)

    configs = Config(parser.parse_args())
    opt = configs.params
    torch_cudnn()

    # create and setup loggers
    setup_logger(None, opt["path"]["log"], "train", screen=True)
    setup_logger("val", opt["path"]["log"], "val")

    # get logger instances
    logger = logging.getLogger("base")
    val_logger = logging.getLogger("val")

    # create wandb logger
    wandb_logger = WandbLogger(opt)


    # get transformation
    transformation = get_transformation_by_name(opt["data"]["transformation"])

    # create dataset
    logger.info("Creating datasets.")
    data_handler = DataHandler(opt["data"]["dataroot"], opt["data"]["variables"], opt["data"]["dataroot"],
                               opt["data"]["months_subset"], opt["data"]["transform_groups"], transformation,
                               opt["data"]["train_min_date"], opt["data"]["train_max_date"],
                               opt["data"]["val_min_date"], opt["data"]["val_max_date"], opt["data"]["val_batch_size"],
                               opt["data"]["batch_size"], opt["data"]["use_shuffle"], opt["data"]["num_workers"])

    train_loader, val_loader, metadata, data_transformer = data_handler.process_data()

    # get model optimizer
    optimizer = get_optimizer(opt["train"]["optimizer"]["type"])

    # create model defined in configs
    model = create_model(opt, optimizer)

    # use loaded model if resuming training
    curr_iter, curr_epoch = model.get_loaded_iter(), model.get_loaded_epoch()
    if opt["path"]["resume_state"]:
        if opt["phase"] == "train":
            logger.info(f"Resuming training from epoch: {curr_epoch}, iter: {curr_iter}.")
        elif opt["phase"] == "val":
            logger.info(f"Resuming validation from epoch: {curr_epoch}, iter: {curr_iter}.")
        else:
            raise NotImplementedError(f"Phase: {opt['phase']} not implemented.")

    model.prepare_to_train()

    # create validation and training metrics
    val_metrics = ValidationMetrics(create_metric_dict(torch_device=configs.device))
    train_metrics = TrainMetrics()

    # start training or validation
    if opt["phase"] == "train":
        train(train_loader, model, logger, data_transformer, train_metrics, val_metrics, wandb_logger, curr_iter,
              curr_epoch, opt, val_loader, metadata)
    elif opt["phase"] == "val":
        validate(val_loader, model, logger, data_transformer, val_metrics, wandb_logger, opt, curr_iter, curr_epoch, metadata)
    else:
        logger.info("Wrong phase")

if __name__ == "__main__":
    main()
