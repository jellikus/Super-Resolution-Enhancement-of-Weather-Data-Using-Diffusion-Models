import torch
import wandb

from tqdm import tqdm
import argparse
import os

from configs.config import Config

import logging
from logger.python_logging import setup_logger

from data.transforms import get_transformation_by_name
from data.dataset_builder import DataHandler

from training.visualization import ImageContainer
from training.metrics import PSNR, RMSE, SSIM, MSE, MR, MAE
from training.utils import get_optimizer
from training.utils import set_seeds
from logger.wandb import WandbLogger

import torch.nn.functional as F


def train(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch over the train dataset.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training data.
        criterion (function): Loss function used for training.
        optimizer (Optimizer): Optimizer used for training.
        device (torch.device): Device on which the model and data are stored.

    Returns:
        float: Average training loss for the epoch.
        int: Number of training iterations.
    """
    model.train()
    train_loss = 0.0
    train_iter = 0

    for batch in tqdm(train_loader):
        inputs = batch[0]["LR"]
        targets = batch[0]["HR"]

        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_iter += 1

    return (train_loss / len(train_loader)), train_iter


def evaluate(model, dataloader, device, data_transformer):
    """
    Evaluate the model using metrics like PSNR, RMSE, SSIM, etc.

    Args:
        model (torch.nn.Module): The trained model for evaluation.
        dataloader (DataLoader): DataLoader for the validation or test data.
        device (torch.device): Device to perform computation on.
        data_transformer: A transformer to apply inverse data transformations.

    Returns:
        tuple: Evaluation metrics (PSNR, RMSE, SSIM, MSE, MR, MAE).
    """
    model.eval()
    psnr = PSNR()
    rmse = RMSE()
    ssim = SSIM()
    mse = MSE()
    mr = MR()
    mae = MAE()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation', leave=False):
            months = batch[1]
            batch = batch[0]

            inputs, targets = batch["LR"].to(device), batch["HR"].to(device)
            outputs = model(inputs)

            visuals_dict = {"SR": outputs.to("cpu"), "HR": targets.to("cpu"), "INTERPOLATED": batch["SR"].to("cpu")}
            reversed_visuals = data_transformer.inverse_transform(visuals_dict, months)
            psnr.update(reversed_visuals["SR"], reversed_visuals["HR"])
            rmse.update(reversed_visuals["SR"], reversed_visuals["HR"])
            ssim.update(reversed_visuals["SR"], reversed_visuals["HR"])
            mse.update(reversed_visuals["SR"], reversed_visuals["HR"])
            mr.update(reversed_visuals["SR"], reversed_visuals["HR"])
            mae.update(reversed_visuals["SR"], reversed_visuals["HR"])

    return psnr.compute(), rmse.compute(), ssim.compute(), mse.compute(), mr.compute(), mae.compute()


# Save the model prediction results
def save_result(model, dataloader, device, path, metadata, data_transformer, max_images=15):
    """
    Save model prediction results for a subset of data from dataloader.
    Result are Interpolated, Super-Resolved and High-Resolution images in one plot.

    Args:
        model (torch.nn.Module): The trained model for generating predictions.
        dataloader (DataLoader): DataLoader for the data.
        device (torch.device): Device to perform computation on.
        path (str): Directory path to save the results.
        metadata: Metadata for the data used.
        data_transformer: Transformer for data normalization and denormalization.
        max_images (int): Maximum number of images to save.

    Returns:
        None
    """
    model.eval()
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), desc='Test', leave=False):
            if idx >= max_images:
                break

            months = batch[1]
            batch = batch[0]
            outputs = model(batch["LR"].to(device))

            batch["INF"] = batch["SR"]
            batch["SR"] = outputs.to('cpu')

            reversed_visuals = data_transformer.inverse_transform(batch, months)
            image_container = ImageContainer(reversed_visuals, metadata)
            image_container.save_it_sr_hr_plot(path + f'/it_sr_hr_plot_{idx}.png')


def get_model(opt):
    """
    Create SimpleCNN or RRDBNet model based on the configuration.
    Args:
        opt: Configuration dictionary.

    Returns:
        Model and loss function.
    """
    if opt["model"]["name"] == 'SimpleSR':
        from models.simple_cnn.Simple_CNN import SimpleCNN
        from models.simple_cnn.loss import image_compare_loss
        model = SimpleCNN(scale_factor=4, channels=opt["model"]["in_channel"])
        criterion = image_compare_loss
    elif opt["model"]["name"] == 'RRDBNet':
        from models.rrdb_encoder.RRDBNet import RRDBNet
        model = RRDBNet(in_nc=opt["model"]["in_channel"], out_nc=opt["model"]["out_channel"],
                        nf=opt["model"]["hidden_size"], nb=opt["model"]["num_block"],
                        gc=opt["model"]["hidden_size"] // 2)
        criterion = F.l1_loss
    else:
        raise ValueError(f"Unknown model name: {opt['model']['name']}")

    return model, criterion


def main():
    """
    Main function to handle the workflow of training, evaluating, or saving results of the model.
    """
    set_seeds()

    # Enable CuDNN GPU acceleration
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="JSON file for configuration")
    parser.add_argument("-p", "--phase", type=str, choices=["train", "val"],
                        help="Run either training or validation(inference).", default="train")
    parser.add_argument("-gpu", "--gpu_ids", type=str, default=None)

    configs = Config(parser.parse_args())
    opt = configs.params

    # setup logger
    setup_logger(None, opt["path"]["log"], "train", screen=True)
    logger = logging.getLogger("base")

    device = int(opt["gpu_ids"])
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    transformation = get_transformation_by_name(opt["data"]["transformation"])

    # create data handler
    logger.info("Creating datasets.")
    data_handler = DataHandler(opt["data"]["dataroot"], opt["data"]["variables"], opt["data"]["dataroot"],
                               opt["data"]["months_subset"], opt["data"]["transform_groups"], transformation,
                               opt["data"]["train_min_date"], opt["data"]["train_max_date"],
                               opt["data"]["val_min_date"], opt["data"]["val_max_date"], opt["data"]["val_batch_size"],
                               opt["data"]["batch_size"], opt["data"]["use_shuffle"], opt["data"]["num_workers"])

    # create data loaders for training and validation
    train_loader, val_loader, metadata, data_transformer = data_handler.process_data()

    # get model optimizer

    _, channels_lr, lr_height, lr_width = data_handler.train_dataset[0][0][0][0].shape
    _, channels_hr, hr_height, hr_width = data_handler.train_dataset[0][1][0][0].shape
    scale_factor = hr_height // lr_height
    assert hr_width // lr_width == scale_factor and channels_lr == channels_hr
    logger.info(f"Scale factor: {scale_factor}")

    # Create models, loss functions and optimizers
    model, criterion = get_model(opt)
    model = model.to(device)

    if opt["path"]["resume_state"]:
        logger.info('Loading pretrained model [', opt["path"]["resume_state"], "]")
        model.load_state_dict(torch.load(opt["path"]["resume_state"]))

    Optimizer = get_optimizer(opt["train"]["optimizer"]["type"])
    optimizer = Optimizer(model.parameters(), lr=opt["train"]["optimizer"]["lr"],
                          amsgrad=opt["train"]["optimizer"]["amsgrad"])

    # Training and evaluation models, and save the model weights along with prediction images
    if opt["phase"] == 'train':
        wandb_logger = WandbLogger(opt)

        logger.info('Start training')
        iteration = 0
        for epoch in range(opt["train"]["epoch"]):
            train_loss, train_iter = train(model, train_loader, criterion, optimizer, device)
            val_psnr, val_rmse, val_ssim, val_mse, val_mr, val_mae = evaluate(model, val_loader, device,
                                                                              data_transformer)
            iteration += train_iter
            logger.info('Epoch [{}/{}],Iter {} ,Train Loss: {:.4f}, '
                        'Val PSNR: {:.4f}, SSIM: {:.4f}, RMSE: {:.4f},  MSE: {:.4f}'.format(epoch + 1,
                                                                                            opt["train"]["epoch"],
                                                                                            iteration,
                                                                                            train_loss, val_psnr,
                                                                                            val_ssim,
                                                                                            val_rmse, val_mse))

            # log metrics to wandb
            wandb_logger.log_metrics({'epoch': epoch + 1}, commit=False, step=iteration)
            wandb_logger.log_train_metrics({"loss": train_loss}, commit=False, step=iteration)
            wandb_logger.log_val_metrics(
                {"MSE": val_mse, "RMSE": val_rmse, "MAE": val_mae, "MR": val_mr, "PSNR": val_psnr, "SSIM": val_ssim},
                commit=False, step=iteration)
            wandb_logger.log_metrics({}, commit=True, step=iteration)

            torch.save(model.state_dict(),
                       os.path.join(opt["path"]["checkpoint"], f'pretrain_{opt["diffusion"]["name"]}_E{epoch}_gen.pth'))

    # perform evaluation on validation set
    elif opt["phase"] == 'val':
        logger.info('Start testing')
        val_psnr, val_rmse, val_ssim, val_mse, val_mr, val_mae = evaluate(model, val_loader, device, data_transformer)
        logger.info(
            'Val PSNR: {:.4f}, SSIM: {:.4f}, RMSE: {:.4f},  MSE: {:.4f}, MAE: {:.4f}, MR: {:.4f}'.format(val_psnr,
                                                                                                         val_ssim,
                                                                                                         val_rmse,
                                                                                                         val_mse,
                                                                                                         val_mae,
                                                                                                         val_mr))

    else:
        raise ValueError(f"Unknown phase: {opt['phase']}")

    # save images generated by the model
    save_result(model, val_loader, device, opt["path"]["validation_results_path"], metadata, data_transformer,
                max_images=opt["save_images"])


if __name__ == '__main__':
    main()
