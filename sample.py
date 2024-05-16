import argparse
import torch

# data
from data.transforms import get_transformation_by_name
from data.dataset_builder import DataHandler
from data.utils import get_month_idx
from datetime import timedelta

from configs.config import Config
from models.base_model import create_model

from training.utils import get_optimizer
from training.visualization import ImageContainer



def main():
    """Main function for data visualization."""

    # Enable CuDNN GPU acceleration
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="JSON file for configuration", required=True)
    parser.add_argument("-p", "--model_path", type=str, default=None, help="Path to trained model")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Path to save ouput images")
    parser.add_argument("-gpu", "--gpu_ids", type=str, default=None, help="GPU ids to use")
    parser.add_argument("-n", "--number_of_samples", type=int, default=1, help="Number of samples to visualize")
    image_types = ['HR', 'SR', 'LR', 'INTERPOLATED', 'DELTA', 'AE']
    parser.add_argument('-t', '--image_types', nargs='+',
                        help=f'List of image types to process. Choose from: {", ".join(image_types)}',
                        default=['SR'],
                        choices=image_types)
    parser.add_argument("-m", "--color_map", type=str, default='coolwarm', help="Color map for visualization",
                        choices=['coolwarm', 'heat_muted'])
    parser.add_argument("-d", "--date", type=str, default=None, help="Date to visualize in (YYYY-MM-DD-HH) format")



    parser.parse_args()
    args = parser.parse_args()
    configs = Config(args, False)
    opt = configs.params
    number_of_samples = args.number_of_samples

    if args.date:
        opt["data"]["months_subset"] = [get_month_idx(args.date)]
        opt["data"]["val_min_date"] = args.date
        opt["data"]["val_max_date"] = args.date + timedelta(hours=1)

    print("Preparing data.")
    data_handler = DataHandler(opt["data"]["dataroot"], opt["data"]["variables"], opt["data"]["dataroot"],
                               opt["data"]["months_subset"], opt["data"]["transform_groups"],
                               get_transformation_by_name(opt["data"]["transformation"]), opt["data"]["val_min_date"],
                               opt["data"]["val_max_date"],
                               opt["data"]["val_min_date"], opt["data"]["val_max_date"], opt["data"]["val_batch_size"],
                               opt["data"]["batch_size"], opt["data"]["use_shuffle"], opt["data"]["num_workers"])

    _, val_loader, metadata, data_transformer = data_handler.process_data()



    # load trained model
    if args.model_path:
        opt["path"]["resume_state"] = args.model_path

    if not opt["path"]["resume_state"]:
        raise ValueError("Model path not provided.")

    # create model
    print("Creating model.")
    model = create_model(opt, get_optimizer(opt["train"]["optimizer"]["type"]))
    model.prepare_to_eval()

    if args.date:
        data_to_feed = data_handler.get_data_by_date(args.date)
    else:
        data_to_feed = next(iter(val_loader))

    model.feed_data(data_to_feed)
    model.generate_sr()
    visuals = model.get_images(need_LR=True)
    reversed_visuals = data_transformer.inverse_transform(visuals, model.get_months())
    image_container = ImageContainer(reversed_visuals, metadata, number_of_samples)
    image_container.set_min_max(220, 315)

    print("Making visualizations.")
    image_container.save_all_images(path=args.output_path, cmap_list=[args.color_map],
                                    image_types=args.image_types)


if __name__ == "__main__":
    main()
