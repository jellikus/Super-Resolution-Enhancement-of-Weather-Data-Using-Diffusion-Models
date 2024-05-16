import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import os

import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.colors import LinearSegmentedColormap


class ImageContainer:
    def __init__(self, visuals_dict: dict, metadata, n_val_vis=1, mean_image=False):
        """
        Initialize the ImageContainer with a dictionary of images, metadata, and configuration.

        Args:
            visuals_dict (dict): Dictionary of images.
            metadata: Associated metadata for the images.
            n_val_vis (int, optional): Number of validation visualizations. Defaults to 1.
            mean_image (bool, optional): If True, compute the mean image for each batch. Defaults to False.
        """
        self.visuals_dict = visuals_dict
        self.n_val_vis = 1
        self.metadata = metadata

        # for all images in visuals_dict, compute mean image for batch
        if mean_image:
            for key, value in self.visuals_dict.items():
                self.visuals_dict[key] = value.mean(dim=0, keepdim=True)

        self.min = self._compute_min_value()
        self.max = self._compute_max_value()

        self.compute_residual_mask()

        self.CMAPS = CMAPS()

    def compute_residual_mask(self):
        """
        Compute residuals and residual masks for image comparison.
        """
        self.visuals_dict["RESIDUALS"] = self.visuals_dict["SR"] - self.visuals_dict["HR"]
        self.visuals_dict["RESIDUALS_INTERPOLATED"] = self.visuals_dict["INF"] - self.visuals_dict["HR"]
        self.visuals_dict["ABS_RESIDUALS"] = self.visuals_dict["RESIDUALS"].abs()
        self.visuals_dict["ABS_INTERPOLATED"] = self.visuals_dict["RESIDUALS_INTERPOLATED"].abs()

    def set_visuals_dict(self, visuals_dict: dict, n_val_vis=1):
        """
        Set a new visuals dictionary and number of validation visualizations, and recompute metrics.

        Args:
            visuals_dict (dict): New dictionary of images.
            n_val_vis (int, optional): Number of validation visualizations. Defaults to 1.
        """
        self.visuals_dict = visuals_dict
        self.n_val_vis = n_val_vis
        self.min = self._compute_min_value()
        self.max = self._compute_max_value()
        self.compute_residual_mask()

    def _compute_min_value(self):
        """
        Compute the minimum value across all images in the visuals dictionary.

        Returns:
            float: The minimum value found.
        """
        return min([val[:self.n_val_vis].min() for val in self.visuals_dict.values()])

    def _compute_max_value(self):
        """
        Compute the maximum value across all images in the visuals dictionary.

        Returns:
            float: The maximum value found.
        """
        return max([val[:self.n_val_vis].max() for val in self.visuals_dict.values()])

    def set_min_max(self, min_value, max_value):
        """ Set the minimum and maximum values for the images."""
        self.min = min_value
        self.max = max_value

    def _add_batch_index(self, path: str, index: int):
        """Adds the number of batch gotten from data loader to path.

        Args:
            path: The path to which the function needs to add batch index.
            index: The batch index.

        Returns:
            The path with the index appended to the filename.
        """
        try:
            filename, extension = path.split(".")
        except ValueError:
            splitted_parts = path.split(".")
            filename, extension = ".".join(splitted_parts[:-1]), splitted_parts[-1]
        return f"{filename}_{index}.{extension}"

    def save_all_images(self, path: str,
                        image_types=('HR', 'SR', 'LR', 'INTERPOLATED', "DELTA", "AE", "AE_INTER", "AE_TRUTH"),
                        cmap_list=["coolwarm"]):
        """
        Save all images specified by image_type in various formats with specified color maps.

        Args:
            path (str): Base path to save the images.
            image_types (tuple, optional): Tuple of image types to save. Defaults to common types.
            cmap_list (list, optional): List of color maps to use for image saving. Defaults to ['coolwarm'].
        """
        for cmap in cmap_list:
            if 'HR' in image_types:
                self.create_and_save_images(latitude=self.metadata.hr_lat, longitude=self.metadata.hr_lon,
                                            data=self.visuals_dict["HR"][:self.n_val_vis],
                                            path=f"{path}_{cmap}_hr.png", vmin=self.min, vmax=self.max, cmap=cmap)
            if 'SR' in image_types:
                self.create_and_save_images(latitude=self.metadata.hr_lat, longitude=self.metadata.hr_lon,
                                            data=self.visuals_dict["SR"][:self.n_val_vis],
                                            path=f"{path}_{cmap}_sr.png", vmin=self.min, vmax=self.max, cmap=cmap)
            if 'LR' in image_types:
                self.create_and_save_images(latitude=self.metadata.lr_lat, longitude=self.metadata.lr_lon,
                                            data=self.visuals_dict["LR"][:self.n_val_vis],
                                            path=f"{path}_{cmap}_lr.png", vmin=self.min, vmax=self.max, cmap=cmap)
            if 'INTERPOLATED' in image_types:
                self.create_and_save_images(latitude=self.metadata.hr_lat, longitude=self.metadata.hr_lon,
                                            data=self.visuals_dict["INF"][:self.n_val_vis],
                                            path=f"{path}_{cmap}_interpolated.png", vmin=self.min, vmax=self.max,
                                            cmap=cmap)
        if 'DELTA' in image_types:
            self.create_and_save_images(latitude=self.metadata.hr_lat, longitude=self.metadata.hr_lon,
                                        data=self.visuals_dict["RESIDUALS"][:self.n_val_vis],
                                        path=f"{path}_delta.png", vmin=-1, vmax=1,
                                        costline_color="black", cmap="custom",
                                        label=" ")

        if 'WHITE' in image_types:
            self.create_and_save_images(latitude=self.metadata.hr_lat, longitude=self.metadata.hr_lon,
                                        data=(self.visuals_dict["HR"] - self.visuals_dict["HR"])[:self.n_val_vis],
                                        path=f"{path}white.png", vmin=-1, vmax=1,
                                        costline_color="black", cmap="custom",
                                        label=" ")
        if 'AE_TRUTH' in image_types:
            self.create_and_save_images(latitude=self.metadata.hr_lat, longitude=self.metadata.hr_lon,
                                        data=(self.visuals_dict["HR"] - self.visuals_dict["HR"])[:self.n_val_vis],
                                        path=f"{path}_truth_absolute_error.png", vmin=0, vmax=21,
                                        costline_color="black", cmap="custom_ae",
                                        label=" ")
        if 'AE' in image_types:
            self.create_and_save_images(latitude=self.metadata.hr_lat, longitude=self.metadata.hr_lon,
                                        data=self.visuals_dict["ABS_RESIDUALS"][:self.n_val_vis],
                                        path=f"{path}_absolute_error.png", vmin=0, vmax=21,
                                        costline_color="black", cmap="custom_ae",
                                        label=" ")
        if 'AE_INTER' in image_types:
            self.create_and_save_images(latitude=self.metadata.hr_lat, longitude=self.metadata.hr_lon,
                                        data=self.visuals_dict["ABS_INTERPOLATED"][:self.n_val_vis],
                                        path=f"{path}_interpolated_absolute_error.png", vmin=0, vmax=21,
                                        costline_color="black", cmap="custom_ae",
                                        label=" ")


    def _create_and_save_image(self, latitude: np.array, longitude: np.array, single_variable: torch.tensor,
                               path: str, title: str = None, label: str = None, dpi: int = 200,
                               figsize: tuple = (11, 8.5), cmap: str = "coolwarm", vmin=None,
                               vmax=None, costline_color="black"):

        """Create and save a single image from the data with geographical information.


        Args:
            latitude: An array of latitudes.
            longitude: An array of longitudes.
            single_variable: A tensor to visualize.
            path: Path of a directory to save visualization.
            title: Title of the figure.
            label: Label of the colorbar.
            dpi: Resolution of the figure.
            figsize: Tuple of (width, height) in inches.
            cmap: A matplotlib colormap.
            vmin: Minimum value for colormap.
            vmax: Maximum value for colormap.
            costline_color: Matplotlib color.
        """
        single_variable, longitude = add_cyclic_point(single_variable, coord=np.array(longitude))
        plt.figure(dpi=dpi, figsize=figsize)
        projection = ccrs.PlateCarree()
        ax = plt.axes(projection=projection)

        if cmap == "binary":
            # For mask visualization.
            p = plt.contourf(longitude, latitude, single_variable, 60, transform=projection,
                             cmap=(matplotlib.colors.ListedColormap(["white", "gray", "black"])
                                   .with_extremes(over="0.25", under="0.75")),
                             vmin=-1, vmax=1)
            boundaries, ticks = [-1, -0.33, 0.33, 1], [-1, 0, 1]
        elif cmap == "coolwarm":
            p = plt.contourf(longitude, latitude, single_variable, 60, transform=projection, cmap=cmap,
                             levels=np.linspace(vmin, vmax, max(int(np.abs(vmax - vmin)) // 2, 3)))
            boundaries, ticks = None, np.round(np.linspace(vmin, vmax, 7), 2)

        elif cmap == "custom_heatmap_vibrant":
            custom_cmap = self.CMAPS.heat_vibrant()
            p = plt.contourf(
                longitude, latitude, single_variable, 60,
                transform=projection,
                cmap=custom_cmap,
                levels=np.linspace(vmin, vmax, max(int(np.abs(vmax - vmin)) // 2, 3))
            )
            boundaries, ticks = None, np.round(np.linspace(vmin, vmax, 7), 2)

        elif cmap == "heat_muted":
            custom_cmap = self.CMAPS.heat_muted()
            p = plt.contourf(
                longitude, latitude, single_variable, 60,
                transform=projection,
                cmap=custom_cmap,
                levels=np.linspace(vmin, vmax, max(int(np.abs(vmax - vmin)) // 2, 3))
            )
            boundaries, ticks = None, np.round(np.linspace(vmin, vmax, 7), 2)

        elif cmap == "viridis":
            # For temperature visualization.
            p = plt.contourf(longitude, latitude, single_variable, 60, transform=projection, cmap=cmap,
                             levels=np.linspace(vmin, vmax, max(int(np.abs(vmax - vmin)) // 2, 3)))
            boundaries, ticks = None, np.round(np.linspace(vmin, vmax, 7), 2)

        elif cmap == "Greens":
            # For visualization of standard deviation.
            p = plt.contourf(longitude, latitude, single_variable, 60, transform=projection, cmap=cmap,
                             extend="max")
            boundaries, ticks = None, np.linspace(single_variable.min(), single_variable.max(), 5)

        elif cmap == "custom_ae":
            custom_cmap = self.CMAPS.ae_color()

            # Create the main contour plot
            p = plt.contourf(longitude, latitude, single_variable, 60, transform=projection,
                             cmap=custom_cmap, levels=np.linspace(0, 21, 400))

            # Overlay for values above 21
            # Use np.ma.masked_where to ignore values <= 21
            masked_data = np.ma.masked_where(single_variable <= 21, single_variable)
            plt.contourf(longitude, latitude, masked_data, levels=[20.5, 10000000],
                         colors=['#ff0000'], transform=projection)  # Vibrant red for values above 21

            boundaries, ticks = None, [0, 3, 6, 9, 12, 15, 18, 21]


        elif cmap == "custom":
            # For visualization of standard deviation.
            minimum = -25
            maximum = 25
            from matplotlib.colors import Normalize

            cmap = self.CMAPS.abs_color()
            norm = Normalize(vmin=minimum, vmax=maximum)
            p = plt.contourf(longitude, latitude, single_variable, 60, transform=projection, cmap=cmap,
                             norm=norm, levels=np.linspace(minimum, maximum, 100))
            boundaries, ticks = None, np.linspace(minimum, maximum, 11)

        ax.coastlines(color="black")

        plt.colorbar(p, pad=0.01, label=label, orientation="horizontal",
                     boundaries=boundaries, ticks=ticks, aspect=60)
        plt.savefig(path, bbox_inches="tight")
        plt.close("all")

    def create_and_save_images(self, latitude: np.array, longitude: np.array, data: torch.tensor,
                               path: str, title: str = None, label: str = None,
                               dpi: int = 200, figsize: tuple = (11, 8.5), cmap: str = "coolwarm",
                               vmin=None, vmax=None, costline_color="black"):
        """Helper function to create and save a single image visualization for a given variable.


        Args:
            latitude: An array of latitudes.
            longitude: An array of longitudes.
            data: A batch of variables to visualize.
            path: Path of a directory to save visualization.
            title: Title of the figure.
            label: Label of the colorbar.
            dpi: Resolution of the figure.
            figsize: Tuple of (width, height) in inches.
            cmap: A matplotlib colormap.
            vmin: Minimum value for colormap.
            vmax: Maximum value for colormap.
            costline_color: Matplotlib color.
        """
        if len(data.shape) > 2:
            data = data.squeeze()

        # if batch is larger than 1, add batch index to path and create image for each sample
        if len(data.shape) > 2:
            file_path, extension = os.path.splitext(path)
            for batch_index in range(data.shape[0]):
                path_for_sample = f"{file_path}_{batch_index}{extension}"
                self._create_and_save_image(latitude, longitude, data[batch_index], path_for_sample,
                                            title, label, dpi, figsize, cmap, vmin, vmax, costline_color)
        else:
            self._create_and_save_image(latitude, longitude, data, path, title, label, dpi, figsize, cmap,
                                        vmin, vmax, costline_color)

    def save_it_sr_hr_plot(self, path: str):
        """
        Create and save a plot comparing interpolated, super-resolution, and high-resolution data.

        Args:
            path (str): Base path for saving the plot image.
        """
        data = (self.visuals_dict["INF"][-1].squeeze(), self.visuals_dict["SR"][-1].squeeze(),
                self.visuals_dict["HR"][-1].squeeze())
        fig = self._create_it_sr_hr_plot(latitude=self.metadata.hr_lat, longitude=self.metadata.hr_lon, data=data)
        fig.savefig(path, format="png")

    def make_wandb_plot(self):
        """
        Create a plot for visualization in Weights & Biases (wandb).

        This method generates a comparison plot of interpolated, super-resolution, and high-resolution
        data, which is commonly used to evaluate model performance visually. The plot is not saved but
        returned for potentially logging it directly to Weights & Biases.

        Returns:
            matplotlib.figure.Figure: The created figure suitable for wandb logging.
        """

        data = (self.visuals_dict["INF"][-1].squeeze(), self.visuals_dict["SR"][-1].squeeze(),
                self.visuals_dict["HR"][-1].squeeze())
        return self._create_it_sr_hr_plot(latitude=self.metadata.hr_lat, longitude=self.metadata.hr_lon, data=data)

    def _create_it_sr_hr_plot(self, latitude: np.array, longitude: np.array, data: tuple, label=None,
                              dpi: int = 300, figsize: tuple = (22, 6), cmap: str = "coolwarm") -> Figure:
        """Construct tensorboard visualization figure.

        Args:
            latitude: An array of latitudes.
            longitude: An array of longitudes.
            data: A batch of variables to visualize.
            label: Label of the colorbar.
            dpi: Resolution of the figure.
            figsize: Tuple of (width, height) in inches.
            cmap: A matplotlib colormap.

        Returns:
            Matplotlib Figure.
        """

        TITLES = ("Upsampled with interpolation",
                  "Super-resolution reconstruction",
                  "High-resolution original")

        max_value = max((tensor.max() for tensor in data))
        min_value = min((tensor.min() for tensor in data))
        projection = ccrs.PlateCarree()
        axes_class = (GeoAxes, dict(projection=projection))
        fig = plt.figure(figsize=figsize, dpi=dpi)
        axgr = AxesGrid(fig, 111, axes_class=axes_class, nrows_ncols=(1, 3), axes_pad=0.95, cbar_location="bottom",
                        cbar_mode="single", cbar_pad=0.01, cbar_size="2%", label_mode='')
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()

        for i, ax in enumerate(axgr):
            single_variable, lon = add_cyclic_point(data[i], coord=np.array(longitude))
            ax.set_title(TITLES[i])
            ax.gridlines(draw_labels=True, xformatter=lon_formatter, yformatter=lat_formatter,
                         xlocs=np.linspace(-180, 180, 5), ylocs=np.linspace(-90, 90, 5))
            p = ax.contourf(lon, latitude, single_variable, transform=projection, cmap=cmap,
                            vmin=min_value, vmax=max_value)
            ax.coastlines()

        axgr.cbar_axes[0].colorbar(p, pad=0.01, label=label, shrink=0.95)
        plt.close("all")
        return fig

    def save_sr_hr_plot(self, path: str, cmap="coolwarm"):
        """
        Save a comparison plot of super-resolution and high-resolution images.

        Args:
            path (str): Path to save the plot image.
            cmap (str, optional): Color map for visualization. Defaults to 'coolwarm'.

        This method saves a plot comparing super-resolution (SR) and high-resolution (HR) images to
        a specified path. This is useful for visually assessing the performance of super-resolution algorithms.
        """
        data = (self.visuals_dict["HR"][-1].squeeze(), self.visuals_dict["SR"][-1].squeeze())
        fig = self._create_sr_hr_plot(latitude=self.metadata.hr_lat, longitude=self.metadata.hr_lon, data=data,
                                      cmap=cmap)
        fig.savefig(path + f"_sr_hr_{cmap}", format="png")

    def _create_sr_hr_plot(self, latitude: np.array, longitude: np.array, data: tuple, label=None,
                           dpi: int = 300, figsize: tuple = (22, 6), cmap: str = "coolwarm") -> Figure:
        """
        Helper function to create a high-quality visualization figure.

        Args:
            latitude (np.array): Array of latitudes.
            longitude (np.array): Array of longitudes.
            data (tuple): Tuple of image data tensors.
            label (str, optional): Label for the colorbar. Defaults to None.
            dpi (int, optional): Resolution of the figure. Defaults to 300.
            figsize (tuple, optional): Figure size in inches (width, height). Defaults to (22, 6).
            cmap (str, optional): Color map for visualization. Defaults to 'coolwarm'.

        Returns:
            matplotlib.figure.Figure: The created figure, which is suitable for high-quality publications or presentations.
        """

        TITLES = ("High-resolution Ground truth",
                  "Model reconstruction")

        max_value = 315
        min_value = 220

        projection = ccrs.PlateCarree()
        axes_class = (GeoAxes, dict(projection=projection))
        fig = plt.figure(figsize=figsize, dpi=dpi)
        axgr = AxesGrid(fig, 111, axes_class=axes_class, nrows_ncols=(1, 2), axes_pad=0.95, cbar_location="bottom",
                        cbar_mode="single", cbar_pad=0.01, cbar_size="2%", label_mode='')
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        levels = 8
        for i, ax in enumerate(axgr):
            single_variable, lon = add_cyclic_point(data[i], coord=np.array(longitude))
            ax.set_title(TITLES[i])
            ax.gridlines(draw_labels=True, xformatter=lon_formatter, yformatter=lat_formatter,
                         xlocs=np.linspace(-180, 180, 5), ylocs=np.linspace(-90, 90, 5))
            p = ax.contourf(lon, latitude, single_variable, transform=projection, cmap=cmap,
                            vmin=min_value, vmax=max_value, levels=np.linspace(min_value, max_value, 9))
            ax.coastlines()

        axgr.cbar_axes[0].colorbar(p, pad=0.01, label=label, ticks=np.round(np.linspace(min_value, max_value, 9), 1))
        # fig.tight_layout()
        plt.close("all")
        return fig

    def save_sr_hr_abs_plot(self, path: str):
        """
        Save a plot comparing the absolute errors of super-resolution and high-resolution data.

        Args:
            path (str): The file path where the plot will be saved.

        This method generates a plot showing the absolute errors between super-resolution and high-resolution
        images. This is particularly useful for assessing the error distribution and quality of the super-resolution model.
        """
        data = (self.visuals_dict["ABS_INTERPOLATED"][-1].squeeze(), self.visuals_dict["ABS_RESIDUALS"][-1].squeeze())
        fig = self._create_abs_plot(latitude=self.metadata.hr_lat, longitude=self.metadata.hr_lon, data=data)
        fig.savefig(path + "_sr_hr_abs", format="png")

    def _create_abs_plot(self, latitude: np.array, longitude: np.array, data: tuple, label=None,
                         dpi: int = 300, figsize: tuple = (22, 6), cmap: str = "coolwarm") -> Figure:
        """
        Helper function to create a visualization figure of absolute errors.

        Args:
            latitude (np.array): Array of latitudes.
            longitude (np.array): Array of longitudes.
            data (tuple): Tuple of image data tensors showing absolute errors.
            label (str, optional): Label for the colorbar. Defaults to None.
            dpi (int, optional): Resolution of the figure. Defaults to 300.
            figsize (tuple, optional): Figure size in inches (width, height). Defaults to (22, 6).
            cmap (str, optional): Color map for visualization. Defaults to 'coolwarm'.

        Returns:
            matplotlib.figure.Figure: The created figure, which visually represents the absolute errors in a comparative layout.
        """
        TITLES = ("Bicubic interpolation Absolute Error",
                  "Model Absolute Error")

        max_value = 21
        min_value = 0

        colors = [
            (0.0, "darkblue"),  # Dark blue at the minimum
            (0.08, "blue"),  # Blue at 20% of the range
            (0.16, "cyan"),  # Cyan at 40% of the range
            (0.3, "green"),  # Green at 60% of the range
            (0.5, "yellow"),  # Yellow at 80% of the range
            (1.0, "red")  # Red at the maximum
        ]
        from matplotlib.colors import LinearSegmentedColormap
        custom_cmap = LinearSegmentedColormap.from_list("custom_ae", colors)

        projection = ccrs.PlateCarree()
        axes_class = (GeoAxes, dict(projection=projection))
        fig = plt.figure(figsize=figsize, dpi=dpi)
        axgr = AxesGrid(fig, 111, axes_class=axes_class, nrows_ncols=(1, 2), axes_pad=0.95, cbar_location="bottom",
                        cbar_mode="single", cbar_pad=0.01, cbar_size="2%", label_mode='')
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()

        for i, ax in enumerate(axgr):
            single_variable, lon = add_cyclic_point(data[i], coord=np.array(longitude))
            ax.set_title(TITLES[i])
            ax.gridlines(draw_labels=True, xformatter=lon_formatter, yformatter=lat_formatter,
                         xlocs=np.linspace(-180, 180, 5), ylocs=np.linspace(-90, 90, 5))
            p = ax.contourf(lon, latitude, single_variable, transform=projection, cmap=custom_cmap,
                            vmin=min_value, vmax=max_value, levels=np.linspace(0, 21, 400))
            ax.coastlines()

        axgr.cbar_axes[0].colorbar(p, pad=0.01, label=label, ticks=[0, 3, 6, 9, 12, 15, 18, 21])
        plt.close("all")
        return fig

    def save_tensor_it_sr_hr_plot(self, path: str):
        """
        Saves a plot of three tensors: interpolated, super-resolution, and high-resolution.

        Args:
            path (str): The path to save the plot image.
        """

        data = (self.visuals_dict["INF"][-1].squeeze(), self.visuals_dict["SR"][-1].squeeze(),
                self.visuals_dict["HR"][-1].squeeze())
        fig = self._create_tensor_it_sr_hr_plot(data)
        fig.savefig(path, format="png")

    def _create_tensor_it_sr_hr_plot(self, data: tuple, label=None,
                                     dpi: int = 300, figsize: tuple = (22, 6), cmap: str = "gray") -> plt.Figure:
        """Construct a subplot figure for three tensors.

        Args:
            data: A batch of tensors to visualize.
            label: Label of the colorbar.
            dpi: Resolution of the figure.
            figsize: Tuple of (width, height) in inches.
            cmap: A matplotlib colormap.

        Returns:
            Matplotlib Figure.
        """
        TITLES = ("Tensor INTERPOLATED", "Tensor SR", "Tensor HR")

        max_value = max((tensor.max() for tensor in data))
        min_value = min((tensor.min() for tensor in data))

        fig, axs = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

        for i, ax in enumerate(axs):
            ax.set_title(TITLES[i])
            p = ax.imshow(data[i], cmap=cmap, vmin=min_value, vmax=max_value)
            ax.grid(False)

        cbar = fig.colorbar(p, ax=axs, pad=0.01, label=label, shrink=0.95)
        cbar.ax.tick_params(labelsize=8)  # Adjust colorbar label size if needed

        plt.close("all")
        return fig

class CMAPS:
    """
    A class that manages custom color maps for data visualization.
    """

    def get(self, cmap_name: str):
        if cmap_name == "custom_heatmap_vibrant":
            return self.heat_vibrant()
        elif cmap_name == "heat_muted":
            return self.heat_muted()
        elif cmap_name == "custom_ae":
            return self.ae_color()
        elif cmap_name == "custom":
            return self.abs_color()

    def heat_vibrant(self):
        """
        Generate a vibrant heat map color scheme.

        Returns:
            matplotlib.colors.LinearSegmentedColormap: A vibrant heat map color map.
        """
        colors = [
            (0.5, 0, 0.5),  # purple
            (0, 0, 1),  # blue
            (0, 1, 1),  # cyan
            (0, 1, 0),  # green
            (1, 1, 0),  # yellow
            (1, 0.5, 0),  # orange
            (1, 0, 0)  # red
        ]
        from matplotlib.colors import LinearSegmentedColormap
        cmap_name = "custom_heatmap_vibrant"
        n_bins = 100  # Increase this number for a smoother transition between colors
        return LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    def heat_muted(self):
        """
        Generate a muted heat map color scheme.

        Returns:
            matplotlib.colors.LinearSegmentedColormap: A muted heat map color map.
        """
        colors = [
            (0.75, 0.5, 0.75),  # less vibrant purple
            (0.5, 0.5, 1),  # less vibrant blue
            (0.5, 1, 1),  # less vibrant cyan
            (0.5, 1, 0.5),  # less vibrant green
            (1, 1, 0.5),  # less vibrant yellow
            (1, 0.75, 0.5),  # less vibrant orange
            (1, 0.5, 0.5)  # less vibrant red
        ]

        cmap_name = "heat_muted"
        n_bins = 100  # Increase this number for a smoother transition between colors
        return LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    def ae_color(self):
        """
        Generate a color map designed for visualizing absolute error in temperature.

        Returns:
            matplotlib.colors.LinearSegmentedColormap: A color map tailored for absolute error visualization.
        """
        colors = [
            (0.0, "darkblue"),  # Dark blue at the minimum
            (0.08, "blue"),  # Blue at 20% of the range
            (0.16, "cyan"),  # Cyan at 40% of the range
            (0.3, "green"),  # Green at 60% of the range
            (0.5, "yellow"),  # Yellow at 80% of the range
            (1.0, "red")  # Red at the maximum
        ]
        return LinearSegmentedColormap.from_list("custom_ae", colors)

    def abs_color(self):
        """
        Generate a color map for visualizing a range of values with emphasis on specific ranges.

        Returns:
            matplotlib.colors.LinearSegmentedColormap: A custom color map emphasizing mid-range values.
        """
        minimum = -25
        maximum = 25
        zero_norm_position = (0 - minimum) / (maximum - minimum)
        light_blue_position = (-5 - minimum) / (maximum - minimum)  # light blue close to zero
        light_red_position = (5 - minimum) / (maximum - minimum)  # light red close to zero

        cmap = LinearSegmentedColormap.from_list(
            'CustomMap', [
                (0.0, 'darkblue'),  # dark blue at the minimum
                (light_blue_position, 'lightblue'),  # light blue for values close to zero
                (zero_norm_position, 'white'),  # zero to be white
                (light_red_position, 'salmon'),  # light red for values close to zero
                (1.0, 'darkred')  # dark red at the maximum
            ]
        )
        return cmap
