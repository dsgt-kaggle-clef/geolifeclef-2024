# Author: Benjamin Deneu <benjamin.deneu@inria.fr>
#         Theo Larcher <theo.larcher@inria.fr>
#
# License: GPLv3
#
# Python version: 3.10.6

import itertools
import logging
import math
import os
from abc import abstractmethod
from random import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio
from PIL import Image

from .get_jpeg_patches_mean_std import standardize as jpeg_standardize


class PatchProvider(object):
    def __init__(self, size, normalize) -> None:
        self.patch_size = size
        self.normalize = normalize
        self.nb_layers = 0

    @abstractmethod
    def __getitem__(self, item):
        pass

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def __str__(self):
        pass

    def __len__(self):
        return self.nb_layers

    def plot_patch(self, item):
        patch = self[item]
        if self.nb_layers == 1:
            plt.figure(figsize=(10, 10))
            plt.imshow(patch[0])
        else:
            # calculate the number of rows and columns for the subplots grid
            rows = int(math.ceil(math.sqrt(self.nb_layers)))
            cols = int(math.ceil(self.nb_layers / rows))

            # create a figure with a grid of subplots
            fig, axs = plt.subplots(rows, cols, figsize=(10, 10))

            # flatten the subplots array to easily access the subplots
            axs = axs.flatten()

            # loop through the layers of patch data
            for i in range(self.nb_layers):
                # display the layer on the corresponding subplot
                axs[i].imshow(patch[i])
                axs[i].set_title("layer_{}: {}".format(i, self.bands_names[i]))
                axs[i].axis("off")

            # remove empty subplots
            for i in range(self.nb_layers, rows * cols):
                fig.delaxes(axs[i])

        plt.suptitle("Tensor for item: " + str(item), fontsize=16)

        # show the plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


class MetaPatchProvider(PatchProvider):
    def __init__(self, providers, transform=None):
        self.providers = providers
        self.nb_layers = sum([len(provider) for provider in self.providers])
        self.bands_names = list(
            itertools.chain.from_iterable(
                [provider.bands_names for provider in self.providers]
            )
        )
        self.transform = transform

    def __getitem__(self, item):
        patch = np.concatenate([provider[item] for provider in self.providers])
        if self.transform:
            patch = self.transform(patch)
        return patch

    def __str__(self):
        result = "Providers:\n"
        for provider in self.providers:
            result += str(provider)
            result += "\n"
        return result


class RasterPatchProvider(PatchProvider):
    def __init__(
        self,
        raster_path,
        size=128,
        spatial_noise=0,
        normalize=True,
        fill_zero_if_error=False,
        nan_value=0,
    ):
        super().__init__(size, normalize)
        self.spatial_noise = spatial_noise
        self.fill_zero_if_error = fill_zero_if_error
        self.transformer = None
        self.name = os.path.basename(os.path.splitext(raster_path)[0])
        self.normalize = normalize

        # open the tif file with rasterio
        with rasterio.open(raster_path) as src:
            # read the metadata of the file
            meta = src.meta
            meta.update(
                count=src.count
            )  # update the count of the meta to match the number of layers

            # read the data from the raster
            self.data = src.read()

            # get the NoData value from the raster
            self.nodata_value = src.nodatavals

            # iterate through all the layers
            for i in range(src.count):
                # replace the NoData values with np.nan
                self.data = self.data.astype(np.float)
                self.data[i] = np.where(
                    self.data[i] == self.nodata_value[i], np.nan, self.data[i]
                )
                if self.normalize:
                    self.data[i] = (
                        self.data[i] - np.nanmean(self.data[i])
                    ) / np.nanstd(self.data[i])
                self.data[i] = np.where(np.isnan(self.data[i]), nan_value, self.data[i])

            self.nb_layers = src.count

            self.x_min = src.bounds.left
            self.y_min = src.bounds.bottom
            self.x_resolution = src.res[0]
            self.y_resolution = src.res[1]
            self.n_rows = src.height
            self.n_cols = src.width
            self.crs = src.crs
        if self.nb_layers > 1:
            self.bands_names = [
                self.name + "_" + str(i + 1) for i in range(self.nb_layers)
            ]
        else:
            self.bands_names = [self.name]

        self.epsg = self.crs.to_epsg()
        if self.epsg != 4326:
            # create a pyproj transformer object to convert lat, lon to EPSG:32738
            self.transformer = pyproj.Transformer.from_crs(
                "epsg:4326", self.epsg, always_xy=True
            )

    def __getitem__(self, item):
        """
        :param item: dictionary that needs to contains at least the keys latitude and longitude ({'lat': lat, 'lon':lon})
        :return: return the environmental tensor or vector (size>1 or size=1)
        """

        # convert the lat, lon coordinates to EPSG:32738
        if self.transformer:
            lon, lat = self.transformer.transform(item["lon"], item["lat"])
        else:
            lon, lat = (item["lon"], item["lat"])

        # add noise as data augmentation
        if self.spatial_noise > 0:
            lon = lon + ((random() * 2 * self.spatial_noise) - self.spatial_noise)
            lat = lat + ((random() * 2 * self.spatial_noise) - self.spatial_noise)

        # calculate the x, y coordinate of the point of interest
        x = int(self.n_rows - (lat - self.y_min) / self.y_resolution)
        y = int((lon - self.x_min) / self.x_resolution)

        # read the data of the patch from all layers
        if self.patch_size == 1:
            patch_data = [self.data[i, x, y] for i in range(self.nb_layers)]
        else:
            patch_data = [
                self.data[
                    i,
                    x - (self.patch_size // 2) : x + (self.patch_size // 2),
                    y - (self.patch_size // 2) : y + (self.patch_size // 2),
                ]
                for i in range(self.nb_layers)
            ]

        tensor = np.concatenate([patch[np.newaxis] for patch in patch_data])
        if self.fill_zero_if_error and tensor.shape != (
            self.nb_layers,
            self.patch_size,
            self.patch_size,
        ):
            tensor = np.zeros((self.nb_layers, self.patch_size, self.patch_size))
        return tensor

    def __str__(self):
        result = "-" * 50 + "\n"
        result += "n_layers: " + str(self.nb_layers) + "\n"
        result += "x_min: " + str(self.x_min) + "\n"
        result += "y_min: " + str(self.y_min) + "\n"
        result += "x_resolution: " + str(self.x_resolution) + "\n"
        result += "y_resolution: " + str(self.y_resolution) + "\n"
        result += "n_rows: " + str(self.n_rows) + "\n"
        result += "n_cols: " + str(self.n_cols) + "\n"
        result += "-" * 50
        return result


class MultipleRasterPatchProvider(PatchProvider):
    def __init__(
        self,
        rasters_folder,
        select=None,
        size=128,
        spatial_noise=0,
        normalize=True,
        fill_zero_if_error=False,
    ):
        files = os.listdir(rasters_folder)
        # Filter files to include only those with .tif extension
        # rasters_paths = [f for f in files if f.endswith('.tif')]
        # if select:
        #    select = [r+'.tif' for r in select]
        #    rasters_paths = [r for r in rasters_paths if r in select]
        if select:
            rasters_paths = [r + ".tif" for r in select]
        else:
            rasters_paths = [f for f in files if f.endswith(".tif")]
        self.rasters_providers = [
            RasterPatchProvider(
                rasters_folder + path,
                size=size,
                spatial_noise=spatial_noise,
                normalize=normalize,
                fill_zero_if_error=fill_zero_if_error,
            )
            for path in rasters_paths
        ]
        self.nb_layers = sum([len(raster) for raster in self.rasters_providers])
        self.bands_names = list(
            itertools.chain.from_iterable(
                [raster.bands_names for raster in self.rasters_providers]
            )
        )

    def __getitem__(self, item):
        return np.concatenate([raster[item] for raster in self.rasters_providers])

    def __str__(self):
        result = "Rasters in folder:\n"
        for raster in self.rasters_providers:
            result += str(raster)
            result += "\n"
        return result


class JpegPatchProvider(PatchProvider):
    """JPEG patches provider for GLC23.

    Provides tensors of multi-modal patches from JPEG patch files
    of rasters of the GLC23 challenge.

    Attributes:
        PatchProvider (_type_): _description_
    """

    def __init__(
        self,
        root_path,
        select=None,
        normalize=True,
        patch_transform=None,
        size=128,
        dataset_stats="jpeg_patches_stats.csv",
    ):
        """Class constructor.

        Args:
            root_path (str): root path to the directory containing all patches modalities
            channel_list (list, optional): list of channels to provide for the output tensor. Defaults to None.
            normalize (bool, optional): normalize data. Defaults to False.
            patch_transform (callable, optional): custom transformation functions. Defaults to None.
            size (int, optional): default tensor sizes (must match patch sizes). Defaults to 128.
            dataset_stats (str, optional): path to the csv file containing the mean and std values of the
                                           jpeg patches dataset if `normalize` is True. If the file doesn't
                                           exist, the values will be calculated and the file will be created once.
                                           Defaults to 'jpeg_patches_stats.csv'
        """
        super().__init__(size, normalize)
        self.patch_transform = patch_transform
        self.root_path = root_path
        self.ext = ".jpeg"
        self.dataset_stats = os.path.join(self.root_path, dataset_stats)

        self.channel_folder = {
            "red": "rgb",
            "green": "rgb",
            "blue": "rgb",
            "swir1": "swir1",
            "swir2": "swir2",
            "nir": "nir",
        }
        if not select:
            sub_dirs = next(os.walk(root_path))[1]
            select = [k for k, v in self.channel_folder.items() if v in sub_dirs]

        self.channels = [c.lower() for c in select]
        self.nb_layers = len(self.channels)
        self.bands_names = self.channels

    def __getitem__(self, item):
        """Return a tensor composed of every channels of a jpeg patch.

        Args:
            item (dict): dictionary containing the patchID necessary to
                         identify the jpeg patch to return.

        Raises:
            KeyError: the 'patchID' key is missing from item
            Exception: item is not a dictionary as expected

        Returns:
            (tensor): multi-channel patch tensor.
        """
        try:
            id_ = str(int(item["patchID"]))
        except KeyError as e:
            raise KeyError("The patchID key does not exists.")
        except Exception as e:
            raise Exception(
                "An error has occurred when trying to load a patch patchID."
                'Check that the input argument is a dict containing the "patchID" key.'
            )

        # folders that contain patches
        sub_folder_1 = id_[-2:]
        sub_folder_2 = id_[-4:-2]
        list_tensor = {"order": [], "tensors": []}

        for channel in self.channels:
            if channel not in list_tensor["order"]:
                path = os.path.join(
                    self.root_path,
                    self.channel_folder[channel],
                    sub_folder_1,
                    sub_folder_2,
                    id_ + self.ext,
                )
                try:
                    img = np.asarray(Image.open(path))
                    if set(["red", "green", "blue"]).issubset(
                        self.channels
                    ) and channel in ["red", "green", "blue"]:
                        img = img.transpose((2, 0, 1))
                        list_tensor["order"].extend(["red", "green", "blue"])
                    else:
                        if channel in ["red", "green", "blue"]:
                            img = img[:, :, "rgb".find(channel[0])]
                        img = np.expand_dims(img, axis=0)
                        list_tensor["order"].append(channel)
                except Exception as e:
                    logging.critical(
                        "Could not open {} properly. Setting array to 0.".format(path)
                    )
                    img = np.zeros((1, self.patch_size, self.patch_size))
                    list_tensor["order"].append(channel)
                if self.normalize:
                    if os.path.isfile(self.dataset_stats):
                        df = pd.read_csv(self.dataset_stats, sep=";")
                        mean, std = df.loc[0, "mean"], df.loc[0, "std"]
                    else:
                        mean, std = jpeg_standardize(
                            self.root_path, [self.ext], output=self.dataset_stats
                        )
                    img = (img - mean) / std
                for depth in img:
                    list_tensor["tensors"].append(np.expand_dims(depth, axis=0))
        tensor = np.concatenate(list_tensor["tensors"])
        if self.patch_transform:
            for transform in self.patch_transform:
                tensor = transform(tensor)
        self.channels = list_tensor["order"]
        self.n_rows = img.shape[1]
        self.n_cols = img.shape[2]
        return tensor

    def __str__(self):
        result = "-" * 50 + "\n"
        result += "n_layers: " + str(self.nb_layers) + "\n"
        result += "n_rows: " + str(self.n_rows) + "\n"
        result += "n_cols: " + str(self.n_cols) + "\n"
        result += "-" * 50
        return result
