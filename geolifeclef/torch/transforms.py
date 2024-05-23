from collections import Counter

import torch
import torch_dct as dct
from pyspark.ml.linalg import SparseVector, VectorUDT
from pyspark.sql import functions as F
from torchvision.transforms import v2


def collect_sparse_labels(array):
    """Turn a list of numbers into a sparse vector."""
    max_species = 11255

    @F.udf(VectorUDT())
    def func(array):
        if not array:
            return SparseVector(max_species, [])
        return SparseVector(max_species, sorted(Counter(array).items()))

    return func(array)


class ToReshapedLayers(v2.Transform):
    def __init__(self, num_layers, num_features, features=None):
        self.features = features
        self.num_layers = num_layers
        self.num_features = num_features
        super().__init__()

    def forward(self, batch):
        if self.features is not None:
            features = torch.stack([batch[col] for col in self.features])
        else:
            features = batch["features"]

        batch["features"] = features.view(
            -1,
            self.num_layers,
            self.num_features,
            self.num_features,
        )
        if "label" in batch:
            batch["label"] = batch["label"].to_sparse()
        return batch


class IDCTransform(v2.Transform):
    def forward(self, batch):
        features, label = batch["features"], batch["label"]
        # put the features on a 128x128 grid
        zero_pad = torch.zeros(
            features.shape[0], features.shape[1], 128, 128, device=features.device
        )
        zero_pad[:, :, : features.shape[2], : features.shape[3]] = features

        return {
            "features": dct.idct_2d(zero_pad),
            "label": label,
        }


class DCTRandomRotation(v2.Transform):
    def __init__(self, p=0.5):
        self.p = p
        super().__init__()

    def forward(self, batch):
        # just transpose the features
        if torch.rand(1) < self.p:
            batch["features"] = batch["features"].transpose(-1, -2)
        return batch


class DCTRandomHorizontalFlip(v2.Transform):
    def __init__(self, p=0.5):
        self.p = p
        self.odd_factor = -torch.ones((8, 8))
        for i in range(0, 8, 2):
            self.odd_factor[i, :] = 1
        super().__init__()

    def forward(self, batch):
        # just flip the features
        if torch.rand(1) < self.p:
            batch["features"] = batch["features"] * self.odd_factor
        return batch


class DCTRandomVerticalFlip(DCTRandomHorizontalFlip):
    def forward(self, batch):
        # just flip the features
        if torch.rand(1) < self.p:
            batch["features"] = batch["features"] * self.odd_factor.T
        return batch
