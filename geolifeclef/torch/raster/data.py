import os
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch_dct as dct
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.ml.linalg import SparseVector, VectorUDT
from pyspark.sql import functions as F
from torchvision.transforms import v2


def _collect_sparse_labels(array):
    """Turn a list of numbers into a sparse vector."""
    max_species = 11255

    @F.udf(VectorUDT())
    def func(array):
        if not array:
            return SparseVector(max_species, [])
        return SparseVector(max_species, sorted(Counter(array).items()))

    return func(array)


class ToReshapedLayers(v2.Transform):
    def __init__(self, features, num_layers, num_features):
        self.features = features
        self.num_layers = num_layers
        self.num_features = num_features
        super().__init__()

    def forward(self, batch):
        # stack the features together
        return {
            "features": (
                torch.stack([batch[col] for col in self.features]).view(
                    -1,
                    self.num_layers,
                    self.num_features,
                    self.num_features,
                )
            ),
            "label": batch["label"].to_sparse(),
        }


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


class RasterDataModel(pl.LightningDataModule):
    def __init__(
        self,
        spark,
        input_path,
        feature_paths,
        feature_col=["red", "green", "blue", "nir"],
        batch_size=32,
        num_partitions=32,
        workers_count=8,
        cache_dir="file:///mnt/data/tmp",
    ):
        super().__init__()
        spark.conf.set(
            SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF,
            Path(cache_dir).as_posix(),
        )
        self.spark = spark
        self.input_path = input_path
        self.feature_paths = feature_paths
        self.feature_col = feature_col
        self.normalized_feature_col = [
            self._normalize_column_name(x) for x in feature_col
        ]
        self.batch_size = batch_size
        self.num_partitions = num_partitions
        self.workers_count = workers_count

    def _normalize_column_name(self, col):
        return col.lower().replace("-", "_")

    def _load(self):
        metadata = self.spark.read.parquet(self.input_path).where(
            "dataset = 'pa_train'"
        )
        df = self._load_labels(metadata)

        for feature_path in self.feature_paths:
            feature_df = self.spark.read.parquet(feature_path)
            df = df.join(
                feature_df.select(
                    F.col("surveyId").cast("integer").alias("surveyId"),
                    *[
                        F.col(x).alias(self._normalize_column_name(x))
                        for x in feature_df.columns
                        if x in self.feature_col
                    ],
                ).distinct(),
                on="surveyId",
                how="inner",
            )

        return df.select("surveyId", "label", "sample_id", *self.normalized_feature_col)

    def _load_labels(self, df):
        return (
            df.withColumn("surveyId", F.col("surveyId").cast("integer"))
            .groupBy("surveyId")
            .agg(
                vector_to_array(
                    _collect_sparse_labels(F.collect_list("speciesId")),
                    "float32",
                )
                .cast("array<boolean>")
                .alias("label"),
            )
            .withColumn("sample_id", F.crc32(F.col("surveyId").cast("string")) % 100)
        )

    def _prepare_dataframe(self, df):
        return df.select(
            *self.normalized_feature_col,
            "label",
        ).repartition(self.num_partitions)

    def get_shape(self):
        row = self._prepare_dataframe(self.valid_data).first()
        num_layers = len(self.feature_col)
        return (
            num_layers,
            # int(np.sqrt(int(len(row.features)) // num_layers)),
            # 128,
            8,
            int(len(row.label)),
        )

    def setup(self, stage=None):
        df = self._load()
        self.df = df
        self.valid_data = df.where("sample_id >= 90")
        self.train_data = df.where("sample_id < 90")

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.converter_train = make_spark_converter(
                self._prepare_dataframe(self.train_data)
            )
            self.converter_valid = make_spark_converter(
                self._prepare_dataframe(self.valid_data)
            )

    def get_transform(self, augment=True):
        num_layers, _, _ = self.get_shape()
        return v2.Compose(
            [
                ToReshapedLayers(self.normalized_feature_col, num_layers, 8),
                # IDCTransform(),
                # *(
                #     [
                #         v2.RandomHorizontalFlip(),
                #         v2.RandomVerticalFlip(),
                #         v2.RandomResizedCrop(128, scale=(0.8, 1.0)),
                #     ]
                #     if augment
                #     else []
                # ),
                *(
                    [
                        DCTRandomRotation(),
                        DCTRandomHorizontalFlip(),
                        DCTRandomVerticalFlip(),
                    ]
                    if augment
                    else []
                ),
            ]
        )

    def _dataloader(self, converter, augment=True):
        transform = self.get_transform(augment)
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            with converter.make_torch_dataloader(
                batch_size=self.batch_size,
                num_epochs=1,
                workers_count=self.workers_count,
            ) as dataloader:
                for batch in dataloader:
                    yield transform(batch)

    def train_dataloader(self):
        for batch in self._dataloader(self.converter_train):
            yield batch

    def val_dataloader(self):
        for batch in self._dataloader(self.converter_valid, augment=False):
            yield batch
