import os
import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch_dct as dct
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from torchvision.transforms import v2

from geolifeclef.torch.transforms import collect_sparse_labels


class ToReshapedLayers(v2.Transform):
    def __init__(self, num_layers, num_features, feature_names):
        self.num_layers = num_layers
        self.num_features = num_features
        self.feature_names = feature_names
        super().__init__()

    def _reshape(self, batch, prefix):
        features = torch.stack([batch[f"{prefix}_{x}"] for x in self.feature_names])
        return features.view(
            -1,
            self.num_layers,
            self.num_features,
            self.num_features,
        )

    def forward(self, batch):
        return {
            "features": {k: self._reshape(batch, k) for k in ["anchor", "neighbor"]},
            "label": {
                k: label
                for k, label in [
                    ("anchor", batch["anchor_label"]),
                    ("neighbor", batch["neighbor_label"]),
                ]
            },
        }


class IDCTransform(v2.Transform):
    def forward(self, batch):
        # put the features on a 128x128 grid
        features = {}
        for k in ["anchor", "neighbor"]:
            X = batch["features"][k]
            zero_pad = torch.zeros(X.shape[0], X.shape[1], 128, 128, device=X.device)
            zero_pad[:, :, : X.shape[2], : X.shape[3]] = X
            features[k] = dct.idct_2d(zero_pad)

        return {
            "features": features,
            "label": batch["label"],
        }


class AugmentPairs(v2.Transform):
    def __init__(self):
        super().__init__()
        self.transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.RandomResizedCrop(128, scale=(0.8, 1.0)),
            ]
        )

    def forward(self, batch):
        # apply the transform to both the anchor and the neighbor
        for k in batch["features"].keys():
            batch["features"][k] = self.transform(batch["features"][k])
        return batch


class DCTRandomRotation(v2.Transform):
    def __init__(self, p=0.5):
        self.p = p
        super().__init__()

    def forward(self, batch):
        # just transpose the features
        if torch.rand(1) < self.p:
            batch = batch.transpose(-1, -2)
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
            batch = batch * self.odd_factor
        return batch


class DCTRandomVerticalFlip(DCTRandomHorizontalFlip):
    def forward(self, batch):
        # just flip the features
        if torch.rand(1) < self.p:
            batch = batch * self.odd_factor.T
        return batch


class AugmentDCTPairs(v2.Transform):
    def __init__(self):
        super().__init__()
        self.transform = v2.Compose(
            [
                DCTRandomHorizontalFlip(),
                DCTRandomVerticalFlip(),
                DCTRandomRotation(),
            ]
        )

    def forward(self, batch):
        # apply the transform to both the anchor and the neighbor
        for k in batch["features"].keys():
            batch["features"][k] = self.transform(batch["features"][k])
        return batch


class MiniBatchTriplet(v2.Transform):
    """Now that the we've applied randomization to our pairs, we simply a triple that's random from the batch."""

    def forward(self, batch):
        # we want to shuffle the batch
        idx = torch.randperm(batch["features"]["anchor"].shape[0])
        return {
            "features": {
                "anchor": batch["features"]["anchor"],
                "neighbor": batch["features"]["neighbor"],
                "distant": batch["features"]["anchor"][idx],
            },
            "label": {
                "anchor": batch["label"]["anchor"].to_sparse(),
                "neighbor": batch["label"]["neighbor"].to_sparse(),
                "distant": batch["label"]["anchor"][idx].to_sparse(),
            },
        }


class Raster2VecDataModel(pl.LightningDataModule):
    def __init__(
        self,
        spark,
        input_path,
        feature_paths,
        feature_col=["red", "green", "blue", "nir"],
        sample=1.0,
        batch_size=32,
        num_partitions=32,
        workers_count=os.cpu_count(),
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
        self.sample = sample

    def _normalize_column_name(self, col):
        return col.lower().replace("-", "_")

    def _load(self):
        def get_nodes(edges):
            return (
                edges.selectExpr("src as surveyId")
                .union(edges.selectExpr("dst as surveyId"))
                .distinct()
            )

        edges = self.spark.read.parquet(self.input_path).cache()
        edges.printSchema()
        train_edges = self._sample_edges(
            edges, 1e6, seed=42, sample=self.sample
        ).cache()
        train_edges.printSchema()
        train_nodes = get_nodes(train_edges).cache()
        train_nodes.printSchema()
        valid_edges = self._sample_edges(
            edges, 5e4, seed=108, sample=self.sample
        ).cache()

        # now create the features and labels per survey
        nodes = (train_nodes.union(get_nodes(valid_edges)).distinct()).cache()

        # generate the features for each of the nodes in our pairs
        df = nodes
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
        # for col in self.feature_col:
        #     df = df.withColumn(col, array_to_vector(F.col(col).cast("array<float>")))

        edges_subset = edges.join(
            nodes.selectExpr("surveyId as srcSurveyId").distinct(),
            how="inner",
            on="srcSurveyId",
        )
        data = df.select("surveyId", *self.normalized_feature_col).join(
            # only keep data if it's in our neighborhood
            self._get_labels(edges_subset),
            on="surveyId",
            how="left",
        )
        data.printSchema()
        data = self._prepare_dataframe(data).persist()
        data.printSchema()
        return (train_edges, valid_edges), nodes, data

    def _sample_edges(self, edges, limit, seed=42, sample=0.01, filter=None):
        if filter is not None:
            edges = edges.join(
                filter.selectExpr("surveyId as srcSurveyId"), how="left_anti"
            ).join(filter.selectExpr("surveyId as dstSurveyId"), how="left_anti")
        return (
            edges.where("srcDataset = 'po'")
            .where("dstDataset = 'po'")
            .selectExpr("srcSurveyId as src", "dstSurveyId as dst")
            .where("src != dst")
            .distinct()
            .sample(sample, seed=seed)
            .limit(int(limit))
        )

    def _get_labels(self, edges):
        return (
            edges.groupBy(F.expr("srcSurveyId as surveyId"))
            .agg(
                collect_sparse_labels(
                    F.collect_list("dstSpeciesId").cast("array<short>")
                ).alias("labels_sp")
            )
            .withColumn("sample_id", F.crc32(F.col("surveyId").cast("string")) % 100)
        )

    def _prepare_dataframe(self, df):
        """Prepare the DataFrame for training by ensuring correct types and repartitioning"""
        return df.select(
            "surveyId",
            *self.normalized_feature_col,
            vector_to_array("labels_sp", "float32")
            .cast("array<boolean>")
            .alias("label"),
        ).repartition(self.num_partitions)

    def get_shape(self):
        row = self.train_data.first()
        num_layers = len(self.feature_col)
        return (
            num_layers,
            # int(np.sqrt(int(len(row.features)) // num_layers)),
            # 128,
            8,
            int(len(row.anchor_label)),
        )

    def join_edge_data(self, edges, data):
        print("joining edges to data", flush=True)
        return (
            edges.join(
                data.selectExpr(
                    "surveyId as src",
                    "label as src_label",
                    *[f"{x} as src_{x}" for x in self.normalized_feature_col],
                ).distinct(),
                on="src",
                how="inner",
            )
            .join(
                data.selectExpr(
                    "surveyId as dst",
                    "label as dst_label",
                    *[f"{x} as dst_{x}" for x in self.normalized_feature_col],
                ).distinct(),
                on="dst",
                how="inner",
            )
            .selectExpr(
                "src as anchor_id",
                *[f"src_{x} as anchor_{x}" for x in self.normalized_feature_col],
                "src_label as anchor_label",
                "dst as neighbor_id",
                *[f"dst_{x} as neighbor_{x}" for x in self.normalized_feature_col],
                "dst_label as neighbor_label",
            )
        )

    def setup(self, stage=None):
        (train_edges, valid_edges), _, data = self._load()
        print(
            "counts", train_edges.count(), valid_edges.count(), data.count(), flush=True
        )
        self.train_data = self.join_edge_data(train_edges, data)
        self.valid_data = self.join_edge_data(valid_edges, data)
        self.train_data.printSchema()

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.converter_train = make_spark_converter(self.train_data)
            self.converter_valid = make_spark_converter(self.valid_data)

    def get_transform(self, augment=True):
        num_layers, _, _ = self.get_shape()
        return v2.Compose(
            [
                ToReshapedLayers(
                    num_layers, 8, feature_names=self.normalized_feature_col
                ),
                # IDCTransform(),
                # *([AugmentPairs()] if augment else []),
                *([AugmentDCTPairs()] if augment else []),
                MiniBatchTriplet(),
            ]
        )

    def _dataloader(self, converter, augment=True):
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            transform = self.get_transform(augment)
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
