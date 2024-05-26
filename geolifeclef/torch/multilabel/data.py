import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from torchvision.transforms import v2

from geolifeclef.torch.transforms import collect_sparse_labels


# create a transform to convert a list of numbers into a sparse tensor
class ToSparseTensor(v2.Transform):
    def forward(self, batch):
        if "label" in batch:
            batch["label"] = batch["label"].to_sparse()
        return batch


class AddNoise(v2.Transform):
    def __init__(self, noise=5_000):
        super().__init__()
        # add 5km of noise to the featurse
        self.noise = noise

    def forward(self, batch):
        batch["features"] = (
            batch["features"] + torch.randn_like(batch["features"]) * self.noise
        )
        return batch


class GeoSpatialDataModel(pl.LightningDataModule):
    def __init__(
        self,
        spark,
        input_path,
        feature_col=["lat_proj", "lon_proj"],
        pa_only=True,
        batch_size=32,
        num_partitions=32,
        workers_count=os.cpu_count() // 2,
        cache_dir="file:///mnt/data/tmp",
    ):
        super().__init__()
        spark.conf.set(
            SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF,
            Path(cache_dir).as_posix(),
        )
        self.spark = spark
        self.input_path = input_path
        self.feature_col = feature_col
        self.batch_size = batch_size
        self.num_partitions = num_partitions
        self.workers_count = workers_count

        df = spark.read.parquet(self.input_path)
        if pa_only:
            df = df.where(F.col("dataset").like("pa%"))
        self.df = df.cache()

    def _load(self):
        # create a smaller subset of rows to keep
        # only keep rows where there are more than 5 species
        # this should bring us down from 3.9m rows to 150k rows, but doesn't cover
        # all of the species
        subset = (
            self.df.where("dataset='po'")
            .groupBy("surveyId")
            .agg(F.countDistinct("speciesId").alias("n_species"))
            .where("n_species > 5")
            .select("surveyId")
        ).union(
            self.df.where(F.col("dataset").like("pa%")).select("surveyId").distinct()
        )

        return (
            self.df.join(subset, on="surveyId", how="right")
            .groupBy("surveyId")
            .agg(
                F.mean("lat_proj").alias("lat_proj"),
                F.mean("lon_proj").alias("lon_proj"),
                collect_sparse_labels(
                    F.collect_list("speciesId").cast("array<short>")
                ).alias("labels_sp"),
                F.first("dataset").alias("dataset"),
            )
            .withColumn("sample_id", F.crc32(F.col("surveyId").cast("string")) % 100)
        )

    def _prepare_dataframe(self, df, include_label=True):
        """Prepare the DataFrame for training by ensuring correct types and repartitioning"""
        asm = VectorAssembler(inputCols=self.feature_col, outputCol="features")
        res = asm.transform(df)
        return res.select(
            "surveyId",
            vector_to_array("features").alias("features"),
            *(
                [vector_to_array("labels_sp").cast("array<boolean>").alias("label")]
                if include_label
                else []
            ),
        )

    def compute_weights(self):
        df = self.df
        num_classes = int(
            df.select(F.max("speciesId").alias("num_classes")).first().num_classes + 1
        )
        counts = (
            df.where("speciesId is not null")
            .groupBy("speciesId")
            .agg(F.count("surveyId").alias("n"))
            .orderBy("speciesId")
            .collect()
        )
        vec = torch.ones(num_classes)
        for count in counts:
            vec[int(count.speciesId)] += count.n
        return vec / vec.sum()

    def setup(self, stage=None):
        df = self._load().cache()
        self.valid_data = (
            df.where("sample_id >= 90 and dataset='pa_train'")
            .repartition(self.num_partitions)
            .cache()
        )
        self.train_data = (
            df.join(self.valid_data.select("surveyId"), on="surveyId", how="left_anti")
            .repartition(self.num_partitions)
            .cache()
        )
        self.predict_data = (
            df.where("dataset='pa_test'").repartition(self.num_partitions).cache()
        )

        self.converter_train = make_spark_converter(
            self._prepare_dataframe(self.train_data)
        )
        self.converter_valid = make_spark_converter(
            self._prepare_dataframe(self.valid_data)
        )
        self.converter_predict = make_spark_converter(
            self._prepare_dataframe(self.predict_data, include_label=False)
        )

    def get_shape(self):
        row = self._prepare_dataframe(self.valid_data).first()
        return int(len(row.features)), int(len(row.label))

    def _dataloader(self, converter, augment=True):
        transform = v2.Compose(
            [
                ToSparseTensor(),
                *([AddNoise()] if augment else []),
            ]
        )
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

    def predict_dataloader(self):
        for batch in self._dataloader(self.converter_predict, augment=False):
            yield batch
