import os
from collections import Counter
from pathlib import Path

import pytorch_lightning as pl
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import SparseVector, VectorUDT
from pyspark.sql import functions as F


def _collect_sparse_labels(array):
    """Turn a list of numbers into a sparse vector."""
    max_species = 11255

    @F.udf(VectorUDT())
    def func(array):
        if not array:
            return SparseVector(max_species, [])
        return SparseVector(max_species, sorted(Counter(array).items()))

    return func(array)


class GeoSpatialDataModel(pl.LightningDataModule):
    def __init__(
        self,
        spark,
        input_path,
        feature_col,
        limit_species=None,
        species_image_count=100,
        batch_size=32,
        num_partitions=32,
        workers_count=os.cpu_count(),
    ):
        super().__init__()
        cache_dir = "file:///mnt/data/tmp"
        spark.conf.set(
            SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF,
            Path(cache_dir).as_posix(),
        )
        self.spark = spark
        self.input_path = input_path
        self.feature_col = feature_col
        self.limit_species = limit_species
        self.species_image_count = species_image_count
        self.batch_size = batch_size
        self.num_partitions = num_partitions
        self.workers_count = workers_count

    def _load(self, spark):
        # now we can convert this into a multi-label problem by surveyId
        return self._prepare_dataset(
            spark.read.parquet(self.input_path)
            .groupBy("surveyId")
            .agg(
                F.mean("lat_proj").alias("lat_proj"),
                F.mean("lon_proj").alias("lon_proj"),
                _collect_sparse_labels(F.collect_list("speciesId")).alias("labels_sp"),
                F.sort_array(F.collect_set("speciesId")).alias("labels"),
                F.first("dataset").alias("dataset"),
            )
            .withColumn("sample_id", F.crc32(F.col("surveyId").cast("string")) % 100)
            .cache(),
            self.num_partitions,
        )

    def _prepare_dataframe(self, df, partitions=32):
        """Prepare the DataFrame for training by ensuring correct types and repartitioning"""
        pipeline = Pipeline(
            stages=[
                VectorAssembler(
                    inputCols=["lat_proj", "lon_proj"],
                    outputCol="features",
                )
            ]
        )
        res = pipeline.fit(df).transform(df)
        return res.select(
            vector_to_array("features").alias("features"),
            vector_to_array("labels_sp").alias("labels"),
        ).repartition(partitions)

    def setup(self, stage=None):
        df = self._load(self.spark).cache()
        self.train_data = df.where("sample_id < 80")
        self.valid_data = df.where("sample_id >= 80")
        self.converter_train = make_spark_converter(self.train_data)
        self.converter_valid = make_spark_converter(self.valid_data)

    def _dataloader(self, converter):
        with converter.make_torch_dataloader(
            batch_size=self.batch_size,
            num_epochs=1,
            workers_count=self.workers_count,
        ) as dataloader:
            for batch in dataloader:
                yield batch

    def train_dataloader(self):
        for batch in self._dataloader(self.converter_train):
            yield batch

    def val_dataloader(self):
        for batch in self._dataloader(self.converter_valid):
            yield batch
