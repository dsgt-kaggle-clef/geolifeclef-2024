import warnings
from pathlib import Path

import pytorch_lightning as pl
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.sql import functions as F
from torchvision.transforms import v2

from geolifeclef.torch.transforms import (
    DCTRandomHorizontalFlip,
    DCTRandomRotation,
    DCTRandomVerticalFlip,
    IDCTransform,
    ToReshapedLayers,
    collect_sparse_labels,
)


class RasterDataModel(pl.LightningDataModule):
    def __init__(
        self,
        spark,
        input_path,
        feature_paths,
        feature_col=["red", "green", "blue", "nir"],
        use_idct=False,
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
        self.use_idct = use_idct
        self.batch_size = batch_size
        self.num_partitions = num_partitions
        self.workers_count = workers_count

    def _normalize_column_name(self, col):
        return col.lower().replace("-", "_")

    def _load(self):
        metadata = self.spark.read.parquet(self.input_path).where(
            F.col("dataset") != "po"
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

        return df.select(
            "surveyId", "dataset", "label", "sample_id", *self.normalized_feature_col
        )

    def _load_labels(self, df):
        surveys = (
            df.select("surveyId", "dataset")
            .distinct()
            .withColumn("sample_id", F.crc32(F.col("surveyId").cast("string")) % 100)
        )
        labels = (
            df.withColumn("surveyId", F.col("surveyId").cast("integer"))
            .groupBy("surveyId")
            .agg(collect_sparse_labels(F.collect_list("speciesId")).alias("label"))
        )
        return surveys.join(labels, on="surveyId", how="left")

    def _prepare_dataframe(self, df, include_labels=True):
        """Prepare the DataFrame for training by ensuring correct types and repartitioning"""
        # asm = VectorAssembler(
        #     inputCols=self.normalized_feature_col,
        #     outputCol="features",
        # )
        # res = asm.transform(df)
        return df.select(
            "surveyId",
            *self.normalized_feature_col,
            # vector_to_array("features", "float32").alias("features"),
            *(
                [
                    vector_to_array("label", "float32")
                    .cast("array<boolean>")
                    .alias("label")
                ]
                if include_labels
                else []
            ),
        ).repartition(self.num_partitions)

    def get_shape(self):
        row = self._prepare_dataframe(self.valid_data).first()
        num_layers = len(self.feature_col)
        return (
            num_layers,
            128 if self.use_idct else 8,
            int(len(row.label)),
        )

    def setup(self, stage=None):
        df = self._load().cache()
        self.df = df
        self.valid_data = df.where("dataset = 'pa_train'").where("sample_id >= 90")
        self.train_data = df.where("dataset = 'pa_train'").where("sample_id < 90")
        self.test_data = df.where("dataset = 'pa_test'")

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.converter_train = make_spark_converter(
                self._prepare_dataframe(self.train_data)
            )
            self.converter_valid = make_spark_converter(
                self._prepare_dataframe(self.valid_data)
            )
            self.converter_predict = make_spark_converter(
                self._prepare_dataframe(self.test_data, include_labels=False)
            )

    def get_transform(self, augment=True):
        num_layers, _, _ = self.get_shape()
        return v2.Compose(
            [
                ToReshapedLayers(num_layers, 8, features=self.normalized_feature_col),
                *([IDCTransform()] if self.use_idct else []),
                *(
                    (
                        [
                            v2.RandomHorizontalFlip(),
                            v2.RandomVerticalFlip(),
                            v2.RandomResizedCrop(128, scale=(0.8, 1.0)),
                        ]
                        if self.use_idct
                        else [
                            DCTRandomRotation(),
                            DCTRandomHorizontalFlip(),
                            DCTRandomVerticalFlip(),
                        ]
                    )
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

    def predict_dataloader(self):
        for batch in self._dataloader(self.converter_predict, augment=False):
            yield batch
