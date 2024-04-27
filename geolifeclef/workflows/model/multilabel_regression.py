import json
import os
from argparse import ArgumentParser
from collections import Counter

import luigi
import numpy as np
from contexttimer import Timer
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import MultilabelClassificationEvaluator
from pyspark.ml.feature import DCT, StandardScaler, VectorAssembler, VectorSlicer
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import SparseVector, VectorUDT
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder
from pyspark.sql import functions as F
from xgboost.spark import SparkXGBRegressor

from geolifeclef.utils import spark_resource

from ..utils import RsyncGCSFiles, maybe_gcs_target
from .base_tasks import BaseFitModel
from .transformer import (
    ExtractLabelsFromVector,
    ReconstructDCTCoefficients,
    ThresholdMultiClassToMultiLabel,
)


class FitMultiLabelModel(BaseFitModel):
    max_species = luigi.IntParameter(default=11255)
    labels_dim = luigi.IntParameter(default=8)
    label = luigi.Parameter(default="speciesId")

    device = luigi.Parameter(default="cpu")
    subsample = luigi.FloatParameter(default=1.0)
    num_workers = luigi.IntParameter(default=os.cpu_count())
    sampling_method = luigi.Parameter(default="uniform")

    def _target_mapping(self, df, src, dst):
        return df

    def _collect_sparse_labels(self, array):
        """Turn a list of numbers into a sparse vector."""

        @F.udf(VectorUDT())
        def func(array):
            if not array:
                return SparseVector(self.max_species, [])
            return SparseVector(self.max_species, sorted(Counter(array).items()))

        return func(array)

    def _subset_df(self, df):
        return df.limit(self.k)

    def _load(self, spark):
        df = super()._load(spark)
        # now we can convert this into a multi-label problem by surveyId
        return (
            df.groupBy("surveyId")
            .agg(
                F.mean("lat_proj").alias("lat_proj"),
                F.mean("lon_proj").alias("lon_proj"),
                self._collect_sparse_labels(F.collect_list("speciesId")).alias(
                    "labels_sp"
                ),
                F.sort_array(F.collect_set("speciesId")).alias("labels"),
            )
            .withColumn("is_validation", F.rand(seed=42) < 0.1)
        )

    def _pipeline(self):
        return Pipeline(
            stages=[
                VectorAssembler(
                    inputCols=self.features,
                    outputCol="features",
                ),
                # create a new label for each of the coefficients
                StandardScaler(inputCol="features", outputCol="scaled_features"),
                DCT(inputCol="labels_sp", outputCol="labels_dct", inverse=False),
                # slice the first k coefficients
                VectorSlicer(
                    inputCol="labels_dct",
                    outputCol="labels_dct_sliced",
                    indices=list(range(self.labels_dim)),
                ),
                ExtractLabelsFromVector(
                    inputCol="labels_dct_sliced",
                    outputColPrefix="label",
                    indexDim=self.labels_dim,
                ),
                *[
                    SparkXGBRegressor(
                        features_col="scaled_features",
                        label_col=f"label_{i:03d}",
                        prediction_col=f"prediction_{i:03d}",
                        device=self.device,
                        num_workers=self.num_workers,
                        subsample=self.subsample,
                        sampling_method=self.sampling_method,
                        early_stopping_rounds=10,
                        validation_indicator_col="is_validation",
                    )
                    for i in range(self.labels_dim)
                ],
                # now project the labels back by concatenating the predictions
                # together and taking the inverse
                ReconstructDCTCoefficients(
                    inputCols=[f"prediction_{i:03d}" for i in range(self.labels_dim)],
                    outputCol="dct_prediction",
                    indexDim=self.labels_dim,
                ),
                # it's not really a probabiity, but more of a quantization to the nearest
                # integer
                DCT(inputCol="dct_prediction", outputCol="probability", inverse=True),
                {
                    "threshold": ThresholdMultiClassToMultiLabel(
                        primaryKeyCol="surveyId",
                        labelCol="labels",
                        inputCol="probability",
                        outputCol="prediction",
                        isPreCollated=True,
                    ),
                }[self.multilabel_strategy],
            ]
        )

    def _param_grid(self, pipeline):
        # from the pipeline, let's extract the logistic regression model
        return ParamGridBuilder().build()

    def _evaluator(self):
        return MultilabelClassificationEvaluator(
            predictionCol="prediction",
            labelCol="labels",
            metricName="microF1Measure",
        )


class MakePrediction(luigi.Task):
    model_path = luigi.Parameter()
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    max_species = luigi.IntParameter(default=11255)
    shuffle_partitions = luigi.IntParameter(default=200)

    def output(self):
        return luigi.LocalTarget(f"{self.output_path}/predictions.csv")

    def _collect_sparse_labels(self, array):
        """Turn a list of numbers into a sparse vector."""

        @F.udf(VectorUDT())
        def func(array):
            if not array:
                return SparseVector(self.max_species, [])
            return SparseVector(self.max_species, sorted(Counter(array).items()))

        return func(array)

    def _load(self, spark):
        df = spark.read.parquet(self.input_path)
        # now we can convert this into a multi-label problem by surveyId
        return (
            df.groupBy("surveyId")
            .agg(
                F.mean("lat_proj").alias("lat_proj"),
                F.mean("lon_proj").alias("lon_proj"),
                self._collect_sparse_labels(F.collect_list("speciesId")).alias(
                    "labels_sp"
                ),
                F.sort_array(F.collect_set("speciesId")).alias("labels"),
                F.first("dataset").alias("dataset"),
            )
            .withColumn("is_validation", F.rand(seed=42) < 0.1)
        )

    def _predictions_to_labels(self, array):
        pass

    def run(self):
        with spark_resource(
            **{"spark.sql.shuffle.partitions": self.shuffle_partitions},
        ) as spark:
            df = self._load(spark).where(F.col("dataset") != "po")
            model = CrossValidatorModel.load(self.model_path)
            result = (
                model.transform(df)
                # drop unnecessarily expensive columns
                .drop(
                    "features",
                    "scaled_features",
                    "labels_sp",
                    "labels_dct",
                    "probability",
                )
            ).cache()

            # now to convert the prediction column...
            result.printSchema()
            result.show(vertical=True, n=1)

            evaluator = MultilabelClassificationEvaluator(
                labelCol="labels",
                predictionCol="prediction",
                metricName="microF1Measure",
            )
            with Timer() as train_timer:
                f1 = evaluator.evaluate(result.where(F.col("dataset") == "pa_train"))

            with maybe_gcs_target(f"{self.output_path}/perf.json").open("w") as f:
                json.dump({"f1": f1, "time": train_timer.elapsed}, f)

            # now write out the predictions to parquet and the final competition format
            result.write.parquet(f"{self.output_path}/predictions", mode="overwrite")

            # only write out the test data in the correct format
            (
                result.where(F.col("dataset") == "pa_test")
                .select(
                    F.col("surveyId").cast("integer"),
                    F.array_join(F.col("prediction").cast("array<integer>"), " ").alias(
                        "predictions"
                    ),
                )
                .toPandas()
                .to_csv(f"{self.output_path}/predictions.csv", index=False)
            )

            # write success
            with self.output().open("w") as f:
                f.write("")


class Workflow(luigi.Task):
    remote_root = luigi.Parameter(default="gs://dsgt-clef-geolifeclef-2024/data")
    local_root = luigi.Parameter(default="/mnt/data/geolifeclef-2024/data")
    device = luigi.ChoiceParameter(choices=["cpu", "cuda"], default="cuda")

    def run(self):
        yield RsyncGCSFiles(
            src_path=f"{self.remote_root}/processed/metadata_clean/v1",
            dst_path=f"{self.local_root}/processed/metadata_clean/v1",
        )

        # v1 - initial implementation
        params = (
            {
                "device": self.device,
                "num_workers": 1,
                "subsample": 0.1,
                "sampling_method": "gradient_based",
            }
            if self.device == "cuda"
            else {
                "device": self.device,
                "num_workers": os.cpu_count(),
                "subsample": 0.5,
                "sampling_method": "uniform",
            }
        )
        yield [
            [
                FitMultiLabelModel(
                    k=1000,
                    multilabel_strategy=strategy,
                    num_folds=3,
                    labels_dim=labels_dim,
                    input_path=f"{self.local_root}/processed/metadata_clean/v1",
                    output_path=f"{self.local_root}/models/xgboost_multilabel_dct_{self.device}_{labels_dim}/v1_test",
                    **params,
                )
            ]
            for strategy in ["threshold"]
            for labels_dim in [1, 2]
        ]
        yield [
            [
                FitMultiLabelModel(
                    multilabel_strategy=strategy,
                    num_folds=3,
                    labels_dim=labels_dim,
                    input_path=f"{self.local_root}/processed/metadata_clean/v1",
                    output_path=f"{self.local_root}/models/xgboost_multilabel_dct_{self.device}_{labels_dim}/v1",
                    **params,
                ),
                MakePrediction(
                    model_path=f"{self.local_root}/models/xgboost_multilabel_dct_{self.device}_{labels_dim}/v1/model",
                    input_path=f"{self.local_root}/processed/metadata_clean/v1",
                    output_path=f"{self.local_root}/models/xgboost_multilabel_dct_{self.device}_{labels_dim}/v1/predictions",
                ),
            ]
            for strategy in ["threshold"]
            for labels_dim in [1, 2]
        ]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scheduler-host", default="localhost")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    luigi.build(
        [Workflow(device=args.device)],
        scheduler_host=args.scheduler_host,
        workers=1,
    )
