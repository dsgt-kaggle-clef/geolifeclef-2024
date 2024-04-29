import json
import os
from argparse import ArgumentParser
from collections import Counter

import luigi
import numpy as np
from contexttimer import Timer
from pynndescent import NNDescent
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import MultilabelClassificationEvaluator
from pyspark.ml.feature import DCT, StandardScaler, VectorAssembler, VectorSlicer
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import SparseVector, VectorUDT
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.sql import Row
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


def _collect_sparse_labels(array):
    """Turn a list of numbers into a sparse vector."""
    max_species = 11255

    @F.udf(VectorUDT())
    def func(array):
        if not array:
            return SparseVector(max_species, [])
        return SparseVector(max_species, sorted(Counter(array).items()))

    return func(array)


class FitSVDMultiLabelModel(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    k_neighbors = luigi.IntParameter(default=15)
    sample = luigi.OptionalFloatParameter()

    def output(self):
        return luigi.LocalTarget(f"{self.output_path}/_SUCCESS")

    def run(self):
        with spark_resource(memory="24g") as spark:
            self._run(spark)

    def _pipeline(self):
        return Pipeline(
            stages=[
                VectorAssembler(
                    inputCols=["lat_proj", "lon_proj"], outputCol="features"
                ),
                StandardScaler(inputCol="features", outputCol="features_scaled"),
                *[
                    SparkXGBRegressor(
                        features_col="features_scaled",
                        label_col=f"labels_svd_{i:03d}",
                        prediction_col=f"prediction_{i:03d}",
                        num_workers=os.cpu_count(),
                        early_stopping_rounds=10,
                        validation_indicator_col="is_validation",
                        multi_strategy="multi_output_tree",
                    )
                    for i in range(2)
                ],
                VectorAssembler(
                    inputCols=[f"prediction_{i:03d}" for i in range(2)],
                    outputCol="prediction",
                ),
            ]
        )

    def _run(self, spark):
        metadata = spark.read.parquet(self.input_path)
        df = (
            metadata.groupBy("surveyId")
            .agg(
                F.mean("lat_proj").alias("lat_proj"),
                F.mean("lon_proj").alias("lon_proj"),
                _collect_sparse_labels(F.collect_list("speciesId")).alias("labels_sp"),
                F.sort_array(F.collect_set("speciesId")).alias("labels"),
                F.first("dataset").alias("dataset"),
            )
            .withColumn("sample_id", F.crc32(F.col("surveyId").cast("string")) % 100)
            .withColumn("is_validation", F.col("sample_id") < 10)
            .withColumn("is_train", F.col("sample_id") < 80)
            .withColumn("is_test", F.col("sample_id") >= 80)
            .orderBy("surveyId")
            .cache()
        )
        if self.sample is not None:
            df = df.where(F.col("dataset") != "po")
            df = df.sample(self.sample).orderBy("surveyId").cache()

        train = df.where(F.col("dataset") != "pa_test")
        scaled_labels = (
            StandardScaler(
                inputCol="labels_sp",
                outputCol="labels_centered",
                withStd=False,
                withMean=True,
            )
            .fit(train)
            .transform(train)
            .cache()
        )
        print(f"scaled_labels: {scaled_labels.count()}")
        X = RowMatrix(
            scaled_labels.rdd.map(lambda x: Vectors.fromML(x.labels_centered)).cache()
        )
        with Timer() as t_svd:
            svd = X.computeSVD(2, computeU=True)
            print(svd.s)
            print(svdgit.V.toArray().shape)
            print(svd.U.rows)

        print(f"svd computed in {t_svd.elapsed:.2f}s")

        labels_svd = (
            svd.U.rows.zipWithIndex()
            .map(lambda x: (x[1], x[0]))
            .toDF(["index", "labels_svd"])
            .join(
                df.select("surveyId")
                .rdd.zipWithIndex()
                .map(lambda x: (x[1], x[0].surveyId))
                .toDF(["index", "surveyId"]),
                on="index",
            )
        )
        labels_svd = (
            ExtractLabelsFromVector(
                inputCol="labels_svd",
                outputColPrefix="labels_svd",
                indexDim=2,
            )
            .transform(labels_svd)
            .cache()
        )

        full_train = train.join(labels_svd, on="surveyId")
        pipeline = self._pipeline()
        model = pipeline.fit(full_train.where("is_train"))
        predictions = (
            model.transform(df)
            .drop("labels_sp", "labels_sp_scaled")
            .orderBy("surveyId")
            .cache()
        )
        predictions.printSchema()

        U = np.stack(labels_svd.toPandas().labels_svd)
        index = NNDescent(U, metric="cosine")

        pdf = predictions.select(
            vector_to_array("prediction").alias("prediction")
        ).toPandas()
        # np.stack(pdf.prediction.values)
        idx, _ = index.query(np.stack(pdf.prediction.values), k=self.k_neighbors)

        pred_labels = (
            spark.createDataFrame(
                [
                    Row(
                        index=i,
                        pred_index=int(row[0]),
                    )
                    for i, row in enumerate(idx)
                ]
            )
            .join(
                predictions.select("index", "surveyId", "labels", "dataset"),
                on="index",
            )
            .join(
                predictions.select(
                    F.col("index").alias("pred_index"),
                    F.col("labels").alias("pred_labels"),
                ),
                on="pred_index",
            )
        )

        with Timer() as t:
            res = MultilabelClassificationEvaluator(
                predictionCol="pred_labels",
                labelCol="labels",
                metricName="microF1Measure",
            ).evaluate(pred_labels.where(F.col("dataset") == "pa_train"))

        # write out stats
        with open(f"{self.output_path}/perf.json", "w") as f:
            json.dump({"f1": res, "time": t.elapsed, "time_svd": t_svd}, f)

        # now write out the predictions to parquet and the final competition format
        # only write out the test data in the correct format
        (
            pred_labels.where(F.col("dataset") == "pa_test")
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

        # v2 - SVD
        yield [
            FitSVDMultiLabelModel(
                sample=0.01,
                input_path=f"{self.local_root}/processed/metadata_clean/v1",
                output_path=f"{self.local_root}/models/xgboost_multilabel_svd/v1_test",
            )
        ]
        # yield [
        #     FitSVDMultiLabelModel(
        #         input_path=f"{self.local_root}/processed/metadata_clean/v1",
        #         output_path=f"{self.local_root}/models/xgboost_multilabel_svd/v1",
        #     ),
        # ]


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
