from functools import reduce

import luigi
import numpy as np
from contexttimer import Timer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MultilabelClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F

from geolifeclef.functions import get_projection_udf
from geolifeclef.utils import spark_resource

from ..utils import maybe_gcs_target
from .transformer import NaiveMultiClassToMultiLabel, ThresholdMultiClassToMultiLabel


class CleanMetadata(luigi.Task):
    """Generate a new metadata that we can use for training that only has the columns we want."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()

    def output(self):
        return maybe_gcs_target(f"{self.output_path}/_SUCCESS")

    def _load(self, spark):
        po_suffix = "PresenceOnlyOccurrences/GLC24-PO-metadata-train.csv"
        pa_train_suffix = "PresenceAbsenceSurveys/GLC24-PA-metadata-train.csv"
        pa_test_suffix = "PresenceAbsenceSurveys/GLC24-PA-metadata-test.csv"

        return [
            spark.read.csv(f"{self.input_path}/{suffix}", header=True, inferSchema=True)
            for suffix in [po_suffix, pa_train_suffix, pa_test_suffix]
        ]

    def _select(self, df, dataset):
        # projection to espg:32738 should be useful later down the line
        proj_udf = get_projection_udf()
        return df.withColumn("proj", proj_udf("lat", "lon")).select(
            F.lit(dataset).alias("dataset"),
            "surveyId",
            F.expr("proj.lat").alias("lat_proj"),
            F.expr("proj.lon").alias("lon_proj"),
            "lat",
            "lon",
            "year",
            "geoUncertaintyInM",
            (
                "speciesId"
                if "speciesId" in df.columns
                else F.lit(None).alias("speciesId")
            ),
        )

    def run(self):
        with spark_resource() as spark:
            # why use many variables when lexically-scoped do trick?
            (
                reduce(
                    lambda a, b: a.union(b),
                    [
                        self._select(df, dataset)
                        for df, dataset in zip(
                            self._load(spark),
                            ["po", "pa_train", "pa_test"],
                        )
                    ],
                )
                .orderBy("dataset", "surveyId")
                .repartition(8)
                .write.parquet(self.output_path, mode="overwrite")
            )


class BaseFitModel(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()

    features = luigi.ListParameter(default=["lat_proj", "lon_proj"])
    label = luigi.Parameter(default="target")
    multilabel_strategy = luigi.ChoiceParameter(
        choices=["naive", "threshold"], default="naive"
    )
    k = luigi.OptionalIntParameter(default=None)

    num_folds = luigi.IntParameter(default=3)
    seed = luigi.IntParameter(default=42)
    shuffle_partitions = luigi.IntParameter(default=200)

    def output(self):
        return maybe_gcs_target(f"{self.output_path}/_SUCCESS")

    def _load(self, spark):
        df = spark.read.parquet(self.input_path)
        if self.k is not None:
            df = self._subset_df(df)
        return (
            self._target_mapping(df, "speciesId", self.label)
            .where(F.col(self.label).isNotNull())
            .repartition(self.shuffle_partitions, "surveyId")
        )

    def _target_mapping(self, df, src, dst):
        """Create a mapping from the speciesId to a unique identifier"""
        mapping = (
            df.select(src)
            .distinct()
            .orderBy(src)
            .withColumn(dst, F.monotonically_increasing_id().astype("double"))
        )
        return df.join(mapping, src)

    def _subset_df(self, df):
        return df.join(
            (
                df.groupBy("speciesId")
                .count()
                .where(F.col("count") > 10)  # make sure there are enough examples
                .orderBy(F.rand(self.seed))
                .limit(self.k)
                .cache()
            ),
            "speciesId",
        )

    def _classifier(self, featuresCol, labelCol):
        raise NotImplementedError()

    def _param_grid(self, pipeline):
        return ParamGridBuilder().build()

    def _pipeline(self):
        multilabel = {
            "naive": NaiveMultiClassToMultiLabel(
                primaryKeyCol="surveyId",
                labelCol=self.label,
                inputCol="prediction",
                outputCol="prediction",
            ),
            "threshold": ThresholdMultiClassToMultiLabel(
                primaryKeyCol="surveyId",
                labelCol=self.label,
                inputCol="probability",
                outputCol="prediction",
            ),
        }
        return Pipeline(
            stages=[
                VectorAssembler(
                    inputCols=self.features,
                    outputCol="features",
                ),
                self._classifier("features", self.label),
                multilabel[self.multilabel_strategy],
            ]
        )

    def _evaluator(self):
        return MultilabelClassificationEvaluator(
            predictionCol="prediction",
            labelCol=self.label,
            metricName="microF1Measure",
        )

    def run(self):
        with spark_resource(
            **{
                # increase shuffle partitions to avoid OOM
                "spark.sql.shuffle.partitions": self.shuffle_partitions,
            },
        ) as spark:
            train = self._load(spark).cache()

            # write the model to disk
            pipeline = self._pipeline()
            cv = CrossValidator(
                estimator=pipeline,
                estimatorParamMaps=self._param_grid(pipeline),
                evaluator=self._evaluator(),
                numFolds=self.num_folds,
            )

            with Timer() as train_timer:
                model = cv.fit(train)
                model.write().overwrite().save(f"{self.output_path}/model")

            # write the results to disk
            perf = spark.createDataFrame(
                [
                    {
                        "train_time": train_timer.elapsed,
                        "avg_metrics": np.array(model.avgMetrics).tolist(),
                        "std_metrics": np.array(model.avgMetrics).tolist(),
                        "metric_name": model.getEvaluator().getMetricName(),
                    }
                ],
                schema="""
                    train_time double,
                    avg_metrics array<double>,
                    std_metrics array<double>,
                    metric_name string
                """,
            )
            perf.write.json(f"{self.output_path}/perf", mode="overwrite")
            perf.show()

        # write the output
        with self.output().open("w") as f:
            f.write("")
