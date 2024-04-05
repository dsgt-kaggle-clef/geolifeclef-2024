from functools import reduce

import luigi
import numpy as np
from contexttimer import Timer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MultilabelClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder
from pyspark.sql import functions as F

from geolifeclef.functions import get_projection_udf
from geolifeclef.utils import spark_resource

from .transformer import NaiveMultiClassToMultiLabel, ThresholdMultiClassToMultiLabel
from .utils import maybe_gcs_target


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
    label = luigi.Parameter(default="speciesId")
    multilabel_strategy = luigi.ChoiceParameter(
        choices=["naive", "threshold"], default="naive"
    )
    k = luigi.IntParameter(default=None)

    num_folds = luigi.IntParameter(default=3)
    seed = luigi.IntParameter(default=42)
    shuffle_partitions = luigi.IntParameter(default=500)

    def output(self):
        return maybe_gcs_target(f"{self.output_path}/_SUCCESS")

    def _load(self, spark):
        df = spark.read.parquet(self.input_path)
        if self.k is not None:
            df = self._subset_df(df)
        return df.where(F.col(self.label).isNotNull()).repartition(
            self.shuffle_partitions, "surveyId"
        )

    def _subset_df(self, df):
        return df.join(
            (
                df.groupBy("speciesId")
                .count()
                .where(F.col("count") > 10)  # make sure there are enough examples
                .orderBy(F.rand(self.seed))
                .limit(self.k)
                .withColumn(
                    "speciesSubsetId", F.monotonically_increasing_id().cast("double")
                )
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

    def _calculate_multilabel_stats(self, train):
        """Calculate statistics about the number of rows with multiple labels"""

        train.groupBy("surveyId").count().describe().write.csv(
            f"{self.output_path}/multilabel_stats/dataset=train", mode="overwrite"
        )

    def run(self):
        with spark_resource(
            **{
                # increase shuffle partitions to avoid OOM
                "spark.sql.shuffle.partitions": self.shuffle_partitions,
            },
        ) as spark:
            train = self._load(spark).cache()
            self._calculate_multilabel_stats(train)

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


class FitLogisticModel(BaseFitModel):
    max_iter = luigi.ListParameter(default=[100])
    reg_param = luigi.ListParameter(default=[0.0])
    elastic_net_param = luigi.ListParameter(default=[0.0])

    def _classifier(self, featuresCol, labelCol):
        return Pipeline(
            stages=[
                StandardScaler(inputCol=featuresCol, outputCol="scaled_features"),
                LogisticRegression(featuresCol="scaled_features", labelCol=labelCol),
            ]
        )

    def _param_grid(self, pipeline):
        # from the pipeline, let's extract the logistic regression model
        lr = pipeline.getStages()[-2].getStages()[-1]
        return (
            ParamGridBuilder()
            .addGrid(lr.maxIter, self.max_iter)
            .addGrid(lr.regParam, self.reg_param)
            .addGrid(lr.elasticNetParam, self.elastic_net_param)
            .build()
        )


class LogisticWorkflow(luigi.Task):
    def run(self):
        data_root = "gs://dsgt-clef-geolifeclef-2024/data"
        yield CleanMetadata(
            input_path=f"{data_root}/downloaded/2024",
            output_path=f"{data_root}/processed/metadata_clean/v1",
        )

        # v1 - multi-class w/ test-train split and cv
        # v2 - conversion to multilabel
        # v3 - drop custom test-train split and rely on cv
        # v4 - add threshold multilabel strategy, add a faster training phase
        yield [
            # these runs are meant to validate that the pipeline works as expected before expensive runs
            FitLogisticModel(
                k=3,
                max_iter=[5],
                num_folds=2,
                multilabel_strategy=strategy,
                input_path=f"{data_root}/processed/metadata_clean/v1",
                output_path=f"{data_root}/models/subset_logistic_{strategy}/v4_test",
            )
            for strategy in ["naive", "threshold"]
        ]

        # now fit this on a larger dataset to see if this works in a more realistic setting
        yield [
            FitLogisticModel(
                k=20,
                multilabel_strategy=strategy,
                input_path=f"{data_root}/processed/metadata_clean/v1",
                output_path=f"{data_root}/models/subset_logistic_{strategy}/v4",
            )
            for strategy in ["naive", "threshold"]
        ]

        yield [
            FitLogisticModel(
                multilabel_strategy=strategy,
                input_path=f"{data_root}/processed/metadata_clean/v1",
                output_path=f"{data_root}/models/logistic_{strategy}/v4",
            )
            for strategy in ["naive", "threshold"]
        ]


if __name__ == "__main__":
    luigi.build(
        [LogisticWorkflow()],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
        workers=1,
    )
