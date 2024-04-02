from functools import reduce

import luigi
from contexttimer import Timer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MultilabelClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder
from pyspark.sql import functions as F

from geolifeclef.functions import get_projection_udf
from geolifeclef.utils import spark_resource

from .transformer import NaiveMultiClassToMultiLabel
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


class FitLogisticModel(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()

    features = luigi.ListParameter(default=["lat_proj", "lon_proj"])
    label = luigi.Parameter(default="speciesId")

    num_folds = luigi.IntParameter(default=3)
    max_iter = luigi.ListParameter(default=[100])
    reg_param = luigi.ListParameter(default=[0.0])
    elastic_net_param = luigi.ListParameter(default=[0.0])
    seed = luigi.IntParameter(default=42)

    def output(self):
        return maybe_gcs_target(f"{self.output_path}/_SUCCESS")

    def _load(self, spark):
        return spark.read.parquet(self.input_path).where(F.col(self.label).isNotNull())

    def _train_test_split(self, df, train_size=0.8):
        # use the surveyId to split the data
        sample_id = df.withColumn(
            "sample_id", F.crc32(F.col("surveyId").cast("string")) % 100
        )
        train = (
            sample_id.where(F.col("sample_id") < train_size * 100)
            .drop("sample_id")
            .cache()
        )
        test = (
            sample_id.where(F.col("sample_id") >= train_size * 100)
            .drop("sample_id")
            .cache()
        )
        return train, test

    def _pipeline(self):
        return Pipeline(
            stages=[
                VectorAssembler(
                    inputCols=self.features,
                    outputCol="features",
                ),
                StandardScaler(inputCol="features", outputCol="scaled_features"),
                LogisticRegression(featuresCol="scaled_features", labelCol=self.label),
                NaiveMultiClassToMultiLabel(
                    primaryKeyCol="surveyId",
                    labelCol=self.label,
                    inputCol="prediction",
                    outputCol="prediction",
                ),
            ]
        )

    def _evaluator(self):
        return MultilabelClassificationEvaluator(
            predictionCol="prediction",
            labelCol=self.label,
            metricName="microF1Measure",
        )

    def _param_grid(self, lr):
        return (
            ParamGridBuilder()
            .addGrid(lr.maxIter, self.max_iter)
            .addGrid(lr.regParam, self.reg_param)
            .addGrid(lr.elasticNetParam, self.elastic_net_param)
            .build()
        )

    def _calculate_multilabel_stats(self, train, test):
        """Calculate statistics about the number of rows with multiple labels"""

        train.groupBy("surveyId").count().describe().write.csv(
            f"{self.output_path}/multilabel_stats/dataset=train", mode="overwrite"
        )
        test.groupBy("surveyId").count().describe().write.csv(
            f"{self.output_path}/multilabel_stats/dataset=test", mode="overwrite"
        )

    def run(self):
        with spark_resource() as spark:
            train, test = self._train_test_split(self._load(spark), train_size=0.8)
            self._calculate_multilabel_stats(train, test)

            # write the model to disk
            pipeline = self._pipeline()
            lr = [
                stage
                for stage in pipeline.getStages()
                if isinstance(stage, LogisticRegression)
            ][0]
            cv = CrossValidator(
                estimator=pipeline,
                estimatorParamMaps=self._param_grid(lr),
                evaluator=self._evaluator(),
                numFolds=self.num_folds,
            )

            with Timer() as train_timer:
                model = cv.fit(train)
                model.write().overwrite().save(f"{self.output_path}/model")

            # evaluate on test set
            with Timer() as eval_timer:
                model = CrossValidatorModel.load(f"{self.output_path}/model")
                predictions = model.transform(test)
                score = model.getEvaluator().evaluate(predictions)

            # write the results to disk
            perf = spark.createDataFrame(
                [
                    {
                        "train_time": train_timer.elapsed,
                        "eval_time": eval_timer.elapsed,
                        "avg_metrics": model.avgMetrics,
                        "std_metrics": model.avgMetrics,
                        "test_metric": score,
                        "metric_name": model.getEvaluator().getMetricName(),
                    }
                ]
            )
            perf.write.json(f"{self.output_path}/perf", mode="overwrite")
            perf.show()

        # write the output
        with self.output().open("w") as f:
            f.write("")


class FitSubsetLogisticModel(FitLogisticModel):
    """A logistic model that uses a subset of labels for training.

    This should ideally save us a bit of time. However, we'll need
    to compute a few statistics to ensure we're doing this in a rigorous
    way. In particular, we'll want to know if any of these rows will end
    up with multiple labels. We can simply count this at the beginning
    to make sure we see something reasonable.
    """

    k = luigi.IntParameter(default=20)
    seed = luigi.IntParameter(default=42)

    def _load(self, spark):
        df = super()._load(spark)
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


class LogisticWorkflow(luigi.Task):
    def run(self):
        data_root = "gs://dsgt-clef-geolifeclef-2024/data"
        yield CleanMetadata(
            input_path=f"{data_root}/downloaded/2024",
            output_path=f"{data_root}/processed/metadata_clean/v1",
        )

        yield FitSubsetLogisticModel(
            input_path=f"{data_root}/processed/metadata_clean/v1",
            output_path=f"{data_root}/models/subset_logistic/v2",
        )

        yield FitLogisticModel(
            input_path=f"{data_root}/processed/metadata_clean/v1",
            output_path=f"{data_root}/models/logistic/v2",
        )


if __name__ == "__main__":
    luigi.build(
        [LogisticWorkflow()],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
        workers=1,
    )
