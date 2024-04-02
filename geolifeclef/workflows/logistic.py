from functools import reduce

import luigi
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import BucketedRandomProjectionLSH, VectorAssembler
from pyspark.sql import functions as F

from geolifeclef.functions import get_projection_udf
from geolifeclef.utils import spark_resource

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


class GeoLSH(luigi.Task):
    """Find the nearest neighbor of each survey in the metadata."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()

    def output(self):
        return maybe_gcs_target(f"{self.output_path}/_SUCCESS")

    def _load(self, spark):
        return (
            spark.read.parquet(self.input_path)
            .where(F.col("speciesId").isNotNull())
            .repartition(32)
        ).cache()

    def _pipeline(self):
        return Pipeline(
            stages=[
                VectorAssembler(
                    inputCols=["lat_proj", "lon_proj"], outputCol="features"
                ),
                BucketedRandomProjectionLSH(
                    inputCol="features",
                    outputCol="hashes",
                    bucketLength=20,
                    numHashTables=5,
                ),
            ]
        )

    def run(self):
        with spark_resource() as spark:
            train = self._load(spark)

            # write the model to disk
            model = self._pipeline().fit(train)
            model.write().overwrite().save(f"{self.output_path}/model")
            model = PipelineModel.load(f"{self.output_path}/model")

            # transform the data and write it to disk
            model.transform(train).write.parquet(
                f"{self.output_path}/data", mode="overwrite"
            )

        # write the output
        with self.output().open("w") as f:
            f.write("")


class LSHSimilarityTest(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()

    def output(self):
        return maybe_gcs_target(f"{self.output_path}/_SUCCESS")

    def run(self):
        with spark_resource() as spark:
            model = PipelineModel.load(f"{self.input_path}/model")
            data = spark.read.parquet(f"{self.input_path}/data")

            # now for different values of the threshold, we see the average number of neighbors
            # the euclidean distance is measured in meters, so let's choose reasonable values based
            # on the size of europe
            for threshold in [
                10,
                50,
                100,
                500,
                1_000,
                5_000,
                10_000,
                50_000,
                100_000,
                500_000,
            ]:
                (
                    model.stages[-1]
                    .approxSimilarityJoin(
                        data,
                        data,
                        threshold,
                        distCol="euclidean",
                    )
                    .groupBy(
                        F.col("datasetA.speciesId").alias("src"),
                        F.col("datasetB.speciesId").alias("dst"),
                    )
                    .agg(F.count("*").alias("count"))
                    .write.parquet(
                        f"{self.output_path}/edges/threshold={threshold}",
                        mode="overwrite",
                    )
                )
                (
                    spark.read.parquet(
                        f"{self.output_path}/edges/threshold={threshold}"
                    )
                    .describe()
                    .repartition(1)
                    .write.parquet(
                        f"{self.output_path}/stats/threshold={threshold}",
                        mode="overwrite",
                    )
                )
                spark.read.parquet(
                    f"{self.output_path}/stats/threshold={threshold}"
                ).show()

        with self.output().open("w") as f:
            f.write("")


class LogisticWorkflow(luigi.Task):
    def run(self):
        data_root = "gs://dsgt-clef-geolifeclef-2024/data"
        yield CleanMetadata(
            input_path=f"{data_root}/downloaded/2024",
            output_path=f"{data_root}/processed/metadata_clean/v1",
        )
        yield GeoLSH(
            input_path=f"{data_root}/processed/metadata_clean/v1",
            output_path=f"{data_root}/processed/geolsh/v1",
        )
        yield LSHSimilarityTest(
            input_path=f"{data_root}/processed/geolsh/v1",
            output_path=f"{data_root}/processed/geolsh_graph/v1",
        )


if __name__ == "__main__":
    luigi.build(
        [
            LogisticWorkflow(),
        ],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
        workers=1,
    )
