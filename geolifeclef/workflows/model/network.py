from functools import reduce

import luigi
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import BucketedRandomProjectionLSH, VectorAssembler
from pyspark.sql import functions as F

from geolifeclef.utils import spark_resource

from ..utils import maybe_gcs_target
from .logistic import CleanMetadata


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
    threshold = luigi.IntParameter(default=100)

    def output(self):
        return [
            maybe_gcs_target(
                f"{self.output_path}/edges/threshold={self.threshold}/_SUCCESS"
            ),
            maybe_gcs_target(
                f"{self.output_path}/stats/threshold={self.threshold}/_SUCCESS"
            ),
        ]

    def run(self):
        with spark_resource() as spark:
            model = PipelineModel.load(f"{self.input_path}/model")
            data = spark.read.parquet(f"{self.input_path}/data")

            # now for different values of the threshold, we see the average number of neighbors
            # the euclidean distance is measured in meters, so let's choose reasonable values based
            # on the size of europe
            edges_path = f"{self.output_path}/edges/threshold={self.threshold}"
            stats_path = f"{self.output_path}/stats/threshold={self.threshold}"
            (
                model.stages[-1]
                .approxSimilarityJoin(
                    data,
                    data,
                    self.threshold,
                    distCol="euclidean",
                )
                .groupBy(
                    F.col("datasetA.speciesId").alias("src"),
                    F.col("datasetB.speciesId").alias("dst"),
                )
                .agg(F.count("*").alias("count"))
                .write.parquet(edges_path, mode="overwrite")
            )
            (
                spark.read.parquet(edges_path)
                .describe()
                .repartition(1)
                .write.parquet(stats_path, mode="overwrite")
            )
            spark.read.parquet(stats_path).show()


class NetworkWorkflow(luigi.Task):
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
        lsh_sim_test = []
        for threshold in [
            10,
            50,
            100,
            500,
            1_000,
            5_000,
            # 10_000,
            # 50_000,
            # 100_000,
            # 500_000,
        ]:
            task = LSHSimilarityTest(
                input_path=f"{data_root}/processed/geolsh/v1",
                output_path=f"{data_root}/processed/geolsh_graph/v1",
                threshold=threshold,
            )
            lsh_sim_test.append(task)
        yield lsh_sim_test


if __name__ == "__main__":
    luigi.build(
        [NetworkWorkflow()],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
        workers=1,
    )
