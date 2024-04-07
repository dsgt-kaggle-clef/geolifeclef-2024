import json
from argparse import ArgumentParser
from functools import reduce

import luigi
from contexttimer import Timer
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import BucketedRandomProjectionLSH, VectorAssembler
from pyspark.sql import functions as F

from geolifeclef.utils import spark_resource

from ..utils import RsyncGCSFiles, maybe_gcs_target


class GeoLSH(luigi.Task):
    """Find the nearest neighbor of each survey in the metadata."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()

    def output(self):
        return maybe_gcs_target(f"{self.output_path}/_SUCCESS")

    def _load(self, spark):
        return (spark.read.parquet(self.input_path).repartition(32)).cache()

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


class LSHSimilarityJoin(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    threshold = luigi.IntParameter(default=100)
    num_partitions = luigi.IntParameter(default=200)

    def output(self):
        return [
            maybe_gcs_target(
                f"{self.output_path}/edges/threshold={self.threshold}/_SUCCESS"
            ),
            maybe_gcs_target(
                f"{self.output_path}/timing.json",
            ),
        ]

    def run(self):
        with spark_resource(
            **{
                # set the number of shuffle partitions to be high
                "spark.sql.shuffle.partitions": self.num_partitions,
            }
        ) as spark:
            model = PipelineModel.load(f"{self.input_path}/model")
            data = (
                spark.read.parquet(f"{self.input_path}/data")
                .select("dataset", "surveyId", "speciesId", "features")
                .repartition(self.num_partitions)
                .cache()
            )
            data.printSchema()

            # now for different values of the threshold, we see the average number of neighbors
            # the euclidean distance is measured in meters, so let's choose reasonable values based
            # on the size of europe. We should compute this once for a large threshold, and then
            # start to whittle down after having the results materialized.
            with Timer() as timer:
                edges_path = f"{self.output_path}/edges/threshold={self.threshold}"
                edges = (
                    model.stages[-1]
                    .approxSimilarityJoin(
                        data,
                        data,
                        self.threshold,
                        distCol="euclidean",
                    )
                    .select(
                        F.col("datasetA.dataset").alias("srcDataset"),
                        F.col("datasetA.surveyId").alias("srcSurveyId"),
                        F.col("datasetA.speciesId").alias("srcSpeciesId"),
                        F.col("datasetB.dataset").alias("dstDataset"),
                        F.col("datasetB.surveyId").alias("dstSurveyId"),
                        F.col("datasetB.speciesId").alias("dstSpeciesId"),
                        "euclidean",
                    )
                )
                edges.printSchema()
                edges.write.parquet(edges_path, mode="overwrite")

            with self.output()[1].open("w") as f:
                res = {"threshold": self.threshold, "time": timer.elapsed}
                f.write(json.dumps(res))
                print(res)


class GenerateEdges(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    threshold = luigi.IntParameter(default=100)
    num_partitions = luigi.IntParameter(default=64)

    def output(self):
        return [
            maybe_gcs_target(
                f"{self.output_path}/survey_edges/threshold={self.threshold}/_SUCCESS"
            ),
            maybe_gcs_target(
                f"{self.output_path}/species_edges/threshold={self.threshold}/_SUCCESS"
            ),
            maybe_gcs_target(
                f"{self.output_path}/timing/threshold={self.threshold}/result.json",
            ),
        ]

    def _generate_edgelist(self, df, src, dst):
        return df.groupBy(src, dst).agg(F.count("*").alias("n"))

    def _process(self, spark, df, name, src, dst):
        edges_path = f"{self.output_path}/{name}_edges/threshold={self.threshold}"
        stats_path = f"{self.output_path}/{name}_stats/threshold={self.threshold}"
        (
            self._generate_edgelist(df, src, dst)
            .repartition(self.num_partitions)
            .write.parquet(edges_path, mode="overwrite")
        )
        spark.read.parquet(edges_path).describe().write.parquet(
            f"{stats_path}/name=freq", mode="overwrite"
        )
        spark.read.parquet(edges_path).groupBy("n").count().describe().write.parquet(
            f"{stats_path}/name=degree", mode="overwrite"
        )
        spark.read.parquet(stats_path).show()

    def run(self):
        with spark_resource() as spark:
            df = spark.read.parquet(self.input_path).where(
                f"euclidean < {self.threshold}"
            )

            # survey edges
            with Timer() as t1:
                self._process(spark, df, "survey", "srcSurveyId", "dstSpeciesId")

            # species edges
            with Timer() as t2:
                self._process(spark, df, "species", "srcSpeciesId", "dstSpeciesId")

            with self.output()[-1].open("w") as f:
                json.dump(
                    {
                        "time_survey": t1.elapsed,
                        "time_species": t2.elapsed,
                        "df_count": df.count(),
                    },
                    f,
                )


class NetworkWorkflow(luigi.Task):
    remote_root = luigi.Parameter(default="gs://dsgt-clef-geolifeclef-2024/data")
    local_root = luigi.Parameter(default="/mnt/data/geolifeclef-2024/data")
    threshold = luigi.IntParameter(default=100_000)

    def run(self):
        yield RsyncGCSFiles(
            src_path=f"{self.remote_root}/processed/metadata_clean/v1",
            dst_path=f"{self.local_root}/processed/metadata_clean/v1",
        )
        yield GeoLSH(
            input_path=f"{self.local_root}/processed/metadata_clean/v1",
            output_path=f"{self.local_root}/processed/geolsh/v1",
        )
        # everything within 100km is considered similar here
        yield LSHSimilarityJoin(
            input_path=f"{self.local_root}/processed/geolsh/v1",
            output_path=f"{self.local_root}/processed/geolsh_graph/v1",
            threshold=self.threshold,
        )

        # now let's compute edges in 10km increments
        yield [
            GenerateEdges(
                input_path=f"{self.local_root}/processed/geolsh_graph/v1/edges/threshold={self.threshold}",
                output_path=f"{self.local_root}/processed/geolsh_nn_graph/v2",
                threshold=threshold,
            )
            for threshold in [(i + 1) * 10_000 for i in range(10)]
        ]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scheduler-host", default="localhost")
    args = parser.parse_args()

    luigi.build(
        [NetworkWorkflow()],
        scheduler_host=args.scheduler_host,
        workers=1,
    )
