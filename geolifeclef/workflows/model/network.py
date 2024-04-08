import json
from argparse import ArgumentParser
from functools import reduce

import luigi
from contexttimer import Timer
from node2vec.spark import Node2VecSpark
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


class SubsetEdges(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    src = luigi.Parameter()
    dst = luigi.Parameter()
    k = luigi.IntParameter(default=100)
    seed = luigi.IntParameter(default=42)

    def output(self):
        return maybe_gcs_target(f"{self.output_path}/_SUCCESS")

    def run(self):
        with spark_resource() as spark:
            edges = spark.read.parquet(self.input_path)

            # randomly sample src and edges
            srcs = (
                edges.select(self.src)
                .distinct()
                .orderBy(F.rand(self.seed))
                .limit(self.k)
            )
            dst = (
                edges.select(self.dst)
                .distinct()
                .orderBy(F.rand(self.seed))
                .limit(self.k)
            )
            edges = edges.join(srcs, on=self.src).join(dst, on=self.dst)
            edges.write.parquet(self.output_path, mode="overwrite")


class Node2VecBase(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()

    # node2vec params
    num_walks = luigi.IntParameter(default=10)
    walk_length = luigi.IntParameter(default=10)
    p = luigi.FloatParameter(default=1.0)
    q = luigi.FloatParameter(default=1.0)
    max_out_degree = luigi.IntParameter(default=10_000)
    vector_size = luigi.IntParameter(default=128)
    checkpoint_dir = luigi.Parameter(default="/mnt/data/tmp/checkpoints")

    def _get_node2vec(self, spark):
        return Node2VecSpark(
            spark,
            {
                "num_walks": self.num_walks,
                "walk_length": self.walk_length,
                "return_param": self.p,
                "inout_param": self.q,
            },
            {},
            max_out_degree=self.max_out_degree,
            vector_size=self.vector_size,
            checkpoint_dir=self.checkpoint_dir,
        )


class Node2VecIndex(Node2VecBase):
    src = luigi.Parameter(default="srcSurveyId")
    dst = luigi.Parameter(default="dstSpeciesId")
    weight = luigi.Parameter(default="n")
    log_scale = luigi.BoolParameter(default=True)

    def output(self):
        return [
            maybe_gcs_target(f"{self.output_path}/name_id/_SUCCESS"),
            maybe_gcs_target(f"{self.output_path}/df/_SUCCESS"),
            maybe_gcs_target(f"{self.output_path}/df_adj/_SUCCESS"),
            maybe_gcs_target(f"{self.output_path}/index_timing.json"),
        ]

    def run(self):
        with spark_resource() as spark:
            df = spark.read.parquet(self.input_path).select(
                F.col(self.src).alias("src"),
                F.col(self.dst).alias("dst"),
                (F.log1p(self.weight) if self.log_scale else self.weight).alias(
                    "weight"
                ),
            )
            g2v = self._get_node2vec(spark)
            with Timer() as timer:
                g2v.preprocess_input_graph(df, indexed=False, directed=False)
                g2v.name_id.write.parquet(
                    f"{self.output_path}/name_id", mode="overwrite"
                )
                g2v.df.write.parquet(f"{self.output_path}/df", mode="overwrite")
                g2v.df_adj.write.parquet(f"{self.output_path}/df_adj", mode="overwrite")
        with self.output()[-1].open("w") as f:
            json.dump({"time": timer.elapsed}, f)


class Node2VecWalk(Node2VecBase):
    shuffle_partitions = luigi.IntParameter(default=3000)

    def output(self):
        return [
            maybe_gcs_target(f"{self.output_path}/walks/_SUCCESS"),
            maybe_gcs_target(f"{self.output_path}/walk_timing.json"),
        ]

    def run(self):
        with spark_resource(
            **{
                "spark.sql.shuffle.partitions": self.shuffle_partitions,
            }
        ) as spark:
            g2v = self._get_node2vec(spark)
            g2v.name_id = (
                spark.read.parquet(f"{self.input_path}/name_id")
                .repartition(self.shuffle_partitions)
                .cache()
            )
            g2v.df = (
                spark.read.parquet(f"{self.input_path}/df")
                .repartition(self.shuffle_partitions)
                .cache()
            )
            g2v.df_adj = (
                spark.read.parquet(f"{self.input_path}/df_adj")
                .repartition(self.shuffle_partitions)
                .cache()
            )
            with Timer() as timer:
                g2v.random_walk().write.parquet(
                    f"{self.output_path}/walks", mode="overwrite"
                )
        with self.output()[-1].open("w") as f:
            json.dump({"time": timer.elapsed}, f)


class Node2VecTrain(Node2VecBase):
    def output(self):
        return [
            maybe_gcs_target(f"{self.output_path}/embedding/_SUCCESS"),
            maybe_gcs_target(f"{self.output_path}/embedding_timing.json"),
        ]

    def run(self):
        with spark_resource() as spark:
            g2v = self._get_node2vec(spark)
            g2v.name_id = spark.read.parquet(f"{self.input_path}/name_id").cache()
            g2v.df = spark.read.parquet(f"{self.input_path}/df")
            g2v.df_adj = spark.read.parquet(f"{self.input_path}/df_adj")
            walks = spark.read.parquet(f"{self.output_path}/walks").cache()
            with Timer() as timer:
                g2v.fit(walks)
                g2v.embedding().write.parquet(
                    f"{self.output_path}/embedding", mode="overwrite"
                )
        with self.output()[-1].open("w") as f:
            json.dump({"time": timer.elapsed}, f)


class Node2VecWorkflow(Node2VecBase):
    src = luigi.Parameter(default="srcSurveyId")
    dst = luigi.Parameter(default="dstSpeciesId")
    weight = luigi.Parameter(default="n")

    @property
    def _output_path_keyed(self):
        return "/".join(
            [
                self.output_path,
                f"p={self.p}",
                f"q={self.q}",
                f"walks={self.num_walks}",
                f"length={self.walk_length}",
                f"vector_size={self.vector_size}",
            ]
        )

    def output(self):
        return [
            maybe_gcs_target(f"{self._output_path_keyed}/embedding/_SUCCESS"),
            maybe_gcs_target(f"{self._output_path_keyed}/embedding_timing.json"),
        ]

    def run(self):
        params = dict(
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            p=self.p,
            q=self.q,
            max_out_degree=self.max_out_degree,
            vector_size=self.vector_size,
        )
        yield Node2VecIndex(
            input_path=self.input_path,
            output_path=f"{self.output_path}/index",
            src=self.src,
            dst=self.dst,
            weight=self.weight,
            log_scale=True,
            **params,
        )
        yield Node2VecWalk(
            input_path=f"{self.output_path}/index",
            output_path=self._output_path_keyed,
            **params,
        )
        yield Node2VecTrain(
            input_path=f"{self.output_path}/index",
            output_path=self._output_path_keyed,
            **params,
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

        # let's run node2vec on a subset of data for testing
        yield SubsetEdges(
            input_path=f"{self.local_root}/processed/geolsh_nn_graph/v2/survey_edges/threshold={self.threshold}",
            output_path=f"{self.local_root}/processed/geolsh_nn_graph/v2_subset/survey_edges/threshold=50000",
            src="srcSurveyId",
            dst="dstSpeciesId",
            k=100,
        )
        yield Node2VecWorkflow(
            input_path=f"{self.local_root}/processed/geolsh_nn_graph/v2_subset/survey_edges/threshold=50000",
            output_path=f"{self.local_root}/processed/survey_node2vec/v2_subset",
            num_walks=10,
            walk_length=10,
            p=1.0,
            q=1.0,
            max_out_degree=100,
            vector_size=32,
        )

        # now run this for real
        yield [
            Node2VecWorkflow(
                input_path=f"{self.local_root}/processed/geolsh_nn_graph/v2/{gtype}_edges/threshold=50000",
                output_path=f"{self.local_root}/processed/{gtype}_node2vec/v2",
                src="srcSurveyId" if gtype == "survey" else "srcSpeciesId",
                dst="dstSpeciesId",
                num_walks=30,
                walk_length=10,
                p=p,
                q=q,
                max_out_degree=10_000,
                vector_size=d,
            )
            for (p, q) in [
                (1.0, 0.5),
                (1.0, 1.0),
                (1.0, 2.0),
            ]
            for d in [64]
            for gtype in ["survey", "species"]
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
