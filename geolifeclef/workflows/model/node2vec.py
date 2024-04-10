import json
import logging
from argparse import ArgumentParser

import luigi
from contexttimer import Timer
from node2vec.spark import Node2VecSpark
from pyspark.sql import functions as F

from geolifeclef.utils import spark_resource

from ..utils import RsyncGCSFiles, maybe_gcs_target


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
    checkpoint_interval = luigi.IntParameter(default=5)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # enable logging
        logging.basicConfig(level=logging.INFO)

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
            # checkpoint_dir=self.checkpoint_dir,
            # checkpoint_interval=self.checkpoint_interval,
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
    shuffle_partitions = luigi.IntParameter(default=500)

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


class Node2VecTask(Node2VecBase):
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


class Node2VecWorkflow(luigi.Task):
    remote_root = luigi.Parameter(default="gs://dsgt-clef-geolifeclef-2024/data")
    local_root = luigi.Parameter(default="/mnt/data/geolifeclef-2024/data")
    threshold = luigi.IntParameter(default=100_000)

    def run(self):
        # TODO: this workflow isn't ready for prime-time
        yield RsyncGCSFiles(
            src_path=f"{self.remote_root}/processed/metadata_clean/v1",
            dst_path=f"{self.local_root}/processed/metadata_clean/v1",
        )

        yield Node2VecTask(
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
            Node2VecTask(
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
