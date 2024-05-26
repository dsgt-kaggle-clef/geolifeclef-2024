import json
import logging
from argparse import ArgumentParser

import luigi
from contexttimer import Timer
from pyspark.ml.evaluation import MultilabelClassificationEvaluator
from pyspark.sql import Window
from pyspark.sql import functions as F

from geolifeclef.utils import spark_resource

from ..utils import RsyncGCSFiles, maybe_gcs_target


class KNNGraphModel(luigi.Task):
    input_path = luigi.Parameter()
    edges_path = luigi.Parameter()
    output_path = luigi.Parameter()
    target_column = luigi.Parameter(default="dist")
    threshold = luigi.IntParameter()

    def output(self):
        return maybe_gcs_target(f"{self.output_path}/_SUCCESS")

    def run(self):
        with spark_resource() as spark:
            df = spark.read.parquet(self.input_path)
            edges = spark.read.parquet(self.edges_path)

            # first get the actual labels for the pa dataset
            pa_df = (
                df.where(F.col("dataset") != "po")
                .groupBy("dataset", "surveyId")
                .agg(F.array_sort(F.collect_set("speciesId")).alias("labels"))
            )

            # also always make at minimum a single prediction
            predictions_min = (
                edges.withColumn(
                    "rank",
                    F.row_number().over(
                        Window.partitionBy("srcSurveyId").orderBy(
                            F.asc(self.target_column)
                        )
                    ),
                )
                .where("rank = 1")
                .drop("rank")
            )
            predictions = (
                edges.where(F.col(self.target_column) < self.threshold)
                .union(predictions_min)
                .groupBy(F.col("srcSurveyId").alias("surveyId"))
                .agg(F.array_sort(F.collect_set("dstSpeciesId")).alias("prediction"))
            )

            result = (
                pa_df.join(predictions, "surveyId", "left")
                .withColumn("prediction", F.coalesce("prediction", F.array()))
                .withColumn("labels", F.coalesce("labels", F.array()))
            ).cache()

            # now test the results using pa_train
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

    def run(self):
        yield [
            RsyncGCSFiles(
                src_path=f"{self.remote_root}/processed/metadata_clean/v1",
                dst_path=f"{self.local_root}/processed/metadata_clean/v1",
            ),
            RsyncGCSFiles(
                src_path=f"{self.remote_root}/processed/geolsh_knn_graph/v2",
                dst_path=f"{self.local_root}/processed/geolsh_knn_graph/v2",
            ),
            RsyncGCSFiles(
                src_path=f"{self.remote_root}/processed/geolsh_nn_graph/v2",
                dst_path=f"{self.local_root}/processed/geolsh_nn_graph/v2",
            ),
        ]

        suffix = "threshold=50000"
        yield [
            *[
                KNNGraphModel(
                    input_path=f"{self.local_root}/processed/metadata_clean/v1",
                    edges_path=f"{self.local_root}/processed/geolsh_nn_graph/v2/survey_edges/{suffix}",
                    output_path=f"{self.local_root}/models/geolsh_nn_graph/v1/threshold={threshold}",
                    threshold=threshold,
                )
                for threshold in [5_000, 10_000, 50_000]
            ],
            *[
                KNNGraphModel(
                    input_path=f"{self.local_root}/processed/metadata_clean/v1",
                    edges_path=f"{self.local_root}/processed/geolsh_knn_graph/v2/survey_edges/{suffix}/k=10",
                    output_path=f"{self.local_root}/models/knn_graph/v1/threshold={threshold}",
                    threshold=threshold,
                )
                for threshold in [5_000, 10_000, 50_000]
            ],
        ]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scheduler-host", default="localhost")
    args = parser.parse_args()

    luigi.build(
        [Workflow()],
        scheduler_host=args.scheduler_host,
        workers=1,
    )
