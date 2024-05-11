import os
from argparse import ArgumentParser

import luigi
from luigi.contrib.gcs import GCSTarget
from pyspark.sql import functions as F

from geolifeclef.utils import spark_resource

from ..utils import RsyncGCSFiles
from .xgboost import FitXGBoostModel


class LogSplitData(luigi.Task):
    input_path = luigi.Parameter(
        default="gs://dsgt-clef-geolifeclef-2024/data/processed/metadata_clean/v1"
    )
    output_path = luigi.Parameter(
        default="gs://dsgt-clef-geolifeclef-2024/data/processed/metadata_split"
    )
    log_base = luigi.IntParameter(default=2)

    def output(self):
        return GCSTarget(os.path.join(self.output_path), "_SUCCESS")

    def run(self):
        with spark_resource() as spark:
            metadata = spark.read.parquet(self.input_path)
            log_base_name = f"log_bin_{self.log_base}"
            species_count = metadata.groupBy("speciesId").count()
            log_df = species_count.withColumn(
                log_base_name,
                F.floor(F.log10(F.col("count")) / F.log10(F.lit(self.log_base))),
            )
            metadata_count = metadata.join(log_df, on="speciesId", how="inner")
            metadata_count.write.partitionBy(log_base_name).format("parquet").save(
                self.output_path
            )


class XGBoostSplitWorkflows(luigi.Task):
    input_path = luigi.Parameter(
        default="gs://dsgt-clef-geolifeclef-2024/data/processed/metadata_clean/v1"
    )
    split_output_path = luigi.Parameter(
        default="gs://dsgt-clef-geolifeclef-2024/data/processed/metadata_split"
    )
    remote_root = luigi.Parameter(default="gs://dsgt-clef-geolifeclef-2024/data")
    local_root = luigi.Parameter(default="/mnt/data/geolifeclef-2024/data")
    output_path = luigi.Parameter(default="")
    log_base = luigi.IntParameter(default=2)

    def requires(self):
        return LogSplitData(self.input_path, self.split_output_path, self.log_base)

    def run(self):
        yield RsyncGCSFiles(
            src_path=self.split_output_path,
            dst_path=f"{self.local_root}/processed/metadata_split/",
        )

        yield [
            FitXGBoostModel(
                num_workers=8,
                subsample=0.1,
                multilabel_strategy=strategy,
                input_path=f"{self.local_root}/processed/metadata_split/log_bin_2=10",
                output_path=f"{self.local_root}/models/baseline_xgboost_{strategy}/log_bin_2=10",
            )
            for strategy in ["naive"]
        ]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scheduler-host", default="localhost")
    args = parser.parse_args()

    luigi.build(
        [XGBoostSplitWorkflows()],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
        workers=1,
    )
