from argparse import ArgumentParser

import luigi
from pyspark.ml.classification import RandomForestClassifier
from xgboost.spark import SparkXGBClassifier

from .logistic import BaseFitModel, FitLogisticModel
from .utils import RsyncGCSFiles


class FitRandomForestModel(BaseFitModel):
    num_trees = luigi.ListParameter(default=[20])

    def _classifier(self, featuresCol: str, labelCol: str):
        return RandomForestClassifier(
            featuresCol=featuresCol, labelCol=labelCol, numTrees=self.num_trees[0]
        )


class FitXGBoostModel(BaseFitModel):
    device = luigi.Parameter(default="cpu")

    def _classifier(self, featuresCol: str, labelCol: str):
        return SparkXGBClassifier(
            features_col=featuresCol, label_col=labelCol, device=self.device
        )


class BenchmarkClassificationWorkflow(luigi.Task):
    remote_root = luigi.Parameter(default="gs://dsgt-clef-geolifeclef-2024/data")
    local_root = luigi.Parameter(default="/mnt/data/geolifeclef-2024/data")

    def run(self):
        yield RsyncGCSFiles(
            src_path=f"{self.remote_root}/processed/metadata_clean/v1",
            dst_path=f"{self.local_root}/processed/metadata_clean/v1",
        )

        for k in [3, 20, 100]:
            yield [
                # these runs are meant to validate that the pipeline works as expected before expensive runs
                FitLogisticModel(
                    k=k,
                    shuffle_partitions=32,
                    label="speciesSubsetId",
                    input_path=f"{self.local_root}/processed/metadata_clean/v1",
                    output_path=f"{self.local_root}/models/benchmark_classification/v2/model=logistic/k={k}",
                ),
                FitRandomForestModel(
                    k=k,
                    shuffle_partitions=32,
                    label="speciesSubsetId",
                    input_path=f"{self.local_root}/processed/metadata_clean/v1",
                    output_path=f"{self.local_root}/models/benchmark_classification/v2/model=random_forest/k={k}",
                ),
                FitXGBoostModel(
                    k=k,
                    shuffle_partitions=32,
                    label="speciesSubsetId",
                    input_path=f"{self.local_root}/processed/metadata_clean/v1",
                    output_path=f"{self.local_root}/models/benchmark_classification/v2/model=xgboost/k={k}",
                ),
            ]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scheduler-host", default="localhost")
    args = parser.parse_args()

    luigi.build(
        [BenchmarkClassificationWorkflow()],
        scheduler_host=args.scheduler_host,
        workers=1,
    )
