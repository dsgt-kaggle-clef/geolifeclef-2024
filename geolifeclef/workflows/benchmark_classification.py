import luigi
from .logistic import FitLogisticModel, BaseFitModel
from .utils import RsyncGCSFiles
from argparse import ArgumentParser
from pyspark.ml.classification import RandomForestClassifier


class FitRandomForestModel(BaseFitModel):
    num_trees = luigi.ListParameter(default=[20])

    def _classifier(self, featuresCol: str, labelCol: str):
        return RandomForestClassifier(
            featuresCol=featuresCol, labelCol=labelCol, numTrees=self.num_trees[0]
        )


class BenchmarkClassificationWorkflow(luigi.Task):
    remote_root = luigi.Parameter(default="gs://dsgt-clef-geolifeclef-2024/data")
    local_root = luigi.Parameter(default="/mnt/data/geolifeclef-2024/data")

    def run(self):
        yield RsyncGCSFiles(
            src_path=f"{self.remote_root}/processed/metadata_clean/v1",
            dst_path=f"{self.local_root}/processed/metadata_clean/v1",
        )

        yield [
            # these runs are meant to validate that the pipeline works as expected before expensive runs
            FitLogisticModel(
                k=3,
                max_iter=[5],
                num_folds=2,
                shuffle_partitions=8,
                label="speciesSubsetId",
                input_path=f"{self.local_root}/processed/metadata_clean/v1",
                output_path=f"{self.local_root}/models/benchmark_classification/v1_test/logistic",
            ),
            FitRandomForestModel(
                k=3,
                num_trees=[5],
                num_folds=2,
                shuffle_partitions=8,
                label="speciesSubsetId",
                input_path=f"{self.local_root}/processed/metadata_clean/v1",
                output_path=f"{self.local_root}/models/benchmark_classification/v1_test/random_forest",
            ),
        ]

        # now fit this on a larger dataset to see if this works in a more realistic setting
        yield [
            FitLogisticModel(
                k=20,
                shuffle_partitions=8,
                label="speciesSubsetId",
                input_path=f"{self.local_root}/processed/metadata_clean/v1",
                output_path=f"{self.local_root}/models/benchmark_classification/v1/logistic",
            ),
            FitRandomForestModel(
                k=20,
                shuffle_partitions=8,
                label="speciesSubsetId",
                input_path=f"{self.local_root}/processed/metadata_clean/v1",
                output_path=f"{self.local_root}/models/benchmark_classification/v1/random_forest",
            ),
        ]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scheduler-host", default="localhost")
    args = parser.parse_args()

    luigi.build(
        [BenchmarkClassificationWorkflow()],
        scheduler_host=args.scheduler_host,
        workers=4,
    )
