import os

import luigi
from xgboost.spark import SparkXGBClassifier

from ..utils import RsyncGCSFiles
from .base_tasks import BaseFitModel


class FitXGBoostModel(BaseFitModel):
    device = luigi.Parameter(default="cpu")
    num_workers = luigi.IntParameter(default=os.cpu_count())

    def _classifier(self, featuresCol: str, labelCol: str):
        return SparkXGBClassifier(
            features_col=featuresCol,
            label_col=labelCol,
            device=self.device,
            num_workers=self.num_workers,
        )


class BaselineWorkflow(luigi.Task):
    remote_root = luigi.Parameter(default="gs://dsgt-clef-geolifeclef-2024/data")
    local_root = luigi.Parameter(default="/mnt/data/geolifeclef-2024/data")

    def run(self):
        yield RsyncGCSFiles(
            src_path=f"{self.remote_root}/processed/metadata_clean/v1",
            dst_path=f"{self.local_root}/processed/metadata_clean/v1",
        )

        # fit this on a smaller dataset first for debugging
        yield [
            FitXGBoostModel(
                k=3,
                multilabel_strategy=strategy,
                input_path=f"{self.local_root}/processed/metadata_clean/v1",
                output_path=f"{self.local_root}/models/baseline_xgboost_{strategy}/v1_test",
            )
            for strategy in ["naive"]
        ]

        yield [
            FitXGBoostModel(
                multilabel_strategy=strategy,
                input_path=f"{self.local_root}/processed/metadata_clean/v1",
                output_path=f"{self.local_root}/models/baseline_xgboost_{strategy}/v1",
            )
            for strategy in ["naive"]
        ]


if __name__ == "__main__":
    luigi.build(
        [BaselineWorkflow()],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
        workers=1,
    )
