import os
from argparse import ArgumentParser
from pathlib import Path

import luigi
import luigi.contrib.gcs
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from geolifeclef.torch.multilabel.data import GeoSpatialDataModel
from geolifeclef.torch.multilabel.model import MultiLabelClassifier
from geolifeclef.utils import spark_resource

from ..utils import RsyncGCSFiles, maybe_gcs_target
from .base_tasks import CleanMetadata


class TrainMultiLabelClassifier(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    batch_size = luigi.IntParameter(default=32)
    num_partitions = luigi.IntParameter(default=32)

    def output(self):
        # save the model run
        return maybe_gcs_target(f"{self.output_path}/checkpoints/last.ckpt")

    def run(self):
        with spark_resource() as spark:
            # data module
            data_module = GeoSpatialDataModel(
                spark,
                self.input_path,
                batch_size=self.batch_size,
                num_partitions=self.num_partitions,
            )
            weights = data_module.compute_weights()
            data_module.setup()

            # get parameters for the model
            row = data_module.train_data.first()
            num_features = int(len(row.features))
            num_classes = int(len(row.label))

            assert weights.shape[0] == num_classes, (weights.shape, num_classes)

            # model module
            model = MultiLabelClassifier(num_features, num_classes)

            trainer = pl.Trainer(
                max_epochs=20,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                reload_dataloaders_every_n_epochs=1,
                default_root_dir=self.output_path,
                logger=WandbLogger(
                    project="geolifeclef-2024",
                    name=Path(self.output_path).name,
                    save_dir=self.output_path,
                    config={
                        "batch_size": self.batch_size,
                        "input_path": self.input_path,
                    },
                ),
                callbacks=[
                    EarlyStopping(monitor="val_loss", mode="min", min_delta=1e-4),
                    EarlyStopping(monitor="val_f1", mode="max"),
                    ModelCheckpoint(
                        dirpath=os.path.join(self.output_path, "checkpoints"),
                        monitor="val_f1",
                        mode="max",
                        save_last=True,
                    ),
                ],
            )
            trainer.fit(model, data_module)


class Workflow(luigi.Task):
    remote_root = luigi.Parameter(default="gs://dsgt-clef-geolifeclef-2024/data")
    local_root = luigi.Parameter(default="/mnt/data/geolifeclef-2024/data")

    def run(self):
        yield [
            RsyncGCSFiles(
                src_path=f"{self.remote_root}/downloaded/2024/{name}",
                dst_path=f"{self.local_root}/downloaded/2024/{name}",
            )
            for name in [
                "PresenceOnlyOccurrences",
                "PresenceAbsenceSurveys",
            ]
        ]
        yield CleanMetadata(
            input_path=f"{self.local_root}/downloaded/2024",
            output_path=f"{self.local_root}/processed/metadata_clean/v2",
        )
        yield RsyncGCSFiles(
            src_path=f"{self.local_root}/processed/metadata_clean/v2",
            dst_path=f"{self.remote_root}/processed/metadata_clean/v2",
        )

        yield [
            # these runs are meant to validate that the pipeline works as expected before expensive runs
            # v1 - first model, 70 it/s on epoch 1+
            # v2 - set 90/10 train/valid split, increase number of partitions, 20 epochs max
            #   - 22 it/s on epoch 0, 80 it/s on epoch 1+
            # v3 - remove sigmoid layer
            # v4 - proper early stopping and use weights
            TrainMultiLabelClassifier(
                input_path=f"{self.local_root}/processed/metadata_clean/v2",
                output_path=f"{self.local_root}/models/multilabel_classifier/v4",
            )
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
