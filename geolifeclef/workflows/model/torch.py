import os
from argparse import ArgumentParser
from pathlib import Path

import luigi
import luigi.contrib.gcs
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateFinder, StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from geolifeclef.torch.multilabel.data import GeoSpatialDataModel
from geolifeclef.torch.multilabel.model import MultiLabelClassifier
from geolifeclef.torch.raster2vec.data import Raster2VecDataModel
from geolifeclef.torch.raster2vec.model import Raster2Vec
from geolifeclef.torch.raster2vec_clf.data import Raster2VecClassifierDataModel
from geolifeclef.torch.raster2vec_clf.model import Raster2VecClassifier
from geolifeclef.torch.raster.data import RasterDataModel
from geolifeclef.torch.raster.model import RasterClassifier
from geolifeclef.utils import spark_resource

from ..utils import RsyncGCSFiles, maybe_gcs_target
from .base_tasks import CleanMetadata


class TrainMultiLabelClassifier(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    batch_size = luigi.IntParameter(default=128)
    num_partitions = luigi.IntParameter(default=200)

    def output(self):
        # save the model run
        return maybe_gcs_target(f"{self.output_path}/checkpoints/last.ckpt")

    def run(self):
        with spark_resource(memory="24g") as spark:
            # data module
            data_module = GeoSpatialDataModel(
                spark,
                self.input_path,
                batch_size=self.batch_size,
                num_partitions=self.num_partitions,
                pa_only=True,
            )
            weights = data_module.compute_weights()
            data_module.setup()

            # get parameters for the model
            num_features, num_classes = data_module.get_shape()

            assert weights.shape[0] == num_classes, (weights.shape, num_classes)

            # model module
            model = MultiLabelClassifier(num_features, num_classes, weights=weights)

            trainer = pl.Trainer(
                max_epochs=20,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                reload_dataloaders_every_n_epochs=1,
                default_root_dir=self.output_path,
                logger=WandbLogger(
                    project="geolifeclef-2024",
                    name="-".join(reversed(Path(self.output_path).parts[-2:])),
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
                    LearningRateFinder(),
                ],
            )
            trainer.fit(model, data_module)


class TrainRasterClassifier(luigi.Task):
    input_path = luigi.Parameter()
    feature_paths = luigi.ListParameter()
    feature_cols = luigi.ListParameter()
    output_path = luigi.Parameter()
    use_idct = luigi.BoolParameter(default=False)
    batch_size = luigi.IntParameter(default=250)
    num_partitions = luigi.IntParameter(default=200)

    def output(self):
        # save the model run
        return maybe_gcs_target(f"{self.output_path}/checkpoints/last.ckpt")

    def run(self):
        with spark_resource() as spark:
            # data module
            data_module = RasterDataModel(
                spark,
                self.input_path,
                self.feature_paths,
                self.feature_cols,
                use_idct=self.use_idct,
                batch_size=self.batch_size,
                num_partitions=self.num_partitions,
            )
            data_module.setup()

            num_layers, num_features, num_classes = data_module.get_shape()
            model = RasterClassifier(num_layers, num_features, num_classes)

            trainer = pl.Trainer(
                max_epochs=20,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                reload_dataloaders_every_n_epochs=1,
                default_root_dir=self.output_path,
                logger=WandbLogger(
                    project="geolifeclef-2024",
                    name="-".join(reversed(Path(self.output_path).parts[-2:])),
                    save_dir=self.output_path,
                    config={
                        "feature_paths": self.feature_paths,
                        "feature_cols": self.feature_cols,
                        "batch_size": self.batch_size,
                        "input_path": self.input_path,
                    },
                ),
                callbacks=[
                    EarlyStopping(monitor="val_loss", mode="min"),
                    EarlyStopping(monitor="val_f1", mode="max"),
                    # StochasticWeightAveraging(swa_lrs=1e-2),
                    ModelCheckpoint(
                        dirpath=os.path.join(self.output_path, "checkpoints"),
                        monitor="val_f1",
                        mode="max",
                        save_last=True,
                    ),
                    LearningRateFinder(),
                ],
            )
            trainer.fit(model, data_module)


class TrainRaster2Vec(luigi.Task):
    input_path = luigi.Parameter()
    feature_paths = luigi.ListParameter()
    feature_cols = luigi.ListParameter()
    output_path = luigi.Parameter()
    batch_size = luigi.IntParameter(default=250)
    workers_count = luigi.IntParameter(default=8)
    num_partitions = luigi.IntParameter(default=200)

    def output(self):
        # save the model run
        return maybe_gcs_target(f"{self.output_path}/checkpoints/last.ckpt")

    def run(self):
        with spark_resource(memory="16g") as spark:
            # data module
            data_module = Raster2VecDataModel(
                spark,
                self.input_path,
                self.feature_paths,
                self.feature_cols,
                batch_size=self.batch_size,
                num_partitions=self.num_partitions,
                workers_count=self.workers_count,
            )
            data_module.setup()

            num_layers, num_features, num_classes = data_module.get_shape()
            model = Raster2Vec(num_layers, num_features, num_classes, disable_asl=True)

            trainer = pl.Trainer(
                max_epochs=20,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                reload_dataloaders_every_n_epochs=1,
                default_root_dir=self.output_path,
                logger=WandbLogger(
                    project="geolifeclef-2024",
                    name="-".join(reversed(Path(self.output_path).parts[-2:])),
                    group=Path(self.output_path).parts[-2],
                    save_dir=self.output_path,
                    config={
                        "feature_paths": self.feature_paths,
                        "feature_cols": self.feature_cols,
                        "batch_size": self.batch_size,
                        "input_path": self.input_path,
                    },
                ),
                callbacks=[
                    EarlyStopping(monitor="train_loss", mode="min"),
                    # StochasticWeightAveraging(swa_lrs=1e-2),
                    ModelCheckpoint(
                        dirpath=os.path.join(self.output_path, "checkpoints"),
                        monitor="val_loss",
                        mode="min",
                        save_last=True,
                    ),
                    # LearningRateFinder(),
                ],
            )
            trainer.fit(model, data_module)


class TrainRaster2VecClassifier(luigi.Task):
    input_path = luigi.Parameter()
    base_model = luigi.Parameter()
    feature_paths = luigi.ListParameter()
    feature_cols = luigi.ListParameter()
    output_path = luigi.Parameter()
    batch_size = luigi.IntParameter(default=250)
    num_partitions = luigi.IntParameter(default=200)

    def output(self):
        return [
            maybe_gcs_target(f"{self.output_path}/checkpoints/last.ckpt"),
        ]

    def run(self):
        with spark_resource() as spark:
            data_module = RasterDataModel(
                spark,
                self.input_path,
                self.feature_paths,
                self.feature_cols,
                batch_size=self.batch_size,
                num_partitions=self.num_partitions,
            )
            data_module.setup()

            backbone = Raster2Vec.load_from_checkpoint(self.base_model)
            num_layers, num_features, num_classes = data_module.get_shape()
            model = Raster2VecClassifier(num_layers, num_features, num_classes)
            model.model = backbone.model

            trainer = pl.Trainer(
                max_epochs=20,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                reload_dataloaders_every_n_epochs=1,
                default_root_dir=self.output_path,
                logger=WandbLogger(
                    project="geolifeclef-2024",
                    name="-".join(reversed(Path(self.output_path).parts[-2:])),
                    save_dir=self.output_path,
                    config={
                        "feature_paths": self.feature_paths,
                        "feature_cols": self.feature_cols,
                        "batch_size": self.batch_size,
                        "input_path": self.input_path,
                        "base_model": self.base_model,
                    },
                ),
                callbacks=[
                    EarlyStopping(monitor="val_loss", mode="min"),
                    EarlyStopping(monitor="val_f1", mode="max"),
                    # StochasticWeightAveraging(swa_lrs=1e-2),
                    ModelCheckpoint(
                        dirpath=os.path.join(self.output_path, "checkpoints"),
                        monitor="val_f1",
                        mode="max",
                        save_last=True,
                    ),
                    LearningRateFinder(),
                ],
            )
            trainer.fit(model, data_module)


class PredictClassifier(luigi.Task):
    input_path = luigi.Parameter()
    base_model = luigi.Parameter()
    model_name = luigi.ChoiceParameter(choices=["raster", "raster2vec"])
    use_idct = luigi.BoolParameter(default=False)
    feature_paths = luigi.ListParameter()
    feature_cols = luigi.ListParameter()
    output_path = luigi.Parameter()
    batch_size = luigi.IntParameter(default=250)
    num_partitions = luigi.IntParameter(default=200)

    def output(self):
        return [
            maybe_gcs_target(f"{self.output_path}/predictions_topk.csv"),
            maybe_gcs_target(f"{self.output_path}/predictions_threshold.csv"),
        ]

    def run(self):
        with spark_resource() as spark:
            # data module
            data_module = Raster2VecClassifierDataModel(
                spark,
                self.input_path,
                self.feature_paths,
                self.feature_cols,
                use_idct=self.use_idct,
                batch_size=self.batch_size,
                num_partitions=self.num_partitions,
            )
            data_module.setup()

            model_cls = {
                "raster": RasterClassifier,
                "raster2vec": Raster2VecClassifier,
            }[self.model_name]

            model = model_cls.load_from_checkpoint(self.base_model)

            trainer = pl.Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                reload_dataloaders_every_n_epochs=1,
                default_root_dir=self.output_path,
            )
            predictions = trainer.predict(model, data_module)

            rows = []
            for batch in predictions:
                for surveyId, prediction in zip(
                    batch["surveyId"], batch["predictions"]
                ):
                    row = {"surveyId": int(surveyId)}
                    # get all the top 20 indices
                    indices = torch.argsort(prediction, descending=True)[:20]
                    row["predictions_topk"] = " ".join(
                        [str(x) for x in indices.tolist()]
                    )
                    row["predictions_sigmoid_topk"] = prediction[indices].tolist()
                    # get all the indices where value is greater than 0.5
                    indices = (prediction > 0.5).nonzero().flatten()
                    row["predictions_threshold"] = " ".join(
                        [str(x) for x in indices.tolist()]
                    )
                    rows.append(row)

            df = pd.DataFrame(rows).sort_values("surveyId")
            # rename predictions_topk to predictions
            df[["surveyId", "predictions_topk"]].rename(
                columns={"predictions_topk": "predictions"}
            ).to_csv(f"{self.output_path}/predictions_topk.csv", index=False)

            # same for predictions_threshold
            df[["surveyId", "predictions_threshold"]].rename(
                columns={"predictions_threshold": "predictions"}
            ).to_csv(f"{self.output_path}/predictions_threshold.csv", index=False)

            df.to_csv(f"{self.output_path}/predictions.csv", index=False)
            print(df.head())


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

        # download satellite imagery
        yield [
            RsyncGCSFiles(
                src_path=f"{self.remote_root}/processed/{name}",
                dst_path=f"{self.local_root}/processed/{name}",
            )
            for name in [
                "tiles/po/satellite",
                "tiles/pa-train/satellite",
                "tiles/pa-train/LandCover/LandCover_MODIS_Terra-Aqua_500m",
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
            # v5 - weights are only from the pa_train set
            # v6 - set pa_only to false, also reduce the number of train samples incorporated
            # v7 - autotuning
            # v8 - actually add the weights, set pa_only to true
            # v9 - ASL loss
            # v10 - ASL loss with sigmoid
            # v11 - no early stopping, try to reproduce v8
            # v12 - Hill loss
            # v13 - hill loss, but with batch norm because loss is nan
            # v14 - sigmoidf1 loss - not as great as I would have wanted
            # v15 - ASL loss again
            TrainMultiLabelClassifier(
                input_path=f"{self.local_root}/processed/metadata_clean/v2",
                output_path=f"{self.local_root}/models/multilabel_classifier/v15",
            ),
            # v1 - first model, 18it/s on epoch 0, 69it/s on epoch 1+
            # v2 - flatten the input
            # v3 - use 2d convolution
            # v4 - batch norm - necessary to get a loss
            # v5 - increase hidden layer size - very little effect
            # v6 - trying efficientnetv2 backbone now...
            #       this is pretty slow at 3it/s on epoch 0
            # v7 - idct and larger batch size
            # v8 - ASL loss
            # v9 - ASL loss using DCT coefficients
            # v10 - Hill loss using DCT coefficients
            #       there are too many knobs to turn
            # v11 - use efficientnet with coefficients
            # v12 - cnn with idct images (random rotations) - very good
            # v13 - remove augmentations - less good
            # v14 - add augmentations back, but don't augment the validation - even better
            # v15 - add another convolutional layer
            # v16 - use augmentations with dct coefficients, same as v14
            # v17 - use idct inside the model
            # v18 - v14 with SWA
            # TODO: v19 - add more coefficients for landcover/modis
            #       fixed a very serious bug in the data module that throws off validation results
            # v20 - repeat v18 with fixed bug
            # v21 - larger batch size 100 -> 250
            # v22 - use coefficients to learn
            # v23 - disable early stopping, disable swa
            TrainRasterClassifier(
                input_path=f"{self.local_root}/processed/metadata_clean/v2",
                feature_paths=[
                    f"{self.local_root}/processed/tiles/pa-train/satellite/v3",
                    # f"{self.local_root}/processed/tiles/pa-train/LandCover/LandCover_MODIS_Terra-Aqua_500m/v3",
                ],
                feature_cols=(
                    ["red", "green", "blue", "nir"]
                    # + [f"LandCover_MODIS_Terra-Aqua_500m_{i}" for i in [9, 10, 11]]
                ),
                output_path=f"{self.local_root}/models/raster_classifier/v23",
            ),
            # v24 - train for predictions
            TrainRasterClassifier(
                input_path=f"{self.local_root}/processed/metadata_clean/v2",
                feature_paths=[
                    f"{self.local_root}/processed/tiles/pa-*/satellite/v3",
                    # f"{self.local_root}/processed/tiles/pa-train/LandCover/LandCover_MODIS_Terra-Aqua_500m/v3",
                ],
                feature_cols=(
                    ["red", "green", "blue", "nir"]
                    # + [f"LandCover_MODIS_Terra-Aqua_500m_{i}" for i in [9, 10, 11]]
                ),
                output_path=f"{self.local_root}/models/raster_classifier/v24_rbgir_dct",
            ),
            # v1 - initial model
            # v2 - fix more bugs
            # v3 - use po dataset
            # v4 - use coefficient space and increase batch size
            # v5 - disable asl loss
            # v6 - redo, what happened?
            TrainRaster2Vec(
                batch_size=500,
                workers_count=16,
                input_path=f"{self.local_root}/processed/geolsh_graph/v1/edges/threshold=100000",
                feature_paths=[
                    f"{self.local_root}/processed/tiles/po/satellite/v3",
                ],
                feature_cols=["red", "green", "blue", "nir"],
                output_path=f"{self.local_root}/models/raster2vec/v6",
            ),
            # v1 - embedding with no asl loss
            TrainRaster2VecClassifier(
                input_path=f"{self.local_root}/processed/metadata_clean/v2",
                base_model=f"{self.local_root}/models/raster2vec/v6/checkpoints/last.ckpt",
                feature_paths=[
                    f"{self.local_root}/processed/tiles/pa-*/satellite/v3",
                ],
                feature_cols=["red", "green", "blue", "nir"],
                output_path=f"{self.local_root}/models/raster2vec_classifier/v1",
            ),
            # for the other model with asl
            TrainRaster2VecClassifier(
                input_path=f"{self.local_root}/processed/metadata_clean/v2",
                base_model=f"{self.local_root}/models/raster2vec/v4/checkpoints/last.ckpt",
                feature_paths=[
                    f"{self.local_root}/processed/tiles/pa-*/satellite/v3",
                ],
                feature_cols=["red", "green", "blue", "nir"],
                output_path=f"{self.local_root}/models/raster2vec_classifier_asl/v1",
            ),
        ]

        yield [
            PredictClassifier(
                input_path=f"{self.local_root}/processed/metadata_clean/v2",
                model_name="raster",
                base_model=f"{self.local_root}/models/raster_classifier/v24_rbgir_dct/checkpoints/last.ckpt",
                feature_paths=[
                    f"{self.local_root}/processed/tiles/pa-*/satellite/v3",
                ],
                feature_cols=["red", "green", "blue", "nir"],
                output_path=f"{self.local_root}/models/raster_classifier/v24_rbgir_dct_pred",
            ),
            PredictClassifier(
                input_path=f"{self.local_root}/processed/metadata_clean/v2",
                model_name="raster2vec",
                base_model=f"{self.local_root}/models/raster2vec_classifier/v1/checkpoints/last.ckpt",
                feature_paths=[
                    f"{self.local_root}/processed/tiles/pa-*/satellite/v3",
                ],
                feature_cols=["red", "green", "blue", "nir"],
                output_path=f"{self.local_root}/models/raster2vec_classifier/v1_pred",
            ),
            PredictClassifier(
                input_path=f"{self.local_root}/processed/metadata_clean/v2",
                model_name="raster2vec",
                base_model=f"{self.local_root}/models/raster2vec_classifier_asl/v1/checkpoints/last.ckpt",
                feature_paths=[
                    f"{self.local_root}/processed/tiles/pa-*/satellite/v3",
                ],
                feature_cols=["red", "green", "blue", "nir"],
                output_path=f"{self.local_root}/models/raster2vec_classifier_asl/v1_pred",
            ),
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
