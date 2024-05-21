import pytest
import pytorch_lightning as pl
import torch

from geolifeclef.torch.raster.data import RasterDataModel
from geolifeclef.torch.raster.model import RasterClassifier


def test_raster_data_model(tmp_path, spark, metadata_v2, raster_features):
    print(metadata_v2)
    raster_feature_path, raster_feature_col = raster_features
    dm = RasterDataModel(spark, metadata_v2, raster_feature_path, raster_feature_col)
    dm.setup()
    batch = next(dm.train_dataloader())
    assert set(batch.keys()) == {"features", "label"}
    assert batch["label"].device.type == "cpu"


# run this both gpu and cpu, but only the gpu if it's available
# pytest parametrize
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_raster_classifier_validation_model(
    tmp_path, spark, metadata_v2, raster_features, device
):
    if device == "gpu" and not torch.cuda.is_available():
        pytest.skip()

    raster_feature_path, raster_feature_col = raster_features
    data_module = RasterDataModel(
        spark, metadata_v2, raster_feature_path, raster_feature_col, batch_size=2
    )
    data_module.setup()

    # get parameters for the model
    num_layers, num_features, num_classes = data_module.get_shape()
    model = RasterClassifier(num_layers, num_features, num_classes)

    trainer = pl.Trainer(
        accelerator=device,
        default_root_dir=tmp_path,
        fast_dev_run=True,
    )
    trainer.fit(model, data_module)
