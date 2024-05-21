import pytest
import pytorch_lightning as pl
import torch

from geolifeclef.torch.raster2vec.data import Raster2VecDataModel
from geolifeclef.torch.raster2vec.model import Raster2Vec


def test_raster_data_model(tmp_path, spark, geolsh_graph_v1, raster_features):
    print(geolsh_graph_v1)
    raster_feature_path, raster_feature_col = raster_features
    dm = Raster2VecDataModel(
        spark,
        geolsh_graph_v1,
        raster_feature_path,
        raster_feature_col,
        sample=1.0,
    )
    dm.setup()
    batch = next(dm.train_dataloader())
    print(batch)
    assert set(batch.keys()) == {"features", "label"}
    # check that the features dictionary has anchor, neighbor, and distant keys
    assert set(batch["features"].keys()) == {"anchor", "neighbor", "distant"}
    # check labels also have the same properties
    assert set(batch["label"].keys()) == {"anchor", "neighbor", "distant"}


# run this both gpu and cpu, but only the gpu if it's available
# pytest parametrize
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_raster_classifier_validation_model(
    tmp_path, spark, geolsh_graph_v1, raster_features, device
):
    if device == "gpu" and not torch.cuda.is_available():
        pytest.skip()

    raster_feature_path, raster_feature_col = raster_features
    data_module = Raster2VecDataModel(
        spark,
        geolsh_graph_v1,
        raster_feature_path,
        raster_feature_col,
        batch_size=3,
        sample=1.0,
    )
    data_module.setup()

    # get parameters for the model
    num_layers, num_features, num_classes = data_module.get_shape()
    model = Raster2Vec(num_layers, num_features, num_classes)

    trainer = pl.Trainer(
        accelerator=device,
        default_root_dir=tmp_path,
        fast_dev_run=True,
    )
    trainer.fit(model, data_module)
