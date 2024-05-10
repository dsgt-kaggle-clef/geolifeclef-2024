import pytest
import pytorch_lightning as pl
import torch

from geolifeclef.torch.multilabel.data import GeoSpatialDataModel, ToSparseTensor
from geolifeclef.torch.multilabel.model import MultiLabelClassifier


def test_to_sparse_tensor():
    num_classes = 5
    dense = torch.tensor(
        [
            [1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 0, 0],
        ]
    )
    test_data = {
        "features": torch.ones(4, 2),
        "label": dense,
    }

    transform = ToSparseTensor()
    result = transform(test_data)
    assert result["label"].shape == (4, num_classes)
    assert (dense - result["label"]).sum() == 0


def test_geospatial_data_model(tmp_path, spark, metadata_v2):
    print(metadata_v2)
    dm = GeoSpatialDataModel(spark, metadata_v2)
    dm.setup()
    batch = next(dm.train_dataloader())
    assert set(batch.keys()) == {"features", "label"}
    assert batch["label"].device.type == "cpu"


# run this both gpu and cpu, but only the gpu if it's available
# pytest parametrize
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_classifier_validation_model(tmp_path, spark, metadata_v2, device):
    if device == "gpu" and not torch.cuda.is_available():
        pytest.skip()

    data_module = GeoSpatialDataModel(spark, metadata_v2)
    data_module.setup()

    # get parameters for the model
    row = data_module.train_data.first()
    num_features = int(len(row.features))
    num_classes = int(len(row.label))
    model = MultiLabelClassifier(num_features, num_classes)

    trainer = pl.Trainer(
        accelerator=device,
        default_root_dir=tmp_path,
        fast_dev_run=True,
    )
    trainer.fit(model, data_module)
