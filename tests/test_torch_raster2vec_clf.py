import pandas as pd
import pytest
import pytorch_lightning as pl
import torch

from geolifeclef.torch.raster2vec.data import Raster2VecDataModel
from geolifeclef.torch.raster2vec.model import Raster2Vec
from geolifeclef.torch.raster2vec_clf.data import Raster2VecClassifierDataModel
from geolifeclef.torch.raster2vec_clf.model import Raster2VecClassifier


@pytest.fixture
def raster2vec_model(tmp_path, spark, geolsh_graph_v1, raster_features):
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
        accelerator="cpu",
        default_root_dir=tmp_path,
        fast_dev_run=True,
    )
    trainer.fit(model, data_module)

    output_path = tmp_path / "raster2vec/checkpoints/last.ckpt"
    trainer.save_checkpoint(output_path)

    # assert that the file exists
    return output_path


def test_raster2vec(raster2vec_model):
    assert raster2vec_model.exists()


def test_raster2vec_clf_data_model(spark, metadata_v2, raster_features):
    raster_feature_path, raster_feature_col = raster_features
    dm = Raster2VecClassifierDataModel(
        spark,
        metadata_v2,
        raster_feature_path,
        raster_feature_col,
    )
    dm.setup()
    batch = next(dm.train_dataloader())
    print(batch)
    assert set(batch.keys()) == {"features", "label", "surveyId"}


# run this both gpu and cpu, but only the gpu if it's available
# pytest parametrize
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_raster2vec_clf_validate_model(
    tmp_path, spark, metadata_v2, raster_features, device, raster2vec_model
):
    if device == "gpu" and not torch.cuda.is_available():
        pytest.skip()

    raster_feature_path, raster_feature_col = raster_features
    data_module = Raster2VecClassifierDataModel(
        spark,
        metadata_v2,
        raster_feature_path,
        raster_feature_col,
        batch_size=3,
    )
    data_module.setup()

    # load the pretrained model
    backbone = Raster2Vec.load_from_checkpoint(raster2vec_model)

    num_layers, num_features, num_classes = data_module.get_shape()
    model = Raster2VecClassifier(num_layers, num_features, num_classes)
    model.model = backbone.model

    trainer = pl.Trainer(
        accelerator=device,
        default_root_dir=tmp_path,
        fast_dev_run=True,
    )
    trainer.fit(model, data_module)

    # also test the predictions
    predictions = trainer.predict(model, data_module)

    rows = []
    for batch in predictions:
        for surveyId, prediction in zip(batch["surveyId"], batch["predictions"]):
            row = {"surveyId": int(surveyId)}
            # get all the indices where value is greater than 0.5
            indices = torch.where(prediction > 0.5)
            row["predictions"] = " ".join(indices[0].tolist())
            rows.append(row)
    df = pd.DataFrame(rows).sort_values("surveyId")
    # write csv
    output_path = tmp_path / "predictions.csv"
    df.to_csv(output_path, index=False)

    assert set(df.columns) == {"surveyId", "predictions"}
    print(output_path)
    print(output_path.read_text())
