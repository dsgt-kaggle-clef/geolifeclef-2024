import numpy as np
import pytest

from geolifeclef.utils import spark_resource


@pytest.fixture
def spark(tmp_path):
    with spark_resource(local_dir=tmp_path.as_posix()) as spark:
        yield spark


@pytest.fixture
def metadata_v2(spark, tmp_path):
    metadata_path = (tmp_path / "metadata").as_posix()
    df = spark.createDataFrame(
        [
            {
                "dataset": ["pa_train", "pa_test"][x % 2],
                "surveyId": x,
                "lat_proj": 1,
                "lon_proj": 1,
                "speciesId": x % 3,
            }
            for x in range(20)
        ]
    )
    df.write.parquet(metadata_path)
    yield metadata_path


@pytest.fixture
def geolsh_graph_v1(spark, tmp_path):
    graph_path = (tmp_path / "graph").as_posix()
    df = spark.createDataFrame(
        [
            {
                "srcDataset": "po",
                "srcSurveyId": x,
                "srcSpeciesId": x % 3,
                "dstDataset": "po",
                "dstSurveyId": (x + 1) % 10,
                "dstSpeciesId": (x + 1) % 3,
            }
            for x in range(10)
        ]
    )
    df.write.parquet(graph_path)
    yield graph_path


@pytest.fixture
def raster_features(spark, tmp_path):
    raster_path = (tmp_path / "raster").as_posix()
    df = spark.createDataFrame(
        [
            {
                "surveyId": x,
                "red": np.ones(64).tolist(),
                "green": np.ones(64).tolist(),
                "blue": np.ones(64).tolist(),
                "nir": np.ones(64).tolist(),
            }
            for x in range(20)
        ]
    )
    df.write.parquet(raster_path)

    other_raster_path = (tmp_path / "raster2").as_posix()
    df = spark.createDataFrame(
        [
            {
                "surveyId": x,
                "other": np.ones(64).tolist(),
            }
            for x in range(20)
        ]
    )
    df.write.parquet(other_raster_path)

    return [raster_path, other_raster_path], ["red", "green", "blue", "nir", "other"]
