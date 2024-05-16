import numpy as np
import pytest

from geolifeclef.utils import spark_resource


@pytest.fixture
def spark():
    with spark_resource() as spark:
        yield spark


@pytest.fixture
def metadata_v2(spark, tmp_path):
    metadata_path = (tmp_path / "metadata").as_posix()
    df = spark.createDataFrame(
        [
            {
                "dataset": "pa_train",
                "surveyId": x,
                "lat_proj": 1,
                "lon_proj": 1,
                "speciesId": x % 3,
            }
            for x in range(10)
        ]
    )
    df.write.parquet(metadata_path)
    yield metadata_path


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
            for x in range(10)
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
            for x in range(10)
        ]
    )
    df.write.parquet(other_raster_path)

    return [raster_path, other_raster_path], ["red", "green", "blue", "nir", "other"]
