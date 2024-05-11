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
