import luigi
import luigi.contrib.gcs
from pyspark.sql import functions as F

from geolifeclef.utils import spark_resource


class SampleRandomTask(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    sample_size = luigi.IntParameter(default=10_000)

    def output(self):
        return luigi.contrib.gcs.GCSTarget(f"{self.output_path}/_SUCCESS")

    def run(self):
        with spark_resource(
            **{"spark.sql.parquet.enableVectorizedReader": False}
        ) as spark:
            df = spark.read.parquet(self.input_path)
            # 5m divided by 100 is 50k
            sample = df.sample(fraction=0.01).limit(self.sample_size)
            sample.write.parquet(self.output_path, mode="overwrite")


class FilterTopSpeciesTask(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    top_k = luigi.IntParameter(default=10)

    def output(self):
        return luigi.contrib.gcs.GCSTarget(f"{self.output_path}/_SUCCESS")

    def run(self):
        with spark_resource(
            **{"spark.sql.parquet.enableVectorizedReader": False}
        ) as spark:
            df = spark.read.parquet(self.input_path)
            # this could potentially be impossibly slow
            top_species = (
                df.groupBy("target")
                .count()
                .orderBy(F.desc("count"))
                .limit(self.top_k)
                .select("target")
            ).cache()
            subset = df.join(F.broadcast(top_species.select("target")), on="target")
            subset.write.parquet(self.output_path, mode="overwrite")


if __name__ == "__main__":
    luigi.build(
        [
            FilterTopSpeciesTask(
                input_path="gs://dsgt-clef-geolifeclef-2024/data/processed/tiles/v2",
                output_path="gs://dsgt-clef-geolifeclef-2024/data/processed/sample_tiles/species_v1",
            ),
            SampleRandomTask(
                input_path="gs://dsgt-clef-geolifeclef-2024/data/processed/tiles/v2",
                output_path="gs://dsgt-clef-geolifeclef-2024/data/processed/sample_tiles/random_v1",
            ),
        ],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
        workers=1,
    )
