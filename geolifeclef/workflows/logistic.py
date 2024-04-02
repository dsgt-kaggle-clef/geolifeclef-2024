from functools import reduce

import luigi
from pyspark.sql import functions as F

from geolifeclef.functions import get_projection_udf
from geolifeclef.utils import spark_resource

from .utils import maybe_gcs_target


class CleanMetadata(luigi.Task):
    """Generate a new metadata that we can use for training that only has the columns we want."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()

    def output(self):
        return maybe_gcs_target(f"{self.output_path}/_SUCCESS")

    def _load(self, spark):
        po_suffix = "PresenceOnlyOccurrences/GLC24-PO-metadata-train.csv"
        pa_train_suffix = "PresenceAbsenceSurveys/GLC24-PA-metadata-train.csv"
        pa_test_suffix = "PresenceAbsenceSurveys/GLC24-PA-metadata-test.csv"

        return [
            spark.read.csv(f"{self.input_path}/{suffix}", header=True, inferSchema=True)
            for suffix in [po_suffix, pa_train_suffix, pa_test_suffix]
        ]

    def _select(self, df, dataset):
        # projection to espg:32738 should be useful later down the line
        proj_udf = get_projection_udf()
        return df.withColumn("proj", proj_udf("lat", "lon")).select(
            F.lit(dataset).alias("dataset"),
            "surveyId",
            F.expr("proj.lat").alias("lat_proj"),
            F.expr("proj.lon").alias("lon_proj"),
            "lat",
            "lon",
            "year",
            "geoUncertaintyInM",
            (
                "speciesId"
                if "speciesId" in df.columns
                else F.lit(None).alias("speciesId")
            ),
        )

    def run(self):
        with spark_resource() as spark:
            # why use many variables when lexically-scoped do trick?
            (
                reduce(
                    lambda a, b: a.union(b),
                    [
                        self._select(df, dataset)
                        for df, dataset in zip(
                            self._load(spark),
                            ["po", "pa_train", "pa_test"],
                        )
                    ],
                )
                .orderBy("dataset", "surveyId")
                .repartition(8)
                .write.parquet(self.output_path, mode="overwrite")
            )


if __name__ == "__main__":
    luigi.build(
        [
            CleanMetadata(
                input_path="gs://dsgt-clef-geolifeclef-2024/data/downloaded/2024",
                output_path="gs://dsgt-clef-geolifeclef-2024/data/processed/metadata_clean",
            )
        ],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
        workers=1,
    )
