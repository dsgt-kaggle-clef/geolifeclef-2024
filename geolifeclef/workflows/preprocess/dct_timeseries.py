import os
from pathlib import Path

import faiss
import luigi
import luigi.contrib.gcs
import numpy as np
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql import types as T
from scipy.fft import dct
from tqdm import tqdm

from geolifeclef.utils import get_spark


class DCT_Timeseries(luigi.Task):
    timeseries_path = luigi.Parameter()
    output_path = luigi.Parameter()
    output_name = luigi.Parameter()
    dct_length = luigi.IntParameter(default=64)

    def output(self):
        return luigi.contrib.gcs.GCSTarget(f"{self.output_path}/_SUCCESS")

    def dct_op(self, x):
        x = np.array(x)
        for i in range(len(x) - 1, -1, -1):
            if x[i] != None:
                break
        x = x[: i + 1]
        x_dct = dct(x)
        dct_filter = x_dct.copy()
        dct_filter = dct_filter[: self.dct_length]
        return dct_filter.tolist()

    def run(self):
        spark = get_spark()
        df = spark.read.csv(self.timeseries_path, header=True, inferSchema=True)

        columns = [
            F.col(column_name)
            for column_name in df.columns
            if column_name != "surveyId"
        ]
        df = df.withColumn("TimeSeries", F.array(columns))
        df = df.select("surveyId", "TimeSeries")
        dct_udf = F.udf(self.dct_op, T.ArrayType(T.FloatType()))
        df = df.withColumn("DCT", dct_udf(F.col("TimeSeries")))

        df.write.parquet(
            os.path.join(self.output_path, self.output_name), mode="overwrite"
        )
        with self.output().open("w") as f:
            f.write("")


class CombineDCTResults(luigi.Task):
    timeseries_paths = luigi.ListParameter()
    metadata_path = luigi.Parameter(
        default="gs://dsgt-clef-geolifeclef-2024/data/processed/metadata_clean/v1"
    )
    output_path = luigi.Parameter()
    output_name = luigi.Parameter()

    def requires(self):
        tasks = []
        for path in self.timeseries_paths:
            name = os.path.basename(path).split(".")[0]
            output_path = os.path.join(self.output_path, name)
            tasks.append(
                DCT_Timeseries(
                    timeseries_path=path, output_path=output_path, output_name=name
                )
            )
        return tasks

    def output(self):
        return luigi.contrib.gcs.GCSTarget(
            f"{self.output_path}/{self.output_name}/_SUCCESS"
        )

    def run(self):
        spark = get_spark()

        combined_df = None

        for task in self.input():
            name = os.path.basename(os.path.dirname(task.path))
            parquet_path = os.path.join(os.path.dirname(task.path), name)
            df = spark.read.parquet(parquet_path)

            for col in df.columns:
                if col.startswith(
                    "DCT"
                ):  # Adjusted based on your column naming convention
                    df = df.withColumnRenamed(col, f"ts_{name.split('-')[-1]}")

            if combined_df is None:
                combined_df = df
            else:
                combined_df = combined_df.join(
                    df.select("surveyId", f"ts_{name.split('-')[-1]}"),
                    on="surveyId",
                    how="outer",
                )

        metadata_df = spark.read.parquet(self.metadata_path).filter(
            F.col("dataset") != "po"
        )

        # Add lon and lat to combined_df by joining with metadata_df
        combined_df_with_coords = combined_df.join(
            metadata_df.select("surveyId", "lon", "lat"), on="surveyId", how="left"
        )

        # Find unmatched rows
        unmatched_df = metadata_df.join(
            combined_df, on="surveyId", how="left_anti"
        ).select(
            metadata_df.surveyId.alias("unmatched_surveyId"),
            metadata_df.lon.alias("unmatched_lon"),
            metadata_df.lat.alias("unmatched_lat"),
        )
        unmatched_df = unmatched_df.sample(False, 0.001)
        # Convert combined_df_with_coords to Pandas DataFrame
        combined_pdf = combined_df_with_coords.select(
            "surveyId", "lon", "lat"
        ).collect()
        combined_coords = np.array(
            [(row["lon"], row["lat"]) for row in combined_pdf], dtype="float32"
        )
        combined_survey_ids = np.array([row["surveyId"] for row in combined_pdf])

        # Create FAISS index
        d = 2  # dimension (lon, lat)
        index = faiss.IndexFlatL2(d)  # L2 distance (Euclidean)
        index.add(combined_coords)  # add vectors to the index

        # Initialize tqdm progress bar
        total_unmatched = unmatched_df.count()
        progress_bar = tqdm(total=total_unmatched, desc="Processing unmatched records")

        # Perform vector search
        nearest_neighbors = []
        for row in unmatched_df.collect():
            query = np.array(
                [[row["unmatched_lon"], row["unmatched_lat"]]], dtype="float32"
            )
            D, I = index.search(query, 1)
            nearest_survey_id = combined_survey_ids[I[0][0]].item()
            nearest_neighbors.append((row["unmatched_surveyId"], nearest_survey_id))
            progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()
        schema = T.StructType(
            [
                T.StructField("surveyId", T.IntegerType(), True),
                T.StructField("closest_surveyId", T.IntegerType(), True),
            ]
        )
        # Create DataFrame for nearest neighbors
        nearest_neighbors_df = spark.createDataFrame(nearest_neighbors, schema)
        nearest_neighbors_df.show()
        # Join nearest_neighbors_df with combined_df to get the nearest neighbors with time series data
        nearest_combined_df = nearest_neighbors_df.join(
            combined_df.withColumnRenamed("surveyId", "CsurveyId"),
            nearest_neighbors_df.closest_surveyId == F.col("CsurveyId"),
            "left",
        ).drop("closest_surveyId", "CsurveyId")
        nearest_combined_df.show()
        # Union combined_df with nearest_combined_df
        final_df = combined_df.unionByName(
            nearest_combined_df, allowMissingColumns=True
        )

        final_df.write.parquet(
            os.path.join(self.output_path, self.output_name), mode="overwrite"
        )

        with self.output().open("w") as f:
            f.write("")

        spark.stop()


if __name__ == "__main__":
    filenames = [
        "data/downloaded/2024/SatelliteTimeSeries/GLC24-PO-train-landsat-time-series-blue.csv",
        "data/downloaded/2024/SatelliteTimeSeries/GLC24-PO-train-landsat-time-series-green.csv",
        "data/downloaded/2024/SatelliteTimeSeries/GLC24-PO-train-landsat-time-series-nir.csv",
        "data/downloaded/2024/SatelliteTimeSeries/GLC24-PO-train-landsat-time-series-red.csv",
        "data/downloaded/2024/SatelliteTimeSeries/GLC24-PO-train-landsat-time-series-swir1.csv",
        "data/downloaded/2024/SatelliteTimeSeries/GLC24-PO-train-landsat-time-series-swir2.csv",
    ]

    prefix = "gs://dsgt-clef-geolifeclef-2024"
    timeseries_paths = [os.path.join(prefix, filename) for filename in filenames]

    output_path = "gs://dsgt-clef-geolifeclef-2024/data/processed/dct_timeseries"
    output_name = "combined_timeseries_v4"

    combine_task = CombineDCTResults(
        timeseries_paths=timeseries_paths,
        output_path=output_path,
        output_name=output_name,
    )

    luigi.build(
        [combine_task],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
        workers=1,
    )
