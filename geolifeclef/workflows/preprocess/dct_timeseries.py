import os
from pathlib import Path

import luigi
import luigi.contrib.gcs
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql import types as T
from scipy.fft import dct

from geolifeclef.utils import get_spark


class DCT_Timeseries(luigi.Task):
    timeseries_path = luigi.Parameter()
    output_path = luigi.Parameter()
    output_name = luigi.Parameter()

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
        dct_filter = dct_filter[:8]
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
    tasks_args = []
    for filename in filenames:
        args = {}
        args["timeseries_path"] = os.path.join(prefix, filename)
        name = Path(filename).stem
        args["output_path"] = os.path.join(
            "gs://dsgt-clef-geolifeclef-2024/data/processed/dct_timeseries", name
        )
        args["output_name"] = name
        tasks_args.append(args)
    tasks = []
    for args in tasks_args:
        tasks.append(
            DCT_Timeseries(
                timeseries_path=args["timeseries_path"],
                output_path=args["output_path"],
                output_name=args["output_name"],
            )
        )

    luigi.build(
        tasks,
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
        workers=1,
    )
