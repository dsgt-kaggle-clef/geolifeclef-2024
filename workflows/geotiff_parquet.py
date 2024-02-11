import os
from pathlib import Path

import luigi
import rasterio
from luigi.contrib.external_program import ExternalProgramTask
from pyspark.sql import Row
from pyspark.sql.types import (
    BinaryType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from rasterio.windows import Window

from geolifeclef.utils import get_spark


class LoadDataset(ExternalProgramTask):
    script_path = "script/download_dataset.sh"
    url = luigi.Parameter(
        default="gs://dsgt-clef-geolifeclef-2024/data/raw/EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010"
    )
    download_path = luigi.Parameter(default="/mnt/data/download")

    def program_args(self):
        return [self.script_path, self.url, self.download_path]


class PreProcess(ExternalProgramTask):
    script_path = "script/preprocess.sh"
    tile_size = luigi.IntParameter(default=256)
    process_path = luigi.Parameter(default="/mnt/data/processed")
    download_path = luigi.Parameter(default="/mnt/data/download")

    def requires(self):
        return [LoadDataset(download_path=self.download_path)]

    def program_args(self):
        return [
            self.script_path,
            self.process_path,
            self.download_path,
            str(self.tile_size),
            str(self.tile_size),
        ]


class ParquetConversion(luigi.Task):
    output_path = luigi.Parameter(default="paraquet_local")
    process_dir = luigi.Parameter(default="/mnt/data/processed")

    def requires(self):
        return [PreProcess()]

    def read_tiled_tiff(self, tiff_path):
        with rasterio.open(tiff_path) as src:
            print(src.meta)
            for band in range(src.meta["count"]):
                for _, window in src.block_windows(band):
                    data = src.read(window=window)
                    binary_data = data.tobytes()

                    tile_x, tile_y = window.col_off, window.row_off
                    tile_x, tile_y = int(tile_x), int(tile_y)
                    yield tile_x, tile_y, binary_data, (band + 1)

    def run(self):
        spark = get_spark()
        schema = StructType(
            [
                StructField("x", IntegerType(), True),
                StructField("y", IntegerType(), True),
                StructField("filename", StringType(), True),
                StructField("band", IntegerType(), True),
                StructField("binary_data", BinaryType(), True),
            ]
        )

        # Assuming you have a list of TIFF paths
        process_dir = Path(self.process_dir)
        rows = []
        for path in process_dir.glob("*.tif"):
            for x, y, data, band in self.read_tiled_tiff(path):
                rows.append(
                    Row(
                        x=x,
                        y=y,
                        filename=os.path.basename(path),
                        band=band,
                        binary_data=data,
                    )
                )
        df = spark.createDataFrame(rows, schema)
        df.write.parquet(self.output_path)


if __name__ == "__main__":
    luigi.run(["ParquetConversion", "--local-scheduler"])
