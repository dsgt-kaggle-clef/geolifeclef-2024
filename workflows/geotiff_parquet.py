"""Convert tif files into parquet files that contain tiles of the original image.

usage:
    python -m workflows.geotiff_parquet
"""

import os
import shutil
from pathlib import Path

import gdal2tiles
import luigi
import luigi.contrib.gcs

from geolifeclef.utils import spark_resource


# https://github.com/tehamalab/gdal2tiles
class GDALTilingTask(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    zoom = luigi.IntParameter(default=7)
    nb_processes = luigi.IntParameter(default=2)
    tmp_dir = luigi.Parameter(default="/mnt/data/tmp")

    def output(self):
        return luigi.LocalTarget((Path(self.output_path) / "_SUCCESS").as_posix())

    def run(self):
        # move things over into the tmp directory
        copy_input_path = Path(self.tmp_dir) / Path(self.input_path).name
        copy_input_path.parent.mkdir(parents=True, exist_ok=True)

        if self.input_path.startswith("gs://"):
            client = luigi.contrib.gcs.GCSClient()
            fp = client.download(self.input_path)
            copy_input_path.write_bytes(fp.read())
            fp.close()
        else:
            shutil.copy(self.input_path, copy_input_path)

        # now actually tile the
        gdal2tiles.generate_tiles(
            copy_input_path.as_posix(),
            self.output_path,
            zoom=self.zoom,
            nb_processes=self.nb_processes,
        )


class ParquetTask(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    partitions = luigi.IntParameter(default=8)

    def output(self):
        return luigi.contrib.gcs.GCSFlagTarget(f"{self.output_path}/")

    def run(self):
        with spark_resource() as spark:
            df = (
                spark.read.format("binaryFile")
                .option("pathGlobFilter", "*.png")
                .option("recursiveFileLookup", "true")
                .load(self.input_path)
            )
            df.repartition(self.partitions).write.parquet(self.output_path)


class Workflow(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    zoom = luigi.IntParameter(default=7)

    def run(self):
        tiling_task = GDALTilingTask(
            input_path=self.input_path,
            output_path=self.output_path,
            zoom=self.zoom,
        )
        yield tiling_task


if __name__ == "__main__":
    luigi.build(
        [
            Workflow(
                input_path="gs://dsgt-clef-geolifeclef-2024/data/raw/EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010/bio1.tif",
                output_path="/mnt/data/tmp/test_tiles",
            ),
        ],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
        workers=os.cpu_count(),
    )
