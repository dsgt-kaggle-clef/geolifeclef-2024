"""Convert tif files into parquet files that contain tiles of the original image.

usage:
    python -m workflows.geotiff_parquet
"""

import os
from multiprocessing import Pool
from pathlib import Path

import luigi
import luigi.contrib.gcs
import pandas as pd
import torch
import tqdm

from geolifeclef.loaders.GLC23Datasets import PatchesDataset
from geolifeclef.loaders.GLC23PatchesProviders import (
    JpegPatchProvider,
    MultipleRasterPatchProvider,
    RasterPatchProvider,
)
from geolifeclef.utils import spark_resource


class TilingTask(luigi.Task):
    input_path = luigi.Parameter()
    meta_path = luigi.Parameter()
    output_path = luigi.Parameter()
    batch_size = luigi.IntParameter(default=100)
    num_workers = luigi.IntParameter(default=os.cpu_count())

    def output(self):
        return luigi.LocalTarget((Path(self.output_path) / "_SUCCESS").as_posix())

    def write_batch(self, band_names, x, target, item, path):
        # n x k x 128 x 128 tensor
        if path.exists():
            return
        x = x.numpy()
        target = target.numpy()
        rows = []
        for i in range(x.shape[0]):
            zipped = list(zip(band_names, x[i]))
            row = {k: v.reshape(-1).astype(float).tolist() for k, v in zipped}
            row.update({k: v[i].item() for k, v in item.items()})
            row.update({"target": int(target[i])})
            rows.append(row)
        df = pd.DataFrame(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    def mapfn(self, args):
        self.write_batch(*args)

    def run(self):
        input_path = Path(self.input_path)
        meta_path = Path(self.meta_path)
        # take only bio1 and bio2 from bioclimatic rasters (2 rasters from the 3 in the folder)
        p_bioclim = MultipleRasterPatchProvider(
            (
                input_path
                / "EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010"
            ).as_posix(),
            select=["bio1", "bio2"],
        )
        # take the human footprint 2009 summarized raster (a single raster)
        p_hfp_s = RasterPatchProvider(
            (
                input_path
                / "EnvironmentalRasters/HumanFootprint/summarized/HFP2009_WGS84.tif"
            ).as_posix()
        )
        # take all sentinel imagery layers (r,g,b,nir = 4 layers)
        p_rgb = JpegPatchProvider(
            (input_path / "SatellitePatches/").as_posix(),
            # very inefficient normalization in the data loading library
            normalize=False,
        )

        dataset = PatchesDataset(
            occurrences=(
                meta_path / "PresenceOnlyOccurrences/GLC24-PO-metadata-train.csv"
            ).as_posix(),
            providers=[
                p_bioclim,
                p_hfp_s,
                p_rgb,
            ],
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers // 4,
        )

        # Write the batch to parquet in many small fragments.
        # We put them into subfolders to avoid too many files in a single folder
        # parallelize writing batches. If we use starmap, we find that we run
        # into a strange memory leak issue.

        with Pool(self.num_workers) as p:
            for _ in p.imap(
                self.mapfn,
                (
                    (
                        dataset.provider.bands_names,
                        x,
                        target,
                        item,
                        Path(self.output_path)
                        / f"{i//self.batch_size:04d}"
                        / f"{i:06d}.parquet",
                    )
                    for i, (x, target, item) in tqdm.tqdm(
                        enumerate(dataloader), total=len(dataloader)
                    )
                ),
            ):
                pass

        with self.output().open("w") as f:
            f.write("")


class ConsolidateParquet(luigi.Task):
    input_path = luigi.Parameter()
    meta_path = luigi.Parameter()
    intermediate_path = luigi.Parameter()
    output_path = luigi.Parameter()
    num_partitions = luigi.IntParameter(default=100)

    def requires(self):
        return TilingTask(
            input_path=self.input_path,
            meta_path=self.meta_path,
            output_path=self.intermediate_path,
        )

    def output(self):
        return luigi.contrib.gcs.GCSFlagTarget(f"{self.output_path}/")

    def run(self):
        with spark_resource() as spark:
            df = spark.read.parquet(
                Path(self.intermediate_path).as_posix() + "/*/*.parquet"
            )
            df.printSchema()
            df.repartition(self.num_partitions).write.parquet(
                self.output_path, mode="overwrite"
            )


if __name__ == "__main__":
    luigi.build(
        [
            ConsolidateParquet(
                input_path="/mnt/data/raw",
                meta_path="/mnt/data/downloaded",
                intermediate_path="/mnt/data/intermediate/tiles",
                output_path="gs://dsgt-clef-geolifeclef-2024/data/processed/tiles/v1",
            ),
        ],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
        workers=1,
    )