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
import os

from geolifeclef.loaders.GLC23Datasets import PatchesDataset
from geolifeclef.loaders.GLC23PatchesProviders import (
    JpegPatchProvider,
    MultipleRasterPatchProvider,
    RasterPatchProvider,
)
from geolifeclef.utils import spark_resource
from .utils import RsyncGCSFiles
import itertools


class TilingTask(luigi.Task):
    input_path = luigi.Parameter()
    meta_path = luigi.Parameter()
    intermediate_path = luigi.Parameter()
    output_path = luigi.Parameter()

    batch_size = luigi.IntParameter(default=100)
    num_workers = luigi.IntParameter(default=os.cpu_count())

    def output(self):
        return luigi.contrib.gcs.GCSTarget(f"{self.output_path}/_SUCCESS")

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

        # testing code for limited number of batches
        # iterable = itertools.islice(enumerate(dataloader),4)
        iterable = enumerate(dataloader)
        with Pool(self.num_workers) as p:
            for _ in p.imap(
                self.mapfn,
                (
                    (
                        dataset.provider.bands_names,
                        x,
                        target,
                        item,
                        Path(self.intermediate_path)
                        / f"{i//self.batch_size:08d}"
                        / f"{i:08d}.parquet",
                    )
                    for i, (x, target, item) in tqdm.tqdm(
                        iterable, total=len(dataloader)
                    )
                ),
            ):
                pass

        sync_up = RsyncGCSFiles(
            src_path=self.intermediate_path,
            dst_path=self.output_path,
        )
        yield sync_up

        with self.output().open("w") as f:
            f.write("")


class ConsolidateParquet(luigi.Task):
    input_path = luigi.Parameter()
    meta_path = luigi.Parameter()
    intermediate_path = luigi.Parameter()
    intermediate_remote_path = luigi.Parameter()
    output_path = luigi.Parameter()
    num_partitions = luigi.IntParameter(default=400)
    memory = luigi.Parameter(default="10g")
    sync_local = luigi.BoolParameter(default=False)

    def requires(self):
        return TilingTask(
            input_path=self.input_path,
            meta_path=self.meta_path,
            intermediate_path=self.intermediate_path,
            output_path=self.intermediate_remote_path,
        )

    def output(self):
        return luigi.contrib.gcs.GCSTarget(f"{self.output_path}/_SUCCESS")

    def run(self):
        if self.sync_local:
            # and then sync it back down to the local filesystem
            sync_down = RsyncGCSFiles(
                src_path=sync_up.dst_path,
                dst_path=sync_up.src_path,
            )
            yield sync_down

        with spark_resource(
            cores=os.cpu_count(),
            memory=self.memory,
            **{"spark.sql.shuffle.partitions": self.num_partitions},
        ) as spark:
            df = spark.read.parquet(f"{self.intermediate_remote_path}/*/*.parquet")
            df.printSchema()
            print(f"row count: {df.count()}")
            df.coalesce(self.num_partitions).write.parquet(
                self.output_path, mode="overwrite"
            )


if __name__ == "__main__":
    luigi.build(
        [
            ConsolidateParquet(
                input_path="/mnt/data/raw",
                meta_path="/mnt/data/downloaded",
                intermediate_path="/mnt/data/intermediate/tiles",
                intermediate_remote_path="gs://dsgt-clef-geolifeclef-2024/data/intermediate/tiles/v2",
                output_path="gs://dsgt-clef-geolifeclef-2024/data/processed/tiles/v2",
                memory="24g",
            ),
        ],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
        workers=1,
    )
