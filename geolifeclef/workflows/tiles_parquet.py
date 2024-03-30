"""Convert tif files into parquet files that contain tiles of the original image.

usage:
    python -m geolifeclef.workflows.tiles_parquet
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
from scipy.fftpack import dctn

from geolifeclef.loaders.GLC23Datasets import PatchesDataset
from geolifeclef.loaders.GLC23PatchesProviders import (
    JpegPatchProvider,
    MultipleRasterPatchProvider,
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
    num_workers = luigi.IntParameter(default=os.cpu_count() // 2)

    def output(self):
        return luigi.contrib.gcs.GCSTarget(f"{self.output_path}/_SUCCESS")

    def dctn_filter(self, tile, k=8):
        coeff = dctn(tile)
        coeff_subset = coeff[:k, :k]
        return coeff_subset.flatten()

    def write_batch(self, band_names, x, target, item, path):
        # n x k x 128 x 128 tensor
        if path.exists():
            return
        x = x.numpy()
        target = target.numpy()
        rows = []
        for i in range(x.shape[0]):
            zipped = list(zip(band_names, x[i]))
            row = {
                k: self.dctn_filter(v).reshape(-1).astype(float).tolist()
                for k, v in zipped
            }
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
        # 19 bioclimatic rasters
        p_bioclim = MultipleRasterPatchProvider(
            (
                input_path
                / "EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010"
            ).as_posix(),
        )
        # ignore climatic monthly for now because there are way too many of them
        # 1 elevation
        p_elevation = MultipleRasterPatchProvider(
            (input_path / "EnvironmentalRasters/Elevation").as_posix()
        )
        # human footprint detailed (14 rasters)
        p_hfp_d = MultipleRasterPatchProvider(
            (input_path / "EnvironmentalRasters/HumanFootprint/detailed").as_posix()
        )
        # human footprint summarized (2 rasters)
        p_hfp_s = MultipleRasterPatchProvider(
            (input_path / "EnvironmentalRasters/HumanFootprint/summarized").as_posix()
        )
        # 1 raster
        p_landcover = MultipleRasterPatchProvider(
            (input_path / "EnvironmentalRasters/LandCover").as_posix()
        )
        # 9 rasters
        p_soilgrids = MultipleRasterPatchProvider(
            (input_path / "EnvironmentalRasters/SoilGrids").as_posix()
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
                p_elevation,
                p_hfp_d,
                p_hfp_s,
                p_landcover,
                p_soilgrids,
                p_rgb,
            ],
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=(self.num_workers // 4) if self.num_workers > 1 else 1,
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
                src_path=self.intermediate_remote_path,
                dst_path=self.intermediate_path,
            )
            yield sync_down

        with spark_resource(
            **{"spark.sql.shuffle.partitions": self.num_partitions}
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
                intermediate_remote_path="gs://dsgt-clef-geolifeclef-2024/data/intermediate/tiles/v3",
                output_path="gs://dsgt-clef-geolifeclef-2024/data/processed/tiles/v3",
            ),
        ],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
        workers=1,
    )
