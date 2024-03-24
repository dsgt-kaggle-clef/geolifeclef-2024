"""Convert tif files into parquet files that contain tiles of the original image.

usage:
    python -m workflows.geotiff_parquet
"""

import os
from pathlib import Path

import luigi
import luigi.contrib.gcs
import pandas as pd
import torch

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

    def output(self):
        return luigi.LocalTarget((Path(self.output_path) / "_SUCCESS").as_posix())

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
            providers=[p_bioclim, p_hfp_s, p_rgb],
        )

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

        # write the batch to parquet in many small fragments
        # we put them into subfolders to avoid too many files in a single folder
        for i, (x, target, item) in enumerate(dataloader):
            if i > 10:
                break
            # n x k x 128 x 128 tensor
            x = x.numpy()
            target = target.numpy()
            rows = []
            for j in range(x.shape[0]):
                zipped = list(zip(dataset.provider.bands_names, x[i]))
                row = {k: v[j].astype(float).tolist() for k, v in zipped}
                row.update({k: v[j].item() for k, v in item.items()})
                row.update({"target": int(target[j])})
                rows.append(row)
            df = pd.DataFrame(rows)
            subfolder = Path(self.output_path) / f"{i%self.batch_size:04d}"
            subfolder.mkdir(parents=True, exist_ok=True)
            path = subfolder / f"{i:06d}.parquet"
            df.to_parquet(path, index=False)

        with self.output().open("w") as f:
            f.write("")


if __name__ == "__main__":
    luigi.build(
        [
            TilingTask(
                input_path="/mnt/data/raw",
                meta_path="/mnt/data/downloaded",
                output_path="/mnt/data/intermediate/tiles",
            ),
        ],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
        workers=os.cpu_count(),
    )
