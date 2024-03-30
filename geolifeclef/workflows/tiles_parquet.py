"""Convert tif files into parquet files that contain tiles of the original image.

usage:
    python -m geolifeclef.workflows.tiles_parquet
"""

import itertools
import os
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path
import shutil

import luigi
import luigi.contrib.gcs
import pandas as pd
import torch
import tqdm
from scipy.fftpack import dctn

from geolifeclef.loaders.GLC23Datasets import PatchesDataset
from geolifeclef.loaders.GLC23PatchesProviders import (
    JpegPatchProvider,
    RasterPatchProvider,
)
from geolifeclef.utils import spark_resource

from .utils import RsyncGCSFiles


class BaseTilingTask(luigi.Task):
    input_path = luigi.Parameter()
    occurrences_path = luigi.Parameter()
    intermediate_path = luigi.Parameter()
    output_path = luigi.Parameter()

    batch_size = luigi.IntParameter(default=1000)
    num_workers = luigi.IntParameter(default=2)
    test_mode = luigi.BoolParameter(default=False)

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

    def load_dataset(self):
        raise NotImplementedError()

    def run(self):
        dataset = self.load_dataset()

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=(self.num_workers // 2) if self.num_workers > 1 else 1,
        )

        # Write the batch to parquet in many small fragments.
        # We put them into subfolders to avoid too many files in a single folder
        # parallelize writing batches. If we use starmap, we find that we run
        # into a strange memory leak issue.

        # testing code for limited number of batches
        if self.test_mode:
            iterable = itertools.islice(enumerate(dataloader), 4)
        else:
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

        # delete the intermediate files
        shutil.rmtree(self.intermediate_path)

        with self.output().open("w") as f:
            f.write("")


class RasterTilingTask(BaseTilingTask):
    def load_dataset(self):
        return PatchesDataset(
            occurrences=self.occurrences_path,
            providers=[RasterPatchProvider(self.input_path)],
        )


class JpegTilingTask(BaseTilingTask):
    def load_dataset(self):
        return PatchesDataset(
            occurrences=self.occurrences_path,
            providers=[
                JpegPatchProvider(
                    self.input_path,
                    normalize=False,
                )
            ],
        )


class ConsolidateParquet(luigi.Task):
    input_path = luigi.Parameter()
    input_type = luigi.ChoiceParameter(choices=["raster", "jpeg"])
    occurrences_path = luigi.Parameter()
    intermediate_path = luigi.Parameter()
    intermediate_remote_path = luigi.Parameter()
    output_path = luigi.Parameter()
    num_partitions = luigi.IntParameter(default=400)
    sync_local = luigi.BoolParameter(default=False)
    test_mode = luigi.BoolParameter(default=False)

    resources = {"max_workers": 1}

    def requires(self):
        tiling_task = {
            "raster": RasterTilingTask,
            "jpeg": JpegTilingTask,
        }[self.input_type]
        return tiling_task(
            input_path=self.input_path,
            occurrences_path=self.occurrences_path,
            intermediate_path=self.intermediate_path,
            output_path=self.intermediate_remote_path,
            test_mode=self.test_mode,
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
            **{"spark.sql.shuffle.partitions": self.num_partitions},
        ) as spark:
            df = spark.read.parquet(f"{self.intermediate_remote_path}/*/*.parquet")
            df.printSchema()
            print(f"row count: {df.count()}")
            df.coalesce(self.num_partitions).write.parquet(
                self.output_path, mode="overwrite"
            )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--test-mode", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    name = "tiles" if not args.test_mode else "tiles_test"
    version = "v3"
    raw_root = Path("/mnt/data")
    raster_tifs = Path("/mnt/data/raw/EnvironmentalRasters").glob("**/*.tif")

    raster_tifs = [
        p
        for p in raster_tifs
        # too many climatic rasters
        if ("Climatic_Monthly" not in p.as_posix())
        # leftover macos files
        and ("__MACOSX" not in p.as_posix())
        # nothing larger than 1gb
        and p.stat().st_size < 1e9
    ]
    luigi.build(
        [
            ConsolidateParquet(
                input_path=p.as_posix(),
                input_type="raster",
                occurrences_path="/mnt/data/downloaded/PresenceOnlyOccurrences/GLC24-PO-metadata-train.csv",
                intermediate_path=f"/mnt/data/intermediate/{name}/po/{p.parts[-2]}/{p.stem}/{version}",
                intermediate_remote_path=f"gs://dsgt-clef-geolifeclef-2024/data/intermediate/{name}/po/{p.parts[-2]}/{p.stem}/{version}",
                output_path=f"gs://dsgt-clef-geolifeclef-2024/data/processed/{name}/po/{p.parts[-2]}/{p.stem}/{version}",
                num_partitions=200 if not args.test_mode else 4,
                test_mode=args.test_mode,
            )
            for p in raster_tifs
        ]
        + [
            ConsolidateParquet(
                input_path="/mnt/data/raw/SatellitePatches",
                input_type="jpeg",
                occurrences_path="/mnt/data/downloaded/PresenceOnlyOccurrences/GLC24-PO-metadata-train.csv",
                intermediate_path=f"/mnt/data/intermediate/{name}/po/satellite/{version}",
                intermediate_remote_path=f"gs://dsgt-clef-geolifeclef-2024/data/intermediate/{name}/po/satellite/{version}",
                output_path=f"gs://dsgt-clef-geolifeclef-2024/data/processed/{name}/po/satellite/{version}",
                num_partitions=400 if not args.test_mode else 4,
                test_mode=args.test_mode,
            )
        ],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
        workers=4,
    )
