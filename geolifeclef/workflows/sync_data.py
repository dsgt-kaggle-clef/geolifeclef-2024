"""Download the dataset from GCS and unzip the files in the data directory.

This script prepares analysis for using the spatial rasters using the included dataloaders.
"""

import os
import tempfile
from pathlib import Path
from textwrap import dedent

import luigi
from luigi.contrib.external_program import ExternalProgramTask


class RsyncGCSFiles(ExternalProgramTask):
    """Download using the gcloud command-line tool."""

    remote_path = luigi.Parameter()
    local_path = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget((Path(self.local_path) / "_SUCCESS").as_posix())

    def program_args(self):
        """Use gcloud to download the files."""
        script_text = dedent(
            f"""
            #!/bin/bash
            set -eux -o pipefail
            gcloud storage rsync -r {self.remote_path} {self.local_path}
            touch {self.output().path}
            """
        ).strip()
        script_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        script_file.write(script_text)
        script_file.close()
        print(f"Script file: {script_file.name}")
        print(script_text)
        return ["/bin/bash", script_file.name]


class UnzipFiles(ExternalProgramTask):
    """Unzip the files in the data directory."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()

    def output(self):
        stem = Path(self.input_path).stem
        return luigi.LocalTarget(
            (Path(self.output_path) / f"_SUCCESS.{stem}").as_posix()
        )

    def program_args(self):
        """Use pigz to unzip the files."""
        # generate a shell script in a temporary file
        parent = Path(self.output_path)
        script_text = dedent(
            f"""
            #!/bin/bash
            set -eux -o pipefail
            mkdir -p {parent}
            unzip -q -o {self.input_path} -d {parent}
            touch {self.output().path}
            """
        ).strip()
        script_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        script_file.write(script_text)
        script_file.close()
        print(f"Script file: {script_file.name}")
        print(script_text)
        return ["/bin/bash", script_file.name]


class Workflow(luigi.Task):
    input_path = luigi.Parameter()
    intermediate_path = luigi.Parameter()
    output_path = luigi.Parameter()

    def requires(self):
        return RsyncGCSFiles(
            remote_path=self.input_path,
            local_path=self.intermediate_path,
        )

    def output(self):
        return luigi.LocalTarget((Path(self.output_path) / "_SUCCESS").as_posix())

    def run(self):
        paths = Path(self.intermediate_path).glob("*/*.zip")
        unzip_tasks = []
        for path in paths:
            unzip_tasks.append(
                UnzipFiles(
                    input_path=path.as_posix(),
                    output_path=(
                        self.output_path
                        / path.relative_to(self.intermediate_path).parent
                    ).as_posix(),
                )
            )
        yield unzip_tasks

        # touch output
        Path(self.output().path).touch()


if __name__ == "__main__":
    luigi.build(
        [
            Workflow(
                input_path="gs://dsgt-clef-geolifeclef-2024/data/downloaded/2024",
                intermediate_path="/mnt/data/downloaded/",
                output_path="/mnt/data/raw",
            ),
        ],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
        workers=os.cpu_count(),
    )
