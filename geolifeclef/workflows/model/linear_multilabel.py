from argparse import ArgumentParser
from collections import Counter

import luigi
import numpy as np
import scipy.sparse as sp
from pyspark.ml import Pipeline
from pyspark.ml.feature import DCT, StandardScaler, VectorAssembler, VectorSlicer
from pyspark.ml.linalg import SparseVector, VectorUDT
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import functions as F

from ..utils import RsyncGCSFiles
from .base_tasks import BaseFitModel
from .transformer import (
    ExtractLabelsFromVector,
    NaiveMultiClassToMultiLabel,
    ReconstructDCTCoefficients,
    ThresholdMultiClassToMultiLabel,
)


class FitLinearMultiLabelModel(BaseFitModel):
    max_iter = luigi.ListParameter(default=[100])
    reg_param = luigi.ListParameter(default=[1e-5])
    elastic_net_param = luigi.ListParameter(default=[0.5])
    max_species = luigi.IntParameter(default=11255)
    labels_dim = luigi.IntParameter(default=16)
    label = luigi.Parameter(default="speciesId")

    def _target_mapping(self, df, src, dst):
        return df

    def _collect_sparse_labels(self, array):
        """Turn a list of numbers into a sparse vector."""

        @F.udf(VectorUDT())
        def func(array):
            items = sorted(Counter(array).items())
            indices = np.array([i for i, _ in items])
            values = np.array([v for _, v in items])
            return SparseVector(self.max_species, indices, values)

        return func(array)

    def _load(self, spark):
        df = super()._load(spark)
        # now we can convert this into a multi-label problem by surveyId
        return df.groupBy("surveyId").agg(
            F.mean("lat_proj").alias("lat_proj"),
            F.mean("lon_proj").alias("lon_proj"),
            self._collect_sparse_labels(F.collect_list("speciesId")).alias("labels"),
        )

    def _pipeline(self):
        multilabel = {
            "naive": NaiveMultiClassToMultiLabel(
                primaryKeyCol="surveyId",
                labelCol=self.label,
                inputCol="prediction",
                outputCol="prediction",
            ),
            "threshold": ThresholdMultiClassToMultiLabel(
                primaryKeyCol="surveyId",
                labelCol=self.label,
                inputCol="probability",
                outputCol="prediction",
            ),
        }
        return Pipeline(
            stages=[
                VectorAssembler(
                    inputCols=self.features,
                    outputCol="features",
                ),
                # create a new label for each of the coefficients
                StandardScaler(inputCol="features", outputCol="scaled_features"),
                DCT(inputCol="labels", outputCol="labels_dct", inverse=False),
                # slice the first k coefficients
                VectorSlicer(
                    inputCol="labels_dct",
                    outputCol="labels_dct_sliced",
                    indices=list(range(self.labels_dim)),
                ),
                ExtractLabelsFromVector(
                    inputCol="labels_dct_sliced",
                    outputColPrefix="label",
                    indexDim=self.labels_dim,
                ),
                *[
                    LinearRegression(featuresCol="scaled_features", labelCol=col)
                    for col in [f"label_{i:03d}" for i in range(self.labels_dim)]
                ],
                # now project the labels back by concatenating the predictions
                # together and taking the inverse
                ReconstructDCTCoefficients(
                    inputCols=[f"label_{i:03d}" for i in range(self.labels_dim)],
                    outputCol="dct_prediction",
                    indexDim=self.labels_dim,
                ),
                # it's not really a probabiity, but more of a quantization to the nearest
                # integer
                DCT(inputCol="dct_prediction", outputCol="probability", inverse=True),
                multilabel[self.multilabel_strategy],
            ]
        )

    def _param_grid(self, pipeline):
        # from the pipeline, let's extract the logistic regression model
        builder = ParamGridBuilder()
        for stage in pipeline.getStages():
            if not isinstance(stage, LinearRegression):
                continue
            builder = (
                builder.addGrid(stage.maxIter, self.max_iter)
                .addGrid(stage.regParam, self.reg_param)
                .addGrid(stage.elasticNetParam, self.elastic_net_param)
            )
        return builder.build()


class Workflow(luigi.Task):
    remote_root = luigi.Parameter(default="gs://dsgt-clef-geolifeclef-2024/data")
    local_root = luigi.Parameter(default="/mnt/data/geolifeclef-2024/data")

    def run(self):
        yield RsyncGCSFiles(
            src_path=f"{self.remote_root}/processed/geolsh_knn_graph/v2",
            dst_path=f"{self.local_root}/processed/geolsh_knn_graph/v2",
        )

        # v1 - initial implementation
        yield [
            FitLinearMultiLabelModel(
                multilabel_strategy=strategy,
                input_path=f"{self.local_root}/processed/metadata_clean/v1",
                output_path=f"{self.local_root}/models/linear_multilabel_dct/v1",
            )
            for strategy in ["threshold"]
        ]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scheduler-host", default="localhost")
    args = parser.parse_args()

    luigi.build(
        [Workflow()],
        scheduler_host=args.scheduler_host,
        workers=1,
    )
