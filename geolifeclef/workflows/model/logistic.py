import luigi
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StandardScaler
from pyspark.ml.tuning import ParamGridBuilder

from .base_tasks import BaseFitModel, CleanMetadata


class FitLogisticModel(BaseFitModel):
    max_iter = luigi.ListParameter(default=[100])
    reg_param = luigi.ListParameter(default=[0.0])
    elastic_net_param = luigi.ListParameter(default=[0.0])

    def _classifier(self, featuresCol, labelCol):
        return Pipeline(
            stages=[
                StandardScaler(inputCol=featuresCol, outputCol="scaled_features"),
                LogisticRegression(featuresCol="scaled_features", labelCol=labelCol),
            ]
        )

    def _param_grid(self, pipeline):
        # from the pipeline, let's extract the logistic regression model
        lr = pipeline.getStages()[-2].getStages()[-1]
        return (
            ParamGridBuilder()
            .addGrid(lr.maxIter, self.max_iter)
            .addGrid(lr.regParam, self.reg_param)
            .addGrid(lr.elasticNetParam, self.elastic_net_param)
            .build()
        )


class LogisticWorkflow(luigi.Task):
    def run(self):
        data_root = "gs://dsgt-clef-geolifeclef-2024/data"
        yield CleanMetadata(
            input_path=f"{data_root}/downloaded/2024",
            output_path=f"{data_root}/processed/metadata_clean/v1",
        )

        # v1 - multi-class w/ test-train split and cv
        # v2 - conversion to multilabel
        # v3 - drop custom test-train split and rely on cv
        # v4 - add threshold multilabel strategy, add a faster training phase
        yield [
            # these runs are meant to validate that the pipeline works as expected before expensive runs
            FitLogisticModel(
                k=3,
                max_iter=[5],
                num_folds=2,
                multilabel_strategy=strategy,
                input_path=f"{data_root}/processed/metadata_clean/v1",
                output_path=f"{data_root}/models/subset_logistic_{strategy}/v4_test",
            )
            for strategy in ["naive", "threshold"]
        ]

        # now fit this on a larger dataset to see if this works in a more realistic setting
        yield [
            FitLogisticModel(
                k=20,
                multilabel_strategy=strategy,
                input_path=f"{data_root}/processed/metadata_clean/v1",
                output_path=f"{data_root}/models/subset_logistic_{strategy}/v4",
            )
            for strategy in ["naive", "threshold"]
        ]

        yield [
            FitLogisticModel(
                multilabel_strategy=strategy,
                input_path=f"{data_root}/processed/metadata_clean/v1",
                output_path=f"{data_root}/models/logistic_{strategy}/v4",
            )
            for strategy in ["naive", "threshold"]
        ]


if __name__ == "__main__":
    luigi.build(
        [LogisticWorkflow()],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
        workers=1,
    )
