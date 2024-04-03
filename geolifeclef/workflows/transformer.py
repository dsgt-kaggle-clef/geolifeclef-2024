from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol, HasLabelCol, HasOutputCol, HasThreshold
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class HasPrimaryKeyCol(Params):
    primaryKeyCol = Param(
        Params._dummy(),
        "primaryKeyCol",
        "Primary key column",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self):
        super().__init__()

    def getPrimaryKeyCol(self):
        return self.getOrDefault(self.primaryKeyCol)


class BaseMultiClassToMultiLabel(
    Transformer,
    HasInputCol,
    HasOutputCol,
    HasPrimaryKeyCol,
    HasLabelCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    def __init__(
        self,
        primaryKeyCol="surveyId",
        labelCol="speciesId",
        inputCol="prediction",
        outputCol="prediction",
    ):
        super().__init__()
        self._setDefault(
            primaryKeyCol=primaryKeyCol,
            labelCol=labelCol,
            inputCol=inputCol,
            outputCol=outputCol,
        )

    def _transform(self, df: DataFrame) -> DataFrame:
        raise NotImplementedError()


class NaiveMultiClassToMultiLabel(BaseMultiClassToMultiLabel):
    def _transform(self, df: DataFrame) -> DataFrame:
        return df.groupBy(self.getPrimaryKeyCol()).agg(
            F.array_sort(F.collect_set(self.getInputCol())).alias(self.getOutputCol()),
            F.array_sort(F.collect_set(self.getLabelCol())).alias(self.getLabelCol()),
        )


class ThresholdMultiClassToMultiLabel(BaseMultiClassToMultiLabel, HasThreshold):
    def __init__(self, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self._setDefault(threshold=threshold)

    def _transform(self, df: DataFrame) -> DataFrame:
        """We look at the probability array, and keep all the classes that are above the threshold.

        We assume that the labels of the species maps directly to the index in the predictions array.
        """

        # explode the predictions and get all of the classes above a certain threshold
        predictions = (
            df.select(
                self.getPrimaryKeyCol(),
                F.posexplode(self.getInputColumn()).alias("class", "probability"),
            )
            .where(F.col("probability") >= self.getThreshold())
            .groupBy(self.getPrimaryKeyCol())
            .agg(F.sort_array(F.collect_set("class")).alias(self.getOutputCol()))
        )

        # join the predictions with the labels from the original dataframe
        return (
            df.groupBy(self.getPrimaryKeyCol())
            .agg(
                F.array_sort(F.collect_set(self.getLabelCol())).alias(
                    self.getLabelCol()
                ),
            )
            .join(predictions, self.getPrimaryKeyCol(), "left")
            .fillna([], subset=[self.getOutputCol()])
        )
