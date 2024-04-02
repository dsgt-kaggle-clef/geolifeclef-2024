from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol, HasLabelCol, HasOutputCol
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


class NaiveMultiClassToMultiLabel(
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
        labelCol="label",
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
        return df.groupBy(self.getPrimaryKeyCol()).agg(
            F.array_sort(F.collect_set(self.getInputCol())).alias(self.getOutputCol()),
            F.array_sort(F.collect_set(self.getLabelCol())).alias(self.getLabelCol()),
        )
