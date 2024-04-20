import numpy as np
from pyspark.ml import Transformer
from pyspark.ml.functions import vector_to_array
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import (
    HasInputCol,
    HasInputCols,
    HasLabelCol,
    HasOutputCol,
    HasThreshold,
)
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


class HasNumPartitions(Params):
    numPartitions = Param(
        Params._dummy(),
        "numPartitions",
        "Number of partitions",
        typeConverter=TypeConverters.toInt,
    )

    def __init__(self) -> None:
        super().__init__()
        self._setDefault(numPartitions=200)

    def getNumPartitions(self):
        return self.getOrDefault(self.numPartitions)


class HasIndexDim(Params):
    indexDim = Param(
        Params._dummy(),
        "indexDim",
        "Index dimension",
        typeConverter=TypeConverters.toInt,
    )

    def __init__(self):
        super().__init__()

    def getIndexDim(self):
        return self.getOrDefault(self.indexDim)


class HasOutputColPrefix(Params):
    outputColPrefix = Param(
        Params._dummy(),
        "outputColPrefix",
        "Prefix for output column",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self):
        super().__init__()

    def getOutputColPrefix(self):
        return self.getOrDefault(self.outputColPrefix)


class ExtractLabelsFromVector(
    Transformer, HasInputCol, HasOutputColPrefix, HasIndexDim
):
    def __init__(self, inputCol="prediction", outputColPrefix="prediction", indexDim=0):
        super().__init__()
        self._setDefault(
            inputCol=inputCol,
            outputColPrefix=outputColPrefix,
            indexDim=indexDim,
        )

    def _transform(self, df: DataFrame) -> DataFrame:
        arr_col = "tmp_array"
        df = df.withColumn(arr_col, vector_to_array(self.getInputCol()))
        for i in range(self.getIndexDim()):
            df = df.withColumn(
                f"{self.getOutputColPrefix()}_{i:03d}", F.col(arr_col)[i]
            )
        df = df.drop(arr_col)
        return df


class ReconstructDCTCoefficients(Transformer, HasInputCols, HasOutputCol, HasIndexDim):
    def __init__(self, inputCols=None, outputCol="prediction", indexDim=0):
        super().__init__()
        self._setDefault(inputCols=inputCols, outputCol=outputCol, indexDim=indexDim)

    def _transform(self, df: DataFrame) -> DataFrame:
        return df.withColumn(
            self.getOutputCol(),
            F.array_append(
                F.array(*[F.col(col) for col in self.getInputCols()]),
                # fill the rest with zero
                F.udf(
                    lambda: np.zeros(self.getIndexDim() - len(self.getInputCols())),
                    "array<double>",
                )(),
            ),
        )


class BaseMultiClassToMultiLabel(
    Transformer,
    HasInputCol,
    HasOutputCol,
    HasNumPartitions,
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
        numPartitions=200,
    ):
        super().__init__()
        self._setDefault(
            primaryKeyCol=primaryKeyCol,
            labelCol=labelCol,
            inputCol=inputCol,
            outputCol=outputCol,
            numPartitions=numPartitions,
        )

    def _transform(self, _: DataFrame) -> DataFrame:
        raise NotImplementedError()


class NaiveMultiClassToMultiLabel(BaseMultiClassToMultiLabel):
    def _transform(self, df: DataFrame) -> DataFrame:
        return (
            df.repartition(self.getNumPartitions(), self.getPrimaryKeyCol())
            .groupBy(self.getPrimaryKeyCol())
            .agg(
                F.array_sort(F.collect_set(self.getInputCol())).alias(
                    self.getOutputCol()
                ),
                F.array_sort(F.collect_set(self.getLabelCol())).alias(
                    self.getLabelCol()
                ),
            )
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
            df.repartition(self.getNumPartitions(), self.getPrimaryKeyCol())
            .select(
                self.getPrimaryKeyCol(),
                F.posexplode(vector_to_array(self.getInputCol())).alias(
                    "class", "probability"
                ),
            )
            .withColumn("class", F.col("class").cast("double"))
            .where(F.col("probability") >= self.getThreshold())
            .groupBy(self.getPrimaryKeyCol())
            .agg(F.sort_array(F.collect_set("class")).alias(self.getOutputCol()))
        )

        # join the predictions with the labels from the original dataframe
        joined = (
            df.repartition(self.getNumPartitions(), self.getPrimaryKeyCol())
            .groupBy(self.getPrimaryKeyCol())
            .agg(
                F.array_sort(F.collect_set(self.getLabelCol())).alias(
                    self.getLabelCol()
                ),
            )
            .join(predictions, self.getPrimaryKeyCol(), "left")
            # fill output with empty arrays for evaluation, otherwise we nullpointer issues
            .withColumn(
                self.getOutputCol(),
                F.coalesce(F.col(self.getOutputCol()), F.array().cast("array<double>")),
            )
        )

        return joined
