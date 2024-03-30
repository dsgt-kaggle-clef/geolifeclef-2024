import numpy as np
from pyspark.ml.functions import predict_batch_udf
from pyspark.sql.types import ArrayType, FloatType
from scipy.fftpack import dctn


def make_dctn_filter_fn(k=8):
    def dctn_filter(tile, k):
        coeff = dctn(tile)
        coeff_subset = coeff[:k, :k]
        return coeff_subset.flatten()

    def predict(inputs: np.ndarray) -> np.ndarray:
        # inputs is a 3D array of shape (batch_size, img_dim, img_dim)
        return np.array([dctn_filter(x, k=k) for x in inputs])

    return predict


# batch prediction UDF
dctn_filter_udf = predict_batch_udf(
    make_predict_fn=make_dctn_filter_fn,
    return_type=ArrayType(FloatType()),
    batch_size=4,
    input_tensor_shapes=[[128, 128]],
)
