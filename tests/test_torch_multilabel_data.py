import torch

from geolifeclef.torch.multilabel.data import ToSparseTensor


def test_to_sparse_tensor():
    num_classes = 5
    dense = torch.tensor(
        [
            [1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 0, 0],
        ]
    )
    test_data = {
        "features": torch.ones(4, 2),
        "labels": dense,
    }

    transform = ToSparseTensor()
    result = transform(test_data)
    assert result["labels"].shape == (4, num_classes)
    assert (dense - result["labels"]).sum() == 0
