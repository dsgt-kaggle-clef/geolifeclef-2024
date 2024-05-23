from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.classification import MultilabelF1Score

from ..losses import AsymmetricLossOptimized

# https://www.kaggle.com/code/rejpalcz/best-loss-function-for-f1-score-metric
# https://stackoverflow.com/questions/65318064/can-i-trainoptimize-on-f1-score-loss-with-pytorch


class RasterClassifier(pl.LightningModule):
    def __init__(
        self,
        num_layers: int,
        num_features: int,
        num_classes: int,
        weights: Optional[torch.Tensor] = None,
        hidden_layer_size: int = 256,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.num_classes = num_classes
        self.weights = weights if weights is not None else torch.ones(num_classes)
        self.learning_rate = 2e-3
        self.save_hyperparameters()
        # https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html#torchvision.models.efficientnet_v2_s
        # net = get_model("efficientnet_v2_s")
        # self.model = nn.Sequential(
        #     # get the appropriate input size
        #     nn.Conv2d(num_layers, 3, kernel_size=1),
        #     *list(net.children())[:-1],
        #     nn.Flatten(),
        #     # dropout
        #     nn.Dropout(0.2, inplace=True),
        #     nn.Linear(1280, num_classes),
        # )
        self.model = nn.Sequential(
            # convolve the layers
            nn.Conv2d(num_layers, num_layers, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_layers),
            nn.ReLU(inplace=True),
            # now reduce to a single embedding layer
            nn.Conv2d(num_layers, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(num_features**2, hidden_layer_size),
            nn.BatchNorm1d(hidden_layer_size),
            nn.ReLU(inplace=True),
            # learn the logits for the classes
            nn.Linear(hidden_layer_size, num_classes),
        )
        # print the model architecture
        print(self.model, flush=True)
        self.f1_score = MultilabelF1Score(num_classes, average="micro")
        self.loss = AsymmetricLossOptimized()
        # torch.nn.functional.multilabel_soft_margin_loss

    # def _transform(self, batch):
    #     features, label = batch["features"], batch["label"]
    #     # put the features on a 128x128 grid
    #     zero_pad = torch.zeros(
    #         features.shape[0], features.shape[1], 128, 128, device=features.device
    #     )
    #     zero_pad[:, :, : features.shape[2], : features.shape[3]] = features

    #     return {
    #         "features": dct.idct_2d(zero_pad),
    #         "label": label,
    #     }

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _run_step(self, batch, batch_idx, step_name):
        # stupid hack, squeeze the first batch dimension out
        # batch = self._transform(batch)
        x, y = batch["features"], batch["label"].to_dense()
        logits = self(x)
        loss = self.loss(logits, y)
        self.log(f"{step_name}_loss", loss, prog_bar=True)
        self.log(
            f"{step_name}_f1",
            self.f1_score(logits, y),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._run_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._run_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._run_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return {
            "predictions": F.sigmoid(self(batch["features"])),
            "surveyId": batch["surveyId"],
        }
