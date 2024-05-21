from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional
import torch.nn.functional as F
import torch_dct as dct
from torch import nn
from torchmetrics.classification import MultilabelF1Score
from torchvision.models import get_model

from ..losses import AsymmetricLossOptimized, Hill

# https://www.kaggle.com/code/rejpalcz/best-loss-function-for-f1-score-metric
# https://stackoverflow.com/questions/65318064/can-i-trainoptimize-on-f1-score-loss-with-pytorch


class Raster2VecClassifier(pl.LightningModule):
    def __init__(
        self,
        num_layers: int,
        num_features: int,
        num_classes: int,
        weights: Optional[torch.Tensor] = None,
        hidden_layer_size: int = 256,
        disable_asl: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.num_classes = num_classes
        self.weights = weights if weights is not None else torch.ones(num_classes)
        self.learning_rate = 2e-5
        self.disable_asl = disable_asl
        self.save_hyperparameters()

        self.model = nn.Sequential(
            # convolve the layers
            nn.Conv2d(num_layers, num_layers, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_layers),
            nn.ReLU(inplace=True),
            # now reduce to a single embedding layer
            nn.Conv2d(num_layers, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(num_features**2, hidden_layer_size),
        )
        self.classifier_layer = nn.Sequential(
            nn.BatchNorm1d(hidden_layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layer_size, num_classes),
        )
        # print the model architecture
        print(self.model, flush=True)
        self.f1_score = MultilabelF1Score(num_classes, average="micro")
        self.asl_loss = AsymmetricLossOptimized()

    def forward(self, x):
        z = self.model(x)
        z = self.classifier_layer(z)
        return z

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _run_step(self, batch, batch_idx, step_name):
        x, y = batch["features"], batch["label"]
        logits = self(x)
        loss = self.asl_loss(logits, y)
        self.log(f"{step_name}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            f"{step_name}_f1",
            self.f1_score(logits, y.to_dense()),
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
            "predictions": self(batch["features"]),
            "surveyId": batch["surveyId"],
        }
