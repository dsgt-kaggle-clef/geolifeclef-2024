from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional
from torch import nn
from torchmetrics.classification import MultilabelF1Score


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
        self.weights = weights or torch.ones(num_classes)
        self.learning_rate = 2e-3
        self.save_hyperparameters()
        self.model = nn.Sequential(
            # convolutional layer
            # we have batch_size x num_layers x num_features and want to go down to a hidden layer size
            nn.Conv1d(num_layers, 1, 1),
            nn.Flatten(),
            nn.ReLU(inplace=True),
            nn.Linear(num_features, hidden_layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layer_size, num_classes),
        )
        # print the model architecture
        print(self.model, flush=True)
        self.f1_score = MultilabelF1Score(num_classes, average="micro")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _run_step(self, batch, batch_idx, step_name):
        # stupid hack, squeeze the first batch dimension out
        x, y = batch["features"], batch["label"].to_dense()
        logits = self(x)
        loss = torch.nn.functional.multilabel_soft_margin_loss(
            logits, y, weight=self.weights.to(y.device)
        )
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
