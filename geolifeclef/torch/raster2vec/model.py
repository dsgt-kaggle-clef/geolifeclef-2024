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


class Raster2Vec(pl.LightningModule):
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
        self.learning_rate = 2e-5
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

    def triplet_loss(self, patch, neighbor, distant, margin=0.1, l2=0):
        """
        Computes loss for each batch.
        """
        z_p, z_n, z_d = (
            self.model(patch),
            self.model(neighbor),
            self.model(distant),
        )
        l_n = torch.norm(z_p - z_n, dim=1)
        l_d = torch.norm(z_p - z_d, dim=1)
        l_nd = l_n - l_d
        penalty = (
            torch.norm(z_p, dim=1) + torch.norm(z_n, dim=1) + torch.norm(z_d, dim=1)
        )
        loss = torch.mean(F.relu(l_nd + margin) + l2 * penalty)
        return loss, torch.mean(l_n), torch.mean(l_d), torch.mean(l_nd)

    def _run_step(self, batch, batch_idx, step_name):
        x, y = batch["features"], batch["label"]

        # we do the loss for each of the items in our triplet
        keys = ["anchor", "neighbor", "distant"]
        logits = {k: self(x[k]) for k in keys}
        asl = {k: self.asl_loss(logits[k], y[k].to_dense()) for k in keys}
        triplet, triple_n, triplet_d, triplet_nd = self.triplet_loss(
            *[x[k] for k in keys]
        )
        asl_sum = sum(asl.values())
        loss = triplet + asl_sum
        self.log(f"{step_name}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        # all of the individual losses
        self.log(f"{step_name}_asl_loss", asl_sum, on_step=False, on_epoch=True)
        for k, v in asl.items():
            self.log(f"{step_name}_{k}_asl_loss", v, on_step=False, on_epoch=True)
        self.log(f"{step_name}_triplet_loss", triplet, on_step=False, on_epoch=True)
        for k, v in zip(
            ["triplet_n", "triplet_d", "triplet_nd"],
            [triple_n, triplet_d, triplet_nd],
        ):
            self.log(f"{step_name}_{k}_loss", v, on_step=False, on_epoch=True)

        # how do we track the f1 score? of each pair/triplet?
        for k in keys:
            self.log(
                f"{step_name}_{k}_f1",
                self.f1_score(logits[k], y[k].to_dense()),
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
