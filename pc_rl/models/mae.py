from __future__ import annotations

from typing import Literal

import pytorch_lightning as pl
import torch
from torch import Tensor

from pc_rl.models.modules.embedder import Embedder
from pc_rl.models.modules.mae_prediction_head import MaePredictionHead
from pc_rl.models.modules.masked_decoder import MaskedDecoder
from pc_rl.models.modules.masked_encoder import MaskedEncoder
from pc_rl.utils.aux_loss import get_loss_fn


class MaskedAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        embedder: Embedder,
        masked_encoder: MaskedEncoder,
        masked_decoder: MaskedDecoder,
        mae_prediction_head: MaePredictionHead,
        learning_rate: float,
        weight_decay: float,
        loss_fn: Callable,
    ):
        super().__init__()
        self.embedder = embedder
        self.masked_encoder = masked_encoder
        self.masked_decoder = masked_decoder
        self.mae_prediction_head = mae_prediction_head
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn
        self.chamfer_loss = get_loss_fn("chamfer")

    def forward(self, pos: Tensor, batch: Tensor):
        x, neighborhoods, center_points = self.embedder(pos, batch)
        x_vis, mask = self.masked_encoder(x, center_points)
        x_recovered, padding_mask = self.masked_decoder(x_vis, mask, center_points)
        pos_recovered = self.mae_prediction_head(x_recovered)

        return pos_recovered, neighborhoods, mask, padding_mask, center_points

    def training_step(self, data, batch_idx):
        prediction, neighborhoods, mask, padding_mask, _ = self.forward(
            data.pos, data.batch
        )
        B, M, G, _ = prediction.shape
        padding_mask = padding_mask.view(B, -1, 1, 1).expand(-1, -1, G, 3)
        padding_mask = padding_mask[mask]
        ground_truth = neighborhoods[mask].reshape(B * M, -1, 3)

        prediction = prediction.reshape(B * M, -1, 3)

        prediction[padding_mask] = 0.0
        ground_truth[padding_mask] = 0.0

        loss = self.loss_fn(prediction, ground_truth)

        self.log("train/loss", loss.item(), batch_size=B)
        return loss

    @torch.no_grad()
    def validation_step(self, data, batch_idx):
        (
            self.test_prediction,
            self.test_neighborhoods,
            self.test_mask,
            self.test_padding_mask,
            self.test_center_points,
        ) = self.forward(data.pos, data.batch)
        B, M, G, _ = self.test_prediction.shape
        self.test_padding_mask = self.test_padding_mask.view(B, -1, 1, 1).expand(
            -1, -1, G, 3
        )
        self.test_padding_mask = self.test_padding_mask[self.test_mask]
        self.test_ground_truth = self.test_neighborhoods[self.test_mask].reshape(
            B * M, -1, 3
        )

        self.test_prediction = self.test_prediction.reshape(B * M, -1, 3)

        self.test_prediction[self.test_padding_mask] = 0.0
        self.test_ground_truth[self.test_padding_mask] = 0.0
        loss = self.loss_fn(self.test_prediction, self.test_ground_truth)
        chamfer_loss = self.chamfer_loss(self.test_prediction, self.test_ground_truth)

        self.log("val/loss", loss.item(), batch_size=B)
        self.log("val/chamfer_loss", chamfer_loss.item(), batch_size=B)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
