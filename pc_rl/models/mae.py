from __future__ import annotations

from typing import Callable, Literal

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.grads import grad_norm
from torch import Tensor
from torch.nn.modules import padding

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
        prediction, neighborhoods, mask, padding_mask, center_points = self.forward(
            data.pos, data.batch
        )
        B, M, G, _ = prediction.shape
        center_points_mask = torch.all(
            center_points == torch.tensor([0.0, 0.0, 0.0], device=center_points.device),
            dim=-1,
        )
        center_points_mask = center_points_mask.reshape(B, -1)
        padding_mask = center_points_mask[mask].reshape(B, -1)

        ground_truth = neighborhoods[mask].reshape(B, M, -1, 3)

        prediction = prediction.reshape(B, M, -1, 3)

        prediction[padding_mask] = 0.0
        ground_truth[padding_mask] = 0.0

        loss = self.loss_fn(
            prediction.reshape(B * M, -1, 3), ground_truth.reshape(B * M, -1, 3)
        )

        self.log("train/loss", loss.item(), batch_size=B)
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        norms = grad_norm(self.masked_encoder, norm_type=2)
        self.log_dict(norms)

    @torch.no_grad()
    def validation_step(self, data, batch_idx):
        (
            self.prediction,
            self.neighborhoods,
            self.ae_mask,
            self.padding_mask,
            self.center_points,
        ) = self.forward(data.pos, data.batch)
        self.B, self.M, self.G, _ = self.prediction.shape
        B, M, G = self.B, self.M, self.G
        self.center_points_mask = torch.all(
            self.center_points
            == torch.tensor([0.0, 0.0, 0.0], device=self.center_points.device),
            dim=-1,
        )
        self.center_points_mask = self.center_points_mask.reshape(B, -1)
        self.padding_mask_without_masked_tokens = self.center_points_mask[
            self.ae_mask
        ].reshape(B, -1)

        self.ground_truth = self.neighborhoods[self.ae_mask].reshape(B, -1, G, 3)
        self.prediction[self.padding_mask_without_masked_tokens] = 0.0
        self.ground_truth[self.padding_mask_without_masked_tokens] = 0.0
        self.prediction = self.prediction.reshape(B * M, -1, 3)
        self.ground_truth = self.ground_truth.reshape(B * M, -1, 3)

        loss = self.loss_fn(
            self.prediction,
            self.ground_truth,
        )
        chamfer_loss = self.chamfer_loss(self.prediction, self.ground_truth)

        self.log("val/loss", loss.item(), batch_size=B)
        self.log("val/chamfer_loss", chamfer_loss.item(), batch_size=B)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
