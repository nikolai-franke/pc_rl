from __future__ import annotations

from typing import Callable

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather
from pytorch_lightning.utilities.grads import grad_norm
from torch import Tensor

from pc_rl.models.modules.embedder import Embedder
from pc_rl.models.modules.mae_prediction_head import MaePredictionHead
from pc_rl.models.modules.masked_decoder import MaskedDecoder
from pc_rl.models.modules.masked_encoder import MaskedEncoder
from pc_rl.utils.aux_loss import get_loss_fn


class ColorMaskedAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        embedder: Embedder,
        masked_encoder: MaskedEncoder,
        masked_decoder: MaskedDecoder,
        mae_prediction_head: MaePredictionHead,
        learning_rate: float,
        weight_decay: float,
        color_loss_coeff: float = 0.1,
    ):
        super().__init__()
        self.embedder = embedder
        self.masked_encoder = masked_encoder
        self.masked_decoder = masked_decoder
        self.mae_prediction_head = mae_prediction_head
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.color_loss_coeff = color_loss_coeff
        self.loss_fn = get_loss_fn("chamfer", loss_kwargs={"return_x_nn": True})

    def forward(self, pos: Tensor, batch: Tensor, color: Tensor):
        x, neighborhoods, center_points = self.embedder(pos, batch, color)
        x_vis, ae_mask, padding_mask = self.masked_encoder(x, center_points)
        x_recovered = self.masked_decoder(x_vis, ae_mask, center_points)
        pos_recovered = self.mae_prediction_head(x_recovered)

        return pos_recovered, neighborhoods, ae_mask, padding_mask, center_points

    def training_step(self, data, batch_idx):
        prediction, neighborhoods, mask, padding_mask, _ = self.forward(
            data.pos, data.batch, data.x
        )
        B, M, *_ = prediction.shape
        padding_mask = padding_mask.reshape(B, -1)
        padding_mask = padding_mask[mask].reshape(B, -1)

        ground_truth = neighborhoods[mask].reshape(B, M, -1, 6)

        prediction = prediction.reshape(B, M, -1, 6)

        prediction[padding_mask] = 0.0
        ground_truth[padding_mask] = 0.0

        prediction = prediction.reshape(B * M, -1, 6)
        ground_truth = ground_truth.reshape(B * M, -1, 6)

        chamfer_distance, *_, x_idx = self.loss_fn(
            prediction[..., :3], ground_truth[..., :3]
        )
        prediction_nearest_neighbor = knn_gather(ground_truth, x_idx).reshape(
            B, M, -1, 6
        )
        color_loss = (
            F.mse_loss(
                prediction[..., 3:].reshape(B, -1, 3),
                prediction_nearest_neighbor[..., 3:].reshape(B, -1, 3),
            )
            * self.color_loss_coeff
        )
        loss = chamfer_distance + color_loss

        self.log("train/loss", loss.item(), batch_size=B)
        self.log("train/chamfer_loss", chamfer_distance.item(), batch_size=B)
        self.log("train/color_loss", color_loss.item(), batch_size=B)
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
        ) = self.forward(data.pos, data.batch, data.x)
        self.B, self.M, self.G, _ = self.prediction.shape
        B, M, G = self.B, self.M, self.G
        self.padding_mask = self.padding_mask.reshape(B, -1)
        self.padding_mask_without_masked_tokens = self.padding_mask[
            self.ae_mask
        ].reshape(B, -1)

        self.ground_truth = self.neighborhoods[self.ae_mask].reshape(B, -1, G, 6)
        self.prediction[self.padding_mask_without_masked_tokens] = 0.0
        self.ground_truth[self.padding_mask_without_masked_tokens] = 0.0
        self.prediction = self.prediction.reshape(B * M, -1, 6)
        self.ground_truth = self.ground_truth.reshape(B * M, -1, 6)

        chamfer_distance, *_, x_idx = self.loss_fn(
            self.prediction[..., :3], self.ground_truth[..., :3]
        )
        prediction_nearest_neighbor = knn_gather(self.ground_truth, x_idx)
        color_loss = (
            F.mse_loss(
                self.prediction[..., 3:].reshape(B, -1, 3),
                prediction_nearest_neighbor[..., 3:].reshape(B, -1, 3),
            )
            * self.color_loss_coeff
        )
        loss = chamfer_distance + color_loss

        self.log("val/loss", loss.item(), batch_size=B)
        self.log("val/chamfer_loss", chamfer_distance.item(), batch_size=B)
        self.log("val/color_loss", color_loss.item(), batch_size=B)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
