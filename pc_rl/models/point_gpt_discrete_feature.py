from __future__ import annotations

import functools

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather
from pytorch_lightning.utilities.grads import grad_norm
from torch import Tensor

from pc_rl.models.modules.gpt_decoder import GPTDecoder
from pc_rl.models.modules.gpt_encoder import GPTEncoder
from pc_rl.models.modules.gpt_tokenizer import GPTTokenizer
from pc_rl.models.modules.prediction_head import PredictionHead
from pc_rl.utils.chamfer import chamfer_distance


class PointGPTDiscreteFeature(pl.LightningModule):
    def __init__(
        self,
        tokenizer: GPTTokenizer,
        encoder: GPTEncoder,
        decoder: GPTDecoder,
        prediction_head: PredictionHead,
        learning_rate: float,
        weight_decay: float,
        color_loss_coeff: float = 1.0,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.decoder = decoder
        self.prediction_head = prediction_head
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.color_loss_coeff = color_loss_coeff
        self.loss_fn = functools.partial(chamfer_distance, return_x_nn=True)
        self.ground_truth_point_dim = 4  # TODO: hardcoded

    def forward(self, pos: Tensor, batch: Tensor, color: Tensor | None = None):
        x, neighborhoods, center_points = self.tokenizer(pos, batch, color)
        x, padding_mask, attn_mask = self.encoder(x, center_points)
        x_recovered = self.decoder(
            x, center_points, padding_mask=padding_mask, attn_mask=attn_mask
        )
        pos_recovered = self.prediction_head(x_recovered)

        return pos_recovered, neighborhoods, padding_mask, center_points

    def training_step(self, data, batch_idx):
        prediction, neighborhoods, padding_mask, _ = self.forward(
            data.pos, data.batch, data.x
        )
        B, M, *_, C = prediction.shape

        # only allow one discrete feature for now
        assert neighborhoods.shape[-1] == self.ground_truth_point_dim

        # when tokenizer.point_dim != prediction_head.point_dim
        neighborhoods = neighborhoods[..., :C]

        padding_mask = padding_mask.reshape(B, -1)
        ground_truth = neighborhoods.reshape(B, M, -1, self.ground_truth_point_dim)
        prediction = prediction.reshape(B, M, -1, C)

        prediction[padding_mask] = 0.0
        ground_truth[padding_mask] = 0.0

        prediction = prediction.reshape(B * M, -1, C)
        ground_truth = ground_truth.reshape(B * M, -1, 4)

        loss, *_, x_idx = self.loss_fn(prediction[..., :3], ground_truth[..., :3])
        self.log("train/chamfer_loss", loss.item(), batch_size=B)

        # if additional feature channels
        if C > 3:
            prediction_nearest_neighbor = knn_gather(ground_truth, x_idx)
            classification_loss = F.cross_entropy(
                input=prediction[..., 3:].reshape(-1, C - 3),
                target=prediction_nearest_neighbor[..., -1].reshape(-1).to(torch.long),
            )
            loss += classification_loss
            self.log(
                "train/classification_loss",
                classification_loss.item(),
                batch_size=B,
            )

        self.log("train/loss", loss.item(), batch_size=B)
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        norms = grad_norm(self.encoder, norm_type=2)
        self.log_dict(norms)

    @torch.no_grad()
    def validation_step(self, data, batch_idx):
        (
            self.prediction,
            self.neighborhoods,
            self.padding_mask,
            self.center_points,
        ) = self.forward(data.pos, data.batch, data.x)
        # save dimensions for point cloud logger
        self.B, self.M, self.G, self.C = self.prediction.shape
        B, M, G, C = self.B, self.M, self.G, self.C

        self.neighborhoods = self.neighborhoods[..., :C]

        self.padding_mask = self.padding_mask.reshape(B, -1)

        self.ground_truth = self.neighborhoods.reshape(B, -1, G, 4)
        self.prediction = self.prediction.reshape(B, M, -1, C)
        self.prediction[self.padding_mask] = 0.0
        self.ground_truth[self.padding_mask] = 0.0
        self.prediction = self.prediction.reshape(B * M, -1, C)
        self.ground_truth = self.ground_truth.reshape(B * M, -1, 4)

        loss, *_, x_idx = self.loss_fn(
            self.prediction[..., :3], self.ground_truth[..., :3]
        )
        self.log("val/chamfer_loss", loss.item(), batch_size=B)
        if C > 3:
            prediction_nearest_neighbor = knn_gather(self.ground_truth, x_idx)
            classification_loss = F.cross_entropy(
                input=self.prediction[..., 3:].reshape(-1, C - 3),
                target=prediction_nearest_neighbor[..., -1].reshape(-1).to(torch.long),
            )
            loss += classification_loss
            self.log(
                "train/classification_loss",
                classification_loss.item(),
                batch_size=B,
            )

        self.log("val/loss", loss.item(), batch_size=B)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
