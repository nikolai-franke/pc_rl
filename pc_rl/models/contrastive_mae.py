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
        aux_loss: Literal["chamfer", "sinkhorn"],
    ):
        super().__init__()
        self.embedder = embedder
        self.masked_encoder = masked_encoder
        self.masked_decoder = masked_decoder
        self.mae_prediction_head = mae_prediction_head
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = get_loss_fn(aux_loss)
        self.dim = self.embedder.embedding_size
        self.similarity_param = torch.nn.Parameter(
            torch.randn(self.dim, self.embedder.embedding_size)
        )
        self.contrastive_loss = torch.nn.CrossEntropyLoss()
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, self.dim))
        self.cls_pos = torch.nn.Parameter(torch.randn(1, 1, self.dim))

    def forward(self, pos: Tensor, batch: Tensor):
        x, neighborhoods, center_points = self.embedder(pos, batch)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        cls_pos = self.cls_pos.expand(x.shape[0], -1, -1)
        x_vis, mask, x_cls = self.masked_encoder(x, center_points, cls_tokens, cls_pos)
        x_cls = torch.cat([x_cls, x[mask].max(1)[0]], dim=-1)
        _, mask, x_cls2 = self.masked_encoder(x, center_points, cls_tokens, cls_pos)
        x_cls2 = torch.cat([x_cls2, x[mask].max(1)[0]], dim=-1)
        x_recovered, padding_mask = self.masked_decoder(x_vis, mask, center_points)
        pos_recovered = self.mae_prediction_head(x_recovered)

        return (
            pos_recovered,
            neighborhoods,
            mask,
            padding_mask,
            center_points,
            x_cls,
            x_cls2,
        )

    def training_step(self, data, batch_idx):
        prediction, neighborhoods, mask, padding_mask, _, x_cls, x_cls2 = self.forward(
            data.pos, data.batch
        )
        projection = x_cls @ self.similarity_param
        logits = projection @ x_cls2
        labels = torch.arange(logits.shape[0])
        contrastive_loss = self.contrastive_loss(logits, labels)

        B, M, G, _ = prediction.shape
        padding_mask = padding_mask.view(B, -1, 1, 1).expand(-1, -1, G, 3)
        padding_mask = padding_mask[mask]
        ground_truth = neighborhoods[mask].reshape(B * M, -1, 3)

        prediction = prediction.reshape(B * M, -1, 3)

        prediction[padding_mask] = 0.0
        ground_truth[padding_mask] = 0.0

        loss = self.loss_fn(prediction, ground_truth)

        self.log("train/loss", loss.item(), batch_size=B)
        self.log("train/contrastive_loss", loss.item(), batch_size=B)
        return loss + contrastive_loss

    @torch.no_grad()
    def validation_step(self, data, batch_idx):
        (
            self.test_prediction,
            self.test_neighborhoods,
            self.test_mask,
            self.test_padding_mask,
            self.test_center_points,
            *_,
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

        self.log("val/loss", loss.item(), batch_size=B)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
