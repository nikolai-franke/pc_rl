from __future__ import annotations

import torch.nn as nn

from pc_rl.models.modules.mae_prediction_head import MaePredictionHead
from pc_rl.models.modules.masked_decoder import MaskedDecoder
from pc_rl.models.modules.masked_encoder import MaskedEncoder


class RLMae(nn.Module):
    def __init__(
        self,
        masked_encoder: MaskedEncoder,
        masked_decoder: MaskedDecoder,
        mae_prediction_head: MaePredictionHead,
    ):
        super().__init__()
        self.masked_encoder = masked_encoder
        self.masked_decoder = masked_decoder
        self.mae_prediction_head = mae_prediction_head
        self.dim = self.masked_encoder.dim

    def forward(self, x, neighborhoods, center_points):
        x_vis, ae_mask, padding_mask = self.masked_encoder(x, center_points)
        x_recovered = self.masked_decoder(x_vis, ae_mask, center_points)
        prediction = self.mae_prediction_head(x_recovered)
        B, M, *_ = x_recovered.shape
        *_, C = neighborhoods.shape

        padding_mask = padding_mask.reshape(B, -1)
        padding_mask = padding_mask[ae_mask].reshape(B, -1)

        ground_truth = neighborhoods[ae_mask].reshape(B, M, -1, C)
        prediction = prediction.reshape(B, M, -1, C)

        ground_truth[padding_mask] = 0.0
        prediction[padding_mask] = 0.0

        return prediction, ground_truth
