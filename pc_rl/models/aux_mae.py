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
        x_vis, ae_mask = self.masked_encoder(x, center_points)
        pos_recovered, padding_mask = self.masked_decoder(x_vis, ae_mask, center_points)
        pos_recovered = self.mae_prediction_head(pos_recovered)
        B, M, G, _ = pos_recovered.shape
        padding_mask = padding_mask.view(B, -1, 1, 1).expand(-1, -1, G, 3)
        padding_mask = padding_mask[ae_mask]
        pos_ground_truth = neighborhoods[ae_mask].reshape(B * M, -1, 3)
        pos_prediction = pos_recovered.reshape(B * M, -1, 3)
        pos_prediction[padding_mask] = 0.0
        pos_ground_truth[padding_mask] = 0.0
        return pos_prediction, pos_ground_truth
