from __future__ import annotations

import torch.nn as nn

from pc_rl.models.modules.gpt_decoder import GPTDecoder
from pc_rl.models.modules.gpt_encoder import GPTEncoder
from pc_rl.models.modules.prediction_head import PredictionHead


class AuxGPT(nn.Module):
    def __init__(
        self,
        gpt_encoder: GPTEncoder,
        gpt_decoder: GPTDecoder,
        prediction_head: PredictionHead,
    ):
        super().__init__()
        self.gpt_encoder = gpt_encoder
        self.gpt_decoder = gpt_decoder
        self.prediction_head = prediction_head
        self.dim = self.gpt_encoder.dim

    def forward(self, x, neighborhoods, center_points):
        x, padding_mask, attn_mask = self.gpt_encoder(x, center_points)
        x_recovered = self.gpt_decoder(
            x, center_points, padding_mask=padding_mask, attn_mask=attn_mask
        )
        prediction = self.prediction_head(x_recovered)
        B, M, *_ = x_recovered.shape
        *_, C = neighborhoods.shape

        padding_mask = padding_mask.reshape(B, -1)

        ground_truth = neighborhoods.reshape(B, M, -1, C)
        prediction = prediction.reshape(B, M, -1, C)

        ground_truth[padding_mask] = 0.0
        prediction[padding_mask] = 0.0

        return prediction, ground_truth
