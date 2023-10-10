from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from pc_rl.models.modules.transformer import TransformerEncoder


class FinetuneEncoder(nn.Module):
    def __init__(
        self,
        transformer_encoder: TransformerEncoder,
        pos_embedder: nn.Module,
    ) -> None:
        super().__init__()
        self.transformer_encoder = transformer_encoder
        self.pos_embedder = pos_embedder
        self.dim = self.transformer_encoder.dim
        self.norm = nn.LayerNorm(self.dim)
        self.attention_pool = nn.Linear(self.dim, 1)

    def get_additional_parameters(self):
        """
        Returns an iterator containing all parameters that are NOT part of the AuxMAE class.
        """
        return itertools.chain(self.norm.parameters(), self.attention_pool.parameters())

    def get_core_parameters(self):
        return itertools.chain(
            self.transformer_encoder.parameters(), self.pos_embedder.parameters()
        )

    def forward(self, x, center_points):
        pos = self.pos_embedder(center_points)
        x = self.transformer_encoder(x, pos)
        x = self.norm(x)
        x = torch.matmul(
            F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x
        ).squeeze(-2)
        return x
