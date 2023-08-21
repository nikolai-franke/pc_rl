from __future__ import annotations

import torch
import torch.nn as nn

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
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.dim))

    def forward(self, x, center_points, _=None):
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        cls_pos = self.cls_pos.expand(x.shape[0], -1, -1)
        pos = self.pos_embedder(center_points)
        pos = torch.cat((cls_pos, pos), dim=1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer_encoder(x, pos)
        x = self.norm(x)
        # out = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        return x[:, 0]
