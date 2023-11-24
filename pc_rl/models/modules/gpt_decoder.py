from __future__ import annotations

import torch
import torch.nn as nn

from .transformer import TransformerDecoder


class GPTDecoder(nn.Module):
    def __init__(
        self,
        transformer_decoder: TransformerDecoder,
        pos_embedder: nn.Module,
        padding_value: float = -1.0,
    ):
        super().__init__()
        self.transformer_decoder = transformer_decoder
        assert hasattr(
            self.transformer_decoder, "dim"
        ), f"Decoder {self.transformer_decoder} does not have a 'dim' attribute"
        self.dim = self.transformer_decoder.dim
        self.pos_embedder = pos_embedder
        self.padding_value = padding_value
        self.norm = nn.LayerNorm(self.dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, center_points, padding_mask=None, attn_mask=None):
        relative_pos = center_points[:, 1:, :] - center_points[:, :-1, :]
        relative_pos_norm = torch.linalg.vector_norm(relative_pos, dim=-1, keepdim=True)
        # add small value to avoid division by 0 for padding center points
        relative_direction = torch.nan_to_num(relative_pos / (relative_pos_norm))
        # prepend absolute position of first center point
        relative_direction = torch.cat(
            [center_points[:, 0, :].unsqueeze(1), relative_direction], dim=1
        )
        relative_pos = self.pos_embedder(relative_direction)
        x = self.transformer_decoder(
            x, relative_pos, padding_mask=padding_mask, attn_mask=attn_mask
        )
        x = self.norm(x)
        return x
