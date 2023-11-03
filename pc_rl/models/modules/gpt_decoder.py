from __future__ import annotations

import torch
import torch.nn as nn

from .transformer import TransformerDecoder


class GptDecoder(nn.Module):
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
        assert (
            pos_dim := self.pos_embedder.channel_list[-1]
        ) == self.dim, f"pos_embedder and decoder don't have matching dimensions: {pos_dim} != {self.dim}"

        # mask token is Parameter so it gets automatically put on the correct device
        self.padding_value = padding_value
        self.norm = nn.LayerNorm(self.dim)

    def forward(self, x, center_points, padding_mask=None, attn_mask=None):
        relative_pos = center_points[:, 1:, :] - center_points[:, :-1, :]
        # prepend absolute position of first center point
        relative_pos = torch.cat(
            [center_points[:, 0, :].unsqueeze(1), relative_pos], dim=1
        )
        relative_pos = self.pos_embedder(relative_pos)
        x = self.transformer_decoder(
            x, relative_pos, padding_mask=padding_mask, attn_mask=attn_mask
        )
        x = self.norm(x)
        return x
