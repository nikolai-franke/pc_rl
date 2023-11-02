from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        attention: MultiheadAttention,
        mlp: nn.Module,
        NormLayer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.attention = attention
        self.dim = self.attention.embed_dim
        self.mlp = mlp
        self.norm_1 = NormLayer(self.dim)
        self.norm_2 = NormLayer(self.dim)
        self._check_mlp()

    def _check_mlp(self):
        with torch.no_grad():
            input = torch.randn((self.dim,))
            try:
                out = self.mlp(input)
                out = self.norm_1(out)
            except RuntimeError as e:
                raise ValueError(
                    f"The first and the layer of the MLP must have the same size as the embedding_dim: {self.dim}"
                ) from e

    def forward(self, x, padding_mask=None, attn_mask=None):
        x = self.norm_1(x)
        x = (
            x
            + self.attention(
                x,
                x,
                x,
                need_weights=False,
                key_padding_mask=padding_mask,
                attn_mask=attn_mask,
            )[0]
        )
        x = self.norm_2(x)
        x = x + self.mlp(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, blocks: list[TransformerBlock]) -> None:
        super().__init__()
        self.dim = blocks[0].dim
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, pos, padding_mask=None, attn_mask=None):
        for block in self.blocks:
            x = block(x + pos, padding_mask=padding_mask, attn_mask=attn_mask)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        blocks: list[TransformerBlock],
        NormLayer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.dim = blocks[0].dim
        self.blocks = nn.ModuleList(blocks)
        self.norm = NormLayer(self.dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num, padding_mask=None, attn_mask=None):
        for block in self.blocks:
            x = block(x + pos, padding_mask=padding_mask, attn_mask=attn_mask)

        x = self.norm(x[:, -return_token_num:])
        return x
