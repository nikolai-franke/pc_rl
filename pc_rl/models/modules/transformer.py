from typing import Callable, Type

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import MultiheadAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        attention: MultiheadAttention,
        mlp: Callable[[Tensor], Tensor],
        NormLayer: Type[nn.Module] = nn.LayerNorm,
        padding_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention = attention
        self.dim = self.attention.embed_dim
        self.mlp = mlp
        self.padding_token = torch.full((1, self.dim), padding_value)
        # the first and last channel of the mlp must have the same size as the attention layer
        assert (
            self.mlp.channel_list[0] == self.dim
            and self.mlp.channel_list[-1] == self.dim
        )

        self.norm_1 = NormLayer(self.dim)
        self.norm_2 = NormLayer(self.dim)

    def forward(self, x):
        padding_mask = torch.all(x == self.padding_token, dim=-1)
        x = self.norm_1(x)
        attn = self.attention(x, x, x, need_weights=True, key_padding_mask=padding_mask)
        # attn = self.attention(x, x, x, need_weights=True)
        x = x + attn[0]
        x = self.norm_2(x)
        x = x + self.mlp(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, blocks: nn.ModuleList) -> None:
        super().__init__()
        self.blocks = blocks
        self.dim = self.blocks[0].dim

    def forward(self, x, pos):
        for block in self.blocks:
            x = block(x + pos)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        blocks: nn.ModuleList,
        NormLayer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.dim = self.blocks[0].dim
        self.norm = NormLayer(self.dim)
        self.head = nn.Identity()  # TODO: maybe remove this if we don't need a head

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for block in self.blocks:
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))
        return x


class MaskedEncoder(nn.Module):
    def __init__(
        self,
        mask_ratio: float,
        transformer_encoder: Callable[[Tensor, Tensor], Tensor],
        pos_embedder: Callable[[Tensor], Tensor],
        mask_type: str = "rand",  # TODO:check if we need different mask_types
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.pos_embedder = pos_embedder
        self.transformer_encoder = transformer_encoder
        assert hasattr(
            self.transformer_encoder, "dim"
        ), f"Encoder {self.transformer_encoder} does not have a 'dim' attribute "
        self.embedding_size = self.transformer_encoder.dim
        self.norm = nn.LayerNorm(self.embedding_size)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        if noaug or self.mask_ratio == 0:
            return torch.ones(center.shape[:2].bool())
        mask_idx = []
        for points in center:
            points = points.unsqueeze(0)
            index = torch.randint(points.shape[1] - 1, (1,))
            distance_matrix = torch.norm(
                points[:, index].reshape(1, 1, 3) - points, p=2, dim=-1
            )

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]
            mask_num = int(self.mask_ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        overall_mask = ~torch.stack(mask_idx).to(center.device)

        return overall_mask

    def _mask_center_rand(self, center, noaug=False):
        B, G, _ = center.shape
        if noaug or self.mask_ratio == 0:
            return torch.ones(center.shape[:2]).bool()

        self.num_masks = int(self.mask_ratio * G)

        overall_mask = torch.zeros([B, G])
        for i in range(B):
            mask = torch.hstack(
                [
                    torch.zeros(self.num_masks),
                    torch.ones(G - self.num_masks),
                ]
            )
            rand_idx = torch.randperm(len(mask))
            mask = mask[rand_idx]
            overall_mask[i, :] = mask

        return overall_mask.bool().to(center.device)

    def forward(self, x, center_points, noaug=False):
        # TODO: check if we really need the noaug parameter

        if self.mask_type == "rand":
            ae_mask = self._mask_center_rand(center_points, noaug=noaug)
        else:
            ae_mask = self._mask_center_block(center_points, noaug=noaug)

        batch_size, _, C = x.shape

        x_vis = x[ae_mask].reshape(batch_size, -1, C)
        masked_center = center_points[ae_mask].reshape(batch_size, -1, 3)
        pos = self.pos_embedder(masked_center)

        x_vis = self.transformer_encoder(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, ae_mask


class MaskedDecoder(nn.Module):
    def __init__(
        self,
        transformer_decoder: Callable[[Tensor, Tensor, int], Tensor],
        pos_embedder: Callable[[Tensor], Tensor],
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
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x_vis, mask, center_points):
        B, _, C = x_vis.shape
        pos_embedding_visible = self.pos_embedder(center_points[mask]).reshape(B, -1, C)
        pos_embedding_masked = self.pos_embedder(center_points[~mask]).reshape(B, -1, C)

        _, num_masked_tokens, _ = pos_embedding_masked.shape
        mask_token = self.mask_token.expand(B, num_masked_tokens, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_embedding_visible, pos_embedding_masked], dim=1)

        x_recovered = self.transformer_decoder(x_full, pos_full, num_masked_tokens)

        return x_recovered
