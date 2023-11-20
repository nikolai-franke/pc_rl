from __future__ import annotations

import torch
import torch.nn as nn

from pc_rl.models.modules.sin_pos_embedder import SinusoidalPosEmbedder

from .transformer import TransformerEncoder


class GPTEncoder(nn.Module):
    def __init__(
        self,
        mask_ratio: float,
        keep_first_tokens_ratio: float,
        transformer_encoder: TransformerEncoder,
        pos_embedder: SinusoidalPosEmbedder,
        padding_value: float = -1.0,
        start_token_value: float = 1.0,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.keep_first_tokens_ratio = keep_first_tokens_ratio
        self.pos_embedder = pos_embedder
        self.transformer_encoder = transformer_encoder
        self.dim = self.transformer_encoder.dim
        self.padding_value = padding_value
        self.start_token_value = start_token_value
        self.norm = nn.LayerNorm(self.dim)
        self.num_attention_heads = self.transformer_encoder.blocks[
            0
        ].attention.num_heads  # TODO: maybe just give the number of attention heads as parameter
        self._check_pos_embedder()
        self.apply(self._init_weights)

    def _check_pos_embedder(self):
        with torch.no_grad():
            input = torch.randn((3,))
            try:
                out = self.pos_embedder(input)
                out = self.norm(out)
            except RuntimeError as e:
                raise ValueError(
                    f"The first and the last layer of the pos_embedder must have the same size as the embedding_dim: {self.dim}"
                ) from e

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _mask_center_rand(self, center, padding_mask):
        B, G, _ = center.shape
        if self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        # count how many center points are paddings in each batch
        num_padding_tokens = torch.count_nonzero(padding_mask, dim=-1)
        # calculate how many center points should be masked in each batch (considering that paddings should NOT be masked)
        # fewer real tokens => fewer masks
        num_non_padding_tokens = G - num_padding_tokens
        num_kept_first_tokens = (
            num_non_padding_tokens * self.keep_first_tokens_ratio
        ).int()
        num_masks = (
            (num_non_padding_tokens - num_kept_first_tokens) * self.mask_ratio
        ).int()
        max_num_masks = torch.max(num_masks)

        overall_mask = torch.zeros([B, G])
        for i, (n_non_padding_tokens, n_masks, n_kept_first_tokens) in enumerate(
            zip(num_non_padding_tokens, num_masks, num_kept_first_tokens)
        ):
            mask = torch.hstack(
                [
                    # we only want a random mask in the range [n_kept_first_tokens, non_padding_tokens]
                    torch.ones(n_masks, dtype=torch.bool),  # type: ignore
                    torch.zeros(
                        n_non_padding_tokens - n_masks - n_kept_first_tokens,
                        dtype=torch.bool,
                    ),
                ]
            )
            rand_idx = torch.randperm(len(mask))
            mask = mask[rand_idx]
            # since we want all masks to have the same number of ones and zeros, we first fill each tensor up with ones
            # we also prepend num_kept_first_tokens zeros
            mask = torch.hstack(
                [
                    torch.zeros(n_kept_first_tokens, dtype=torch.bool),
                    mask,
                    torch.ones(max_num_masks - n_masks, dtype=torch.bool),
                ]
            )
            # and then fill each tensor with zeros until each tensor has length G
            mask = torch.hstack([mask, torch.zeros(G - len(mask))])
            overall_mask[i, :] = mask

        return overall_mask.bool().to(center.device)

    def forward(self, x, center_points):
        padding_mask = torch.all(center_points == self.padding_value, dim=-1)

        batch_size, max_num_groups, *_ = x.shape
        start_token = torch.full(
            (batch_size, 1, self.dim), self.start_token_value, device=x.device
        )
        # since the last token is never used during pretraining, we can discard it and the corresponding center_point
        x = torch.cat([start_token, x[:, :-1, :]], dim=1)
        pos = self.pos_embedder(center_points[:, :-1:, :])

        start_token_pos = torch.full(
            (batch_size, 1, self.dim),
            self.start_token_value,
            device=x.device,
        )
        pos = torch.cat([start_token_pos, pos], dim=1)

        # the standard transformer attention mask which only allows attending preceding tokens
        vanilla_mask = torch.triu(
            torch.ones(
                (max_num_groups, max_num_groups), dtype=torch.bool, device=x.device
            ),
            diagonal=1,
        )
        random_mask = self._mask_center_rand(center_points, padding_mask)
        eye_mask = torch.eye(max_num_groups, dtype=torch.bool, device=x.device)

        attn_mask = vanilla_mask | random_mask.unsqueeze(1) & ~eye_mask
        # the attention module expects one mask for every attention head when attn_mask is 3D
        attn_mask = attn_mask.repeat_interleave(self.num_attention_heads, dim=0)  # type: ignore

        x = self.transformer_encoder(
            x, pos, padding_mask=padding_mask, attn_mask=attn_mask
        )
        x = self.norm(x)

        return x, padding_mask, attn_mask
