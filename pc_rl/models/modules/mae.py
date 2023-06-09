import torch
import torch.nn as nn

from .transformer import TransformerDecoder, TransformerEncoder


class MaskedEncoder(nn.Module):
    def __init__(
        self,
        mask_ratio: float,
        transformer_encoder: TransformerEncoder,
        pos_embedder: nn.Module,
        mask_type: str = "rand",  # TODO:check if we need different mask_types
        padding_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.pos_embedder = pos_embedder
        self.transformer_encoder = transformer_encoder
        self.embedding_dim = self.transformer_encoder.dim
        self.padding_token = nn.Parameter(torch.full((1, 3), padding_value))
        self.norm = nn.LayerNorm(self.embedding_dim)
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
                    f"The first and the last layer of the pos_embedder must have the same size as the embedding_dim: {self.embedding_dim}"
                ) from e

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
        # TODO: maybe we don't need this
        raise NotImplementedError
        # if noaug or self.mask_ratio == 0:
        #     return torch.ones(center.shape[:2].bool())
        # mask_idx = []
        # for points in center:
        #     points = points.unsqueeze(0)
        #     index = torch.randint(points.shape[1] - 1, (1,))
        #     distance_matrix = torch.norm(
        #         points[:, index].reshape(1, 1, 3) - points, p=2, dim=-1
        #     )

        #     idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]
        #     mask_num = int(self.mask_ratio * len(idx))
        #     mask = torch.zeros(len(idx))
        #     mask[idx[:mask_num]] = 1
        #     mask_idx.append(mask.bool())

        # overall_mask = torch.stack(mask_idx).to(center.device)

        # return overall_mask

    def _mask_center_rand(self, center, padding_mask, noaug=False):
        B, G, _ = center.shape
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        # count how many center points are paddings in each batch
        num_padding_tokens = torch.count_nonzero(padding_mask, dim=-1)
        # calculate how many center points should be masked in each batch (considering that paddings should NOT be masked)
        # fewer real tokens => fewer masks
        num_non_padding_tokens = G - num_padding_tokens
        num_masks = (num_non_padding_tokens * self.mask_ratio).int()
        max_num_masks = torch.max(num_masks)

        overall_mask = torch.zeros([B, G])
        for i, (n_non_padding_tokens, n_masks) in enumerate(
            zip(num_non_padding_tokens, num_masks)
        ):
            mask = torch.hstack(
                [
                    # we only want a random mask in the range [0, non_padding_tokens]
                    torch.ones(n_masks),  # type: ignore
                    torch.zeros(n_non_padding_tokens - n_masks),
                ]
            )
            rand_idx = torch.randperm(len(mask))
            mask = mask[rand_idx]
            # since we want all masks to have the same number of ones and zeros, we first fill each tensor up with ones
            mask = torch.hstack([mask, torch.ones(max_num_masks - n_masks)])
            # and then fill each tensor with zeros until each tensor has length G
            mask = torch.hstack([mask, torch.zeros(G - len(mask))])
            overall_mask[i, :] = mask

        return overall_mask.bool().to(center.device)

    def forward(self, x, center_points, noaug=False):
        padding_mask = torch.all(center_points == self.padding_token, dim=-1)
        if self.mask_type == "rand":
            ae_mask = self._mask_center_rand(center_points, padding_mask, noaug=noaug)
        else:
            raise NotImplementedError
            # ae_mask = self._mask_center_block(center_points, noaug=noaug)

        batch_size, _, C = x.shape

        x_vis = x[~ae_mask].reshape(batch_size, -1, C)
        center_points_vis = center_points[~ae_mask].reshape(batch_size, -1, 3)
        pos = self.pos_embedder(center_points_vis)

        # recalculate padding mask
        vis_padding_mask = torch.all(center_points_vis == self.padding_token, dim=-1)
        x_vis = self.transformer_encoder(x_vis, pos, vis_padding_mask)
        x_vis = self.norm(x_vis)

        return x_vis, ae_mask


class MaskedDecoder(nn.Module):
    def __init__(
        self,
        transformer_decoder: TransformerDecoder,
        pos_embedder: nn.Module,
        padding_value: float = 0.0,
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
        self.padding_token = nn.Parameter(torch.full((1, 3), padding_value))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x_vis, mask, center_points):
        B, _, C = x_vis.shape
        center_points_visible = center_points[~mask].reshape(B, -1, 3)
        center_points_masked = center_points[mask].reshape(B, -1, 3)
        _, num_masked_tokens, _ = center_points_masked.shape

        center_points_full = torch.cat(
            [center_points_visible, center_points_masked], dim=1
        )
        # since we reordered the center points, we have to recalculate the padding mask
        padding_mask = torch.all(center_points_full == self.padding_token, dim=-1)

        pos_full = self.pos_embedder(center_points_full).reshape(B, -1, C)

        mask_token = self.mask_token.expand(B, num_masked_tokens, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)

        x_recovered = self.transformer_decoder(
            x_full, pos_full, num_masked_tokens, padding_mask
        )

        return x_recovered, padding_mask

class PredictionHead(nn.Module):
    # TODO: maybe rewrite this using the pytorch MessagePassing interface
    def __init__(self, dim: int, group_size: int):
        super().__init__()
        self.head = nn.Conv1d(dim, 3 * group_size, 1)

    def forward(self, x):
        B, M, _ = x.shape
        prediction = self.head(x.transpose(1, 2)).transpose(1, 2)
        prediction = prediction.reshape(B, M, -1, 3)
        return prediction
