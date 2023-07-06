import torch
import torch.nn as nn

from .transformer import TransformerEncoder


class FinetuneEncoder(nn.Module):
    def __init__(
        self,
        transformer_encoder: TransformerEncoder,
        pos_embedder: nn.Module,
        mlp_head: nn.Module,
    ) -> None:
        super().__init__()
        self.transformer_encoder = transformer_encoder
        self.pos_embedder = pos_embedder
        self.dim = self.transformer_encoder.dim
        self.norm = nn.LayerNorm(self.dim)
        self.mlp_head = mlp_head
        self._check_mlp_head()
        self.out_dim = self._get_out_dim()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.dim))

    def _check_mlp_head(self):
        with torch.no_grad():
            input = torch.randn((2 * self.dim,))
            try:
                self.mlp_head(input)
            except RuntimeError as e:
                raise ValueError(
                        f"The first layer of the MLP head must have size 2 * embedding_size: {2 * self.dim}"
                ) from e

    def _get_out_dim(self):
        with torch.no_grad():
            input = torch.randn((2 * self.dim,))
            out = self.mlp_head(input)
        return out.shape[-1]

    def forward(self, x, center_points):
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        cls_pos = self.cls_pos.expand(x.shape[0], -1, -1)
        pos = self.pos_embedder(center_points)
        pos = torch.cat((cls_pos, pos), dim=1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        out = self.mlp_head(concat_f)
        return out
