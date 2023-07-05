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
        self.mlp_head = mlp_head
        self.dim = self.transformer_encoder.dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))  # type: ignore
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.dim))  # type: ignore
        self.norm = nn.LayerNorm(self.dim)  # type: ignore

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
