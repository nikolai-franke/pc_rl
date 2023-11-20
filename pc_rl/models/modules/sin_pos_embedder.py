import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmbedder(nn.Module):
    def __init__(
        self,
        n_dim: int = 1,
        token_dim: int = 384,
        temperature: float = 1.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_dim = n_dim
        self.num_pos_features = token_dim // n_dim // 2 * 2
        self.temperature = temperature
        self.padding = token_dim - self.num_pos_features * self.n_dim
        self.scale = scale * 2 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        assert xyz.shape[-1] == self.n_dim
        dim_t = torch.arange(
            self.num_pos_features, dtype=torch.float32, device=xyz.device
        )
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="trunc") / self.num_pos_features
        )
        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-1).reshape(*xyz.shape[:-1], -1)

        pos_emb = F.pad(pos_emb, (0, self.padding))
        return pos_emb
