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
        self.token_dim = token_dim

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        B, G, _ = xyz.shape
        dim_t = torch.pow(
            self.temperature,
            2
            * torch.arange(0, self.num_pos_features // 2, device=xyz.device)
            / self.num_pos_features,
        )
        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t

        pos_sin = pos_divided.sin()
        pos_cos = pos_divided.cos()
        pos_emb = torch.zeros(
            B, G, self.token_dim, dtype=torch.float32, device=xyz.device
        )
        pos_emb[..., 0::2] = pos_sin.reshape(B, G, -1)
        pos_emb[..., 1::2] = pos_cos.reshape(B, G, -1)

        pos_emb = F.pad(pos_emb, (0, self.padding))
        return pos_emb
