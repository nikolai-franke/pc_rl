import torch
from torch.nn import Module
from torch_geometric.nn import MLP, global_max_pool


class PointNet(Module):
    def __init__(self, point_dim: int) -> None:
        super().__init__()
        self.mlp1 = MLP([point_dim, 128, 256, 512], norm="layer_norm", plain_last=False)
        self.mlp2 = MLP([512, 384])

    def forward(self, pos, batch, color=None):
        input = pos if color is None else torch.hstack((pos, color))
        x = self.mlp1(input)
        x = global_max_pool(x, batch)
        return self.mlp2(x)
