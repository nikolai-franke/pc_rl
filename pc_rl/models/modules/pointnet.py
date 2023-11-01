import torch
from torch.nn import Module
from torch_geometric.nn import MLP, global_max_pool


class PointNet(Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp1 = MLP([6, 128, 256, 512], norm="layer_norm", plain_last=False)
        self.mlp2 = MLP([512, 384])

    def forward(self, pos, batch, color):
        x = self.mlp1(torch.hstack((pos, color)))
        x = global_max_pool(x, batch)
        return self.mlp2(x)
