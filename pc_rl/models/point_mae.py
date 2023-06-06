import torch
import torch.nn as nn
from torch_geometric.nn import MLP, fps, knn

from pc_rl.models.embedder import PointConv


class Embedder(nn.Module):
    def __init__(
        self,
        sampling_ratio: float,
        neighborhood_size: int,
        embedding_size: int,
        random_start: bool = True,  # setting this to False is useful for testing purposes
    ) -> None:
        super().__init__()
        self.sampling_ratio = sampling_ratio
        self.neighborhood_size = neighborhood_size
        self.random_start = random_start
        mlp_1 = MLP([3, 128, 256])
        mlp_2 = MLP([512, 512, embedding_size])
        self.conv = PointConv(
            mlp_1, mlp_2, neighborhood_size=neighborhood_size, add_self_loops=False
        )

    def forward(self, x, pos, batch):
        # get indices of group center points via furthest point sampling
        seed_idx = fps(
            pos, batch, ratio=self.sampling_ratio, random_start=self.random_start
        )
        seeds = pos[seed_idx]
        batch_y = batch[seed_idx]
        # calculate neighborhood via KNN
        from_idx, to_idx = knn(
            pos, seeds, self.neighborhood_size, batch_x=batch, batch_y=batch_y
        )
        edges = torch.stack([to_idx, from_idx], dim=0)
        print(edges)
        x = self.conv((None, None), (pos, pos[seed_idx]), edges)
        return x
