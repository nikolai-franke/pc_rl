from typing import Callable, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.nn import fps, knn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset


class Embedder(MessagePassing):
    def __init__(
        self,
        mlp_1: Callable,
        mlp_2: Callable,
        neighborhood_size: int,
        sampling_ratio: float,
        random_start: bool = True,
        **kwargs,
    ):
        """
        Embedding module, which divides a point cloud into groups and uses the two MLPs to embed each group.

        :param mlp_1: first embedding MLP
        :param mlp_2: second embedding MLP
        :param neighborhood_size: the number of points contained in each neighborhood
        :param sampling_ratio: sampling ratio of the furthest point sampling algorithm
        :param random_start: whether or not to use a random point as the first neighborhood center
        """
        kwargs.setdefault("aggr", "max")
        super().__init__(**kwargs)

        self.mlp_1 = mlp_1
        self.mlp_2 = mlp_2
        self.neighborhood_size = neighborhood_size
        self.sampling_ratio = sampling_ratio
        self.random_start = random_start

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.mlp_1)
        reset(self.mlp_2)

    def forward(
        self, pos: Tensor, batch: Tensor  # type: ignore
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Takes points as input, selects center points via furthest point
        sampling, creates local neighborhoods via k-nearest-neighbors sampling,
        and embeds the local neighborhoods with the two MLPs.

        B: batch size
        N: number of points
        M: neighborhood size
        G: number of groups
        E: embedding size

        :param pos: [B * N, 3] Tensor containing the points
        :param batch: [B * N, 1] Tensor assigning each point in 'pos' to a batch
        :returns:
            - x - [B, G, M, E] Tensor containing the embeddings
            - neighborhoods - [B, G, N, 3] Tensor containing the neighborhoods in local coordinates (with respect to the neighborhood center)
            - center_points - [B, G, 3] Tensor containing the center points of each neighborhood
        """
        center_points_idx = fps(
            pos, batch, ratio=self.sampling_ratio, random_start=self.random_start
        )
        center_points = pos[center_points_idx]
        batch_y = batch[center_points_idx]

        from_idx, to_idx = knn(
            pos, center_points, self.neighborhood_size, batch_x=batch, batch_y=batch_y
        )
        edges = torch.stack([to_idx, from_idx], dim=0)

        x, neighborhoods = self.propagate(edges, pos=(pos, center_points), size=None)
        # reshape into [B, N, X]
        x = x.reshape(-1, self.neighborhood_size, x.shape[-1])
        # reshape into [B, G, N, 3]
        neighborhoods = neighborhoods.reshape(
            x.shape[0], -1, self.neighborhood_size, neighborhoods.shape[-1]
        )
        # reshape into [B, G, 3]
        center_points = center_points.reshape(
            -1, self.neighborhood_size, center_points.shape[-1]
        )

        return x, neighborhoods, center_points

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ):
        msg, relative_pos = inputs
        return super().aggregate(msg, index, ptr, dim_size), relative_pos

    def message(self, pos_i: Tensor, pos_j: Tensor):
        relative_pos = pos_j - pos_i

        msg = self.mlp_1(relative_pos)
        # reshape into shape [n_groups, neighborhood_size, MLP_out_dim]
        msg = msg.reshape(-1, self.neighborhood_size, msg.shape[-1])
        # get max over neighborhood
        msg_max = torch.max(msg, dim=1, keepdim=True)[0]
        # add the neighborhood max to the original msg for each node
        msg = torch.cat([msg_max.expand(-1, self.neighborhood_size, -1), msg], dim=2)
        msg = self.mlp_2(msg.reshape(-1, msg.shape[-1]))

        return msg, relative_pos

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mlp_1={self.mlp_1},(mlp_2={self.mlp_2})"
