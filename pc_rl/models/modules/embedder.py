from __future__ import annotations

import sys
from typing import Optional, Tuple

import parllel.logger as logger
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import MLP, fps, knn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.utils import unbatch


class Embedder(MessagePassing):
    def __init__(
        self,
        mlp_1: MLP,
        mlp_2: MLP,
        group_size: int,
        sampling_ratio: float,
        random_start: bool = True,
        padding_value: float = 0.0,
        **kwargs,
    ):
        """
        Embedding module, which divides a point cloud into groups and uses the two MLPs to embed each group.

        :param hidden_layers: the dimensions of the hidden layers of both MLPs combined.
        :param embedding_size: the size of the embeddings
        :param group_size: the number of points contained in each neighborhood
        :param sampling_ratio: sampling ratio of the furthest point sampling algorithm
        :param random_start: whether or not to use a random point as the first neighborhood center
        """
        kwargs.setdefault("aggr", "max")
        super().__init__(**kwargs)

        self.group_size = group_size
        self.sampling_ratio = sampling_ratio
        self.random_start = random_start
        self.padding_value = padding_value

        self.mlp_1 = mlp_1
        self.mlp_2 = mlp_2
        self.color_embedder = MLP([3, 128, 384], act="gelu", norm=None)
        self.points_dim = 6

        assert (
            self.mlp_1.channel_list[-1] * 2 == self.mlp_2.channel_list[0]
        ), f"The last layer of mlp_1 (size {self.mlp_1.channel_list[-1]}) must be half the size of the first layer of mlp_2 (size {self.mlp_2.channel_list[0]})"
        self.embedding_size = self.mlp_2.channel_list[-1]

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.mlp_1)
        reset(self.mlp_2)

    def forward(
        self, pos: Tensor, batch: Tensor, color: Tensor | None = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Takes points as input, selects center points via furthest point
        sampling, creates local neighborhoods via k-nearest-neighbors sampling,
        and embeds the local neighborhoods with the two MLPs.

        B: batch size
        N: number of points
        G: number of groups
        M: neighborhood size
        E: embedding size

        :param pos: [B * N, 3] Tensor containing the points
        :param batch: [B * N, 1] Tensor assigning each point in 'pos' to a batch
        :returns:
            - x - [B, G, M, E] Tensor containing the embeddings
            - neighborhoods - [B, G, N, 3] Tensor containing the neighborhoods in local coordinates (with respect to the neighborhood center)
            - center_points - [B, G, 3] Tensor containing the center points of each neighborhood
        """
        _, count = torch.unique(batch, return_counts=True)

        assert torch.all(count >= self.group_size), f"COUNT: {count}"

        center_points_idx = fps(
            pos, batch, ratio=self.sampling_ratio, random_start=self.random_start
        )
        center_points = pos[center_points_idx]
        batch_y = batch[center_points_idx]

        from_idx, to_idx = knn(
            pos, center_points, self.group_size, batch_x=batch, batch_y=batch_y
        )
        edges = torch.stack([to_idx, from_idx], dim=0)
        color = (color, color[center_points_idx]) if color is not None else (None, None)
        try:
            x, neighborhoods = self.propagate(edges, pos=(pos, center_points), x=color)
        except RuntimeError as e:
            torch.set_printoptions(threshold=sys.maxsize)
            logger.log(
                f"POS SHAPE: {pos.shape}\n BATCH SHAPE: {batch.shape}\n CENTER POINTS SHAPE: {center_points.shape}\n EDGES SHAPE: {edges.shape}\n"
            )
            raise e

        # pad and reshape into [B, M, E]
        x = pad_sequence(
            unbatch(x, batch_y),
            padding_value=self.padding_value,
            batch_first=True,
        )

        # pad and reshape into [B, G, M, 3]
        neighborhoods = pad_sequence(
            unbatch(
                neighborhoods.reshape(-1, self.group_size, self.points_dim), batch_y
            ),
            padding_value=self.padding_value,
            batch_first=True,
        )
        # pad and reshape into [B, G, 3]
        center_points = pad_sequence(
            unbatch(center_points, batch_y),
            padding_value=self.padding_value,
            batch_first=True,
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

    def message(self, pos_i: Tensor, pos_j: Tensor, x_j: Tensor):
        neighborhood = pos_j - pos_i

        # if x_j is not None:
        #     neighborhood = torch.cat([neighborhood, x_j], dim=1)

        msg = self.mlp_1(neighborhood)
        # reshape into shape [G, M, mlp_1_out_dim]
        msg = msg.reshape(-1, self.group_size, msg.shape[-1])
        # get max over neighborhood
        msg_max = torch.max(msg, dim=1, keepdim=True)[0]
        # add the neighborhood max to the original msg for each node
        msg = torch.cat([msg_max.expand(-1, self.group_size, -1), msg], dim=2)
        msg = self.mlp_2(msg.reshape(-1, msg.shape[-1]))

        if x_j is not None:
            color_embedding = self.color_embedder(x_j).reshape(-1, msg.shape[-1])
            msg = msg + color_embedding
            neighborhood = torch.cat([neighborhood, x_j], dim=1)

        return msg, neighborhood

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mlp_1={self.mlp_1},(mlp_2={self.mlp_2})"
