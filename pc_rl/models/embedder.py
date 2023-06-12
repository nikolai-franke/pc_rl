from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch_geometric.nn import MLP, fps, knn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (Adj, OptTensor, PairOptTensor, PairTensor,
                                    SparseTensor, torch_sparse)
from torch_geometric.utils import add_self_loops, remove_self_loops


class Embedder(torch.nn.Module):
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
        x = self.conv((None, None), (pos, pos[seed_idx]), edges)
        return x


class PointConv(MessagePassing):
    def __init__(
        self,
        mlp_1: Callable,
        mlp_2: Callable,
        neighborhood_size: int = 32,
        add_self_loops: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "max")
        super().__init__(**kwargs)

        self.mlp_1 = mlp_1
        self.mlp_2 = mlp_2
        self.add_self_loops = add_self_loops
        self.neighborhood_size = neighborhood_size

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.mlp_1)
        reset(self.mlp_2)

    def forward(
        self,
        x: Union[OptTensor, PairOptTensor],  # type: ignore
        pos: Union[Tensor, PairTensor],  # type: ignore
        edge_index: Adj,
    ) -> Tensor:
        if not isinstance(x, tuple):
            x: PairOptTensor = (x, None)

        if isinstance(pos, Tensor):
            pos: PairTensor = (pos, pos)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(
                    edge_index, num_nodes=min(pos[0].size(0), pos[1].size(0))
                )
            elif isinstance(edge_index, SparseTensor):
                edge_index = torch_sparse.set_diag(edge_index)  # type: ignore

        out = self.propagate(edge_index, x=x, pos=pos, size=None)

        return out

    def message(self, x_j: Optional[Tensor], pos_i: Tensor, pos_j: Tensor) -> Tensor:
        msg = pos_j - pos_i
        if x_j is not None:
            msg = torch.cat([x_j, msg], dim=1)

        msg = self.mlp_1(msg)
        # reshape into shape [n_groups, neighborhood_size, MLP_out_dim]
        msg = msg.reshape(-1, self.neighborhood_size, msg.shape[-1])
        # get max over neighborhood
        msg_max = torch.max(msg, dim=1, keepdim=True)[0]
        # add the neighborhood max to the original msg for each node
        msg = torch.cat([msg_max.expand(-1, self.neighborhood_size, -1), msg], dim=2)
        msg = self.mlp_2(msg.reshape(-1, msg.shape[-1]))

        return msg

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mlp_1={self.mlp_1},(mlp_2={self.mlp_2})"
