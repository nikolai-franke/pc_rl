from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.nn import MLP

from .tokenizer import Tokenizer


class TokenizerSeparateColor(Tokenizer):
    def __init__(
        self,
        mlp_1: MLP,
        mlp_2: MLP,
        color_mlp: MLP,
        group_size: int,
        sampling_ratio: float,
        random_start: bool = True,
        padding_value: float = 0,
        **kwargs,
    ):
        super().__init__(
            mlp_1,
            mlp_2,
            group_size,
            sampling_ratio,
            random_start,
            padding_value,
            **kwargs,
        )

        self.color_mlp = color_mlp
        self.points_dim = self.mlp_1.channel_list[0] + self.color_mlp.channel_list[0]

    def message(self, pos_i: Tensor, pos_j: Tensor, x_j: Tensor):
        neighborhood = pos_j - pos_i

        msg = self.mlp_1(neighborhood)
        # reshape into shape [G, M, mlp_1_out_dim]
        msg = msg.reshape(-1, self.group_size, msg.shape[-1])
        # get max over neighborhood
        msg_max = torch.max(msg, dim=1, keepdim=True)[0]
        # add the neighborhood max to the original msg for each node
        msg = torch.cat([msg_max.expand(-1, self.group_size, -1), msg], dim=2)
        msg = self.mlp_2(msg.reshape(-1, msg.shape[-1]))

        # add color embedding
        color_embedding = self.color_mlp(x_j).reshape(msg.shape)
        msg += color_embedding
        neighborhood = torch.cat([neighborhood, x_j], dim=1)

        return msg, neighborhood
