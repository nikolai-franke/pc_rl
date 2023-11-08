from __future__ import annotations

from typing import Tuple

import parllel.logger as logger
import torch
from torch import Tensor
from torch_geometric.nn import MLP

from pc_rl.models.modules.tokenizer import Tokenizer
from pc_rl.utils.morton_code import get_z_values


class GPTTokenizer(Tokenizer):
    def __init__(
        self,
        mlp_1: MLP,
        mlp_2: MLP,
        group_size: int,
        sampling_ratio: float,
        random_start: bool = True,
        padding_value: float = -1.0,
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
        super().__init__(
            mlp_1=mlp_1,
            mlp_2=mlp_2,
            group_size=group_size,
            sampling_ratio=sampling_ratio,
            random_start=random_start,
            padding_valu=padding_value,
            **kwargs,
        )
        if padding_value != -1.0:
            logger.warn(
                "Padding value must be -1.0 to ensure paddings are at the end of a batch"
            )

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
        x, neighborhoods, center_points = super().forward(pos, batch, color)

        # calculate morton codes
        morton_codes = get_z_values(center_points)
        idxs = torch.argsort(morton_codes, dim=1)

        # reorder center points according to morton codes
        center_points = torch.gather(
            center_points,
            dim=1,
            index=idxs.repeat_interleave(center_points.shape[-1]).reshape(
                center_points.shape
            ),
        )

        # reorder x
        x = torch.gather(
            x,
            dim=1,
            index=idxs.repeat_interleave(x.shape[-1]).reshape(x.shape),
        )

        # reorder neighborhoods
        neighborhoods = torch.gather(
            neighborhoods,
            dim=1,
            index=idxs.repeat_interleave(
                neighborhoods.shape[-2] * neighborhoods.shape[-1]
            ).reshape(neighborhoods.shape),
        )

        return x, neighborhoods, center_points
