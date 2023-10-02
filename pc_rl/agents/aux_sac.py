from __future__ import annotations

import torch
from parllel import ArrayDict, ArrayTree, Index, dict_map
from parllel.torch.distributions.squashed_gaussian import SquashedGaussian
from torch import Tensor

from pc_rl.agents.sac import PcSacAgent
from pc_rl.utils.array_dict import dict_to_batched_data


class AuxPcSacAgent(PcSacAgent):
    def __init__(
        self,
        model: torch.nn.ModuleDict,
        distribution: SquashedGaussian,
        device: torch.device,
        learning_starts: int = 0,
        pretrain_std: float = 0.75,
    ):
        super().__init__(
            model=model,
            distribution=distribution,
            device=device,
            learning_starts=learning_starts,
            pretrain_std=pretrain_std,
        )

    def auto_encoder(self, observation: ArrayTree[Tensor]) -> tuple[Tensor, Tensor]:
        pos, batch, color = dict_to_batched_data(observation)
        x, neighborhoods, center_points = self.model["embedder"](pos, batch, color)
        prediction, ground_truth = self.model["rl_mae"](x, neighborhoods, center_points)
        return prediction, ground_truth
