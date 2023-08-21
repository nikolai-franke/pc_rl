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

    def encode(self, observation: ArrayTree[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        pos, batch = dict_to_batched_data(observation)
        x, _, center_points = self.model["embedder"](pos, batch)
        x = self.model["encoder"](x, center_points)
        return x

    def target_encode(self, observation: ArrayTree[Tensor]) -> Tensor:
        pos, batch = dict_to_batched_data(observation)
        x, _, center_points = self.model["target_embedder"](pos, batch)
        x = self.model["target_encoder"](x, center_points)
        return x

    def auto_encoder(self, observation: ArrayTree[Tensor]) -> tuple[Tensor, Tensor]:
        pos, batch = dict_to_batched_data(observation)
        x, neighborhoods, center_points = self.model["embedder"](pos, batch)
        pos_prediction, pos_ground_truth = self.model["rl_mae"](
            x, neighborhoods, center_points
        )
        return pos_prediction, pos_ground_truth

    @torch.no_grad()
    def step(
        self, observation: ArrayTree[Tensor], *, env_indices: Index = ...
    ) -> tuple[Tensor, ArrayDict[Tensor]]:
        observation = observation.to_ndarray()
        observation = dict_map(torch.from_numpy, observation)
        observation = observation.to(device=self.device)
        x = self.encode(observation)
        dist_params = self.model["pi"](x)
        action = self.distribution.sample(dist_params)
        return action.cpu(), ArrayDict()
