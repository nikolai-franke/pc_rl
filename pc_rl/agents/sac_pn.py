from __future__ import annotations

import copy

import torch
from parllel import ArrayTree
from parllel.torch.agents.sac_agent import SacAgent
from parllel.torch.distributions.squashed_gaussian import SquashedGaussian
from parllel.torch.utils import update_state_dict
from torch import Tensor

from pc_rl.utils.array_dict import dict_to_batched_data


class PointNetSacAgent(SacAgent):
    def __init__(
        self,
        model: torch.nn.ModuleDict,
        distribution: SquashedGaussian,
        device: torch.device,
        learning_starts: int = 0,
        pretrain_std: float = 0.75,  # With squash 0.75 is near uniform.
    ) -> None:
        super().__init__(model, distribution, device, learning_starts, pretrain_std)
        model["target_encoder"] = copy.deepcopy(model["encoder"])
        model["target_encoder"].requires_grad_(False)

    def encode(self, observation: ArrayTree[Tensor]) -> Tensor:
        pos, batch, color = dict_to_batched_data(observation)
        return self.model["encoder"](pos, batch, color)

    def target_encode(self, observation: ArrayTree[Tensor]) -> Tensor:
        pos, batch, color = dict_to_batched_data(observation)
        return self.model["target_encoder"](pos, batch, color)

    def update_target(self, tau: float | int = 1) -> None:
        super().update_target(tau)
        update_state_dict(
            self.model["target_encoder"], self.model["encoder"].state_dict(), tau
        )
