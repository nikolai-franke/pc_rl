from __future__ import annotations

import copy

import torch
from parllel import ArrayTree
from parllel.torch.agents.sac_agent import SacAgent
from parllel.torch.distributions.squashed_gaussian import SquashedGaussian
from parllel.torch.utils import update_state_dict
from torch import Tensor


class ImageSacAgent(SacAgent):
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
        encoder_out = self.model["encoder"](observation)
        return encoder_out

    def target_encode(self, observation: ArrayTree[Tensor]) -> Tensor:
        encoder_out = self.model["target_encoder"](observation)
        return encoder_out

    def update_target(self, tau: float | int = 1) -> None:
        super().update_target(tau)
        update_state_dict(
            self.model["target_encoder"], self.model["encoder"].state_dict(), tau
        )
