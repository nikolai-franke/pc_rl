from __future__ import annotations

import copy

import torch
from parllel import ArrayTree
from parllel.torch.agents.sac_agent import SacAgent
from parllel.torch.distributions.squashed_gaussian import SquashedGaussian
from parllel.torch.utils import update_state_dict
from torch import Tensor

from pc_rl.utils.array_dict import dict_to_batched_data


class PcSacAgent(SacAgent):
    def __init__(
        self,
        model: torch.nn.ModuleDict,
        distribution: SquashedGaussian,
        device: torch.device,
        learning_starts: int = 0,
        pretrain_std: float = 0.75,  # With squash 0.75 is near uniform.
        with_robot_state: bool = False,
    ) -> None:
        super().__init__(model, distribution, device, learning_starts, pretrain_std)
        model["target_tokenizer"] = copy.deepcopy(model["tokenizer"])
        model["target_tokenizer"].requires_grad_(False)
        model["target_encoder"] = copy.deepcopy(model["encoder"])
        model["target_encoder"].requires_grad_(False)
        self.with_robot_state = with_robot_state

    def _encode_point_cloud(self, point_cloud: ArrayTree[Tensor]) -> Tensor:
        pos, batch, color = dict_to_batched_data(point_cloud)
        x, _, center_points = self.model["tokenizer"](pos, batch, color)
        encoder_out = self.model["encoder"](x, center_points)
        return encoder_out

    def _target_encode_point_cloud(self, point_cloud: ArrayTree[Tensor]) -> Tensor:
        pos, batch, color = dict_to_batched_data(point_cloud)
        x, _, center_points = self.model["target_tokenizer"](pos, batch, color)
        encoder_out = self.model["target_encoder"](x, center_points)
        return encoder_out

    def encode(self, observation: ArrayTree[Tensor]) -> Tensor:
        if not self.with_robot_state:
            return self._encode_point_cloud(observation)

        encoder_out = self._encode_point_cloud(observation["point_cloud"])
        return torch.concatenate([encoder_out, observation["state"]], dim=-1)

    def target_encode(self, observation: ArrayTree[Tensor]) -> Tensor:
        if not self.with_robot_state:
            return self._target_encode_point_cloud(observation)

        encoder_out = self._target_encode_point_cloud(observation["point_cloud"])
        return torch.concatenate([encoder_out, observation["state"]], dim=-1)

    def update_target(self, tau: float | int = 1) -> None:
        super().update_target(tau)
        update_state_dict(
            self.model["target_encoder"], self.model["encoder"].state_dict(), tau
        )
        update_state_dict(
            self.model["target_tokenizer"], self.model["tokenizer"].state_dict(), tau
        )
