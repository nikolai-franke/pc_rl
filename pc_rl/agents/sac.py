import copy
from typing import TypedDict, Union

import parllel.logger as logger
import torch
from parllel import Array, ArrayDict, ArrayTree, Index, dict_map
from parllel.torch.agents.agent import TorchAgent
from parllel.torch.distributions.squashed_gaussian import (DistParams,
                                                           SquashedGaussian)
from parllel.torch.utils import update_state_dict
from torch import Tensor

from pc_rl.utils.array_dict import dict_to_batched_data

PiModelOutputs = DistParams


class QModelOutputs(TypedDict):
    q_value: Tensor


class SacAgent(TorchAgent):
    model: torch.nn.ModuleDict
    distribution: SquashedGaussian

    def __init__(
        self,
        model: torch.nn.ModuleDict,
        distribution: SquashedGaussian,
        device: torch.device,
        learning_starts: int = 0,
        pretrain_std: float = 0.75,  # With squash 0.75 is near uniform.
    ) -> None:
        """Saves input arguments; network defaults stored within."""
        model["target_q1"] = copy.deepcopy(model["q1"])
        model["target_q1"].requires_grad_(False)
        model["target_q2"] = copy.deepcopy(model["q2"])
        model["target_q2"].requires_grad_(False)

        super().__init__(model, distribution, device)

        self.learning_starts = learning_starts
        self.pretrain_std = pretrain_std
        self.model = model

    @torch.no_grad()
    def step(
        self,
        observation: ArrayTree[Array],
        *,
        env_indices: Index = ...,
    ) -> tuple[Tensor, ArrayDict[Tensor]]:
        observation = observation.to_ndarray()
        observation = dict_map(torch.from_numpy, observation)
        observation = observation.to(device=self.device)
        pos, batch = dict_to_batched_data(observation)
        x, _, center_points = self.model["embedder"](pos, batch)
        encoder_out = self.model["encoder"](x, center_points)
        pi_out: PiModelOutputs = self.model["pi"](encoder_out)
        action = self.distribution.sample(pi_out)

        return action.cpu(), ArrayDict()

    def encode(self, observation: ArrayTree[Tensor]) -> Tensor:
        pos, batch = dict_to_batched_data(observation)
        x, _, center_points = self.model["embedder"](pos, batch)
        encoder_out = self.model["encoder"](x, center_points)
        return encoder_out

    def q(
        self,
        encoder_out: Tensor,
        action: ArrayTree[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Compute twin Q-values for state/observation and input action
        (with grad)."""
        q1: QModelOutputs = self.model["q1"](encoder_out, action)
        q2: QModelOutputs = self.model["q2"](encoder_out, action)
        return q1["q_value"], q2["q_value"]

    def target_q(
        self,
        encoder_out: Tensor,
        action: ArrayTree[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Compute twin target Q-values for state/observation and input
        action."""
        target_q1: QModelOutputs = self.model["target_q1"](encoder_out, action)
        target_q2: QModelOutputs = self.model["target_q2"](encoder_out, action)
        return target_q1["q_value"], target_q2["q_value"]

    def pi(self, encoder_out: Tensor) -> tuple[Tensor, Tensor]:
        """Compute action log-probabilities for state/observation, and
        sample new action (with grad).  Uses special ``sample_loglikelihood()``
        method of Gaussian distriution, which handles action squashing
        through this process."""
        model_outputs: PiModelOutputs = self.model["pi"](encoder_out)
        action, log_pi = self.distribution.sample_loglikelihood(model_outputs)
        return action, log_pi

    def freeze_q_models(self, freeze: bool) -> None:
        self.model["q1"].requires_grad_(not freeze)
        self.model["q2"].requires_grad_(not freeze)

    def update_target(self, tau: Union[float, int] = 1) -> None:
        update_state_dict(self.model["target_q1"], self.model["q1"].state_dict(), tau)
        update_state_dict(self.model["target_q2"], self.model["q2"].state_dict(), tau)

    def sample_mode(self, elapsed_steps: int) -> None:
        super().sample_mode(elapsed_steps)
        if elapsed_steps == 0:
            logger.debug(
                f"Agent at {elapsed_steps} steps, sample std: {self.pretrain_std}"
            )
        std = None if elapsed_steps >= self.learning_starts else self.pretrain_std
        self.distribution.set_std(std)  # If None: std from policy dist_info.
