import copy
from typing import TypedDict, Union

import parllel.logger as logger
import torch
from parllel import Array, ArrayDict, ArrayTree, Index, dict_map
from parllel.torch.agents.agent import TorchAgent
from parllel.torch.distributions.squashed_gaussian import (DistParams,
                                                           SquashedGaussian)
from torch import Tensor
from parllel.torch.utils import update_state_dict

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
        model["target_q1_mlp"] = copy.deepcopy(model["q1_mlp"])
        model["target_q1_mlp"].requires_grad_(False)
        model["target_q2_mlp"] = copy.deepcopy(model["q2_mlp"])
        model["target_q2_mlp"].requires_grad_(False)

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
        embedder_out = self.model["embedder"](observation)
        encoder_out = self.model["encoder"](embedder_out)
        pi_out: PiModelOutputs = self.model["pi_mlp"](encoder_out)
        action = self.distribution.sample(pi_out)

        return action.cpu(), ArrayDict()

    def q(
        self,
        observation: ArrayTree[Tensor],
        action: ArrayTree[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Compute twin Q-values for state/observation and input action
        (with grad)."""
        embedder_out = self.model["embedder"](observation)
        encoder_out = self.model["encoder"](embedder_out)
        q1: QModelOutputs = self.model["q1_mlp"](encoder_out, action)
        q2: QModelOutputs = self.model["q2_mlp"](encoder_out, action)
        return q1["q_value"], q2["q_value"]

    def target_q(
        self,
        observation: ArrayTree[Tensor],
        action: ArrayTree[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Compute twin target Q-values for state/observation and input
        action."""
        embedder_out = self.model["embedder"](observation)
        encoder_out = self.model["encoder"](embedder_out)
        target_q1: QModelOutputs = self.model["target_q1_mlp"](encoder_out, action)
        target_q2: QModelOutputs = self.model["target_q2_mlp"](encoder_out, action)
        return target_q1["q_value"], target_q2["q_value"]

    def pi(
        self,
        observation: ArrayTree[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Compute action log-probabilities for state/observation, and
        sample new action (with grad).  Uses special ``sample_loglikelihood()``
        method of Gaussian distriution, which handles action squashing
        through this process."""
        embedder_out = self.model["embedder"](observation)
        encoder_out = self.model["encoder"](embedder_out)
        model_outputs: PiModelOutputs = self.model["pi_mlp"](encoder_out)
        action, log_pi = self.distribution.sample_loglikelihood(model_outputs)
        return action, log_pi

    def freeze_q_models(self, freeze: bool) -> None:
        self.model["q1_mlp"].requires_grad_(not freeze)
        self.model["q2_mlp"].requires_grad_(not freeze)

    def update_target(self, tau: Union[float, int] = 1) -> None:
        update_state_dict(
            self.model["target_q1_mlp"], self.model["q1_mlp"].state_dict(), tau
        )
        update_state_dict(
            self.model["target_q2_mlp"], self.model["q2_mlp"].state_dict(), tau
        )

    def train_mode(self, elapsed_steps: int) -> None:
        super().train_mode(elapsed_steps)
        self.model["embedder"].train()
        self.model["encoder"].train()
        self.model["pi_mlp"].train()
        self.model["q1_mlp"].train()
        self.model["q2_mlp"].train()

    def sample_mode(self, elapsed_steps: int) -> None:
        super().sample_mode(elapsed_steps)
        self.model["embedder"].train()
        self.model["encoder"].train()
        self.model["pi_mlp"].train()
        self.model["q1_mlp"].eval()
        self.model["q2_mlp"].eval()
        if elapsed_steps == 0:
            logger.debug(
                f"Agent at {elapsed_steps} steps, sample std: {self.pretrain_std}"
            )
        std = None if elapsed_steps >= self.learning_starts else self.pretrain_std
        self.distribution.set_std(std)  # If None: std from policy dist_info.

    def eval_mode(self, elapsed_steps: int) -> None:
        super().eval_mode(elapsed_steps)
        self.model["embedder"].train()
        self.model["encoder"].train()
        self.model["pi_mlp"].train()
        self.model["q1"].eval()
        self.model["q2"].eval()
        self.distribution.set_std(0.0) #TODO: I don't know if there is ever a reason to do this with SAC 
