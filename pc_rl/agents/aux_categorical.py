from dataclasses import dataclass
from typing import Optional, Union

import torch
from parllel.buffers import Buffer, NamedArrayTupleClass, buffer_asarray
from parllel.handlers.agent import AgentStep
from parllel.torch.agents.categorical import AgentInfo
from parllel.torch.algos.ppo import TorchAgent
from parllel.torch.distributions.categorical import Categorical, DistInfo
from parllel.torch.utils import buffer_to_device, torchify_buffer
from torch import nn

AgentPrediction = NamedArrayTupleClass(
    "MaePgAgentPrediction",
    [
        "dist_info",
        "value",
        "pos_prediction",
        "ground_truth",
    ],
)


@dataclass(frozen=True)
class ModelOutputs:
    pi: Buffer
    value: Buffer
    pos_prediction: Buffer
    ground_truth: Buffer


class MaeCategoricalPgAgent(TorchAgent):
    def __init__(
        self,
        model: nn.Module,
        distribution: Categorical,
        example_obs: Buffer,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(model, distribution, device)
        example_obs = buffer_asarray(example_obs)
        example_obs = torchify_buffer(example_obs)
        exmple_inputs = (example_obs,)
        example_inputs = buffer_to_device(exmple_inputs, device=self.device)

        # TODO: error message
        with torch.no_grad():
            try:
                model_outputs: ModelOutputs = self.model(*example_inputs)
            except TypeError as e:
                raise TypeError("Useful error message!") from e

    @torch.no_grad()
    def step(
        self, observation: Buffer, *, env_indices: Union[int, slice] = ...
    ) -> AgentStep:
        model_inputs = (observation,)
        model_inputs = buffer_to_device(model_inputs, device=self.device)
        model_outputs: ModelOutputs = self.model(*model_inputs)
        dist_info = DistInfo(prob=model_outputs.pi)
        action = self.distribution.sample(dist_info)
        value = model_outputs.value
        # TODO: AgentInfo without previous action?
        agent_info = AgentInfo(dist_info, value, None)
        agent_step = AgentStep(action=action, agent_info=agent_info)

        return buffer_to_device(agent_step, device="cpu")

    @torch.no_grad()
    def value(self, observation: Buffer) -> Buffer:
        model_inputs = (observation,)
        model_inputs = buffer_to_device(model_inputs, device=self.device)
        model_outputs: ModelOutputs = self.model(*model_inputs)
        value = model_outputs.value
        return buffer_to_device(value, device="cpu")

    def predict(self, observation: Buffer, agent_info: AgentInfo) -> AgentPrediction:
        model_inputs = (observation,)
        model_outputs: ModelOutputs = self.model(*model_inputs)
        dist_info = DistInfo(prob=model_outputs.pi)
        value = model_outputs.value
        pos_prediction = model_outputs.pos_prediction 
        ground_truth = model_outputs.ground_truth
        prediction = AgentPrediction(dist_info, value, pos_prediction, ground_truth)
        return prediction
