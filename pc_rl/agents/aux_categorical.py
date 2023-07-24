from typing import TypedDict

import torch
import torch.nn as nn
from parllel import Array, ArrayDict, ArrayTree, Index, dict_map
from parllel.torch.distributions.categorical import Categorical, DistParams
from torch import Tensor
from typing_extensions import NotRequired

from pc_rl.agents.pg import AuxPgAgent, AuxPgPrediction


class ModelOutputs(TypedDict):
    dist_params: DistParams
    value: NotRequired[Tensor]
    pos_prediction: Tensor
    ground_truth: Tensor


class MaeCategoricalPgAgent(AuxPgAgent):
    def __init__(
        self,
        model: nn.Module,
        distribution: Categorical,
        example_obs: ArrayTree[Array],
        device: torch.device | None = None,
    ) -> None:
        super().__init__(model, distribution, device)

        example_obs = example_obs.to_ndarray()
        example_obs = dict_map(torch.from_numpy, example_obs)
        example_obs = example_obs.to(device=self.device)
        example_inputs = (example_obs,)

        # TODO: error message
        with torch.no_grad():
            try:
                model_outputs: ModelOutputs = self.model(*example_inputs)
            except TypeError as e:
                raise TypeError("Useful error message!") from e

    @torch.no_grad()
    def step(
        self, observation: ArrayTree[Array], *, env_indices: Index = ...
    ) -> tuple[ArrayTree[Tensor], ArrayDict[Tensor]]:
        observation = observation.to_ndarray()
        observation = dict_map(torch.from_numpy, observation)
        observation = observation.to(device=self.device)
        model_inputs = (observation,)
        model_outputs: ModelOutputs = self.model(*model_inputs)

        dist_params = model_outputs["dist_params"]
        action = self.distribution.sample(dist_params)

        agent_info = ArrayDict({"dist_params": dist_params})
        if "value" in model_outputs:
            agent_info["value"] = model_outputs["value"]

        return action.cpu(), agent_info.cpu()

    @torch.no_grad()
    def value(self, observation: ArrayTree[Array]) -> Tensor:
        observation = observation.to_ndarray()
        observation = dict_map(torch.from_numpy, observation)
        observation = observation.to(device=self.device)
        model_inputs = (observation,)
        model_outputs: ModelOutputs = self.model(*model_inputs)
        assert "value" in model_outputs
        value = model_outputs["value"]
        return value.cpu()

    def predict(
        self, observation: ArrayTree[Tensor], agent_info: ArrayDict[Tensor]
    ) -> AuxPgPrediction:
        model_inputs = (observation,)
        model_outputs: ModelOutputs = self.model(*model_inputs)
        return model_outputs
