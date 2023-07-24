from __future__ import annotations

from abc import abstractmethod

from parllel import ArrayDict, ArrayOrMapping, ArrayTree
from parllel.torch.agents.pg import PgAgent, PgPrediction
from torch import Tensor
from typing_extensions import NotRequired


class AuxPgPrediction(PgPrediction):
    dist_params: ArrayOrMapping[Tensor]
    value: NotRequired[Tensor]
    pos_prediction: Tensor
    ground_truth: Tensor


class AuxPgAgent(PgAgent):
    @abstractmethod
    def predict(
        self,
        observation: ArrayTree[Tensor],
        agent_info: ArrayDict[Tensor],
        initial_rnn_state: ArrayTree[Tensor] | None,
    ) -> AuxPgPrediction:
        pass
