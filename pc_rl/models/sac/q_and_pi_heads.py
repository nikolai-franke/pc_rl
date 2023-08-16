from typing import Sequence

import torch
import torch.nn as nn
from parllel.torch.models import MlpModel
from parllel.torch.utils import infer_leading_dims, restore_leading_dims
from torch import Tensor

from parllel.torch.agents.sac_agent import PiModelOutputs, QModelOutputs


class PiMlpHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        action_size: int,
        hidden_sizes: int | Sequence[int] | None,
        hidden_nonlinearity: str,  # type: ignore
    ):
        super().__init__()
        hidden_nonlinearity: type[nn.Module] = getattr(nn, hidden_nonlinearity)
        self._action_size = action_size

        self.mlp = MlpModel(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=action_size * 2,
            hidden_nonlinearity=hidden_nonlinearity,
        )

    def forward(self, observation: Tensor) -> PiModelOutputs:
        lead_dim, T, B, _ = infer_leading_dims(observation, 1)
        output = self.mlp(observation.view(T * B, -1))
        mean, log_std = output[:, : self._action_size], output[:, self._action_size :]
        mean, log_std = restore_leading_dims((mean, log_std), lead_dim, T, B)
        return PiModelOutputs(mean=mean, log_std=log_std)


class QMlpHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        action_size: int,
        hidden_sizes: int | Sequence[int] | None,
        hidden_nonlinearity: str,  # type: ignore
    ):
        super().__init__()
        hidden_nonlinearity: type[nn.Module] = getattr(nn, hidden_nonlinearity)

        self.mlp = MlpModel(
            input_size=input_size + action_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            hidden_nonlinearity=hidden_nonlinearity,
        )

    def forward(self, observation: Tensor, action: Tensor) -> QModelOutputs:
        lead_dim, T, B, _ = infer_leading_dims(observation, 1)
        q_input = torch.cat(
            [observation.view(T * B, -1), action.view(T * B, -1)], dim=1
        )
        q = self.mlp(q_input).squeeze(-1)
        q = restore_leading_dims(q, lead_dim, T, B)
        return QModelOutputs(q_value=q)
