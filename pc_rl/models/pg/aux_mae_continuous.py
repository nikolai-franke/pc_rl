from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from parllel.torch.distributions.gaussian import DistParams
from parllel.torch.utils import infer_leading_dims, restore_leading_dims
from torch import Tensor
from typing_extensions import NotRequired

from pc_rl.models.aux_mae import AuxMae
from pc_rl.models.modules.embedder import Embedder
from pc_rl.utils.array_dict import dict_to_batched_data


class ModelOutputs(TypedDict):
    dist_params: DistParams
    value: NotRequired[Tensor]
    pos_prediction: Tensor
    ground_truth: Tensor


class AuxMaeContinuousPgModel(nn.Module):
    def __init__(
        self,
        embedder: Embedder,
        aux_mae: AuxMae,
        pi_mlp: nn.Module,
        value_mlp: nn.Module,
        init_log_std: float,
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.aux_mae = aux_mae
        self.pi_mlp = pi_mlp
        self.value_mlp = value_mlp
        self._check_mlps()
        action_size = self._get_action_size()
        self.log_std = nn.Parameter(torch.full((action_size,), init_log_std))

    @torch.no_grad()
    def _check_mlps(self):
        input = torch.randn((self.aux_mae.dim,))
        try:
            self.pi_mlp(input)
        except RuntimeError as e:
            raise ValueError(
                f"The first layer of the Pi MLP must have the same size as the output of the encoder: {self.encoder.dim}"
            ) from e
        try:
            self.value_mlp(input)
        except RuntimeError as e:
            raise ValueError(
                f"The first layer of the Value MLP must have the same size as the output of the encoder: {self.encoder.dim}"
            ) from e

    @torch.no_grad()
    def _get_action_size(self):
        input = torch.randn((self.aux_mae.dim,))
        return len(self.pi_mlp(input))

    def forward(self, data):
        pos, batch = dict_to_batched_data(data)
        x, neighborhoods, center_points = self.embedder(pos, batch)
        x, pos_prediction, pos_ground_truth = self.aux_mae(
            x, center_points, neighborhoods
        )

        lead_dim, B, T, _ = infer_leading_dims(x, 1)
        x = x.view(T * B, -1)
        mean = self.pi_mlp(x)
        mean = F.softmax(mean, dim=-1)
        value = self.value_mlp(x).squeeze(-1)
        log_std = self.log_std.repeat(T * B, 1)
        mean, value, log_std, pos_prediction, pos_ground_truth = restore_leading_dims(
            (mean, value, log_std, pos_prediction, pos_ground_truth), lead_dim, T, B
        )

        return ModelOutputs(
            dist_params=DistParams(mean=mean, log_std=log_std),
            value=value,
            pos_prediction=pos_prediction,
            ground_truth=pos_ground_truth,
        )
