from __future__ import annotations

from typing import TypedDict

import torch.nn as nn
import torch.nn.functional as F
from parllel.torch.distributions.categorical import DistParams
from parllel.torch.utils import infer_leading_dims, restore_leading_dims
from torch import Tensor
from typing_extensions import NotRequired

from pc_rl.models.aux_mae import RLMae
from pc_rl.models.finetune_encoder import FinetuneEncoder
from pc_rl.models.modules.tokenizer import Tokenizer
from pc_rl.utils.array_dict import dict_to_batched_data


class ModelOutputs(TypedDict):
    dist_params: DistParams
    value: NotRequired[Tensor]
    pos_prediction: Tensor
    ground_truth: Tensor


class AuxMaeCategoricalPgModel(nn.Module):
    def __init__(
        self,
        tokenizer: Tokenizer,
        encoder: FinetuneEncoder,
        aux_mae: RLMae,
        pi_mlp: nn.Module,
        value_mlp: nn.Module,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.aux_mae = aux_mae
        self.pi = pi_mlp
        self.value = value_mlp

    def forward(self, data):
        pos, batch = dict_to_batched_data(data)
        embedder_out, neighborhoods, center_points = self.tokenizer(pos, batch)
        x = self.encoder(embedder_out, center_points)
        pos_prediction, pos_ground_truth = self.aux_mae(
            embedder_out, neighborhoods, center_points
        )

        lead_dim, B, T, _ = infer_leading_dims(x, 1)
        x = x.view(T * B, -1)
        pi = self.pi(x)
        pi = F.softmax(pi, dim=-1)
        value = self.value(x).squeeze(-1)
        pi, value, pos_prediction, pos_ground_truth = restore_leading_dims(
            (pi, value, pos_prediction, pos_ground_truth), lead_dim, T, B
        )

        return ModelOutputs(
            dist_params=DistParams(probs=pi),
            value=value,
            pos_prediction=pos_prediction,
            ground_truth=pos_ground_truth,
        )
