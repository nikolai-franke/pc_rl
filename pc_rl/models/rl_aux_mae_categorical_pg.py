import torch.nn as nn
import torch.nn.functional as F
from parllel.torch.utils import infer_leading_dims, restore_leading_dims

from pc_rl.agents.aux_categorical import ModelOutputs
from pc_rl.models.aux_mae import AuxMae
from pc_rl.models.modules.embedder import Embedder
from pc_rl.models.rl_finetune_categorical_pg import namedtuple_to_batched_data


class AuxMaeCategoricalPgModel(nn.Module):
    def __init__(
        self,
        embedder: Embedder,
        aux_mae: AuxMae,
        pi_mlp: nn.Module,
        value_mlp: nn.Module,
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.aux_mae = aux_mae
        self.pi_mlp = pi_mlp
        self.value_mlp = value_mlp

    def forward(self, data):
        pos, batch = namedtuple_to_batched_data(data)
        x, neighborhoods, center_points = self.embedder(pos, batch)
        x, pos_prediction, pos_ground_truth = self.aux_mae(
            x, center_points, neighborhoods
        )

        lead_dim, B, T, _ = infer_leading_dims(x, 1)
        x = x.view(T * B, -1)
        pi = self.pi_mlp(x)
        pi = F.softmax(pi, dim=-1)
        value = self.value_mlp(x).squeeze(-1)
        pi, value, pos_prediction, pos_ground_truth = restore_leading_dims(
            (pi, value, pos_prediction, pos_ground_truth), lead_dim, T, B
        )

        return ModelOutputs(
            pi=pi,
            value=value,
            pos_prediction=pos_prediction,
            ground_truth=pos_ground_truth,
        )
