import torch.nn.functional as F
from parllel.torch.utils import infer_leading_dims, restore_leading_dims
from torch import nn

from pc_rl.agents.aux_categorical import ModelOutputs
from pc_rl.models.categorical_pg_model import namedtuple_to_batched_data
from pc_rl.models.modules.auxiliarey_mae import AuxiliaryMae
from pc_rl.models.modules.embedder import Embedder


class AuxMaeDiscretePgModel(nn.Module):
    def __init__(
        self,
        embedder: Embedder,
        auxiliary_mae: AuxiliaryMae,
        pi_mlp: nn.Module,
        value_mlp: nn.Module,
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.auxiliary_mae = auxiliary_mae
        self.pi_mlp = pi_mlp
        self.value_mlp = value_mlp

    def forward(self, data):
        pos, batch = namedtuple_to_batched_data(data)
        x, neighborhoods, center_points = self.embedder(pos, batch)
        lead_dim, B, T, _ = infer_leading_dims(x, 1)
        pos_recovered, x = self.auxiliary_mae(x, center_points)

        x = x.view(T * B, -1)
        pi = self.pi_mlp(x)
        pi = F.softmax(pi, dim=-1)
        value = self.value_mlp(x).squeeze(-1)
        pi, value = restore_leading_dims((pi, value), lead_dim, T, B)

        return ModelOutputs(
            pi=pi,
            value=value,
            pos_recovered=pos_recovered,
            center_points=center_points,
            neighborhoods=neighborhoods,
        )
