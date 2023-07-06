import torch
import torch.nn as nn
import torch.nn.functional as F
from parllel.torch.agents.categorical import ModelOutputs
from parllel.arrays.jagged import PointBatch
from parllel.torch.models import MlpModel
from parllel.torch.utils import infer_leading_dims, restore_leading_dims


class CategoricalPgModel(nn.Module):
    def __init__(
        self,
        embedder: nn.Module,
        encoder: nn.Module,
        n_actions: int,
        mlp_hidden_sizes: list[int],
        mlp_act: type[nn.Module],
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder
        self.pi_mlp = MlpModel(
            input_size=1024,
            hidden_sizes=mlp_hidden_sizes,
            hidden_nonlinearity=mlp_act,
            output_size=n_actions,
        )
        self.value_mlp = MlpModel(
            input_size=1024,
            hidden_sizes=mlp_hidden_sizes,
            hidden_nonlinearity=mlp_act,
            output_size=1,
        )

    def forward(self, data):
        pos, batch = namedtuple_to_batched_data(data)
        x, _, center_points = self.embedder(pos, batch)
        x  = self.encoder(x, center_points)
        lead_dim, T, B, _ = infer_leading_dims(x, 1)
        obs_flat = x.view(T * B, -1)
        pi = self.pi_mlp(obs_flat)
        pi = F.softmax(pi, dim=-1)
        value = self.value_mlp(obs_flat).squeeze(-1)
        pi, value = restore_leading_dims((pi, value), lead_dim, T, B)
        return ModelOutputs(pi=pi, value=value)


def namedtuple_to_batched_data(
    namedtup: PointBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos, ptr = namedtup.pos, namedtup.ptr
    num_nodes = ptr[1:] - ptr[:-1]
    batch = torch.repeat_interleave(
        torch.arange(len(num_nodes), device=num_nodes.device),
        repeats=num_nodes,
    )

    return pos, batch
