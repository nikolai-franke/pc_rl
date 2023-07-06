import torch
import torch.nn as nn
import torch.nn.functional as F
from parllel.arrays.jagged import PointBatch
from parllel.torch.agents.categorical import ModelOutputs
from parllel.torch.utils import infer_leading_dims, restore_leading_dims

from pc_rl.models.modules.embedder import Embedder
from pc_rl.models.modules.finetune_encoder import FinetuneEncoder


class CategoricalPgModel(nn.Module):
    def __init__(
        self,
        embedder: Embedder,
        encoder: FinetuneEncoder,
        pi_mlp: nn.Module,
        value_mlp: nn.Module,
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder
        self.pi_mlp = pi_mlp
        self.value_mlp = value_mlp
        self._check_mlps()

    def _check_mlps(self):
        with torch.no_grad():
            input = torch.randn((self.encoder.out_dim,))
            try:
                self.pi_mlp(input)
            except RuntimeError as e:
                raise ValueError(
                    f"The first layer of the Pi MLP must have the same size as the output of the encoder: {self.encoder.out_dim}"
                ) from e
            try:
                self.value_mlp(input)
            except RuntimeError as e:
                raise ValueError(
                    f"The first layer of the Value MLP must have the same size as the output of the encoder: {self.encoder.out_dim}"
                ) from e

    def forward(self, data):
        pos, batch = namedtuple_to_batched_data(data)
        x, _, center_points = self.embedder(pos, batch)
        x = self.encoder(x, center_points)
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
