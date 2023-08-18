import torch
import torch.nn as nn
import torch.nn.functional as F
from parllel.torch.agents.categorical import ModelOutputs
from parllel.torch.distributions.categorical import DistParams
from parllel.torch.utils import infer_leading_dims, restore_leading_dims

from pc_rl.models.finetune_encoder import FinetuneEncoder
from pc_rl.models.modules.embedder import Embedder
from pc_rl.utils.array_dict import dict_to_batched_data


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
            input = torch.randn((self.encoder.dim,))
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

    def forward(self, data):
        pos, batch = dict_to_batched_data(data)
        x, _, center_points = self.embedder(pos, batch)
        x = self.encoder(x, center_points)
        lead_dim, T, B, _ = infer_leading_dims(x, 1)
        obs_flat = x.view(T * B, -1)
        probs = self.pi_mlp(obs_flat)
        probs = F.softmax(probs, dim=-1)
        value = self.value_mlp(obs_flat).squeeze(-1)
        probs, value = restore_leading_dims((probs, value), lead_dim, T, B)
        return ModelOutputs(dist_params=DistParams(probs=probs), value=value)
