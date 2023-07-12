import torch
import torch.nn as nn
from parllel.arrays.jagged import PointBatch
from parllel.torch.agents.gaussian import ModelOutputs
from parllel.torch.utils import infer_leading_dims, restore_leading_dims

from pc_rl.models.modules.embedder import Embedder
from pc_rl.models.finetune_encoder import FinetuneEncoder


class ContinuousPgModel(nn.Module):
    def __init__(
        self,
        embedder: Embedder,
        encoder: FinetuneEncoder,
        mu_mlp: nn.Module,
        value_mlp: nn.Module,
        init_log_std: float,
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder
        self.mu_mlp = mu_mlp
        self.value_mlp = value_mlp
        self._check_mlps()
        action_size = self._get_action_size()
        self.log_std = nn.Parameter(torch.full((action_size,), init_log_std))

    @torch.no_grad()
    def _check_mlps(self):
        input = torch.randn((self.encoder.out_dim,))
        try:
            self.mu_mlp(input)
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

    @torch.no_grad()
    def _get_action_size(self):
        input = torch.randn((self.encoder.out_dim,))
        return len(self.mu_mlp(input))

    def forward(self, data):
        pos, batch = namedtuple_to_batched_data(data)
        x, _, center_points = self.embedder(pos, batch)
        x = self.encoder(x, center_points)
        lead_dim, T, B, _ = infer_leading_dims(x, 1)
        obs_flat = x.view(T * B, -1)
        mu = self.mu_mlp(obs_flat)
        value = self.value_mlp(obs_flat).squeeze(-1)
        log_std = self.log_std.repeat(T * B, 1)
        mu, value, log_std = restore_leading_dims((mu, value, log_std), lead_dim, T, B)
        return ModelOutputs(mean=mu, value=value, log_std=log_std)


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
