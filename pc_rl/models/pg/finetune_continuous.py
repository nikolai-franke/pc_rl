from __future__ import annotations

import torch
import torch.nn as nn
from parllel.torch.agents.gaussian import ModelOutputs
from parllel.torch.distributions.gaussian import DistParams
from parllel.torch.utils import infer_leading_dims, restore_leading_dims

from pc_rl.models.finetune_encoder import FinetuneEncoder
from pc_rl.models.modules.tokenizer import Tokenizer
from pc_rl.utils.array_dict import dict_to_batched_data


class ContinuousPgModel(nn.Module):
    def __init__(
        self,
        tokenizer: Tokenizer,
        encoder: FinetuneEncoder,
        pi_mlp: nn.Module,
        value_mlp: nn.Module,
        init_log_std: float,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.pi_mlp = pi_mlp
        self.value_mlp = value_mlp
        self._check_mlps()
        action_size = self._get_action_size()
        self.log_std = nn.Parameter(torch.full((action_size,), init_log_std))

    @torch.no_grad()
    def _check_mlps(self):
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

    @torch.no_grad()
    def _get_action_size(self):
        input = torch.randn((self.encoder.dim,))
        return len(self.pi_mlp(input))

    def forward(self, data):
        pos, batch = dict_to_batched_data(data)
        x, _, center_points = self.tokenizer(pos, batch)
        x = self.encoder(x, center_points)
        lead_dim, T, B, _ = infer_leading_dims(x, 1)
        obs_flat = x.view(T * B, -1)
        mean = self.pi_mlp(obs_flat)
        value = self.value_mlp(obs_flat).squeeze(-1)
        log_std = self.log_std.repeat(T * B, 1)
        mean, value, log_std = restore_leading_dims(
            (mean, value, log_std), lead_dim, T, B
        )
        return ModelOutputs(
            dist_params=DistParams(mean=mean, log_std=log_std), value=value
        )
