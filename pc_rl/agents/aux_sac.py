from __future__ import annotations

import torch
from parllel import ArrayTree
from parllel.torch.distributions.squashed_gaussian import SquashedGaussian
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import Compose, NormalizeScale, RandomRotate

from pc_rl.agents.sac import PcSacAgent
from pc_rl.utils.array_dict import dict_to_batched_data

transform = Compose(
    [
        RandomRotate(180, axis=0),
        RandomRotate(180, axis=1),
        RandomRotate(180, axis=2),
        NormalizeScale(),
    ]
)


class AuxPcSacAgent(PcSacAgent):
    def __init__(
        self,
        model: torch.nn.ModuleDict,
        distribution: SquashedGaussian,
        device: torch.device,
        learning_starts: int = 0,
        pretrain_std: float = 0.75,
    ):
        super().__init__(
            model=model,
            distribution=distribution,
            device=device,
            learning_starts=learning_starts,
            pretrain_std=pretrain_std,
        )

    def auto_encoder(self, observation: ArrayTree[Tensor]) -> tuple[Tensor, Tensor]:
        pos, batch, color = dict_to_batched_data(observation)
        data = Data(pos=pos, x=color)
        data = transform(data)
        pos, color = data.pos, data.x
        x, neighborhoods, center_points = self.model["tokenizer"](pos, batch, color)
        prediction, ground_truth = self.model["rl_mae"](x, neighborhoods, center_points)
        return prediction, ground_truth
