import torch
from parllel import ArrayTree
from parllel.torch.agents.sac_agent import SacAgent
from parllel.torch.distributions.squashed_gaussian import SquashedGaussian
from torch import Tensor

from pc_rl.utils.array_dict import dict_to_batched_data


class PcSacAgent(SacAgent):
    def __init__(
        self,
        model: torch.nn.ModuleDict,
        distribution: SquashedGaussian,
        device: torch.device,
        learning_starts: int = 0,
        pretrain_std: float = 0.75,  # With squash 0.75 is near uniform.
    ) -> None:
        super().__init__(model, distribution, device, learning_starts, pretrain_std)

    def encode(self, observation: ArrayTree[Tensor]) -> Tensor:
        pos, batch = dict_to_batched_data(observation)
        x, _, center_points = self.model["embedder"](pos, batch)
        encoder_out = self.model["encoder"](x, center_points)
        return encoder_out
