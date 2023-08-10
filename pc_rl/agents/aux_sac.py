import torch
from parllel import ArrayDict, ArrayTree, Index, dict_map
from parllel.torch.distributions.squashed_gaussian import SquashedGaussian
from torch import Tensor

from pc_rl.agents.sac import PcSacAgent


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

    @torch.no_grad()
    def step(
        self, observation: ArrayTree[Tensor], *, env_indices: Index = ...
    ) -> tuple[Tensor, ArrayDict[Tensor]]:
        observation = observation.to_ndarray()
        observation = dict_map(torch.from_numpy, observation)
        observation = observation.to(device=self.device)
        encoding, *_ = self.encode(observation) # TODO: maybe use two separate encode() methods, one for sampling and one for training
        dist_params = self.model["pi"](encoding)
        action = self.distribution.sample(dist_params)
        return action.cpu(), ArrayDict()
