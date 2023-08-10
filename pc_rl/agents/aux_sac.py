import torch
from parllel.torch.agents.agent import TorchAgent
from parllel.torch.distributions.squashed_gaussian import SquashedGaussian

from pc_rl.agents.sac import PcSacAgent


class AuxSacAgent(PcSacAgent):
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
