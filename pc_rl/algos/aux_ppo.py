from typing import Literal

import numpy as np
import torch
from parllel import ArrayDict
from parllel.replays import BatchedDataLoader
from parllel.torch.algos.ppo import PPO
from pytorch3d.loss import chamfer_distance
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from pc_rl.agents.pg import AuxPgAgent, AuxPgPrediction


class AuxPPO(PPO):
    def __init__(
        self,
        agent: AuxPgAgent,
        dataloader: BatchedDataLoader[Tensor],
        optimizer: Optimizer,
        learning_rate_scheduler: _LRScheduler | None,
        value_loss_coeff: float,
        entropy_loss_coeff: float,
        aux_loss_coeff: float,
        clip_grad_norm: float | None,
        epochs: int,
        ratio_clip: float,
        value_clipping_mode: Literal["none", "ratio", "delta", "delta_max"],
        value_clip: float | None = None,
        kl_divergence_limit: float = np.inf,
        **kwargs,
    ) -> None:
        super().__init__(
            agent,
            dataloader,
            optimizer,
            learning_rate_scheduler,
            value_loss_coeff,
            entropy_loss_coeff,
            clip_grad_norm,
            epochs,
            ratio_clip,
            value_clipping_mode,
            value_clip,
            kl_divergence_limit,
            **kwargs,
        )
        self.aux_loss_coeff = aux_loss_coeff
        self.agent = agent

    def loss(self, batch: ArrayDict[Tensor]) -> torch.Tensor:
        loss, agent_prediction = super().loss(batch)
        assert isinstance(agent_prediction, AuxPgPrediction)
        pos_prediction, ground_truth = (
            agent_prediction["pos_prediction"],
            agent_prediction["ground_truth"],
        )

        B, M, *_ = pos_prediction.shape
        pos_prediction = pos_prediction.reshape(B * M, -1, 3)
        ground_truth = ground_truth.reshape(B * M, -1, 3)

        mae_loss = (
            chamfer_distance(pos_prediction, ground_truth, point_reduction="sum")[0]
            * self.aux_loss_coeff
        )

        loss += mae_loss

        self.algo_log_info["mae_loss"].append(mae_loss.item())

        return loss
