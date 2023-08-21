from __future__ import annotations

from itertools import chain
from typing import Literal, Mapping

import torch
from parllel import ArrayDict
from parllel.replays.replay import ReplayBuffer
from parllel.torch.algos.sac import SAC
from parllel.torch.utils import valid_mean
from parllel.types.batch_spec import BatchSpec
from torch import Tensor

from pc_rl.agents.sac import PcSacAgent
from pc_rl.utils.aux_loss import get_loss_fn


class AuxPcSAC(SAC):
    agent: PcSacAgent

    def __init__(
        self,
        batch_spec: BatchSpec,
        agent: PcSacAgent,
        replay_buffer: ReplayBuffer[ArrayDict[Tensor]],
        optimizers: Mapping[str, torch.optim.Optimizer],
        discount: float,
        learning_starts: int,
        replay_ratio: int,  # data_consumption / data_generation target_update_tau: float,  # tau=1 for hard update.
        target_update_tau: float,
        target_update_interval: int,  # 1000 for hard update, 1 for soft.
        ent_coeff: float,
        clip_grad_norm: float,
        aux_loss_coeff: float,
        aux_loss: Literal["chamfer", "sinkhorn"] = "chamfer",
        ent_coeff_lr: float | None = None,
        **kwargs,  # ignore additional arguments
    ):
        super().__init__(
            batch_spec=batch_spec,
            agent=agent,
            replay_buffer=replay_buffer,
            optimizers=optimizers,  # type: ignore
            discount=discount,
            learning_starts=learning_starts,
            replay_ratio=replay_ratio,
            target_update_tau=target_update_tau,
            target_update_interval=target_update_interval,
            ent_coeff=ent_coeff,
            ent_coeff_lr=ent_coeff_lr,
            clip_grad_norm=clip_grad_norm,
        )
        self.aux_loss_fn = get_loss_fn(aux_loss)
        self.aux_loss_coeff = aux_loss_coeff

    def train_once(self, samples: ArrayDict[Tensor]) -> None:
        """
        Computes losses for twin Q-values against the min of twin target Q-values
        and an entropy term.  Computes reparameterized policy loss, and loss for
        tuning entropy weighting, alpha.

        Input samples have leading batch dimension [B,..] (but not time).
        """
        # compute target Q according to formula
        # r + gamma * (1 - d) * (min Q_targ(s', a') - alpha * log pi(s', a'))
        encoder_out = self.agent.encode(samples["observation"])
        new_action, log_prob = self.agent.pi(encoder_out.detach())

        if self.ent_coeff_optimizer is not None:
            entropy_coeff = torch.exp(self._log_ent_coeff.detach())
            ent_coeff_loss = -(
                self._log_ent_coeff * (log_prob + self.target_entropy).detach()
            ).mean()
            self.ent_coeff_optimizer.zero_grad()
            ent_coeff_loss.backward()
            self.ent_coeff_optimizer.step()
            self.algo_log_info["ent_ceff_loss"].append(ent_coeff_loss.item())
        else:
            entropy_coeff = self._ent_coeff

        self.algo_log_info["ent_coeff"].append(entropy_coeff.item())
        # where a' ~ pi(.|s')
        with torch.no_grad():
            next_encoder_out = self.agent.target_encode(samples["next_observation"])
            next_action, next_log_prob = self.agent.pi(next_encoder_out)
            target_q1, target_q2 = self.agent.target_q(next_encoder_out, next_action)
            min_target_q = torch.min(target_q1, target_q2)
            entropy_bonus = -entropy_coeff * next_log_prob
            y = samples["reward"] + self.discount * ~samples["terminated"] * (
                min_target_q + entropy_bonus
            )


        pos_prediction, pos_ground_truth = self.agent.auto_encoder(
            samples["observation"]
        )
        B, M, *_ = pos_prediction.shape
        pos_prediction = pos_prediction.reshape(B * M, -1, 3)
        pos_ground_truth = pos_ground_truth.reshape(B * M, -1, 3)

        mae_loss = self.aux_loss_fn(pos_prediction, pos_ground_truth)
        self.algo_log_info["mae_loss"].append(mae_loss.item())

        q1, q2 = self.agent.q(encoder_out, samples["action"])
        q_loss = 0.5 * valid_mean((y - q1) ** 2 + (y - q2) ** 2)
        # log critic_loss before adding MAE loss
        self.algo_log_info["critic_loss"].append(q_loss.item())
        q_loss += mae_loss * self.aux_loss_coeff


        # update Q model parameters according to Q loss
        self.optimizers["q"].zero_grad()
        q_loss.backward()
        q1_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.agent.model["q1"].parameters(),
            self.clip_grad_norm,
        )
        q2_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.agent.model["q2"].parameters(),
            self.clip_grad_norm,
        )
        self.optimizers["q"].step()

        self.algo_log_info["q1_grad_norm"].append(q1_grad_norm.item())
        self.algo_log_info["q2_grad_norm"].append(q2_grad_norm.item())
        self.algo_log_info["ent_coeff"].append(entropy_coeff.item())
        self.algo_log_info["mean_ent_bonus"].append(entropy_bonus.mean().item())
        self.algo_log_info["max_target_q"].append(min_target_q.max().item())
        self.algo_log_info["min_target_q"].append(min_target_q.min().item())

        # freeze Q models while optimizing policy model
        self.agent.freeze_q_models(True)

        # train policy model by maximizing the predicted Q value
        # maximize (min Q(s, a) - alpha * log pi(a, s))
        # where a ~ pi(.|s)
        q1, q2 = self.agent.q(encoder_out.detach(), new_action)
        min_q = torch.min(q1, q2)
        pi_losses = entropy_coeff * log_prob - min_q
        pi_loss = valid_mean(pi_losses)
        self.algo_log_info["actor_loss"].append(pi_loss.item())
        self.algo_log_info["min_q"].append(min_q.min().item())
        self.algo_log_info["max_q"].append(min_q.max().item())

        # update Pi model parameters according to pi loss
        self.optimizers["pi"].zero_grad()
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.agent.model["pi"].parameters(),
            self.clip_grad_norm,
        )
        self.optimizers["pi"].step()

        # unfreeze Q models for next training iteration
        self.agent.freeze_q_models(False)

        self.algo_log_info["pi_grad_norm"].append(pi_grad_norm.item())
