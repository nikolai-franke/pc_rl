from __future__ import annotations

import functools
from typing import Sequence

import torch
import torch.nn.functional as F
from parllel import ArrayDict
from parllel.replays.replay import ReplayBuffer
from parllel.torch.algos.sac import SAC
from parllel.torch.utils import valid_mean
from parllel.types.batch_spec import BatchSpec
from pytorch3d.ops.knn import knn_gather
from torch import Tensor
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LRScheduler

from pc_rl.agents.sac import PcSacAgent
from pc_rl.utils.chamfer import chamfer_distance


class AuxPcSAC(SAC):
    agent: PcSacAgent

    def __init__(
        self,
        batch_spec: BatchSpec,
        agent: PcSacAgent,
        replay_buffer: ReplayBuffer[ArrayDict[Tensor]],
        q_optimizer: torch.optim.Optimizer,
        pi_optimizer: torch.optim.Optimizer,
        discount: float,
        learning_starts: int,
        replay_ratio: int,  # data_consumption / data_generation
        target_update_tau: float,  # tau=1 for hard update.
        target_update_interval: int,  # 1000 for hard update, 1 for soft.
        ent_coeff: float,
        aux_loss_coeff: float,
        color_loss_coeff: float = 1.0,
        ent_coeff_lr: float | None = None,
        clip_grad_norm: float | None = None,
        learning_rate_schedulers: Sequence[LRScheduler] | None = None,
        detach_encoder: bool = False,
        **kwargs,  # ignore additional arguments
    ):
        super().__init__(
            batch_spec=batch_spec,
            agent=agent,
            replay_buffer=replay_buffer,
            q_optimizer=q_optimizer,
            pi_optimizer=pi_optimizer,
            discount=discount,
            learning_starts=learning_starts,
            replay_ratio=replay_ratio,
            target_update_tau=target_update_tau,
            target_update_interval=target_update_interval,
            ent_coeff=ent_coeff,
            ent_coeff_lr=ent_coeff_lr,
            clip_grad_norm=clip_grad_norm,
            learning_rate_schedulers=learning_rate_schedulers,
        )
        self.aux_loss_fn = functools.partial(chamfer_distance, return_x_nn=True)
        self.aux_loss_coeff = aux_loss_coeff
        self.color_loss_coeff = color_loss_coeff
        self.detach_encoder = detach_encoder

    def train_once(self, samples: ArrayDict[Tensor]) -> None:
        """
        Computes losses for twin Q-values against the min of twin target Q-values
        and an entropy term.  Computes reparameterized policy loss, and loss for
        tuning entropy weighting, alpha.

        Input samples have leading batch dimension [B,..] (but not time).
        """
        # encode once, allowing the agent to reuse its encodings for q and
        # pi predictions.
        # we then later detach the encoding from its computational graph
        # during q prediction, since we only want to optimize the encoder
        # using the gradients from the policy network.
        observation = self.agent.encode(samples["observation"])

        new_action, log_prob = self.agent.pi(observation.detach())

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

        # compute target Q according to formula
        # r + gamma * (1 - d) * (min Q_targ(s', a') - alpha * log pi(s', a'))
        # where a' ~ pi(.|s')
        with torch.no_grad():
            next_observation = self.agent.target_encode(samples["next_observation"])
            next_action, next_log_prob = self.agent.pi(next_observation)
            target_q1, target_q2 = self.agent.target_q(next_observation, next_action)
        min_target_q = torch.min(target_q1, target_q2)
        entropy_bonus = -entropy_coeff * next_log_prob
        y = samples["reward"] + self.discount * ~samples["terminated"] * (
            min_target_q + entropy_bonus
        )
        if not self.detach_encoder:
            q1, q2 = self.agent.q(observation, samples["action"])
        else:
            q1, q2 = self.agent.q(observation.detach(), samples["action"])

        prediction, ground_truth = self.agent.auto_encoder(samples["observation"])
        B, M, *_, C = prediction.shape
        prediction = prediction.reshape(B * M, -1, C)
        ground_truth = ground_truth.reshape(B * M, -1, C)

        mae_loss, _, x_idx = self.aux_loss_fn(  # type: ignore
            prediction[..., :3], ground_truth[..., :3]
        )
        self.algo_log_info["chamfer_loss"].append(mae_loss.item())
        mae_loss *= self.aux_loss_coeff

        # if color
        if C > 3:
            assert x_idx is not None
            prediction_nearest_neighbor = knn_gather(ground_truth, x_idx).reshape(
                B, M, -1, C
            )
            color_loss = (
                F.mse_loss(
                    prediction[..., 3:].reshape(B, M, -1, C - 3),
                    prediction_nearest_neighbor[..., 3:].reshape(B, M, -1, C - 3),
                )
                * self.color_loss_coeff
            )
            self.algo_log_info["color_loss"].append(color_loss.item())
            mae_loss += color_loss

        q_loss = 0.5 * valid_mean((y - q1) ** 2 + (y - q2) ** 2)

        self.algo_log_info["critic_loss"].append(q_loss.item())
        self.algo_log_info["mean_ent_bonus"].append(entropy_bonus.mean().item())
        self.algo_log_info["ent_coeff"].append(entropy_coeff.item())
        self.algo_log_info["mae_loss"].append(mae_loss.item())

        # update Q model parameters according to Q loss
        self.q_optimizer.zero_grad()
        mae_loss.backward()
        q_loss.backward()

        if self.clip_grad_norm is not None:
            q1_grad_norm = clip_grad_norm_(
                self.agent.model["q1"].parameters(),
                self.clip_grad_norm,
            )
            q2_grad_norm = clip_grad_norm_(
                self.agent.model["q2"].parameters(),
                self.clip_grad_norm,
            )
            finetune_encoder_grad_norm = clip_grad_norm_(
                self.agent.model["encoder"].parameters(), self.clip_grad_norm
            )
            # masked_decoder_grad_norm = clip_grad_norm_(
            #     self.agent.model["rl_mae"].masked_decoder.parameters(), self.clip_grad_norm
            # )
            # mae_prediction_head_grad_norm = clip_grad_norm_(
            #     self.agent.model["rl_mae"].mae_prediction_head.parameters(), self.clip_grad_norm
            # )
            embedder_grad_norm = clip_grad_norm_(
                self.agent.model["embedder"].parameters(), self.clip_grad_norm
            )
            self.algo_log_info["q1_grad_norm"].append(q1_grad_norm.item())
            self.algo_log_info["q2_grad_norm"].append(q2_grad_norm.item())
            self.algo_log_info["encoder_grad_norm"].append(
                finetune_encoder_grad_norm.item()
            )
            self.algo_log_info["embedder_grad_norm"].append(embedder_grad_norm.item())
            # self.algo_log_info["masked_decoder_grad_norm"].append(masked_decoder_grad_norm.item())
            # self.algo_log_info["mae_prediction_head_grad_norm"].append(mae_prediction_head_grad_norm.item())

        self.q_optimizer.step()

        # freeze Q models while optimizing policy model
        self.agent.freeze_q_models(True)

        # train policy model by maximizing the predicted Q value
        # maximize (min Q(s, a) - alpha * log pi(a, s))
        # where a ~ pi(.|s)
        q1, q2 = self.agent.q(observation.detach(), new_action)
        min_q = torch.min(q1, q2)
        pi_losses = entropy_coeff * log_prob - min_q
        pi_loss = valid_mean(pi_losses)

        self.algo_log_info["actor_loss"].append(pi_loss.item())

        # update Pi model parameters according to pi loss
        self.pi_optimizer.zero_grad()
        pi_loss.backward()

        if self.clip_grad_norm is not None:
            pi_grad_norm = clip_grad_norm_(
                self.agent.model["pi"].parameters(),
                self.clip_grad_norm,
            )
            self.algo_log_info["pi_grad_norm"].append(pi_grad_norm.item())

        self.pi_optimizer.step()

        # unfreeze Q models for next training iteration
        self.agent.freeze_q_models(False)
