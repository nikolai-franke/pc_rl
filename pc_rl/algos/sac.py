from typing import Mapping

import parllel.logger as logger
import torch
from parllel import ArrayDict
from parllel.replays.replay import ReplayBuffer
from parllel.torch.algos.sac import SAC
from parllel.torch.utils import valid_mean
from parllel.types.batch_spec import BatchSpec
from torch import Tensor

from pc_rl.agents.sac import SacAgent


class PcSac(SAC):
    agent: SacAgent

    def __init__(
        self,
        batch_spec: BatchSpec,
        agent: SacAgent,
        replay_buffer: ReplayBuffer[ArrayDict[Tensor]],
        optimizers: Mapping[str, torch.optim.Optimizer],
        discount: float,
        learning_starts: int,
        replay_ratio: int,  # data_consumption / data_generation
        target_update_tau: float,  # tau=1 for hard update.
        target_update_interval: int,  # 1000 for hard update, 1 for soft.
        ent_coeff: float,
        clip_grad_norm: float,
        **kwargs,  # ignore additional arguments
    ):
        super().__init__(
            batch_spec=batch_spec,
            agent=agent,
            replay_buffer=replay_buffer,
            optimizers=optimizers,
            discount=discount,
            learning_starts=learning_starts,
            replay_ratio=replay_ratio,
            target_update_tau=target_update_tau,
            target_update_interval=target_update_interval,
            ent_coeff=ent_coeff,
            clip_grad_norm=clip_grad_norm,
        )

    def train_once(self, samples: ArrayDict[Tensor]) -> None:
        """
        Computes losses for twin Q-values against the min of twin target Q-values
        and an entropy term.  Computes reparameterized policy loss, and loss for
        tuning entropy weighting, alpha.

        Input samples have leading batch dimension [B,..] (but not time).
        """
        # compute target Q according to formula
        # r + gamma * (1 - d) * (min Q_targ(s', a') - alpha * log pi(s', a'))
        # where a' ~ pi(.|s')
        ptr = samples["observation"]["ptr"]
        num_nodes = ptr[1:] - ptr[:-1]
        assert len(num_nodes) == self.replay_buffer.batch_size, f"PTR:{ptr}\n NUM NODES: {num_nodes}"
        assert torch.all(num_nodes > 0), f"{num_nodes}, {ptr}"

        next_ptr = samples["next_observation"]["ptr"]
        num_nodes = next_ptr[1:] - next_ptr[:-1]
        assert len(num_nodes) == self.replay_buffer.batch_size, f"PTR: {next_ptr}\n NUM NODES: {num_nodes}"
        assert torch.all(num_nodes > 0), f"{num_nodes}, {ptr}"

        with torch.no_grad():
            next_encoder_out = self.agent.encode(samples["next_observation"])
            next_action, next_log_prob = self.agent.pi(next_encoder_out)
            target_q1, target_q2 = self.agent.target_q(next_encoder_out, next_action)
        min_target_q = torch.min(target_q1, target_q2)
        next_q = min_target_q - self._alpha * next_log_prob
        try:
            y = samples["reward"] + self.discount * ~samples["terminated"] * next_q
        except RuntimeError as e:
            logger.warn(
                    f"REWARD SHAPE: {samples['reward'].shape}\n DONE SHAPE: {samples['done'].shape}\n NEXT Q SHAPE: {next_q.shape}, OBSERVATION SHAPE: {samples['observation'].shape}\n, NEXT OBSERVATION SHAPE: {samples['next_observation'].shape}"
            )
            raise e
        encoder_out = self.agent.encode(samples["observation"])
        q1, q2 = self.agent.q(encoder_out.detach(), samples["action"])
        q_loss = 0.5 * valid_mean((y - q1) ** 2 + (y - q2) ** 2)

        # update Q model parameters according to Q loss
        self.optimizers["q"].zero_grad()
        q_loss.backward()
        q1_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.agent.model["q1"].parameters(), self.clip_grad_norm
        )
        q2_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.agent.model["q2"].parameters(), self.clip_grad_norm
        )
        self.optimizers["q"].step()

        self.algo_log_info["critic_loss"].append(q_loss.item())
        self.algo_log_info["q1_grad_norm"].append(q1_grad_norm.item())
        self.algo_log_info["q2_grad_norm"].append(q2_grad_norm.item())

        # freeze Q models while optimizing policy model
        self.agent.freeze_q_models(True)

        # train policy model by maximizing the predicted Q value
        # maximize (min Q(s, a) - alpha * log pi(a, s))
        # where a ~ pi(.|s)
        new_action, log_prob = self.agent.pi(encoder_out)
        q1, q2 = self.agent.q(encoder_out, new_action)
        min_q = torch.min(q1, q2)
        pi_losses = self._alpha * log_prob - min_q
        pi_loss = valid_mean(pi_losses)

        # update Pi model parameters according to pi loss
        self.optimizers["pi"].zero_grad()
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.agent.model["pi"].parameters(), self.clip_grad_norm
        )
        self.optimizers["pi"].step()

        # unfreeze Q models for next training iteration
        self.agent.freeze_q_models(False)

        self.algo_log_info["actor_loss"].append(pi_loss.item())
        self.algo_log_info["q1_grad_norm"].append(pi_grad_norm.item())