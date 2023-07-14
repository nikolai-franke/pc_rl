from typing import Optional

import numpy as np
import parllel.logger as logger
import torch
from parllel.buffers import NamedArrayTupleClass, Samples
from parllel.replays import BatchedDataLoader
from parllel.torch.agents.agent import TorchAgent
from parllel.torch.algos.ppo import PPO
from parllel.torch.utils import explained_variance, valid_mean
from pytorch3d.loss import chamfer_distance
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from pc_rl.agents.aux_categorical import AgentPrediction

SamplesForLoss = NamedArrayTupleClass(
    "SamplesForLoss",
    [
        "observation",
        "agent_info",
        "action",
        "return_",
        "advantage",
        "valid",
        "old_dist_info",
        "old_values",
    ],
)


class AuxPPO(PPO):
    def __init__(
        self,
        agent: TorchAgent,
        dataloader: BatchedDataLoader[SamplesForLoss[np.ndarray]],
        optimizer: Optimizer,
        learning_rate_scheduler: Optional[_LRScheduler],
        value_loss_coeff: float,
        entropy_loss_coeff: float,
        aux_loss_coeff: float,
        clip_grad_norm: Optional[float],
        epochs: int,
        ratio_clip: float,
        value_clipping_mode: str,
        value_clip: Optional[float] = None,
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

    def loss(self, batch: SamplesForLoss) -> torch.Tensor:
        agent_prediction: AgentPrediction = self.agent.predict(
            batch.observation, batch.agent_info
        )
        dist_info, value, pos_prediction, ground_truth = (
            agent_prediction.dist_info,
            agent_prediction.value,
            agent_prediction.pos_prediction,
            agent_prediction.ground_truth,
        )
        dist = self.agent.distribution
        ratio = dist.likelihood_ratio(
            batch.action, old_dist_info=batch.old_dist_info, new_dist_info=dist_info
        )
        surr_1 = ratio * batch.advantage
        clipped_ratio = torch.clamp(ratio, 1.0 - self.ratio_clip, 1.0 + self.ratio_clip)
        surr_2 = clipped_ratio * batch.advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = -valid_mean(surrogate, batch.valid)

        if self.value_clipping_mode == "none":
            # No clipping
            value_error = 0.5 * (value - batch.return_) ** 2
        elif self.value_clipping_mode == "ratio":
            # Clipping the value per time step in respect to the ratio between old and new values
            value_ratio = value / batch.old_values
            clipped_values = torch.where(
                value_ratio > 1.0 + self.value_clip,
                batch.old_values * (1.0 + self.value_clip),
                value,
            )
            clipped_values = torch.where(
                value_ratio < 1.0 - self.value_clip,
                batch.old_values * (1.0 - self.value_clip),
                clipped_values,
            )
            clipped_value_error = 0.5 * (clipped_values - batch.return_) ** 2
            standard_value_error = 0.5 * (value - batch.return_) ** 2
            value_error = torch.max(clipped_value_error, standard_value_error)
        elif self.value_clipping_mode == "delta":
            # Clipping the value per time step with its original (old) value in the boundaries of value_clip
            clipped_values = torch.min(
                torch.max(value, batch.old_values - self.value_clip),
                batch.old_values + self.value_clip,
            )
            value_error = 0.5 * (clipped_values - batch.return_) ** 2
        elif self.value_clipping_mode == "delta_max":
            # Clipping the value per time step with its original (old) value in the boundaries of value_clip
            clipped_values = torch.min(
                torch.max(value, batch.old_values - self.value_clip),
                batch.old_values + self.value_clip,
            )
            clipped_value_error = 0.5 * (clipped_values - batch.return_) ** 2
            standard_value_error = 0.5 * (value - batch.return_) ** 2
            value_error = torch.max(clipped_value_error, standard_value_error)
        else:
            raise ValueError(
                f"Invalid value clipping mode '{self.value_clipping_mode}'"
            )

        value_loss = self.value_loss_coeff * valid_mean(value_error, batch.valid)
        entropy = dist.mean_entropy(dist_info, batch.valid)
        entropy_loss = -self.entropy_loss_coeff * entropy

        # TODO: maybe put this into model
        B, M, *_ = pos_prediction.shape
        pos_prediction = pos_prediction.reshape(B * M, -1, 3)
        ground_truth = ground_truth.reshape(B * M, -1, 3)

        mae_loss = chamfer_distance(
            pos_prediction, ground_truth, point_reduction="sum"
        )[0] * self.aux_loss_coeff

        loss = pi_loss + value_loss + entropy_loss + mae_loss

        # Compute a low-variance estimate of the KL divergence to use for
        # stopping further updates after a KL divergence limit is reached.
        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        # and Schulman blog: http://joschu.net/blog/kl-approx.html
        with torch.no_grad():
            approx_kl_div = torch.mean(ratio - 1 - torch.log(ratio))
            if approx_kl_div >= self.kl_divergence_limit:
                self.early_stopping = True
                logger.info(
                    f"Reached the maximum KL divergence limit of {self.kl_divergence_limit} at step {self.update_counter}, stopping further updates."
                )
                return loss

            perplexity = dist.mean_perplexity(dist_info, batch.valid)
            self.algo_log_info["loss"].append(loss.item())
            self.algo_log_info["policy_gradient_loss"].append(pi_loss.item())
            self.algo_log_info["mae_loss"].append(mae_loss.item())
            self.algo_log_info["approx_kl"].append(approx_kl_div.item())
            clip_fraction = ((ratio - 1).abs() > self.ratio_clip).float().mean().item()
            self.algo_log_info["clip_fraction"].append(clip_fraction)
            if hasattr(dist_info, "log_std"):
                self.algo_log_info["policy_log_std"].append(
                    dist_info.log_std.mean().item()
                )
            self.algo_log_info["entropy_loss"].append(entropy_loss.item())
            self.algo_log_info["entropy"].append(entropy.item())
            self.algo_log_info["perplexity"].append(perplexity.item())
            self.algo_log_info["value_loss"].append(value_loss.item())
            explained_var = explained_variance(value, batch.return_)
            self.algo_log_info["explained_variance"].append(explained_var)

        return loss


def build_dataloader_buffer(sample_buffer: Samples) -> SamplesForLoss:
    dataloader_buffer = SamplesForLoss(
        observation=sample_buffer.env.observation,
        agent_info=sample_buffer.agent.agent_info,
        action=sample_buffer.agent.action,
        return_=sample_buffer.env.return_,
        advantage=sample_buffer.env.advantage,
        valid=None,
        old_dist_info=sample_buffer.agent.agent_info.dist_info,
        old_values=sample_buffer.agent.agent_info.value,
    )
    return dataloader_buffer
