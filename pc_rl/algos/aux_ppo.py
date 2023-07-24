from typing import Literal

import numpy as np
import parllel.logger as logger
import torch
from parllel import Array, ArrayDict
from parllel.replays import BatchedDataLoader
from parllel.torch.algos.ppo import PPO
from parllel.torch.utils import explained_variance, valid_mean
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
        valid = batch.get("valid", None)
        agent_prediction: AuxPgPrediction = self.agent.predict(
            batch["observation"],
            batch["agent_info"],
            batch.get("initial_rnn_state", None),
        )
        dist_params, value, pos_prediction, ground_truth = (
            agent_prediction["dist_params"],
            agent_prediction["value"],
            agent_prediction["pos_prediction"],
            agent_prediction["ground_truth"],
        )
        dist = self.agent.distribution
        ratio = dist.likelihood_ratio(
            batch["action"],
            old_dist_params=batch["old_dist_params"],
            new_dist_params=dist_params,
        )
        surr_1 = ratio * batch["advantage"]
        clipped_ratio = torch.clamp(ratio, 1.0 - self.ratio_clip, 1.0 + self.ratio_clip)
        surr_2 = clipped_ratio * batch.advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = -valid_mean(surrogate, valid)

        if self.value_clipping_mode == "none":
            # No clipping
            value_error = 0.5 * (value - batch["return_"]) ** 2
        elif self.value_clipping_mode == "ratio":
            # Clipping the value per time step in respect to the ratio between old and new values
            value_ratio = value / batch["old_value"]
            clipped_values = torch.where(
                value_ratio > 1.0 + self.value_clip,
                batch["old_values"] * (1.0 + self.value_clip),
                value,
            )
            clipped_values = torch.where(
                value_ratio < 1.0 - self.value_clip,
                batch["old_value"] * (1.0 - self.value_clip),
                clipped_values,
            )
            clipped_value_error = 0.5 * (clipped_values - batch["return_"]) ** 2
            standard_value_error = 0.5 * (value - batch["return_"]) ** 2
            value_error = torch.max(clipped_value_error, standard_value_error)
        elif self.value_clipping_mode == "delta":
            # Clipping the value per time step with its original (old) value in the boundaries of value_clip
            clipped_values = torch.min(
                torch.max(value, batch["old_value"] - self.value_clip),
                batch["old_value"] + self.value_clip,
            )
            value_error = 0.5 * (clipped_values - batch.return_) ** 2
        elif self.value_clipping_mode == "delta_max":
            # Clipping the value per time step with its original (old) value in the boundaries of value_clip
            clipped_values = torch.min(
                torch.max(value, batch["old_value"] - self.value_clip),
                batch["old_value"] + self.value_clip,
            )
            clipped_value_error = 0.5 * (clipped_values - batch["return_"]) ** 2
            standard_value_error = 0.5 * (value - batch["return_"]) ** 2
            value_error = torch.max(clipped_value_error, standard_value_error)
        else:
            raise ValueError(
                f"Invalid value clipping mode '{self.value_clipping_mode}'"
            )

        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)
        entropy = dist.mean_entropy(dist_params, batch.valid)
        entropy_loss = -self.entropy_loss_coeff * entropy

        # TODO: maybe put this into model
        B, M, *_ = pos_prediction.shape
        pos_prediction = pos_prediction.reshape(B * M, -1, 3)
        ground_truth = ground_truth.reshape(B * M, -1, 3)

        mae_loss = (
            chamfer_distance(pos_prediction, ground_truth, point_reduction="sum")[0]
            * self.aux_loss_coeff
        )

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

            perplexity = dist.mean_perplexity(dist_params, batch.valid)
            self.algo_log_info["loss"].append(loss.item())
            self.algo_log_info["policy_gradient_loss"].append(pi_loss.item())
            self.algo_log_info["mae_loss"].append(mae_loss.item())
            self.algo_log_info["approx_kl"].append(approx_kl_div.item())
            clip_fraction = ((ratio - 1).abs() > self.ratio_clip).float().mean().item()
            self.algo_log_info["clip_fraction"].append(clip_fraction)
            if hasattr(dist_params, "log_std"):
                self.algo_log_info["policy_log_std"].append(
                    dist_params.log_std.mean().item()
                )
            self.algo_log_info["entropy_loss"].append(entropy_loss.item())
            self.algo_log_info["entropy"].append(entropy.item())
            self.algo_log_info["perplexity"].append(perplexity.item())
            self.algo_log_info["value_loss"].append(value_loss.item())
            explained_var = explained_variance(value, batch.return_)
            self.algo_log_info["explained_variance"].append(explained_var)

        return loss


def build_loss_sample_tree(
    sample_tree: ArrayDict[Array],
) -> ArrayDict[Array]:
    loss_sample_tree = ArrayDict(
        {
            "observation": sample_tree["observation"],
            "agent_info": sample_tree["agent_info"],
            "action": sample_tree["action"],
            "return_": sample_tree["return_"],
            "advantage": sample_tree["advantage"],
        }
    )

    # move these to the top level for convenience
    # anything else in agent_info is agent-specific state
    loss_sample_tree["old_value"] = loss_sample_tree["agent_info"].pop("value")
    loss_sample_tree["old_dist_params"] = loss_sample_tree["agent_info"].pop(
        "dist_params"
    )

    if "valid" in sample_tree:
        loss_sample_tree["valid"] = sample_tree["valid"]
        assert "initial_rnn_state" in sample_tree
        loss_sample_tree["initial_rnn_state"] = sample_tree["initial_rnn_state"]
    return loss_sample_tree
