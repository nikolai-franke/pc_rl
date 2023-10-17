from __future__ import annotations

from typing import Literal

import gymnasium as gym
import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit
from mani_skill2.utils.sapien_utils import look_at

from pc_rl.envs.wrappers.add_obs_to_info_wrapper import \
    ManiSkillAddObsToInfoWrapper
from pc_rl.envs.wrappers.continuous_task_wrapper import ContinuousTaskWrapper
from pc_rl.envs.wrappers.mani_image_wrapper import ManiSkillImageWrapper
from pc_rl.envs.wrappers.mani_point_cloud_wrapper import \
    ManiSkillPointCloudWrapper
from pc_rl.envs.wrappers.transpose_image_wrapper import TransposeImageWrapper
from pc_rl.utils.point_cloud_post_processing_functions import normalize


class SuccessInfoWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info["successful_task"] = info["success"]
        return observation, reward, terminated, truncated, info


def build(
    env_id: str,
    max_episode_steps: int,
    observation_type: Literal["point_cloud", "color_point_cloud"],
    add_obs_to_info_dict: bool,
    image_shape: list[int],
    control_mode: str,
    reward_mode: str,
    z_far: float | None = None,
    z_near: float | None = None,
    fov: float | None = None,
):
    import mani_skill2.envs

    camera_cfgs = {
        "width": image_shape[0],
        "height": image_shape[1],
    }
    if z_far is not None:
        camera_cfgs.update({"far": z_far})  # type: ignore
    if z_near is not None:
        camera_cfgs.update({"near": z_near})  # type: ignore
    if fov is not None:
        camera_cfgs.update({"fov": fov})  # type: ignore

    env = gym.make(
        env_id,
        obs_mode="pointcloud",
        reward_mode=reward_mode,
        control_mode=control_mode,
        camera_cfgs=camera_cfgs,
    )
    # If we don't set a random seed manually, all parallel environments have the same seed
    env.unwrapped.set_main_rng(np.random.randint(1e9))

    if add_obs_to_info_dict:
        env = ManiSkillAddObsToInfoWrapper(env)

    env = SuccessInfoWrapper(env)
    env = ContinuousTaskWrapper(env)

    env = ManiSkillPointCloudWrapper(
        env,
        use_color=observation_type.startswith("color"),
        post_processing_functions=[normalize],
    )

    env = TimeLimit(env, max_episode_steps)
    return env
