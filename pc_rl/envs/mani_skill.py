from __future__ import annotations

import functools
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
from pc_rl.utils.point_cloud_post_processing_functions import (
    normalize, voxel_grid_sample)


class SuccessInfoWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info["successful_task"] = info["success"]
        return observation, reward, terminated, truncated, info


def build(
    env_id: str,
    camera_name: str,
    max_episode_steps: int,
    observation_type: Literal[
        "point_cloud", "color_point_cloud", "rgb_image", "rgbd_image"
    ],
    add_obs_to_info_dict: bool,
    image_shape: list[int],
    control_mode: str,
    reward_mode: str,
    voxel_grid_size: float,
    z_far: float,
    z_near: float,
    fov: float,
    cam_pos: list[float] | None = None,
    cam_look_at: list[float] | None = None,
):
    import mani_skill2.envs

    camera_config = {
        "width": image_shape[0],
        "height": image_shape[1],
        "far": z_far,
        "near": z_near,
        "fov": fov,
    }
    if cam_pos is not None:
        cam_look_at = cam_look_at or [0, 0, 0]
        cam_pose = look_at(cam_pos, cam_look_at)
        camera_config.update({"p": cam_pose.p, "q": cam_pose.q})

    env = gym.make(
        env_id,
        obs_mode="rgbd",
        reward_mode=reward_mode,
        control_mode=control_mode,
        camera_cfgs={camera_name: camera_config},
    )
    env.set_main_rng(np.random.randint(1e9))
    if add_obs_to_info_dict:
        env = ManiSkillAddObsToInfoWrapper(env, camera_name=camera_name)

    env = SuccessInfoWrapper(env)
    env = ContinuousTaskWrapper(env)

    post_processing_functions = [
        functools.partial(voxel_grid_sample, voxel_grid_size=voxel_grid_size),
        normalize,
    ]
    if observation_type in ("point_cloud", "color_point_cloud"):
        env = ManiSkillPointCloudWrapper(
            env,
            camera_name=camera_name,
            post_processing_functions=post_processing_functions,
            use_color=observation_type == "color_point_cloud",
        )
    elif observation_type == "rgb_image":
        env = ManiSkillImageWrapper(env, camera_name=camera_name)
        env = TransposeImageWrapper(env)
    else:
        raise NotImplementedError
    env = TimeLimit(env, max_episode_steps)
    return env
