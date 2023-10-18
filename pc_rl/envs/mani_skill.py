from __future__ import annotations

import functools
from typing import Literal

import gymnasium as gym
import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit

from pc_rl.envs.wrappers.add_obs_to_info_wrapper import \
    ManiSkillAddObsToInfoWrapper
from pc_rl.envs.wrappers.continuous_task_wrapper import ContinuousTaskWrapper
from pc_rl.envs.wrappers.mani_point_cloud_wrapper import \
    ManiSkillPointCloudWrapper
from pc_rl.utils.point_cloud_post_processing_functions import (
    normalize, voxel_grid_sample)


def build(
    env_id: str,
    max_episode_steps: int,
    observation_type: Literal["point_cloud", "color_point_cloud"],
    add_obs_to_info_dict: bool,
    image_shape: list[int],
    control_mode: str,
    reward_mode: str,
    voxel_grid_size: float | None = None,
    render_mode: str | None = None,
    filter_points_below_z: float | None = None,
    z_far: float | None = None,
    z_near: float | None = None,
    fov: float | None = None,
    sim_freq: int = 500,
    control_freq: int = 20,
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
        render_mode=render_mode,
        sim_freq=sim_freq,
        control_freq=control_freq,
        renderer_kwargs={"offscreen_only": True},
    )
    # If we don't set a random seed manually, all parallel environments have the same seed
    env.unwrapped.set_main_rng(np.random.randint(1e9))

    if add_obs_to_info_dict:
        env = ManiSkillAddObsToInfoWrapper(env)

    env = ContinuousTaskWrapper(env)

    post_processing_functions = []
    if filter_points_below_z is not None:
        post_processing_functions.append(
            lambda point_cloud: point_cloud[point_cloud[..., 2] > filter_points_below_z]
        )

    if voxel_grid_size is not None:
        post_processing_functions.append(
            functools.partial(voxel_grid_sample, voxel_grid_size=voxel_grid_size)
        )
    post_processing_functions.append(normalize)

    env = ManiSkillPointCloudWrapper(
        env,
        use_color=observation_type.startswith("color"),
        post_processing_functions=post_processing_functions,
    )

    env = TimeLimit(env, max_episode_steps)
    return env
