from __future__ import annotations

import functools
from typing import Literal

import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit
from sofa_env.scenes.rope_cutting.rope_cutting_env import (ActionType,
                                                           ObservationType,
                                                           RenderFramework,
                                                           RenderMode,
                                                           RopeCuttingEnv)

from pc_rl.envs.add_obs_to_info_wrapper import AddObsToInfoWrapper
from pc_rl.envs.point_cloud_wrapper import (
    ColorPointCloudWrapper, PointCloudFromDepthImageObservationWrapper)

from .post_processing_functions import normalize, voxel_grid_sample


def build(
    max_episode_steps: int,
    add_obs_to_info_dict: bool,
    render_mode: Literal["headless", "human"],
    action_type: Literal["discrete", "continuous"],
    image_shape: list[int],
    frame_skip: int,
    time_step: float,
    settle_steps: int,
    num_ropes: int,
    num_ropes_to_cut: int,
    reward_amount_dict: dict,
    voxel_grid_size: float | None,
    create_scene_kwargs: dict | None = None,
    use_color: bool = False,
):
    image_shape = tuple(image_shape)  # type: ignore
    render_mode = RenderMode[render_mode.upper()]  # type: ignore
    action_type = ActionType[action_type.upper()]  # type: ignore

    if create_scene_kwargs is not None:
        convert_to_array(create_scene_kwargs)

    env = RopeCuttingEnv(
        observation_type=ObservationType.RGBD,
        render_mode=render_mode,
        action_type=action_type,
        image_shape=image_shape,
        frame_skip=frame_skip,
        time_step=time_step,
        settle_steps=settle_steps,
        num_ropes=num_ropes,
        num_ropes_to_cut=num_ropes_to_cut,
        create_scene_kwargs=create_scene_kwargs,
        reward_amount_dict=reward_amount_dict,
    )

    if add_obs_to_info_dict:
        env = AddObsToInfoWrapper(env)

    post_processing_functions = []
    if voxel_grid_size is not None:
        post_processing_functions.append(
            functools.partial(voxel_grid_sample, voxel_grid_size=voxel_grid_size)
        )
    post_processing_functions.append(normalize)
    if use_color:
        env = ColorPointCloudWrapper(
            env, post_processing_functions=post_processing_functions
        )
    else:
        env = PointCloudFromDepthImageObservationWrapper(
            env,
            post_processing_functions=post_processing_functions,
        )
    env = TimeLimit(env, max_episode_steps)
    return env


def convert_to_array(kwargs_dict):
    for k, v in kwargs_dict.items():
        if isinstance(v, list):
            kwargs_dict[k] = np.asarray(v)
        elif isinstance(v, dict):
            convert_to_array(v)
