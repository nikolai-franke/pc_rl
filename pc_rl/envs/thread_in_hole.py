from __future__ import annotations

from typing import Literal

import numpy as np
from sofa_env.scenes.thread_in_hole.thread_in_hole_env import (ActionType,
                                                               ObservationType,
                                                               RenderMode,
                                                               ThreadInHoleEnv)

from pc_rl.utils.add_env_wrappers import add_env_wrappers


def build(
    max_episode_steps: int,
    add_obs_to_info_dict: bool,
    render_mode: Literal["headless", "human"],
    action_type: Literal["discrete", "continuous"],
    observation_type: Literal[
        "point_cloud", "color_point_cloud", "rgb_image", "rgbd_image"
    ],
    image_shape: list[int],
    frame_skip: int,
    time_step: float,
    insertion_ratio_threshold: float,
    settle_steps: int,
    simple_success_check: bool,
    camera_reset_noise: list | None,
    hole_rotation_reset_noise: list | None,
    hole_position_reset_noise: list | None,
    reward_amount_dict: dict,
    voxel_grid_size: float | None,
    create_scene_kwargs: dict | None = None,
):
    image_shape = tuple(image_shape)  # type: ignore
    render_mode = RenderMode[render_mode.upper()]  # type: ignore
    action_type = ActionType[action_type.upper()]  # type: ignore

    if camera_reset_noise is not None:
        camera_reset_noise = np.asarray(camera_reset_noise)
    if hole_rotation_reset_noise is not None:
        hole_rotation_reset_noise = np.asarray(hole_rotation_reset_noise)
    if hole_position_reset_noise is not None:
        hole_position_reset_noise = np.asarray(hole_position_reset_noise)
    if observation_type in ("point_cloud", "color_point_cloud", "rgbd_image"):
        obs_type = ObservationType.RGBD
    elif observation_type == "rgb_image":
        obs_type = ObservationType.RGB
    else:
        raise ValueError(f"Invalis observation type: {observation_type}")

    if create_scene_kwargs is not None:
        convert_to_array(create_scene_kwargs)

    env = ThreadInHoleEnv(
        observation_type=obs_type,
        render_mode=render_mode,
        action_type=action_type,
        image_shape=image_shape,
        frame_skip=frame_skip,
        time_step=time_step,
        settle_steps=settle_steps,
        create_scene_kwargs=create_scene_kwargs,
        reward_amount_dict=reward_amount_dict,
        camera_reset_noise=camera_reset_noise,
        hole_rotation_reset_noise=hole_rotation_reset_noise,
        hole_position_reset_noise=hole_position_reset_noise,
        insertion_ratio_threshold=insertion_ratio_threshold,
        simple_success_check=simple_success_check,
    )
    env = add_env_wrappers(
        env,
        max_episode_steps=max_episode_steps,
        add_obs_to_info_dict=add_obs_to_info_dict,
        observation_type=observation_type,
        voxel_grid_size=voxel_grid_size,
    )

    return env


def convert_to_array(kwargs_dict):
    for k, v in kwargs_dict.items():
        if isinstance(v, list):
            kwargs_dict[k] = np.asarray(v)
        elif isinstance(v, dict):
            convert_to_array(v)
