from __future__ import annotations

from typing import Literal

import numpy as np
from sofa_env.scenes.tissue_dissection.tissue_dissection_env import (
    ActionType, ObservationType, TissueDissectionEnv)

from pc_rl.utils.add_env_wrappers import add_env_wrappers


def build(
    max_episode_steps: int,
    add_obs_to_info_dict: bool,
    render_mode: Literal["headless", "human"],
    action_type: Literal["discrete", "continuous"],
    observation_type: Literal["color_point_cloud", "rgb_image", "rgbd_image"],
    image_shape: list[int],
    frame_skip: int,
    time_step: float,
    settle_steps: int,
    reward_amount_dict: dict,
    rows_to_cut: int,
    voxel_grid_size: float | None,
    camera_reset_noise: list[float] | None = None,
    create_scene_kwargs: dict | None = None,
):
    image_shape = tuple(image_shape)  # type: ignore
    render_mode = RenderMode[render_mode.upper()]  # type: ignore
    action_type = ActionType[action_type.upper()]  # type: ignore

    if observation_type in ("color_point_cloud", "rgbd_image"):
        obs_type = ObservationType.RGBD
    elif observation_type == "rgb_image":
        obs_type = ObservationType.RGB
    else:
        raise ValueError(f"Invalid observation type: {observation_type}")
    if camera_reset_noise is not None:
        camera_reset_noise = np.asarray(camera_reset_noise)  # type: ignore

    if create_scene_kwargs is not None:
        convert_to_array(create_scene_kwargs)

    env = TissueDissectionEnv(
        observation_type=obs_type,
        render_mode=render_mode,  # type: ignore
        action_type=action_type,  # type: ignore
        image_shape=image_shape,  # type: ignore
        frame_skip=frame_skip,
        rows_to_cut=rows_to_cut,
        time_step=time_step,
        settle_steps=settle_steps,
        create_scene_kwargs=create_scene_kwargs,
        reward_amount_dict=reward_amount_dict,
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
