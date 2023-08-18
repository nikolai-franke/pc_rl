from typing import Literal

import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit
from sofa_env.scenes.thread_in_hole.thread_in_hole_env import (ActionType,
                                                               ObservationType,
                                                               RenderMode,
                                                               ThreadInHoleEnv)

from pc_rl.envs.add_obs_to_info_wrapper import AddObsToInfoWrapper
from pc_rl.envs.point_cloud_wrapper import \
    PointCloudFromDepthImageObservationWrapper


def build(
    max_episode_steps: int,
    add_obs_to_info_dict: bool,
    render_mode: Literal["headless", "human"],
    action_type: Literal["discrete", "continuous"],
    image_shape: list[int],
    frame_skip: int,
    time_step: float,
    discrete_action_magnitude: float,
    camera_reset_noise: list | None,
    hole_rotation_reset_noise: list | None,
    hole_position_reset_noise: list | None,
    reward_amount_dict: dict,
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

    env = ThreadInHoleEnv(
        observation_type=ObservationType.RGBD,
        render_mode=render_mode,
        action_type=action_type,
        image_shape=image_shape,
        frame_skip=frame_skip,
        time_step=time_step,
        create_scene_kwargs=create_scene_kwargs,
        discrete_action_magnitude=discrete_action_magnitude,
        reward_amount_dict=reward_amount_dict,
        camera_reset_noise=camera_reset_noise,
        hole_rotation_reset_noise=hole_rotation_reset_noise,
        hole_position_reset_noise=hole_position_reset_noise,
    )

    if add_obs_to_info_dict:
        env = AddObsToInfoWrapper(env)
    env = PointCloudFromDepthImageObservationWrapper(env)
    env = TimeLimit(env, max_episode_steps)
    return env