from __future__ import annotations

import functools

from gymnasium.wrappers.time_limit import TimeLimit
from sofa_env.scenes.reach.reach_env import (ActionType, ObservationType,
                                             ReachEnv, RenderMode)

from pc_rl.envs.add_obs_to_info_wrapper import AddObsToInfoWrapper
from pc_rl.envs.point_cloud_wrapper import \
    PointCloudFromDepthImageObservationWrapper
from pc_rl.envs.post_processing_functions import normalize, voxel_grid_sample


def build(
    max_episode_steps: int,
    render_mode: str,
    action_type: str,
    image_shape: list[int],
    frame_skip: int,
    time_step: float,
    discrete_action_magnitude: float,
    distance_to_target_threshold: float,
    reward_amount_dict: dict,
    create_scene_kwargs: dict,
    add_obs_to_info_dict: bool,
    voxel_grid_size: float | None,
):
    assert len(image_shape) == 2
    image_shape = tuple(image_shape)  # type: ignore
    render_mode = RenderMode[render_mode.upper()]  # type: ignore
    action_type = ActionType[action_type.upper()]  # type: ignore

    env = ReachEnv(
        observation_type=ObservationType.RGBD,
        render_mode=render_mode,
        action_type=action_type,
        observe_target_position=False,
        image_shape=image_shape,
        frame_skip=frame_skip,
        time_step=time_step,
        discrete_action_magnitude=discrete_action_magnitude,
        distance_to_target_threshold=distance_to_target_threshold,
        reward_amount_dict=reward_amount_dict,
        create_scene_kwargs=create_scene_kwargs,
    )
    if add_obs_to_info_dict:
        env = AddObsToInfoWrapper(env)
    post_processing_functions = []
    post_processing_functions.append(normalize)
    if voxel_grid_size is not None:
        post_processing_functions.append(
            functools.partial(voxel_grid_sample, voxel_grid_size=voxel_grid_size)
        )

    env = PointCloudFromDepthImageObservationWrapper(
        env, post_processing_functions=post_processing_functions
    )
    env = TimeLimit(env, max_episode_steps)
    return env
