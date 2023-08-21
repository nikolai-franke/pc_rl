from __future__ import annotations

from typing import Literal

from gymnasium.wrappers.time_limit import TimeLimit

from pc_rl.envs.add_obs_to_info_wrapper import AddObsToInfoWrapper
from pc_rl.envs.point_cloud_wrapper import \
    PointCloudFromDepthImageObservationWrapper
from pc_rl.envs.sofa_env.pick_and_place import (ActionType, ObservationType,
                                                Phase, PickAndPlaceEnv,
                                                RenderMode)


def build(
    max_episode_steps: int,
    render_mode: Literal["headless", "human"],
    action_type: Literal["discrete", "continuous"],
    image_shape: list[int],
    frame_skip: int,
    time_step: float,
    add_obs_to_info_dict: bool,
    phase_any_rewards: dict,
    phase_pick_rewards: dict,
    phase_place_rewards: dict,
    randomize_torus_position: bool,
    create_scene_kwargs: dict = {},
):
    assert len(image_shape) == 2
    image_shape = tuple(image_shape)  # type: ignore
    render_mode = RenderMode[render_mode.upper()]  # type: ignore
    action_type = ActionType[action_type.upper()]  # type: ignore

    env = PickAndPlaceEnv(
        observation_type=ObservationType.RGBD,
        render_mode=render_mode,
        action_type=action_type,
        image_shape=image_shape,
        frame_skip=frame_skip,
        time_step=time_step,
        randomize_torus_position=randomize_torus_position,
        reward_amount_dict={
            Phase.ANY: phase_any_rewards,
            Phase.PICK: phase_pick_rewards,
            Phase.PLACE: phase_place_rewards,
        },
        create_scene_kwargs=create_scene_kwargs,
    )
    if add_obs_to_info_dict:
        env = AddObsToInfoWrapper(env)

    env = PointCloudFromDepthImageObservationWrapper(env)
    env = TimeLimit(env, max_episode_steps)
    return env
