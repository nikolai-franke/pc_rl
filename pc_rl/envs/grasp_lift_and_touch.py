from __future__ import annotations
import numpy as np

from typing import Literal

from sofa_env.scenes.grasp_lift_touch.grasp_lift_touch_env import (
    ActionType, CollisionEffect, GraspLiftTouchEnv, ObservationType, Phase,
    RenderMode)

from pc_rl.utils.add_env_wrappers import add_env_wrappers


def build(
    max_episode_steps: int,
    render_mode: Literal["headless", "human"],
    action_type: Literal["discrete", "continuous"],
    observation_type: Literal[
        "point_cloud", "color_point_cloud", "rgb_image", "rgbd_image"
    ],
    settle_steps: int,
    image_shape: list[int],
    frame_skip: int,
    time_step: float,
    goal_tolerance: float,
    add_obs_to_info_dict: bool,
    collision_punish_mode: Literal["proportional", "constant", "failure"],
    start_in_phase: Literal["grasp", "lift", "touch", "done", "any"],
    end_in_phase: Literal["grasp", "lift", "touch", "done", "any"],
    phase_any_rewards: dict,
    phase_grasp_rewards: dict,
    phase_touch_rewards: dict,
    voxel_grid_size: float | None,
    create_scene_kwargs: dict | None = None,
    max_depth: float | None = None,
    camera_reset_noise: list | None = None,
):
    assert len(image_shape) == 2
    image_shape = tuple(image_shape)  # type: ignore
    render_mode = RenderMode[render_mode.upper()]  # type: ignore
    action_type = ActionType[action_type.upper()]  # type: ignore
    collision_punish_mode = CollisionEffect[collision_punish_mode.upper()]  # type: ignore
    start_in_phase = Phase[start_in_phase.upper()]  # type: ignore
    end_in_phase = Phase[end_in_phase.upper()]  # type: ignore
    if camera_reset_noise is not None:
        camera_reset_noise = np.asarray(camera_reset_noise)

    if observation_type in ("point_cloud", "color_point_cloud", "rgbd_image"):
        obs_type = ObservationType.RGBD
    elif observation_type == "rgb_image":
        obs_type = ObservationType.RGB
    else:
        raise ValueError(f"Invalid observation type: {observation_type}")

    if create_scene_kwargs is not None:
        convert_to_array(create_scene_kwargs)

    env = GraspLiftTouchEnv(
        observation_type=obs_type,
        render_mode=render_mode,  # type: ignore
        action_type=action_type,  # type: ignore
        image_shape=image_shape,  # type: ignore
        frame_skip=frame_skip,
        time_step=time_step,
        settle_steps=settle_steps,
        create_scene_kwargs=create_scene_kwargs,
        reward_amount_dict={
            Phase.ANY: phase_any_rewards,
            Phase.GRASP: phase_grasp_rewards,
            Phase.TOUCH: phase_touch_rewards,
        },
        goal_tolerance=goal_tolerance,
        start_in_phase=start_in_phase,  # type: ignore
        end_in_phase=end_in_phase,  # type: ignore
        camera_reset_noise=camera_reset_noise,
    )

    env = add_env_wrappers(
        env,
        max_episode_steps=max_episode_steps,
        add_obs_to_info_dict=add_obs_to_info_dict,
        observation_type=observation_type,
        voxel_grid_size=voxel_grid_size,
        max_depth=max_depth,
    )

    return env


def convert_to_array(kwargs_dict):
    for k, v in kwargs_dict.items():
        if isinstance(v, list):
            kwargs_dict[k] = np.asarray(v)
        elif isinstance(v, dict):
            convert_to_array(v)
