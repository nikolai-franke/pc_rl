from typing import Literal

from gymnasium.wrappers.time_limit import TimeLimit
from sofa_env.scenes.grasp_lift_touch.grasp_lift_touch_env import (
    ActionType, CollisionEffect, GraspLiftTouchEnv, ObservationType, Phase,
    RenderMode)
from sofa_env.wrappers.point_cloud import \
    PointCloudFromDepthImageObservationWrapper

from pc_rl.envs.add_obs_to_info_wrapper import AddObsToInfoWrapper


def build(
    max_episode_steps: int,
    render_mode: Literal["headless", "human"],
    action_type: Literal["discrete", "continuous"],
    image_shape: list[int],
    frame_skip: int,
    time_step: float,
    goal_tolerance: float,
    discrete_action_magnitude: float,
    add_obs_to_info_dict: bool,
    collision_punish_mode: Literal["proportional", "constant", "failure"],
    start_in_phase: Literal["grasp", "lift", "touch", "done", "any"],
    end_in_phase: Literal["grasp", "lift", "touch", "done", "any"],
    phase_any_rewards: dict,
    phase_grasp_rewards: dict,
    phase_touch_rewards: dict,
    create_scene_kwargs: dict | None = None,
):
    assert len(image_shape) == 2
    image_shape = tuple(image_shape)  # type: ignore
    render_mode = RenderMode[render_mode.upper()]  # type: ignore
    action_type = ActionType[action_type.upper()]  # type: ignore
    collision_punish_mode = CollisionEffect[collision_punish_mode.upper()]  # type: ignore
    start_in_phase = Phase[start_in_phase.upper()]  # type: ignore
    end_in_phase = Phase[end_in_phase.upper()]  # type: ignore
    reward_amount_dict = {
        Phase.ANY: phase_any_rewards,
        Phase.GRASP: phase_grasp_rewards,
        Phase.TOUCH: phase_touch_rewards,
    }

    env = GraspLiftTouchEnv(
        observation_type=ObservationType.RGBD,
        render_mode=render_mode,
        action_type=action_type,
        image_shape=image_shape,
        frame_skip=frame_skip,
        time_step=time_step,
        discrete_action_magnitude=discrete_action_magnitude,
        create_scene_kwargs=create_scene_kwargs,
        reward_amount_dict=reward_amount_dict,
        goal_tolerance=goal_tolerance,
        start_in_phase=start_in_phase,
        end_in_phase=end_in_phase,
    )

    print(add_obs_to_info_dict)
    if add_obs_to_info_dict:
        env = AddObsToInfoWrapper(env)
    env = PointCloudFromDepthImageObservationWrapper(env)
    env = TimeLimit(env, max_episode_steps)
    return env
