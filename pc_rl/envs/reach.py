from gymnasium.wrappers.time_limit import TimeLimit
import numpy as np
from sofa_env.scenes.reach.reach_env import (ActionType, ObservationType,
                                             ReachEnv, RenderMode)
from sofa_env.wrappers.point_cloud import \
    PointCloudFromDepthImageObservationWrapper


def build(
    max_episode_steps: int,
    render_mode: str,
    action_type: str,
    image_shape: list[int],  # list, since yaml does not support tuples
    frame_skip: int,
    time_step: float,
    discrete_action_magnitude: float,
    distance_to_target_threshold: float,
    reward_amount_dict: dict,
    create_scene_kwargs: dict,
    add_obs_to_info_dict: bool = False,
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
        add_obs_to_info_dict=add_obs_to_info_dict,
    )
    env = PointCloudFromDepthImageObservationWrapper(env)
    env = TimeLimit(env, max_episode_steps)
    return env


def build_reach_env_old(max_episode_steps: int = 100):
    env = ReachEnv(
        observation_type=ObservationType.RGBD,
        render_mode=RenderMode.HUMAN,
        action_type=ActionType.DISCRETE,
        observe_target_position=False,
        image_shape=(128, 128),
        frame_skip=1,
        time_step=0.1,
        discrete_action_magnitude=100.0,
        distance_to_target_threshold=0.02,
        reward_amount_dict={
            "distance_to_target": -0.1,
            "delta_distance_to_target": -10.0,
            "time_step_cost": -0.1,
            "worspace_violation": -1.0,
            "successful_task": 10.0,
        },
        create_scene_kwargs={
            "show_bounding_boxes": False,
            "show_remote_center_of_motion": True,
        },
    )
    env = PointCloudFromDepthImageObservationWrapper(env)
    env = TimeLimit(env, max_episode_steps)

    return env


if __name__ == "__main__":
    import time

    env = build_reach_env_old()


    obs, _ = env.reset()

    terminated = False
    counter = 0
    while not terminated:
        start = time.time()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        np.savetxt("pc.csv", obs, delimiter=",")

        end = time.time()
        if counter % 100 == 0:
            env.reset()
        counter += 1
        terminated = True
