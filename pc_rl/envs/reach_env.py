from collections import deque

import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit
from sofa_env.scenes.reach.reach_env import (ActionType, ObservationType,
                                             ReachEnv, RenderMode)
from sofa_env.wrappers.point_cloud import \
    PointCloudFromDepthImageObservationWrapper

from .point_cloud import PointCloud


def build_reach_env(max_episode_steps: int = 100):
    env = ReachEnv(
        observation_type=ObservationType.RGBD,
        render_mode=RenderMode.HUMAN,
        action_type=ActionType.DISCRETE,
        observe_target_position=False,
        image_shape=(128, 128),
        frame_skip=1,
        time_step=0.1,
        reward_amount_dict={
            "distance_to_target": -1.0,
            "delta_distance_to_target": -1.0,
            "time_step_cost": -1.0,
            "worspace_violation": -1.0,
            "successful_task": 1.0,
        },
    )
    env = PointCloudFromDepthImageObservationWrapper(env)
    env = TimeLimit(env, max_episode_steps)
    env.observation_space = PointCloud(8000, -np.inf, np.inf, feature_shape=(3,))

    return env


if __name__ == "__main__":
    import time

    env = ReachEnv(
        observation_type=ObservationType.RGBD,
        render_mode=RenderMode.HUMAN,
        action_type=ActionType.DISCRETE,
        observe_target_position=False,
        image_shape=(480, 480),
        frame_skip=1,
        time_step=0.1,
        reward_amount_dict={
            "distance_to_target": -1.0,
            "delta_distance_to_target": -1.0,
            "time_step_cost": -1.0,
            "worspace_violation": -1.0,
            "successful_task": 1.0,
        },
    )
    env = PointCloudFromDepthImageObservationWrapper(env)

    obs, _ = env.reset()

    fps_list = deque(maxlen=100)
    terminated = False
    counter = 0
    while not terminated:
        start = time.time()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        end = time.time()
        fps = 1 / (end - start)
        fps_list.append(fps)
        if counter % 100 == 0:
            env.reset()
        counter += 1
