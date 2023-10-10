from __future__ import annotations

import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit
from mani_skill2.utils.wrappers import RecordEpisode

from pc_rl.envs.wrappers.add_obs_to_info_wrapper import \
    ManiSkillAddObsToInfoWrapper

env_id = "PickCube-v0"
obs_mode = "pointcloud"
control_mode = "pd_ee_delta_pose"
reward_mode = "normalized_dense"


class SuccessInfoWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info["successful_task"] = info["success"]
        return observation, reward, terminated, truncated, info


class ManiSkillPointCloudWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space["pointcloud"]["xyzw"]

    def observation(self, observation):
        return observation["pointcloud"]["xyzw"]


def build(max_episode_steps: int, add_obs_to_info_dict: bool = False):
    import mani_skill2.envs

    env = gym.make(
        env_id,
        obs_mode=obs_mode,
        reward_mode=reward_mode,
        control_mode=control_mode,
    )
    env = ManiSkillAddObsToInfoWrapper(env)
    env = ManiSkillPointCloudWrapper(env)
    env = SuccessInfoWrapper(env)
    env = TimeLimit(env, max_episode_steps)
    return env
