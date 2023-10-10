from __future__ import annotations

import gymnasium as gym


class AddObsToInfoWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, key: str = "rendering"):
        super().__init__(env)
        self.key = key

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info[self.key] = observation[..., :3]
        return observation, reward, terminated, truncated, info


class ManiSkillAddObsToInfoWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, key: str = "rendering"):
        super().__init__(env)
        self.key = key

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        print(observation["extra"].keys())
        info[self.key] = observation["extra"]["rgb"]
        return observation, reward, terminated, truncated, info
