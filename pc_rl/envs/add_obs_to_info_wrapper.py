from __future__ import annotations

import gymnasium as gym


class AddObsToInfoWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, key: str = "rendering"):
        super().__init__(env)
        self.key = key

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info[self.key] = observation.transpose(2, 0, 1)
        return observation, reward, terminated, truncated, info
