from __future__ import annotations

import gymnasium as gym

class NormalizePointCloudWrapper(gym.Wrapper):

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        obs_mean = observation.mean(axis=-2)
        observation = observation - obs_mean
        scale = (1 / abs(observation).max()) * 0.999999
        observation = observation * scale
        return observation, reward, terminated, truncated, info

