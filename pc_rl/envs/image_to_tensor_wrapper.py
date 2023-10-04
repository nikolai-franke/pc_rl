from __future__ import annotations

import gymnasium as gym


class ImageToTensorWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_shape[2], obs_shape[0], obs_shape[1])
        )

    def observation(self, observation):
        observation = observation.transpose((2, 0, 1)).astype(float)
        observation *= 1 / 255.0
        return observation
