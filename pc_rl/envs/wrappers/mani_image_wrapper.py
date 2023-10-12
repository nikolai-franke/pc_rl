from __future__ import annotations

import gymnasium as gym


class ManiSkillImageWrapper(gym.ObservationWrapper):
    def __init__(self, env, camera_name: str):
        super().__init__(env)
        self.camera_name = camera_name
        self.observation_space = env.observation_space["image"][self.camera_name]["rgb"]

    def observation(self, observation):
        return observation["image"][self.camera_name]["rgb"]
