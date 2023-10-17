from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np


class ManiSkillPointCloudWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env,
        post_processing_functions: list[Callable] | None = None,
        use_color: bool = False,
    ):
        super().__init__(env)
        num_points = env.observation_space["pointcloud"]["xyzw"].shape[0]
        point_dim = 6 if use_color else 3
        self.observation_space = gym.spaces.Box(
            high=np.inf, low=-np.inf, shape=(num_points, point_dim)
        )
        self.post_processing_functions = post_processing_functions
        self.use_color = use_color
        self.counter = 0

    def observation(self, observation):
        point_cloud = observation["pointcloud"]["xyzw"]
        # only points where w == 1 are valid
        mask = point_cloud[..., -1] == 1
        # we only want xyz in our observation
        point_cloud = point_cloud[mask][..., :3]
        # filter out the floor
        z_mask = point_cloud[..., -1] > 1e-6
        point_cloud = point_cloud[z_mask]

        if self.use_color:
            rgb = observation["pointcloud"]["rgb"].astype(float)
            rgb = rgb[mask][z_mask]
            rgb *= 1 / 255.0
            point_cloud = np.hstack((point_cloud, rgb))

        if self.post_processing_functions is not None:
            for function in self.post_processing_functions:
                point_cloud = function(point_cloud)

        # np.savetxt(
        #     f"obs{self.counter}.csv",
        #     point_cloud,
        #     delimiter=",",
        # )
        # self.counter += 1

        return point_cloud
