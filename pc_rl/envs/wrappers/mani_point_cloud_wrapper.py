from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
import open3d as o3d
from gymnasium import spaces


class ManiSkillPointCloudWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env,
        camera_name: str,
        post_processing_functions: list[Callable] | None = None,
        use_color: bool = False,
    ):
        super().__init__(env)
        self.camera_name = camera_name
        self.image_shape = env.observation_space["image"][self.camera_name]["rgb"].shape  # type: ignore
        self.observation_space = spaces.Box(
            high=np.inf,
            low=-np.inf,
            shape=(
                self.image_shape[0] * self.image_shape[1],
                6 if use_color else 3,
            ),
        )
        self.post_processing_functions = post_processing_functions
        self.use_color = use_color

    def observation(self, observation):
        if self.use_color:
            point_cloud = self._create_color_point_cloud(observation)
        else:
            point_cloud = self._create_point_cloud(observation)

        if self.post_processing_functions is not None:
            for function in self.post_processing_functions:
                point_cloud = function(point_cloud)

        # np.savetxt(
        #     "obs.csv",
        #     point_cloud,
        #     delimiter=",",
        # )
        return point_cloud

    def _create_point_cloud(self, observation):
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.image_shape[0],
            self.image_shape[1],
            observation["camera_param"][self.camera_name]["intrinsic_cv"],
        )
        depth_image = o3d.geometry.Image(
            observation["image"][self.camera_name]["depth"]
        )
        point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image, intrinsic
        )
        return np.asarray(point_cloud.points)

    def _create_color_point_cloud(self, observation):
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.image_shape[0],
            self.image_shape[1],
            observation["camera_param"][self.camera_name]["intrinsic_cv"],
        )
        depth_image = o3d.geometry.Image(
            observation["image"][self.camera_name]["depth"]
        )
        color_image = o3d.geometry.Image(observation["image"][self.camera_name]["rgb"])
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image,
            depth_image,
            convert_rgb_to_intensity=False,
            depth_scale=1.0,
            depth_trunc=1e9,
        )
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic
        )
        return np.hstack(
            (np.asarray(point_cloud.points), np.asarray(point_cloud.colors))
        )
