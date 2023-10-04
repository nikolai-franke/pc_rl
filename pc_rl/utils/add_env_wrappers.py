import functools
from typing import Literal

from gymnasium.wrappers.time_limit import TimeLimit

from pc_rl.envs.add_obs_to_info_wrapper import AddObsToInfoWrapper
from pc_rl.envs.image_to_tensor_wrapper import ImageToTensorWrapper
from pc_rl.envs.point_cloud_wrapper import (
    ColorPointCloudWrapper, PointCloudFromDepthImageObservationWrapper)
from pc_rl.envs.post_processing_functions import normalize, voxel_grid_sample


def add_env_wrappers(
    env,
    max_episode_steps: int,
    add_obs_to_info_dict: bool,
    observation_type: Literal[
        "point_cloud", "color_point_cloud", "rgb_image", "rgbd_image"
    ],
    voxel_grid_size: float | None = None,
):
    if add_obs_to_info_dict:
        env = AddObsToInfoWrapper(env)

    if observation_type in ("rgb_image", "rgbd_image"):
        env = ImageToTensorWrapper(env)
    else:
        assert observation_type in ("point_cloud", "color_point_cloud")
        post_processing_functions = []
        if voxel_grid_size is not None:
            post_processing_functions.append(
                functools.partial(voxel_grid_sample, voxel_grid_size=voxel_grid_size)
            )
        if observation_type == "point_cloud":
            env = PointCloudFromDepthImageObservationWrapper(
                env, post_processing_functions=post_processing_functions
            )
        elif observation_type == "color_point_cloud":
            env = ColorPointCloudWrapper(
                env, post_processing_functions=post_processing_functions
            )
    env = TimeLimit(env, max_episode_steps)
