from __future__ import annotations

import gymnasium as gym
import numpy as np
from sapien.core import Pose

from pc_rl.utils.point_cloud_post_processing_functions import voxel_grid_sample


def apply_pose_to_points(x, pose):
    return to_normal(to_generalized(x) @ pose.to_transformation_matrix().T)


def to_generalized(x):
    if x.shape[-1] == 4:
        return x
    assert x.shape[-1] == 3
    output_shape = list(x.shape)
    output_shape[-1] = 4
    ret = np.ones(output_shape)
    ret[..., :3] = x
    return ret


def to_normal(x):
    if x.shape[-1] == 3:
        return x
    assert x.shape[-1] == 4
    return x[..., :3] / x[..., 3:]


class ManiSkillPointCloudWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env,
        n_goal_points: int | None = None,
        obs_frame: str = "ee",
        use_color: bool = False,
        filter_points_below_z: float | None = None,
        voxel_grid_size: float | None = None,
    ):
        super().__init__(env)
        num_points = env.observation_space["pointcloud"]["xyzw"].shape[0]
        point_dim = 6 if use_color else 3
        self.observation_space = gym.spaces.Box(
            high=np.inf, low=-np.inf, shape=(num_points, point_dim)
        )
        self.env = env
        self.n_goal_points = n_goal_points
        self.obs_frame = obs_frame
        self.use_color = use_color
        self.filter_points_below_z = filter_points_below_z
        self.voxel_grid_size = voxel_grid_size

    def observation(self, observation):
        if self.obs_frame in ["base", "world"]:
            base_pose = observation["agent"]["base_pose"]
            p, q = base_pose[:3], base_pose[3:]
            to_origin = Pose(p=p, q=q).inv()
        elif self.obs_frame == "ee":
            tcp_poses = observation["extra"]["tcp_pose"]
            assert tcp_poses.ndim <= 2
            if tcp_poses.ndim == 2:
                tcp_pose = tcp_poses[
                    0
                ]  # use the first robot hand's tcp pose as the end-effector frame
            else:
                tcp_pose = tcp_poses  # only one robot hand
            p, q = tcp_pose[:3], tcp_pose[3:]
            to_origin = Pose(p=p, q=q).inv()
        else:
            raise ValueError(f"Invalid obs_frame: {self.obs_frame}")

        point_cloud = observation["pointcloud"]["xyzw"]
        mask = point_cloud[..., -1] == 1
        point_cloud = point_cloud[mask][..., :-1]

        filter_mask = None
        if self.filter_points_below_z is not None:
            filter_mask = point_cloud[..., 2] > self.filter_points_below_z
            point_cloud = point_cloud[filter_mask]

        if self.use_color:
            rgb = observation["pointcloud"]["rgb"].astype(float)
            rgb = rgb[mask]
            rgb *= 1 / 255.0
            if filter_mask is not None:
                rgb = rgb[filter_mask]

        if self.n_goal_points is not None:
            assert (goal_pos := observation["extra"]["goal_pos"]) is not None
            assert self.use_color
            goal_pts_xyz = (
                np.random.uniform(low=-1.0, high=1.0, size=(self.n_goal_points, 3))
                * 0.01
            )
            goal_pts_xyz = goal_pts_xyz + goal_pos[None, :]
            goal_pts_rgb = np.zeros_like(goal_pts_xyz)
            goal_pts_rgb[:, 1] = 1
            point_cloud = np.concatenate([point_cloud, goal_pts_xyz])
            rgb = np.concatenate([rgb, goal_pts_rgb])  # type: ignore

        point_cloud = apply_pose_to_points(point_cloud, to_origin)

        if self.use_color:
            point_cloud = np.hstack([point_cloud, rgb])  # type: ignore

        if self.voxel_grid_size is not None:
            point_cloud = voxel_grid_sample(point_cloud, self.voxel_grid_size)

        return point_cloud


# class ManiSkillPointCloudWrapper(gym.ObservationWrapper):
#     def __init__(
#         self,
#         env,
#         post_processing_functions: list[Callable] | None = None,
#         n_goal_points: int = -1,
#         use_color: bool = False,
#     ):
#         super().__init__(env)
#         num_points = env.observation_space["pointcloud"]["xyzw"].shape[0]
#         point_dim = 6 if use_color else 3
#         self.observation_space = gym.spaces.Box(
#             high=np.inf, low=-np.inf, shape=(num_points, point_dim)
#         )
#         self.post_processing_functions = post_processing_functions
#         self.use_color = use_color
#         self.n_goal_points = n_goal_points
#         self.counter = 0
#
#     def observation(self, observation):
#         point_cloud = observation["pointcloud"]["xyzw"]
#         # only points where w == 1 are valid
#         mask = point_cloud[..., -1] == 1
#         # we only want xyz in our observation
#         point_cloud = point_cloud[mask][..., :3]
#
#         if self.use_color:
#             rgb = observation["pointcloud"]["rgb"].astype(float)
#             rgb = rgb[mask]
#             rgb *= 1 / 255.0
#             point_cloud = np.hstack((point_cloud, rgb))
#
#         if self.post_processing_functions is not None:
#             for function in self.post_processing_functions:
#                 point_cloud = function(point_cloud)
#
#         # np.savetxt(
#         #     f"obs{self.counter}.csv",
#         #     point_cloud,
#         #     delimiter=",",
#         # )
#         # self.counter += 1
#
#         return point_cloud
