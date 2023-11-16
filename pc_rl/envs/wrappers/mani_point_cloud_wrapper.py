from __future__ import annotations

from collections import OrderedDict, defaultdict, deque
from typing import Literal

import gymnasium as gym
import numpy as np
from sapien.core import Pose

from pc_rl.utils.point_cloud_post_processing_functions import (
    normalize, voxel_grid_sample)

# These functions and wrappers are adapted/inspired by the ones found in https://github.com/haosulab/ManiSkill2-Learn


def apply_pose_to_points(x, pose):
    return to_normal(to_generalized(x) @ pose.to_transformation_matrix().T)


def to_generalized(x):
    if x.shape[-1] == 4:
        return x
    assert x.shape[-1] == 3
    output_shape = list(x.shape)
    output_shape[-1] = 4
    ret = np.ones(output_shape).astype(np.float32)
    ret[..., :3] = x
    return ret


def to_normal(x):
    if x.shape[-1] == 3:
        return x
    assert x.shape[-1] == 4
    return x[..., :3] / x[..., 3:]


def merge_dicts(ds, asarray=False):
    """Merge multiple dicts with the same keys to a single one."""
    # NOTE(jigu): To be compatible with generator, we only iterate once.
    ret = defaultdict(list)
    for d in ds:
        for k in d:
            ret[k].append(d[k])
    ret = dict(ret)
    # Sanity check (length)
    assert len(set(len(v) for v in ret.values())) == 1, "Keys are not same."
    if asarray:
        ret = {k: np.concatenate(v) for k, v in ret.items()}
    return ret


class PointCloudWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env,
        obs_frame: Literal["world", "base", "ee"],
        center_and_normalize: bool = True,
        use_color: bool = False,
        filter_points_below_z: float | None = None,
        n_goal_points: int = 0,
        voxel_grid_size: float | None = None,
        add_state: bool = False,
    ):
        super().__init__(env)
        self.obs_frame = obs_frame
        self.center_and_normalize = center_and_normalize
        self.use_color = use_color
        self.filter_points_below_z = filter_points_below_z
        self.n_goal_points = n_goal_points
        self.voxel_grid_size = voxel_grid_size
        self.add_state = add_state

        point_dim = 6 if use_color else 3
        max_num_points = self.env.observation_space["pointcloud"]["xyzw"].shape[0]

        point_cloud_space = gym.spaces.Box(
            low=-np.float32("inf"),
            high=np.float32("inf"),
            shape=(max_num_points, point_dim),
        )

        if self.add_state:
            agent_space = self.env.observation_space["agent"]
            # TODO: don't hardcode this
            state_size = (
                agent_space["qpos"].shape[0]
                + agent_space["qvel"].shape[0]
                + agent_space["base_pose"].shape[0]
                + agent_space["base_vel"].shape[0]
                + agent_space["base_ang_vel"].shape[0]
            )
            self.observation_space = gym.spaces.Dict(
                {
                    "point_cloud": point_cloud_space,
                    "state": gym.spaces.Box(
                        low=-np.float32("inf"),
                        high=np.float32("inf"),
                        shape=(state_size,),
                    ),
                }
            )
        else:
            self.observation_space = point_cloud_space

    def observation(self, observation):
        point_cloud = observation["pointcloud"]["xyzw"]
        point_cloud_rgb = observation["pointcloud"]["rgb"].astype(np.float32) / 255.0

        # filter out points beyond z_far of camera
        mask = point_cloud[..., -1] == 1
        point_cloud = point_cloud[mask][..., :3]
        point_cloud_rgb = point_cloud_rgb[mask]

        if self.n_goal_points > 0:
            # TODO: don't hardcode target dict key
            goal_pos = observation["extra"]["target_link_pos"]
            goal_points_pos = (
                np.random.uniform(
                    low=-1.0, high=1.0, size=(self.n_goal_points, 3)
                ).astype(np.float32)
                * 0.05
            )
            goal_points_pos = goal_points_pos + goal_pos
            point_cloud = np.concatenate([point_cloud, goal_points_pos])

            if self.use_color:
                goal_points_rgb = np.zeros_like(goal_points_pos)
                # TODO: don't hardcode color
                goal_points_rgb[:, 1] = 1
                point_cloud_rgb = np.concatenate([point_cloud_rgb, goal_points_rgb])

        if self.use_color:
            point_cloud = np.hstack([point_cloud, point_cloud_rgb])

        if self.filter_points_below_z is not None:
            point_cloud = point_cloud[point_cloud[..., 2] > self.filter_points_below_z]

        if self.obs_frame == "base":
            # TODO: not sure if this is always valid
            base_pose = observation
            base_pose = observation["agent"]["base_pose"]
            p, q = base_pose[:3], base_pose[3:]
            to_origin = Pose(p=p, q=q).inv()
            point_cloud[..., :3] = apply_pose_to_points(point_cloud[..., :3], to_origin)
        elif self.obs_frame == "ee":
            # TODO: what to do with multiple EEs?
            raise NotImplementedError

        if self.voxel_grid_size is not None:
            point_cloud = voxel_grid_sample(point_cloud, self.voxel_grid_size)

        if self.center_and_normalize:
            point_cloud = normalize(point_cloud)

        if self.add_state:
            agent_obs = observation["agent"]
            state = np.concatenate(
                [
                    agent_obs["qpos"],
                    agent_obs["qvel"],
                    agent_obs["base_pose"],
                    agent_obs["base_vel"],
                    agent_obs["base_ang_vel"].reshape(-1),
                ]
            )
            return {"point_cloud": point_cloud, "state": state}

        return point_cloud


class FrameStackWrapper(gym.ObservationWrapper):
    def __init__(self, env, num_frames):
        super().__init__(env)
        self.num_frames = num_frames
        self.frames = deque(maxlen=self.num_frames)

        # TODO: we currently hardcode 3 cameras
        if isinstance(self.env.observation_space, gym.spaces.Box):
            point_cloud_shape = self.env.observation_space.shape
            self.observation_space = gym.spaces.Box(
                low=-np.float32("inf"),
                high=np.float32("inf"),
                shape=(point_cloud_shape[0] * 3, point_cloud_shape[1] + 1),  # type: ignore
            )
        elif isinstance(self.env.observation_space, gym.spaces.Dict):
            point_cloud_shape = self.env.observation_space["point_cloud"].shape
            point_cloud_space = gym.spaces.Box(
                low=-np.float32("inf"),
                high=np.float32("inf"),
                shape=(point_cloud_shape[0] * 3, point_cloud_shape[1] + 1),
            )
            state_space = self.env.observation_space["state"]
            self.observation_space = gym.spaces.Dict(
                {"point_cloud": point_cloud_space, "state": state_space}
            )

    def observation(self, observation):
        for i, frame in enumerate(self.frames):
            frame[..., -1] = (i + 1) / self.num_frames

        return np.concatenate(self.frames)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        if isinstance(observation, dict):
            point_cloud = observation["point_cloud"]
            point_cloud = self._add_column(point_cloud)
            self.frames_append(point_cloud)
            return {
                "point_cloud": self.observation(observation),
                "state": observation["state"],
            }

        observation = self._add_column(observation)
        self.frames.append(observation)
        return self.observation(observation), reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        if isinstance(observation, dict):
            point_cloud = observation["point_cloud"]
            point_cloud = self._add_column(point_cloud)
            self.frames_append(point_cloud)
            return {
                "point_cloud": self.observation(observation),
                "state": observation["state"],
            }, info

        observation = self._add_column(observation)
        for _ in range(self.num_frames):
            self.frames.append(observation)
        return self.observation(observation), info

    def _add_column(self, observation):
        return np.hstack([observation, np.zeros_like(observation[..., None, 0])])


# class ManiFrameStack(gym.ObservationWrapper):
#     def __init__(
#         self,
#         env,
#         image_shape,
#         filter_points_below_z,
#         voxel_grid_size: float | None = None,
#         normalize: bool = True,
#         num_frames: int = 4,
#         num_classes: int = 75,
#         convert_to_ee_frame: bool = True,
#         convert_to_base_frame: bool = False,
#         n_goal_points: None | int = None,
#         add_state: bool = False,
#         add_seg: bool = False,
#     ):
#         super().__init__(env)
#         self.num_frames = num_frames
#         self.num_classes = num_classes
#         self.frames = deque(maxlen=self.num_frames)
#         self.filter_points_below_z = filter_points_below_z
#         self.convert_to_ee_frame = convert_to_ee_frame
#         self.voxel_grid_size = voxel_grid_size
#         self.normalize = normalize
#         self.n_goal_points = n_goal_points
#         self.add_state = add_state
#         self.convert_to_base_frame = convert_to_base_frame
#         self.add_seg = add_seg
#         assert not (convert_to_base_frame and convert_to_ee_frame)
#
#         # shape is hardcoded for 3 cameras and two segmentation masks
#         #
#         point_dim = 6 if add_seg else 4
#         point_cloud_space = gym.spaces.Box(
#             low=-float("inf"),
#             high=float("inf"),
#             shape=(image_shape[0] * image_shape[1] * num_frames * 3, point_dim),
#         )
#         if not self.add_state:
#             self.observation_space = point_cloud_space
#         else:
#             agent_space = self.env.observation_space["agent"]
#             state_size = (
#                 agent_space["qpos"].shape[0]
#                 + agent_space["qvel"].shape[0]
#                 + agent_space["base_pose"].shape[0]
#                 + agent_space["base_vel"].shape[0]
#                 + agent_space["base_ang_vel"].shape[0]
#             )
#             self.observation_space = gym.spaces.Dict(
#                 {
#                     "point_cloud": point_cloud_space,
#                     "state": gym.spaces.Box(
#                         low=-float("inf"), high=float("inf"), shape=(state_size,)
#                     ),
#                 }
#             )
#
#     def _point_cloud(self, observation):
#         image_obs = observation["image"]
#         camera_params = observation["camera_param"]
#         pointcloud_obs = OrderedDict()
#
#         for cam_uid, images in image_obs.items():
#             cam_pcd = {}
#
#             position = images["Position"]
#             position[..., 3] = position[..., 2] < 0
#
#             # Convert to world space
#             cam2world = camera_params[cam_uid]["cam2world_gl"]
#             xyzw = position.reshape(-1, 4) @ cam2world.T
#             cam_pcd["xyzw"] = xyzw
#
#             if self.add_seg:
#                 seg = images["Segmentation"].reshape(-1, 4)[..., :3]
#                 cam_pcd["segmentation"] = seg
#             else:
#                 seg = None
#
#             pointcloud_obs[cam_uid] = cam_pcd
#
#         pointcloud_obs = merge_dicts(pointcloud_obs.values())
#
#         if self.add_seg:
#             seg = np.asarray(pointcloud_obs["segmentation"])[..., :3]
#
#         point_cloud = np.asarray(pointcloud_obs["xyzw"])[..., :3]
#
#         if self.n_goal_points is not None:
#             goal_pos = observation["extra"]["target_link_pos"]
#             goal_pts_xyz = (
#                 np.random.uniform(low=-1.0, high=1.0, size=(3, self.n_goal_points, 3))
#                 * 0.1
#             ).astype(np.float32)
#             goal_pts_xyz = goal_pts_xyz + goal_pos[None, :]
#
#             if self.add_seg:
#                 goal_pts_seg = np.zeros_like(goal_pts_xyz)
#                 goal_pts_seg[..., 1:] = self.num_classes
#                 seg = np.concatenate([seg, goal_pts_seg], axis=-2)
#
#             point_cloud = np.concatenate([point_cloud, goal_pts_xyz], axis=-2)
#
#         if self.add_seg:
#             out = np.dstack(
#                 (
#                     point_cloud,
#                     seg / self.num_classes,
#                 ),
#             )
#         else:
#             out = point_cloud
#
#         out = out[out[..., 2] > self.filter_points_below_z]
#
#         if self.convert_to_ee_frame:
#             tcp_poses = observation["extra"]["tcp_pose"]
#             assert tcp_poses.ndim <= 2
#             if tcp_poses.ndim == 2:
#                 tcp_pose = tcp_poses[
#                     0
#                 ]  # use the first robot hand's tcp pose as the end-effector frame
#             else:
#                 tcp_pose = tcp_poses  # only one robot hand
#             p, q = tcp_pose[:3], tcp_pose[3:]
#             to_origin = Pose(p=p, q=q).inv()
#             out[..., :3] = apply_pose_to_points(out[..., :3], to_origin)
#
#         if self.convert_to_base_frame:
#             base_pose = observation["agent"]["base_pose"]
#             p, q = base_pose[:3], base_pose[3:]
#             to_origin = Pose(p=p, q=q).inv()
#             out[..., :3] = apply_pose_to_points(out[..., :3], to_origin)
#
#         if self.voxel_grid_size is not None:
#             out = voxel_grid_sample(out, self.voxel_grid_size)
#
#         if not self.add_seg:
#             frames = np.zeros(out.shape[:-1] + (1,), dtype=out.dtype)
#             out = np.hstack([out, frames])
#
#         return out.reshape(-1, out.shape[-1])
#
#     def observation(self, observation):
#         for i, frame in enumerate(self.frames):
#             frame[..., -1] = i / self.num_frames
#
#         out = np.concatenate(self.frames)
#
#         # normalize full point cloud
#         if self.normalize:
#             out = normalize(out)
#
#         if self.add_state:
#             agent_obs = observation["agent"]
#             qpos = agent_obs["qpos"]
#             qvel = agent_obs["qvel"]
#             base_pose = agent_obs["base_pose"]
#             base_vel = agent_obs["base_vel"]
#             base_ang_vel = agent_obs["base_ang_vel"].reshape(
#                 -1,
#             )
#             state = np.concatenate([qpos, qvel, base_pose, base_vel, base_ang_vel])
#             out = {"point_cloud": out, "state": state}
#
#         return out
#
#     def step(self, action):
#         observation, reward, terminated, truncated, info = self.env.step(action)
#         self.frames.append(self._point_cloud(observation))
#         return self.observation(observation), reward, terminated, truncated, info
#
#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         [self.frames.append(self._point_cloud(obs)) for _ in range(self.num_frames)]
#         return self.observation(obs), info
#
#
# class ManiSkillPointCloudWrapper(gym.ObservationWrapper):
#     def __init__(
#         self,
#         env,
#         n_goal_points: int | None = None,
#         obs_frame: str = "ee",
#         use_color: bool = False,
#         filter_points_below_z: float | None = None,
#         voxel_grid_size: float | None = None,
#         normalize: bool = True,
#     ):
#         super().__init__(env)
#         num_points = env.observation_space["pointcloud"]["xyzw"].shape[0]
#         point_dim = 6 if use_color else 3
#         self.observation_space = gym.spaces.Box(
#             high=np.inf, low=-np.inf, shape=(num_points, point_dim)
#         )
#         self.env = env
#         self.n_goal_points = n_goal_points
#         self.obs_frame = obs_frame
#         self.use_color = use_color
#         self.filter_points_below_z = filter_points_below_z
#         self.voxel_grid_size = voxel_grid_size
#         self.normalize = normalize
#
#     def observation(self, observation):
#         if self.obs_frame in ["base", "world"]:
#             base_pose = observation["agent"]["base_pose"]
#             p, q = base_pose[:3], base_pose[3:]
#             to_origin = Pose(p=p, q=q).inv()
#         elif self.obs_frame == "ee":
#             tcp_poses = observation["extra"]["tcp_pose"]
#             assert tcp_poses.ndim <= 2
#             if tcp_poses.ndim == 2:
#                 tcp_pose = tcp_poses[
#                     0
#                 ]  # use the first robot hand's tcp pose as the end-effector frame
#             else:
#                 tcp_pose = tcp_poses  # only one robot hand
#             p, q = tcp_pose[:3], tcp_pose[3:]
#             to_origin = Pose(p=p, q=q).inv()
#         else:
#             raise ValueError(f"Invalid obs_frame: {self.obs_frame}")
#
#         point_cloud = observation["pointcloud"]["xyzw"]
#         mask = point_cloud[..., -1] == 1
#         point_cloud = point_cloud[mask][..., :-1]
#
#         rgb = observation["pointcloud"]["rgb"].astype(np.float32)
#         rgb = rgb[mask]
#         rgb *= 1 / 255.0
#
#         segmentation = observation["pointcloud"].get("Segmentation")
#
#         if self.filter_points_below_z is not None:
#             filter_mask = point_cloud[..., 2] > self.filter_points_below_z
#             point_cloud = point_cloud[filter_mask]
#             rgb = rgb[filter_mask]
#             if segmentation is not None:
#                 segmentation = segmentation[filter_mask]
#
#         if self.n_goal_points is not None:
#             assert (goal_pos := observation["extra"]["goal_pos"]) is not None
#             goal_pts_xyz = (
#                 np.random.uniform(low=-1.0, high=1.0, size=(self.n_goal_points, 3))
#                 * 0.01
#             ).astype(np.float32)
#             goal_pts_xyz = goal_pts_xyz + goal_pos[None, :]
#             goal_pts_rgb = np.zeros_like(goal_pts_xyz)
#             goal_pts_rgb[:, 1] = 1
#             point_cloud = np.concatenate([point_cloud, goal_pts_xyz])
#             rgb = np.concatenate([rgb, goal_pts_rgb])  # type: ignore
#
#         point_cloud = apply_pose_to_points(point_cloud, to_origin)
#
#         if self.use_color:
#             point_cloud = np.hstack([point_cloud, rgb])  # type: ignore
#
#         if segmentation is not None:
#             out = np.zeros((point_cloud.shape[0], 6), dtype=np.float32)
#             out[..., :3] = point_cloud
#             out[..., 3] = segmentation[..., 0] / 100.0
#             point_cloud = out
#
#         if self.voxel_grid_size is not None:
#             point_cloud = voxel_grid_sample(point_cloud, self.voxel_grid_size)
#
#         if self.normalize:
#             point_cloud = normalize(point_cloud)
#
#         return point_cloud
