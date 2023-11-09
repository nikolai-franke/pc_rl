from __future__ import annotations

from collections import OrderedDict, defaultdict, deque

import gymnasium as gym
import numpy as np
from sapien.core import Pose

from pc_rl.utils.point_cloud_post_processing_functions import (
    normalize, voxel_grid_sample)


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


class ManiFrameStack(gym.ObservationWrapper):
    def __init__(
        self,
        env,
        image_shape,
        filter_points_below_z,
        voxel_grid_size: float | None = None,
        normalize: bool = True,
        num_frames: int = 4,
        num_classes: int = 75,
        convert_to_ee_frame: bool = True,
    ):
        super().__init__(env)
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.frames = deque(maxlen=self.num_frames)
        self.filter_points_below_z = filter_points_below_z
        self.convert_to_ee_frame = convert_to_ee_frame
        self.voxel_grid_size = voxel_grid_size
        self.normalize = normalize

        # shape is hardcoded for 3 cameras and two segmentation masks
        self.observation_space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(image_shape[0] * image_shape[1] * num_frames * 3, 6),
        )

    def _point_cloud(self, observation):
        image_obs = observation["image"]
        camera_params = observation["camera_param"]
        pointcloud_obs = OrderedDict()

        for cam_uid, images in image_obs.items():
            cam_pcd = {}

            position = images["Position"]
            position[..., 3] = position[..., 2] < 0

            # Convert to world space
            cam2world = camera_params[cam_uid]["cam2world_gl"]
            xyzw = position.reshape(-1, 4) @ cam2world.T
            cam_pcd["xyzw"] = xyzw

            seg = images["Segmentation"].reshape(-1, 4)[..., :3]
            cam_pcd["segmentation"] = seg

            pointcloud_obs[cam_uid] = cam_pcd

        pointcloud_obs = merge_dicts(pointcloud_obs.values())

        out = np.dstack(
            (
                np.asarray(pointcloud_obs["xyzw"])[..., :3],
                np.asarray(pointcloud_obs["segmentation"]) / self.num_classes,
            ),
        )
        out = out[out[..., 2] > self.filter_points_below_z]

        if self.convert_to_ee_frame:
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
            out[..., :3] = apply_pose_to_points(out[..., :3], to_origin)

        if self.voxel_grid_size is not None:
            out = voxel_grid_sample(out, self.voxel_grid_size)

        return out.reshape(-1, out.shape[-1])

    def observation(self, observation):
        for i, frame in enumerate(self.frames):
            frame[..., -1] = i / self.num_frames

        out = np.concatenate(self.frames)

        # normalize full point cloud
        if self.normalize:
            out = normalize(out)
        return out.reshape(-1, out.shape[-1])

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(self._point_cloud(observation))
        return self.observation(None), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        [self.frames.append(self._point_cloud(obs)) for _ in range(self.num_frames)]
        return self.observation(None), info


class ManiSkillPointCloudWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env,
        n_goal_points: int | None = None,
        obs_frame: str = "ee",
        use_color: bool = False,
        filter_points_below_z: float | None = None,
        voxel_grid_size: float | None = None,
        normalize: bool = True,
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
        self.normalize = normalize

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

        rgb = observation["pointcloud"]["rgb"].astype(np.float32)
        rgb = rgb[mask]
        rgb *= 1 / 255.0

        segmentation = observation["pointcloud"].get("Segmentation")

        if self.filter_points_below_z is not None:
            filter_mask = point_cloud[..., 2] > self.filter_points_below_z
            point_cloud = point_cloud[filter_mask]
            rgb = rgb[filter_mask]
            if segmentation is not None:
                segmentation = segmentation[filter_mask]

        if self.n_goal_points is not None:
            assert (goal_pos := observation["extra"]["goal_pos"]) is not None
            goal_pts_xyz = (
                np.random.uniform(low=-1.0, high=1.0, size=(self.n_goal_points, 3))
                * 0.01
            ).astype(np.float32)
            goal_pts_xyz = goal_pts_xyz + goal_pos[None, :]
            goal_pts_rgb = np.zeros_like(goal_pts_xyz)
            goal_pts_rgb[:, 1] = 1
            point_cloud = np.concatenate([point_cloud, goal_pts_xyz])
            rgb = np.concatenate([rgb, goal_pts_rgb])  # type: ignore

        point_cloud = apply_pose_to_points(point_cloud, to_origin)

        if self.use_color:
            point_cloud = np.hstack([point_cloud, rgb])  # type: ignore

        if segmentation is not None:
            out = np.zeros((point_cloud.shape[0], 6), dtype=np.float32)
            out[..., :3] = point_cloud
            out[..., 3] = segmentation[..., 0] / 100.0
            point_cloud = out

        if self.voxel_grid_size is not None:
            point_cloud = voxel_grid_sample(point_cloud, self.voxel_grid_size)

        if self.normalize:
            point_cloud = normalize(point_cloud)

        return point_cloud
