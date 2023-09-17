import numpy as np
import open3d as o3d


def normalize(obs):
    obs_mean = obs.mean(axis=-2)
    obs = obs - obs_mean
    scale = 1 / abs(obs).max() * 0.999999
    obs = obs * scale
    return obs


def voxel_grid_sample(obs, voxel_grid_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obs)
    pcd = pcd.voxel_down_sample(voxel_grid_size)
    return np.asarray(pcd.points)
