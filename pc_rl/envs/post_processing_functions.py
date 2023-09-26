import numpy as np
import open3d as o3d


def normalize(obs):
    pos = obs[:, :3]
    pos_mean = pos.mean(axis=-2)
    pos = pos - pos_mean
    scale = 1 / abs(pos).max() * 0.999999
    obs[:, :3] = pos * scale
    return obs


def voxel_grid_sample(obs, voxel_grid_size):
    assert (shape := obs.shape[-1]) in (3, 6)
    # TODO: find out if we can do this without copying the numpy arrays
    pos = obs[:, :3].copy()
    color = obs[:, 3:].copy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos)
    if shape == 6:
        pcd.colors = o3d.utility.Vector3dVector(color)
    pcd = pcd.voxel_down_sample(voxel_grid_size)
    return np.hstack((np.asarray(pcd.points), np.asarray(pcd.colors)))
