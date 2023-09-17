from __future__ import annotations

import gymnasium as gym
import torch_geometric
import torch
from torch_geometric.utils import scatter

class VoxelGridWrapper(gym.Wrapper):
    def __init__(self, env: Env[ObsType, ActType], size: float):
        self.size = size
        self.env = env

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = torch.tensor(observation)
        c = torch_geometric.nn.voxel_grid(observation, self.size)
        c, perm = torch_geometric.nn.pool.consecutive.consecutive_cluster(c)
        observation = scatter(observation, c, dim=0, reduce="mean")
        return observation.numpy(), reward, terminated, truncated, info

