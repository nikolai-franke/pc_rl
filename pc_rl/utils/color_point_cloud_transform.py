import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class ColorPointCloud(BaseTransform):
    def __call__(self, data: Data) -> Data:
        assert data.x == None
        new_color = np.random.random(3).astype(np.float32)
        full_color = np.broadcast_to(new_color, data.pos.shape)
        data.x = torch.tensor(full_color)
        return data
