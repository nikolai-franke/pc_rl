import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class ColorPointCloud(BaseTransform):
    def __init__(self, additional_channels: int) -> None:
        self.additional_channels = additional_channels

    def __call__(self, data: Data) -> Data:
        assert data.x == None
        new_color = np.random.random(self.additional_channels).astype(np.float32)
        full_color = np.broadcast_to(
            new_color, (data.pos.shape[0], self.additional_channels)
        )
        data.x = torch.tensor(full_color)
        return data
