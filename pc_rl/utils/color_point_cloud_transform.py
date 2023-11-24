import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class AddChannels(BaseTransform):
    def __init__(self, additional_channels: int, randomize: bool = False) -> None:
        self.additional_channels = additional_channels
        self.randomize = randomize

    def __call__(self, data: Data) -> Data:
        assert data.x == None
        if self.randomize:
            new_channel = np.random.random(self.additional_channels).astype(np.float32)
        else:
            new_channel = np.zeros(self.additional_channels).astype(np.float32)
        full_channel = np.broadcast_to(
            new_channel, (data.pos.shape[0], self.additional_channels)
        )
        data.x = torch.tensor(full_channel)  # type: ignore
        return data
