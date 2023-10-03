from __future__ import annotations

import pathlib

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn.models.re_net import Data


class PcInMemoryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.file_paths = [f for f in pathlib.Path(root).iterdir() if f.is_file()]
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["000000.npy", "000001.npy", ...]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        data_list = []
        for file_path in self.file_paths:
            data = torch.from_numpy(np.load(file_path)).float()
            pos = data[..., :3]
            if data.shape[-1] > 3:
                x = data[..., 3:]
            else:
                x = None
            data_list.append(Data(pos=pos, x=x))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
