from __future__ import annotations

import os.path as osp

import numpy as np
import torch
from torch_geometric.data import Dataset
from torch_geometric.nn.models.re_net import Data


class PcOutOfMemoryDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ["000000.npy", "000001.npy", ...]

    @property
    def processed_file_names(self):
        return ["data_1.pt", "data_2.pt", ...]

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            data = torch.from_numpy(np.load(raw_path)).float()

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(
                data, osp.join(self.processed_dir, f"data_{str(idx).zfill(6)}.pt")
            )
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"data_{str(idx).zfill(6)}.pt"))
        return data
