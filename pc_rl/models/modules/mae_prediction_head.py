from __future__ import annotations

import torch.nn as nn


class MaePredictionHead(nn.Module):
    # TODO: maybe rewrite this using the pytorch MessagePassing interface
    def __init__(self, dim: int, group_size: int):
        super().__init__()
        self.head = nn.Conv1d(dim, 3 * group_size, 1)

    def forward(self, x):
        B, M, _ = x.shape
        prediction = self.head(x.transpose(1, 2)).transpose(1, 2)
        prediction = prediction.reshape(B, M, -1, 3)
        return prediction
