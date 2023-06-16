from torch import nn


class MaePredictionHead(nn.Module):
    def __init__(self, dim: int, group_size: int):
        super().__init__()
        self.head = nn.Conv1d(dim, 3 * group_size, 1)

    def forward(self, x):
        B, M, _ = x.shape
        prediction = self.head(x.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)
        return prediction
