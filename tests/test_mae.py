import torch
import torch.nn as nn
import torch_geometric
from knn_cuda import KNN
from pointnet2_ops import pointnet2_utils

from pc_rl.models.point_mae import Embedder


def custom_fps(data, number):
    """
    data B N 3
    number int
    """
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = (
        pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx)
        .transpose(1, 2)
        .contiguous()
    )
    return fps_data


class Encoder(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        """
        point_groups : B G N 3
        -----------------
        feature_global : B G C
        """
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # print("POINT GROUPS", point_groups.transpose(2, 1).shape)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        # print("FEATURE CONV 1", feature.shape)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        # print("FEATURE GLOBAL 1", feature_global.shape)
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1
        )  # BG 512 n
        # print("FEATURE CAT", feature.shape)
        feature = self.second_conv(feature)  # BG 1024 n
        # print("FEATURE", feature.shape)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        # print("GLOBAL", feature_global.shape)
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        """
        input: B N 3
        ---------------------------
        output: B G M 3
        center : B G 3
        """
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = custom_fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = (
            torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        )
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(
            batch_size, self.num_group, self.group_size, 3
        ).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


def init_layers(layers):
    for layer in layers:
        if isinstance(layer, torch.nn.modules.conv.Conv1d) or isinstance(
            layer, torch_geometric.nn.dense.linear.Linear
        ):
            torch.nn.init.uniform_(layer.weight)
            torch.nn.init.uniform_(layer.bias)


def test_embedding():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_points_per_batch = 10
    num_batches = 4
    num_groups = 3
    k = 4
    embedding_size = 5

    input_tensor = torch.rand(num_batches, num_points_per_batch, 3).to(device)
    group = Group(num_groups, k).to(device)
    encoder = Encoder(embedding_size).to(device)
    torch.manual_seed(0)
    init_layers(encoder.first_conv)
    init_layers(encoder.second_conv)
    neighborhood, _ = group.forward(input_tensor)

    sampling_ratio = num_groups / num_points_per_batch
    embedder = Embedder(sampling_ratio, k, embedding_size, random_start=False).to(
        device
    )
    torch.manual_seed(0)
    init_layers(embedder.conv.mlp_1.lins)
    init_layers(embedder.conv.mlp_2.lins)

    out_1 = encoder.forward(neighborhood)
    input_tensor_2 = input_tensor.reshape(num_batches * num_points_per_batch, -1).to(
        device
    )
    batch = torch.arange(num_batches, dtype=torch.long)
    batch = batch.repeat_interleave(num_points_per_batch).to(device)
    out_2 = embedder.forward(None, input_tensor_2, batch)

    assert torch.allclose(out_1.reshape(-1, embedding_size), out_2)
