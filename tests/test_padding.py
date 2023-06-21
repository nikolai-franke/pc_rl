import torch
from torch_geometric.nn import MLP

from pc_rl.models.modules.embedder import Embedder

mlp_1 = MLP([3, 128, 256])
mlp_2 = MLP([512, 512, 512])
padding_value = float("inf")
embedder = Embedder(
    mlp_1, mlp_2, group_size=5, sampling_ratio=0.5, padding_value=padding_value
)


def test_padding():
    t1 = torch.rand(10, 3)
    t2 = torch.rand(8, 3)
    batch = torch.hstack(
        (torch.zeros(10, dtype=torch.long), torch.ones(8, dtype=torch.long))
    )
    pos = torch.cat((t1, t2))
    embedding, neighborhoods, center_points = embedder.forward(pos, batch)

    for e in embedding.flatten()[:-512]:
        assert e != padding_value
    for e in embedding.flatten()[-512:]:
        assert e == padding_value

    for n in neighborhoods.flatten()[: -5 * 3]:
        assert n != padding_value
    for n in neighborhoods.flatten()[-5 * 3 :]:
        assert n == padding_value

    for c in center_points.flatten()[:-3]:
        assert c != padding_value
    for c in center_points.flatten()[-3:]:
        assert c == padding_value


if __name__ == "__main__":
    test_padding()
