import torch
from torch.nn import MultiheadAttention
from torch_geometric.nn import MLP

from pc_rl.models.modules.embedder import Embedder
from pc_rl.models.modules.transformer import TransformerBlock

embed_dim = 512
mlp_1 = MLP([3, 128, 256])
mlp_2 = MLP([512, 512, embed_dim])
padding_value = 0.0
embedder = Embedder(
    mlp_1, mlp_2, group_size=5, sampling_ratio=0.5, padding_value=padding_value
)
attention = MultiheadAttention(embed_dim, 1, batch_first=True, bias=False)
block_mlp = MLP([512, 1532, 512], norm=None)
block = TransformerBlock(attention, block_mlp, padding_value=padding_value)
t1 = torch.rand(10, 3)
t2 = torch.rand(8, 3)
batch = torch.hstack(
    (torch.zeros(10, dtype=torch.long), torch.ones(8, dtype=torch.long))
)
pos = torch.cat((t1, t2))
torch.manual_seed(0)


def test_padding_embedder():
    embedding, neighborhoods, center_points = embedder.forward(pos, batch)

    for e in embedding.flatten()[-512:]:
        assert e == padding_value

    for n in neighborhoods.flatten()[-5 * 3 :]:
        assert n == padding_value

    for c in center_points.flatten()[-3:]:
        assert c == padding_value


def test_padding_block():
    torch.manual_seed(0)
    embedder = Embedder(
        mlp_1,
        mlp_2,
        group_size=5,
        sampling_ratio=0.5,
        padding_value=0.0,
        random_start=False,
    )
    attention = MultiheadAttention(embed_dim, 1, batch_first=True, bias=False)
    block = TransformerBlock(attention, block_mlp, padding_value=0.0)
    token, neighborhoods, center_points = embedder.forward(pos, batch)
    x_1 = block(token)

    torch.manual_seed(0)
    embedder = Embedder(
        mlp_1,
        mlp_2,
        group_size=5,
        sampling_ratio=0.5,
        padding_value=1e9,
        random_start=False,
    )
    attention = MultiheadAttention(embed_dim, 1, batch_first=True, bias=False)
    block_2 = TransformerBlock(attention, block_mlp, padding_value=1e9)
    token, neighborhoods, center_points = embedder.forward(pos, batch)
    x_2 = block_2(token)
    assert torch.equal(x_1[:-1], x_2[:-1])


def test_padding():
    torch.manual_seed(0)
    attention = MultiheadAttention(1, 1, batch_first=True, bias=False)
    input = torch.Tensor([1.0, 2.0, 3.0, 0.0]).unsqueeze(-1)
    input_2 = torch.Tensor([1.0, 2.0, 3.0, -1e32]).unsqueeze(-1)
    mask = torch.Tensor([False, False, False, True]).bool()
    a1 = attention(input, input, input, key_padding_mask=mask)
    a2 = attention(input_2, input_2, input_2, key_padding_mask=mask)
    # a2 = attention(input, input, input)
    print(a1, a2)


if __name__ == "__main__":
    test_padding_embedder()
    # test_padding_block()
    # test_padding()
