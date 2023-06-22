import torch
from torch.nn import ModuleList, MultiheadAttention
from torch_geometric.nn import MLP

from pc_rl.models.modules.embedder import Embedder
from pc_rl.models.modules.prediction_head import MaePredictionHead
from pc_rl.models.modules.transformer import (MaskedDecoder, MaskedEncoder,
                                              TransformerBlock,
                                              TransformerDecoder,
                                              TransformerEncoder)

batch1_size = 80
batch2_size = 40
sampling_ratio = 0.5
mask_ratio = 0.5
group_size = 10
embed_dim = 512
mlp_1 = MLP([3, 128, 256])
mlp_2 = MLP([512, 512, embed_dim])
padding_value = 0.0
torch.manual_seed(0)
t1 = torch.rand(batch1_size, 3)
t2 = torch.rand(batch2_size, 3)
batch = torch.hstack(
    (
        torch.zeros(batch1_size, dtype=torch.long),
        torch.ones(batch2_size, dtype=torch.long),
    )
)
pos = torch.cat((t1, t2))


def test_padding_embedder():
    embedder = Embedder(
        mlp_1,
        mlp_2,
        group_size=group_size,
        sampling_ratio=sampling_ratio,
        padding_value=padding_value,
    )
    embedding, neighborhoods, center_points = embedder.forward(pos, batch)

    for e in embedding.flatten()[-512:]:
        assert e == padding_value

    for n in neighborhoods.flatten()[-5 * 3 :]:
        assert n == padding_value

    for c in center_points.flatten()[-3:]:
        assert c == padding_value


def test_padding_masked_encoder():
    EQUALS_DIM = int((batch1_size - batch2_size) * sampling_ratio * mask_ratio)
    torch.manual_seed(0)
    depth = 8
    pos_embedder = MLP([3, 128, 512], norm=None)
    padding_value = 0.0
    embedder = Embedder(
        mlp_1,
        mlp_2,
        group_size=group_size,
        sampling_ratio=sampling_ratio,
        padding_value=padding_value,
        random_start=False,
    )
    attention = MultiheadAttention(embed_dim, 1, batch_first=True, bias=False)
    block_mlp = MLP([embed_dim, 3 * embed_dim, embed_dim], norm=None)
    blocks = []
    for _ in range(depth):
        block = TransformerBlock(attention, block_mlp)
        blocks.append(block)
    blocks = ModuleList(blocks)
    transformer_encoder = TransformerEncoder(blocks)
    masked_encoder = MaskedEncoder(
        mask_ratio, transformer_encoder, pos_embedder, padding_value=padding_value
    )
    blocks = []
    for _ in range(int(depth / 2)):
        block = TransformerBlock(attention, block_mlp)
        blocks.append(block)
    blocks = ModuleList(blocks)
    transformer_decoder = TransformerDecoder(blocks)
    masked_decoder = MaskedDecoder(transformer_decoder, pos_embedder)
    prediction_head = MaePredictionHead(embed_dim, group_size)

    token, neighborhoods, center_points = embedder.forward(pos, batch)
    x_vis1, ae_mask1 = masked_encoder(token, center_points)
    x_pred1, _ = masked_decoder(x_vis1, ae_mask1, center_points)
    pos_recovered1 = prediction_head(x_pred1)

    torch.manual_seed(0)
    depth = 8
    pos_embedder = MLP([3, 128, 512], norm=None)
    padding_value = 1e9
    embedder = Embedder(
        mlp_1,
        mlp_2,
        group_size=group_size,
        sampling_ratio=sampling_ratio,
        padding_value=padding_value,
        random_start=False,
    )
    attention = MultiheadAttention(embed_dim, 1, batch_first=True, bias=False)
    blocks = []
    block_mlp = MLP([embed_dim, 3 * embed_dim, embed_dim], norm=None)
    for _ in range(depth):
        block = TransformerBlock(attention, block_mlp)
        blocks.append(block)
    blocks = ModuleList(blocks)
    transformer_encoder = TransformerEncoder(blocks)
    masked_encoder2 = MaskedEncoder(
        mask_ratio, transformer_encoder, pos_embedder, padding_value=padding_value
    )
    blocks = []
    for _ in range(int(depth / 2)):
        block = TransformerBlock(attention, block_mlp)
        blocks.append(block)
    blocks = ModuleList(blocks)
    transformer_decoder = TransformerDecoder(blocks)
    masked_decoder = MaskedDecoder(
        transformer_decoder, pos_embedder, padding_value=padding_value
    )
    prediction_head = MaePredictionHead(embed_dim, group_size)

    token, neighborhoods, center_points = embedder.forward(pos, batch)
    x_vis2, ae_mask2 = masked_encoder2(token, center_points)
    x_pred2, _ = masked_decoder(x_vis1, ae_mask1, center_points)
    pos_recovered2 = prediction_head(x_pred2)
    print(pos_recovered2.grad_fn)
    pos_recovered2[0, 0, 0] = 0
    print(pos_recovered2.grad_fn)

    assert torch.equal(x_vis1[0], x_vis2[0])
    assert torch.equal(x_vis1[1, :-EQUALS_DIM], x_vis2[1, :-EQUALS_DIM])
    assert torch.equal(ae_mask1, ae_mask2)
    assert torch.equal(x_pred1[1, :-EQUALS_DIM], x_pred2[1, :-EQUALS_DIM])
    assert torch.equal(pos_recovered1[0], pos_recovered2[0])
    assert torch.equal(pos_recovered1[1, :-EQUALS_DIM], pos_recovered2[1, :-EQUALS_DIM])


def test_padding_encoder():
    EQUALS_DIM = int((batch1_size - batch2_size) * sampling_ratio)
    depth = 8
    pos_embedder = MLP([3, 128, 512], norm=None)
    torch.manual_seed(0)
    padding_value = 0.0
    embedder = Embedder(
        mlp_1,
        mlp_2,
        group_size=5,
        sampling_ratio=sampling_ratio,
        padding_value=padding_value,
        random_start=False,
    )
    attention = MultiheadAttention(embed_dim, 1, batch_first=True, bias=False)
    blocks = []
    block_mlp = MLP([embed_dim, 3 * embed_dim, embed_dim], norm=None)
    for _ in range(depth):
        block = TransformerBlock(attention, block_mlp)
        blocks.append(block)
    blocks = ModuleList(blocks)
    transformer_encoder = TransformerEncoder(blocks)
    token, neighborhoods, center_points = embedder.forward(pos, batch)
    center_points = pos_embedder(center_points)
    padding_token = torch.full((1, embed_dim), padding_value)
    padding_mask = torch.all(token == padding_token, dim=-1)
    x1 = transformer_encoder(token, center_points, padding_mask)

    torch.manual_seed(0)
    padding_value = 1e9
    embedder = Embedder(
        mlp_1,
        mlp_2,
        group_size=5,
        sampling_ratio=sampling_ratio,
        padding_value=padding_value,
        random_start=False,
    )
    attention = MultiheadAttention(embed_dim, 1, batch_first=True, bias=False)
    blocks = []
    block_mlp = MLP([embed_dim, 3 * embed_dim, embed_dim], norm=None)
    for _ in range(depth):
        block = TransformerBlock(attention, block_mlp)
        blocks.append(block)
    blocks = ModuleList(blocks)
    transformer_encoder = TransformerEncoder(blocks)
    token, neighborhoods, center_points = embedder.forward(pos, batch)
    center_points = pos_embedder(center_points)
    padding_token = torch.full((1, embed_dim), padding_value)
    padding_mask = torch.all(token == padding_token, dim=-1)
    x2 = transformer_encoder(token, center_points, padding_mask)

    print(x1 - x2)
    assert torch.equal(x1[:, :-EQUALS_DIM], x2[:, :-EQUALS_DIM])


def test_padding_block():
    EQUALS_DIM = int((batch1_size - batch2_size) * sampling_ratio)
    torch.manual_seed(0)
    padding_value = 0.0
    embedder = Embedder(
        mlp_1,
        mlp_2,
        group_size=5,
        sampling_ratio=sampling_ratio,
        padding_value=padding_value,
        random_start=False,
    )
    attention = MultiheadAttention(embed_dim, 1, batch_first=True, bias=False)
    block_mlp = MLP([embed_dim, 3 * embed_dim, embed_dim], norm=None)
    block = TransformerBlock(attention, block_mlp)
    token, neighborhoods, center_points = embedder.forward(pos, batch)
    padding_token = torch.full((1, embed_dim), padding_value)
    padding_mask = torch.all(token == padding_token, dim=-1)
    x1 = block(token, padding_mask)

    torch.manual_seed(0)
    padding_value = 1e9
    embedder = Embedder(
        mlp_1,
        mlp_2,
        group_size=5,
        sampling_ratio=sampling_ratio,
        padding_value=padding_value,
        random_start=False,
    )
    attention = MultiheadAttention(embed_dim, 1, batch_first=True, bias=False)
    block_2 = TransformerBlock(attention, block_mlp)
    token, neighborhoods, center_points = embedder.forward(pos, batch)
    padding_token = torch.full((1, embed_dim), padding_value)
    padding_mask = torch.all(token == padding_token, dim=-1)
    x2 = block_2(token, padding_mask)
    assert torch.equal(x1[0], x2[0])
    assert torch.equal(x1[1, :-EQUALS_DIM], x2[1, :-EQUALS_DIM])


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
    test_padding_masked_encoder()
    # test_padding_embedder()
    # test_padding_block()
    # test_padding_encoder()
    # test_padding()
