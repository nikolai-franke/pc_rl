from torch import nn
from torch_geometric.nn import MLP

from pc_rl.models.embedder import Embedder
from pc_rl.models.transformer import (Attention, Block, MaskTransformer,
                                      TransformerDecoder, TransformerEncoder)


def build_mask_transformer(conf):
    embedder_conf = conf["embedder"]
    embedder = Embedder(
        hidden_layers=embedder_conf["hidden_layers"],
        embedding_size=embedder_conf["embedding_size"],
        neighborhood_size=embedder_conf["neighborhood_size"],
        sampling_ratio=embedder_conf["sampling_ratio"],
        random_start=embedder_conf["random_start"],
    )

    attention_conf = conf["attention"]

    encoder_conf = conf["encoder"]
    blocks = []
    for _ in range(encoder_conf["depth"]):
        mlp = MLP(
            encoder_conf["mlp_layers"],
            act=encoder_conf["mlp_activation"],
            norm=None,
            dropout=encoder_conf["mlp_dropout_rate"],
        )
        attention = Attention(
            dim=attention_conf["dim"],
            num_heads=attention_conf["num_heads"],
            qkv_bias=attention_conf["qkv_bias"],
            attention_dropout_rate=attention_conf["attention_dropout_rate"],
            projection_dropout_rate=attention_conf["projection_dropout_rate"],
        )
        blocks.append(Block(attention, mlp))

    blocks = nn.ModuleList(blocks)

    encoder = TransformerEncoder(blocks)

    mask_transformer_conf = conf["mask_transformer"]
    pos_embedder = MLP([3, 128, embedder_conf["embedding_size"]], act=nn.GELU())
    mask_transformer = MaskTransformer(
        mask_ratio=mask_transformer_conf["mask_ratio"],
        embedder=embedder,
        encoder=encoder,
        pos_embedder=pos_embedder,
        mask_type=mask_transformer_conf["mask_type"],
    )

    return mask_transformer
