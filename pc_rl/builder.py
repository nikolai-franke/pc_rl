from torch import nn
from torch_geometric.nn import MLP

from pc_rl.models.embedder import Embedder
from pc_rl.models.transformer import (Attention, Block, MaskedEncoder,
                                      TransformerDecoder, TransformerEncoder)


def build_mask_transformer(conf):
    embedder_conf = conf["embedder"]

    mlp_1 = MLP(embedder_conf["mlp_1_layers"], act=embedder_conf["act"])
    mlp_2 = MLP(embedder_conf["mlp_2_layers"], act=embedder_conf["act"])
    embedder = Embedder(
        mlp_1=mlp_1,
        mlp_2=mlp_2,
        neighborhood_size=embedder_conf["neighborhood_size"],
        sampling_ratio=embedder_conf["sampling_ratio"],
        random_start=embedder_conf["random_start"],
    )

    encoder_conf = conf["encoder"]
    attention_conf = conf["attention"]
    blocks = []
    for _ in range(encoder_conf["depth"]):
        mlp = MLP(
            encoder_conf["mlp_layers"],
            act=encoder_conf["act"],
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
    pos_embedder = MLP(
        mask_transformer_conf["pos_embedder_layers"],
        act=mask_transformer_conf["pos_embedder_act"],
    )
    mask_transformer = MaskedEncoder(
        mask_ratio=mask_transformer_conf["mask_ratio"],
        embedder=embedder,
        encoder=encoder,
        pos_embedder=pos_embedder,
        mask_type=mask_transformer_conf["mask_type"],
    )

    return mask_transformer
