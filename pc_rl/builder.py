from torch import nn
from torch_geometric.nn import MLP

from pc_rl.models.modules.embedder import Embedder
from pc_rl.models.modules.transformer import (Attention, Block, MaskedDecoder,
                                              MaskedEncoder,
                                              TransformerDecoder,
                                              TransformerEncoder)


def build_masked_autoencoder(conf):
    embedder_conf = conf["embedder"]

    mlp_1 = MLP(embedder_conf["mlp_1_layers"], act=embedder_conf["act"])
    mlp_2 = MLP(embedder_conf["mlp_2_layers"], act=embedder_conf["act"])
    embedder = Embedder(
        mlp_1=mlp_1,
        mlp_2=mlp_2,
        group_size=embedder_conf["group_size"],
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

    transformer_encoder = TransformerEncoder(blocks)

    masked_encoder_conf = conf["masked_transformer"]
    pos_embedder = MLP(
        masked_encoder_conf["pos_embedder_layers"],
        act=masked_encoder_conf["pos_embedder_act"],
    )
    masked_encoder = MaskedEncoder(
        mask_ratio=masked_encoder_conf["mask_ratio"],
        transformer_encoder=transformer_encoder,
        pos_embedder=pos_embedder,
        mask_type=masked_encoder_conf["mask_type"],
    )

    masked_decoder_conf = conf["masked_decoder"]
    decoder_conf = conf["decoder"]

    blocks = []
    for _ in range(decoder_conf["depth"]):
        mlp = MLP(
            decoder_conf["mlp_layers"],
            act=decoder_conf["act"],
            norm=None,
            dropout=decoder_conf["mlp_dropout_rate"],
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
    pos_embedder = MLP(
        masked_decoder_conf["pos_embedder_layers"],
        act=masked_decoder_conf["pos_embedder_act"],
    )

    transformer_decoder = TransformerDecoder(blocks)
    masked_decoder = MaskedDecoder(transformer_decoder, pos_embedder)
