# from pc_rl.chamfer_distance.chamfer_distance import ChamferDistance
# from chamferdist import ChamferDistance
from pytorch3d.loss import chamfer_distance
from torch import nn
from torch_geometric.nn import MLP

from pc_rl.models.masked_autoencoder import MaskedAutoEncoder
from pc_rl.models.modules.embedder import Embedder
from pc_rl.models.modules.prediction_head import MaePredictionHead
from pc_rl.models.modules.transformer import (Attention, Block, MaskedDecoder,
                                              MaskedEncoder,
                                              TransformerDecoder,
                                              TransformerEncoder)


def build_masked_autoencoder(config):
    embedder_conf = config["embedder"]
    embedding_size = embedder_conf["embedding_size"]
    group_size = embedder_conf["group_size"]

    mlp_1 = MLP(list(embedder_conf["mlp_1_layers"]), act=embedder_conf["act"])
    mlp_2_layers = list(embedder_conf["mlp_2_layers"])
    mlp_2_layers.append(embedding_size)
    mlp_2 = MLP(mlp_2_layers, act=embedder_conf["act"])
    embedder = Embedder(
        mlp_1=mlp_1,
        mlp_2=mlp_2,
        group_size=group_size,
        sampling_ratio=embedder_conf["sampling_ratio"],
        random_start=embedder_conf["random_start"],
    )

    encoder_conf = config["encoder"]
    attention_conf = encoder_conf["attention"]
    blocks = []
    for _ in range(encoder_conf["depth"]):
        mlp = MLP(
            list(encoder_conf["mlp_layers"]),
            act=encoder_conf["act"],
            norm=None,
            dropout=encoder_conf["dropout_rate"],
        )
        # attention = Attention(
        #     dim=embedding_size,
        #     num_heads=attention_conf["num_heads"],
        #     qkv_bias=attention_conf["qkv_bias"],
        #     dropout_rate=attention_conf["dropout_rate"],
        #     proj_dropout_rate=attention_conf["proj_dropout_rate"],
        # )
        attention = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=attention_conf["num_heads"],
            add_bias_kv=attention_conf["qkv_bias"],
            dropout=attention_conf["dropout_rate"],
            batch_first=True,
        )
        blocks.append(Block(attention, mlp))

    blocks = nn.ModuleList(blocks)

    transformer_encoder = TransformerEncoder(blocks)

    masked_encoder_conf = config["masked_encoder"]
    pos_embedder = MLP(
        list(masked_encoder_conf["pos_embedder"]["mlp_layers"]),
        act=masked_encoder_conf["pos_embedder"]["act"],
        norm=None,
    )
    masked_encoder = MaskedEncoder(
        mask_ratio=masked_encoder_conf["mask_ratio"],
        transformer_encoder=transformer_encoder,
        pos_embedder=pos_embedder,
        mask_type=masked_encoder_conf["mask_type"],
    )

    decoder_conf = config["decoder"]
    blocks = []
    for _ in range(decoder_conf["depth"]):
        mlp = MLP(
            list(decoder_conf["mlp_layers"]),
            act=decoder_conf["act"],
            norm=None,
            dropout=decoder_conf["dropout_rate"],
        )
        # attention = Attention(
        #     dim=embedding_size,
        #     num_heads=attention_conf["num_heads"],
        #     qkv_bias=attention_conf["qkv_bias"],
        #     dropout_rate=attention_conf["dropout_rate"],
        #     proj_dropout_rate=attention_conf["proj_dropout_rate"],
        # )
        attention = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=attention_conf["num_heads"],
            add_bias_kv=attention_conf["qkv_bias"],
            dropout=attention_conf["dropout_rate"],
            batch_first=True,
        )
        blocks.append(Block(attention, mlp))

    masked_decoder_conf = config["masked_decoder"]

    blocks = nn.ModuleList(blocks)
    pos_embedder = MLP(
        list(masked_decoder_conf["pos_embedder"]["mlp_layers"]),
        act=masked_decoder_conf["pos_embedder"]["act"],
        norm=None,
    )

    transformer_decoder = TransformerDecoder(blocks)
    masked_decoder = MaskedDecoder(transformer_decoder, pos_embedder)

    loss_function = chamfer_distance
    prediction_head = MaePredictionHead(embedding_size, group_size)
    masked_autoencoder = MaskedAutoEncoder(
        embedder=embedder,
        encoder=masked_encoder,
        decoder=masked_decoder,
        prediction_head=prediction_head,
        loss_function=loss_function,
        learning_rate=config["learning_rate"],
    )

    return masked_autoencoder
