from pytorch3d.loss import chamfer_distance
from torch.nn import ModuleList, MultiheadAttention
from torch_geometric.nn import MLP

from pc_rl.models.masked_autoencoder import MaskedAutoEncoder
from pc_rl.models.modules.embedder import Embedder
from pc_rl.models.modules.prediction_head import MaePredictionHead
from pc_rl.models.modules.transformer import (Block, MaskedDecoder,
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

    model_conf = config["model"]

    attention_conf = model_conf["attention"]
    block_conf = model_conf["transformer_block"]
    blocks = []
    for _ in range(model_conf["encoder_depth"]):
        mlp = MLP(
            list(block_conf["mlp_layers"]),
            act=block_conf["act"],
            norm=None,
            dropout=block_conf["dropout"],
        )
        attention = MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=attention_conf["num_heads"],
            add_bias_kv=attention_conf["qkv_bias"],
            dropout=attention_conf["dropout"],
            bias=attention_conf["bias"],
            batch_first=True,
        )
        blocks.append(Block(attention, mlp))

    blocks = ModuleList(blocks)

    transformer_encoder = TransformerEncoder(blocks)

    pos_embedder = MLP(
        list(model_conf["pos_embedder"]["mlp_layers"]),
        act=model_conf["pos_embedder"]["act"],
        norm=None,
    )
    masked_encoder = MaskedEncoder(
        mask_ratio=model_conf["mask_ratio"],
        transformer_encoder=transformer_encoder,
        pos_embedder=pos_embedder,
        mask_type=model_conf["mask_type"],
    )

    blocks = []
    for _ in range(model_conf["decoder_depth"]):
        mlp = MLP(
            list(block_conf["mlp_layers"]),
            act=block_conf["act"],
            norm=None,
            dropout=block_conf["dropout"],
        )
        attention = MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=attention_conf["num_heads"],
            add_bias_kv=attention_conf["qkv_bias"],
            dropout=attention_conf["dropout"],
            bias=attention_conf["bias"],
            batch_first=True,
        )
        blocks.append(Block(attention, mlp))

    blocks = ModuleList(blocks)
    pos_embedder = MLP(
        list(model_conf["pos_embedder"]["mlp_layers"]),
        act=model_conf["pos_embedder"]["act"],
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
