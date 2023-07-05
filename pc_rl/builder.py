from hydra.utils import instantiate
from torch.nn import MultiheadAttention
from torch_geometric.nn import MLP

from pc_rl.models.masked_autoencoder import MaskedAutoEncoder
from pc_rl.models.modules.embedder import Embedder
from pc_rl.models.modules.transformer import (TransformerBlock,
                                              TransformerDecoder,
                                              TransformerEncoder)
from pc_rl.models.modules.mae import MaskedEncoder, MaskedDecoder, PredictionHead


def build_embedder(
    embedding_size: int,
    mlp_1_layers: list[int],
    mlp_2_layers: list[int],
    mlp_act: str,
    group_size: int,
    sampling_ratio: int,
    random_start: bool,
) -> Embedder:
    mlp_1 = MLP(mlp_1_layers, act=mlp_act)
    mlp_2_layers.append(embedding_size)
    mlp_2 = MLP(mlp_2_layers, act=mlp_act)
    return Embedder(
        mlp_1=mlp_1,
        mlp_2=mlp_2,
        group_size=group_size,
        sampling_ratio=sampling_ratio,
        random_start=random_start,
    )


def build_transformer_block(
    embedding_size: int,
    dropout: float,
    mlp_ratio: int,
    mlp_act: str,
    attention_num_heads: int,
    attention_bias: bool,
    attention_qkv_bias: bool,
) -> TransformerBlock:
    mlp_layers = [embedding_size, int(mlp_ratio * embedding_size), embedding_size]
    mlp = MLP(
        mlp_layers,
        act=mlp_act,
        norm=None,
        dropout=dropout,
    )
    attention = MultiheadAttention(
        embed_dim=embedding_size,
        num_heads=attention_num_heads,
        add_bias_kv=attention_qkv_bias,
        bias=attention_bias,
        dropout=dropout,
        batch_first=True,
    )
    return TransformerBlock(attention, mlp)


def build_pos_embedder(mlp_layers: list[int], act: str) -> MLP:
    return MLP(mlp_layers, act=act, norm=None)


def build_masked_encoder(
    mask_ratio: float,
    mask_type: str,
    transformer_encoder: TransformerEncoder,
    pos_embedder: MLP,
):
    return MaskedEncoder(
        mask_ratio=mask_ratio,
        mask_type=mask_type,
        transformer_encoder=transformer_encoder,
        pos_embedder=pos_embedder,
    )


def build_masked_autoencoder_modules(config):
    embedder_conf = config["model"]["embedder"]
    embedding_size = embedder_conf["embedding_size"]
    group_size = embedder_conf["group_size"]

    embedder = instantiate(config.model.embedder, _convert_="partial")
    model_conf = config["model"]
    blocks = []

    for _ in range(model_conf["encoder_depth"]):
        block = instantiate(
            config.model.transformer_block, embedding_size=embedding_size
        )
        blocks.append(block)

    transformer_encoder = TransformerEncoder(blocks)

    pos_embedder = instantiate(config.model.pos_embedder, _convert_="partial")

    masked_encoder = instantiate(
        config.model.masked_encoder,
        transformer_encoder=transformer_encoder,
        pos_embedder=pos_embedder,
    )

    blocks = []
    for _ in range(model_conf["decoder_depth"]):
        block = instantiate(
            config.model.transformer_block, embedding_size=embedding_size
        )
        blocks.append(block)

    pos_embedder = instantiate(config.model.pos_embedder)
    transformer_decoder = TransformerDecoder(blocks)
    masked_decoder = MaskedDecoder(transformer_decoder, pos_embedder)
    prediction_head = PredictionHead(embedding_size, group_size)

    return embedder, masked_encoder, masked_decoder, prediction_head


def build_masked_autoencoder(config):
    (
        embedder,
        masked_encoder,
        masked_decoder,
        prediction_head,
    ) = build_masked_autoencoder_modules(config)
    masked_autoencoder = MaskedAutoEncoder(
        embedder=embedder,
        encoder=masked_encoder,
        decoder=masked_decoder,
        prediction_head=prediction_head,
        learning_rate=config["learning_rate"],
    )

    return masked_autoencoder
