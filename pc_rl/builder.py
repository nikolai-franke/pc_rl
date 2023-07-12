from collections.abc import Callable

import torch
import torch.nn as nn
from hydra.utils import instantiate
from parllel.torch.models import MlpModel
from torch.nn import MultiheadAttention
from torch_geometric.nn import MLP

from pc_rl.models.finetune_encoder import FinetuneEncoder
from pc_rl.models.mae import MaskedAutoEncoder
from pc_rl.models.modules.embedder import Embedder
from pc_rl.models.modules.masked_decoder import MaskedDecoder
from pc_rl.models.modules.mae_prediction_head import MaePredictionHead
from pc_rl.models.modules.masked_encoder import MaskedEncoder
from pc_rl.models.modules.transformer import (TransformerBlock,
                                              TransformerDecoder,
                                              TransformerEncoder)
from pc_rl.models.rl_finetune_categorical_pg import CategoricalPgModel


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


def build_transformer_encoder(
    transformer_block_factory: Callable[[], TransformerBlock], depth: int
):
    blocks = [transformer_block_factory() for _ in range(depth)]
    return TransformerEncoder(blocks)


def build_transformer_decoder(
    transformer_block_factory: Callable[[], TransformerBlock], depth: int
):
    blocks = [transformer_block_factory() for _ in range(depth)]
    return TransformerDecoder(blocks)


def build_pos_embedder(mlp_layers: list[int], act: type[nn.Module] | str) -> MLP:
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


def build_categorical_pg_model(
    embedder: Embedder,
    finetune_encoder: FinetuneEncoder,
    n_actions: int,
    pi_mlp_hidden_sizes: list[int],
    pi_mlp_act: type[nn.Module] | str,
    value_mlp_hidden_sizes: list[int],
    value_mlp_act: type[nn.Module] | str,
):
    input_size = finetune_encoder.out_dim

    pi_mlp = MlpModel(
        input_size=input_size,
        hidden_sizes=pi_mlp_hidden_sizes,
        # hidden_nonlinearity=activation_resolver(pi_mlp_act),
        hidden_nonlinearity=torch.nn.Tanh,
        output_size=n_actions,
    )

    value_mlp = MlpModel(
        input_size=input_size,
        hidden_sizes=value_mlp_hidden_sizes,
        hidden_nonlinearity=torch.nn.Tanh,
        output_size=1,
    )

    return CategoricalPgModel(
        embedder=embedder,
        encoder=finetune_encoder,
        pi_mlp=pi_mlp,
        value_mlp=value_mlp,
    )


def build_finetune_encoder(
    transformer_encoder: TransformerEncoder,
    pos_embedder: nn.Module,
    mlp_head_hidden_sizes: list[int],
    mlp_head_out_size: int,
    mlp_head_act: type[nn.Module] | str,
):
    mlp_head_sizes = [2 * transformer_encoder.dim] + mlp_head_hidden_sizes
    mlp_head_sizes.append(mlp_head_out_size)
    mlp_head = MLP(channel_list=mlp_head_sizes, act=mlp_head_act, norm="layer_norm")

    return FinetuneEncoder(
        transformer_encoder=transformer_encoder,
        pos_embedder=pos_embedder,
        mlp_head=mlp_head,
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
    prediction_head = MaePredictionHead(embedding_size, group_size)

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
        masked_encoder=masked_encoder,
        masked_decoder=masked_decoder,
        mae_prediction_head=prediction_head,
        learning_rate=config["learning_rate"],
    )

    return masked_autoencoder
