from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import torch
import torch.nn as nn
from parllel.torch.models import MlpModel
from torch.nn import MultiheadAttention
from torch_geometric.nn import MLP

from pc_rl.models.finetune_encoder import FinetuneEncoder
from pc_rl.models.modules.embedder import Embedder
from pc_rl.models.modules.masked_encoder import MaskedEncoder
from pc_rl.models.modules.transformer import (TransformerBlock,
                                              TransformerDecoder,
                                              TransformerEncoder)
from pc_rl.models.pg.aux_mae_categorical import AuxMaeCategoricalPgModel
from pc_rl.models.pg.aux_mae_continuous import AuxMaeContinuousPgModel
from pc_rl.models.pg.finetune_categorical import CategoricalPgModel
from pc_rl.models.pg.finetune_continuous import ContinuousPgModel


def build_embedder(
    embedding_size: int,
    mlp_1_layers: list[int],
    mlp_2_layers: list[int],
    mlp_act: str,
    group_size: int,
    sampling_ratio: float,
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


def build_continuous_pg_model(
    embedder: Embedder,
    finetune_encoder: FinetuneEncoder,
    n_actions: int,
    pi_mlp_hidden_sizes: list[int],
    pi_mlp_act: type[nn.Module] | str,
    pi_out_act: Optional[type[nn.Module] | str],
    value_mlp_hidden_sizes: list[int],
    value_mlp_act: type[nn.Module] | str,
    init_log_std: float,
):
    input_size = finetune_encoder.dim

    pi_hidden_nonlinearity = (
        getattr(torch.nn, pi_mlp_act) if isinstance(pi_mlp_act, str) else pi_mlp_act
    )

    pi_mlp = MlpModel(
        input_size=input_size,
        hidden_sizes=pi_mlp_hidden_sizes,
        hidden_nonlinearity=pi_hidden_nonlinearity,
        output_size=n_actions,
    )
    if pi_out_act is not None:
        pi_out_nonlinearity = (
            getattr(torch.nn, pi_out_act) if isinstance(pi_out_act, str) else pi_out_act
        )
        pi_mlp = nn.Sequential(pi_mlp, pi_out_nonlinearity())

    value_hidden_nonlinearity = (
        getattr(torch.nn, value_mlp_act)
        if isinstance(value_mlp_act, str)
        else value_mlp_act
    )
    value_mlp = MlpModel(
        input_size=input_size,
        hidden_sizes=value_mlp_hidden_sizes,
        hidden_nonlinearity=value_hidden_nonlinearity,
        output_size=1,
    )
    return ContinuousPgModel(
        embedder=embedder,
        encoder=finetune_encoder,
        pi_mlp=pi_mlp,
        value_mlp=value_mlp,
        init_log_std=init_log_std,
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
    input_size = finetune_encoder.dim

    pi_hidden_nonlinearity = (
        getattr(torch.nn, pi_mlp_act) if isinstance(pi_mlp_act, str) else pi_mlp_act
    )

    pi_mlp = MlpModel(
        input_size=input_size,
        hidden_sizes=pi_mlp_hidden_sizes,
        hidden_nonlinearity=pi_hidden_nonlinearity,
        output_size=n_actions,
    )

    value_hidden_nonlinearity = (
        getattr(torch.nn, value_mlp_act)
        if isinstance(value_mlp_act, str)
        else value_mlp_act
    )

    value_mlp = MlpModel(
        input_size=input_size,
        hidden_sizes=value_mlp_hidden_sizes,
        hidden_nonlinearity=value_hidden_nonlinearity,
        output_size=1,
    )

    return CategoricalPgModel(
        embedder=embedder,
        encoder=finetune_encoder,
        pi_mlp=pi_mlp,
        value_mlp=value_mlp,
    )


def build_aux_categorical_pg_model(
    embedder: Embedder,
    aux_mae: AuxMae,
    n_actions: int,
    pi_mlp_hidden_sizes: list[int],
    pi_mlp_act: type[nn.Module] | str,
    value_mlp_hidden_sizes: list[int],
    value_mlp_act: type[nn.Module] | str,
):
    input_size = aux_mae.dim
    pi_hidden_nonlinearity = (
        getattr(torch.nn, pi_mlp_act) if isinstance(pi_mlp_act, str) else pi_mlp_act
    )
    pi_mlp = MlpModel(
        input_size=input_size,
        hidden_sizes=pi_mlp_hidden_sizes,
        # hidden_nonlinearity=activation_resolver(pi_mlp_act),
        hidden_nonlinearity=pi_hidden_nonlinearity,
        output_size=n_actions,
    )

    value_hidden_nonlinearity = (
        getattr(torch.nn, value_mlp_act)
        if isinstance(value_mlp_act, str)
        else value_mlp_act
    )
    value_mlp = MlpModel(
        input_size=input_size,
        hidden_sizes=value_mlp_hidden_sizes,
        hidden_nonlinearity=value_hidden_nonlinearity,
        output_size=1,
    )

    return AuxMaeCategoricalPgModel(
        embedder=embedder,
        aux_mae=aux_mae,
        pi_mlp=pi_mlp,
        value_mlp=value_mlp,
    )


def build_aux_continuous_pg_model(
    embedder: Embedder,
    aux_mae: AuxMae,
    n_actions: int,
    pi_mlp_hidden_sizes: list[int],
    pi_mlp_act: type[nn.Module] | str,
    pi_out_act: Optional[type[nn.Module] | str],
    value_mlp_hidden_sizes: list[int],
    value_mlp_act: type[nn.Module] | str,
    init_log_std: float,
):
    input_size = aux_mae.dim
    pi_hidden_nonlinearity = (
        getattr(torch.nn, pi_mlp_act) if isinstance(pi_mlp_act, str) else pi_mlp_act
    )
    pi_mlp = MlpModel(
        input_size=input_size,
        hidden_sizes=pi_mlp_hidden_sizes,
        # hidden_nonlinearity=activation_resolver(pi_mlp_act),
        hidden_nonlinearity=pi_hidden_nonlinearity,
        output_size=n_actions,
    )

    if pi_out_act is not None:
        pi_out_nonlinearity = (
            getattr(torch.nn, pi_out_act) if isinstance(pi_out_act, str) else pi_out_act
        )
        pi_mlp = nn.Sequential(pi_mlp, pi_out_nonlinearity())

    value_hidden_nonlinearity = (
        getattr(torch.nn, value_mlp_act)
        if isinstance(value_mlp_act, str)
        else value_mlp_act
    )
    value_mlp = MlpModel(
        input_size=input_size,
        hidden_sizes=value_mlp_hidden_sizes,
        hidden_nonlinearity=value_hidden_nonlinearity,
        output_size=1,
    )

    return AuxMaeContinuousPgModel(
        embedder=embedder,
        aux_mae=aux_mae,
        pi_mlp=pi_mlp,
        value_mlp=value_mlp,
        init_log_std=init_log_std,
    )
